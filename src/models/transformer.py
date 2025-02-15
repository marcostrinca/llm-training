import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer_block import Block
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig

class TransformerConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self, 
        context_length: int = 512,
        vocab_size: int = 50304,
        n_embed: int = 128,
        n_head: int = 8,
        N_BLOCKS: int = 2,
        **kwargs):

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.N_BLOCKS = N_BLOCKS
        super().__init__(**kwargs)


# class Transformer(nn.Module):
class Transformer(PreTrainedModel):
    config_class = TransformerConfig

    """
    The main Transformer model.

    This class combines token and position embeddings with a sequence of Transformer blocks
    and a final linear layer for language modeling.

    Args:
        n_head (int): The number of attention heads in each transformer block.
        n_embed (int): The dimensionality of the embedding space.
        context_length (int): The maximum length of the input sequence.
        vocab_size (int): The size of the vocabulary.
        N_BLOCKS (int): The number of transformer blocks in the model.
    """
    def __init__(self, config) -> None:
        """
        Initializes the Transformer model.

        Args:
            n_head (int): Number of attention heads.
            n_embed (int): Embedding dimension.
            context_length (int): Maximum sequence length.
            vocab_size (int): Size of the vocabulary.
            N_BLOCKS (int): Number of transformer blocks.
        """
        super().__init__(config)
        self.context_length = config.context_length
        self.N_BLOCKS = config.N_BLOCKS
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embed = nn.Embedding(config.context_length, config.n_embed)
        self.attn_blocks = nn.ModuleList([Block(config.n_head, config.n_embed, config.context_length) for _ in range(config.N_BLOCKS)])
        self.layer_norm = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        self.register_buffer('pos_idxs', torch.arange(config.context_length))

    def _pre_attn_pass(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Combines token and position embeddings.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Sum of token and position embeddings.
        """
        B, T = idx.shape
        tok_embedding = self.token_embed(idx)
        pos_embedding = self.position_embed(self.pos_idxs[:T])
        return tok_embedding + pos_embedding

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the Transformer.

        Args:
            idx (torch.Tensor): Input token indices.
            targets (torch.Tensor, optional): Target token indices for loss calculation. Defaults to None.

        Returns:
            tuple: Logits and loss (if targets are provided).
        """
        x = self._pre_attn_pass(idx)
        for block in self.attn_blocks:
            x = block(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            flat_logits = logits.view(B * T, C)
            targets = targets.view(B * T).long()
            loss = F.cross_entropy(flat_logits, targets)
        return logits, loss

    def forward_embedding(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass focusing on the embedding and attention blocks.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            tuple: Output after attention blocks and the residual.
        """
        x = self._pre_attn_pass(idx)
        residual = x
        for block in self.attn_blocks:
            x, residual = block.forward_embedding(x)
        return x, residual

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens given a starting sequence.

        Args:
            idx (torch.Tensor): Initial sequence of token indices.
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            torch.Tensor: The extended sequence of tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

