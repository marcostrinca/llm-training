import sys
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import tiktoken
import argparse
sys.path.append('/home/ubuntu/llm-training/config')
from config_500k import default_config as config
sys.path.append('/home/ubuntu/llm-training/src')
from models.transformer import Transformer, TransformerConfig  # Assuming your Transformer class is in this module

# --- Create the configuration ---
t_config = TransformerConfig(
        vocab_size = config['vocab_size'],
        context_length = config['context_length'],
        n_embed = config['n_embed'],
        n_head = config['n_head'],
        N_BLOCKS = config['n_blocks']
        )

def generate_text(model_path: str, input_text: str, max_new_tokens: int = 100, device: str = 'cuda') -> str:
    """
    Generates text using a pre-trained Transformer model.

    Args:
        model_path (str): Path to the saved model checkpoint.
        input_text (str): The initial text to start generation from.
        max_new_tokens (int): The maximum number of new tokens to generate.
        device (str): 'cuda' or 'cpu', the device to run the model on.

    Returns:
        str: The generated text.
    """

    # Initialize the model using the configuration from config.py and Accelerator
    with init_empty_weights():
        model = Transformer(t_config)

    model = load_checkpoint_and_dispatch(
        model, checkpoint=model_path, device_map="sequential"
    )
    model.eval().to(device)
    # for i in model.named_parameters():
        # print(f"{i[0]} -> {i[1].device}")

    # Load the tokenizer
    enc = tiktoken.get_encoding("r50k_base")

    start_ids = enc.encode_ordinary(input_text)
    context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)


    # Generation process
    with torch.no_grad():
        print(model.device)
        print(context.device)
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()


    # Decode the generated tokens
    output_text = enc.decode(generated_tokens)

    return output_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained Transformer model.")
    parser.add_argument('--model_path', type=str, help='Path to the saved model checkpoint.')
    parser.add_argument('--input_text', type=str, help='The initial text to start generation from.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate.')

    args = parser.parse_args()

    generated = generate_text(args.model_path, args.input_text, args.max_new_tokens)
    print(f"Generated text:\n{generated}")

if __name__ == "__main__":
    main()
