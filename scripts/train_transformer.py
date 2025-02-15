import torch
import os, sys
from tqdm import tqdm
import numpy as np

sys.path.append('/home/ubuntu/llm-training/config')
from config_500k import default_config as config
# from config13M import default_config as config
sys.path.append('/home/ubuntu/llm-training/src')
from models.transformer import Transformer, TransformerConfig
sys.path.append('/home/ubuntu/llm-training/data_loader')
from data_loader import get_batch_iterator
from typing import Dict

from accelerate import Accelerator
accel = Accelerator()

# --- Create the configuration ---
t_config = TransformerConfig(
        vocab_size = config['vocab_size'],
        context_length = config['context_length'],
        n_embed = config['n_embed'],
        n_head = config['n_head'],
        N_BLOCKS = config['n_blocks']
        )

# --- Initialize the Model and send to Accelerator ---
model = Transformer(t_config)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params:,}")
model = accel.prepare(model)

# --- Optimizer Setup and Loss Tracking ---
# Set up the AdamW optimizer with the specified learning rate and send to Accelerator
optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])
optimizer = accel.prepare(optimizer)

# List to track loss values during training.
losses = []

# Define a window size for averaging recent losses in the training loop.
AVG_WINDOW = 64

# Helper function to estimate the average loss for training and development data.
@torch.no_grad()
def estimate_loss(steps: int) -> Dict[str, float]:
    """
    Evaluate the model on training and development datasets and calculate average loss.

    Args:
        steps (int): Number of steps to evaluate.

    Returns:
        dict: Dictionary containing average losses for 'train' and 'dev' splits.
    """
    out = {}
    model.eval()  # Set the model to evaluation mode.

    for split in ['train', 'dev']:
    # for split in ['train']:
        # Select the appropriate data path for the current split.
        data_path = config['train_path'] if split == 'train' else config['dev_path']

        # Create a batch iterator for evaluation.
        batch_iterator_eval = get_batch_iterator(
            data_path, config['t_batch_size'], config['t_context_length'], device=config['device']
        )

        # Initialize a tensor to track loss values for each evaluation step.
        losses_eval = torch.zeros(steps)
        for k in range(steps):
            try:
                # Fetch a batch and calculate the loss.
                xb, yb = next(batch_iterator_eval)
                _, loss = model(xb, yb)
                losses_eval[k] = loss.item()
                # losses_eval[k] = loss.sum().item() # tu support multi gpu
            except StopIteration:
                # Handle the case where the data iterator ends early.
                print(f"Warning: Iterator for {split} ended early.")
                break

        # Compute the mean loss for the current split.
        out[split] = losses_eval[:k + 1].mean()

    model.train()  # Restore the model to training mode.
    return out

# --- To save the model ---
def save_model(steps):

    # Create the output directory if it does not exist.
    os.makedirs(config['t_out_path'].split('/')[0], exist_ok=True)

    # Ensure unique model save path in case the file already exists.
    modified_model_out_path = config['t_out_path']
    model_out_name = os.path.splitext(config['t_out_path'])[0]
    modified_model_out_path = model_out_name + f"_{steps}" + ".pt"

    accel.unwrap_model(model).save_pretrained(
              modified_model_out_path,
              is_main_process=accel.is_main_process,
              save_function=accel.save,
              state_dict=accel.get_state_dict(model)
    )
    print(f"Saved model to {modified_model_out_path}")

# --- Training Loop ---

# Create a batch iterator for the training data.
batch_iterator = get_batch_iterator(
    config['train_path'],
    config['t_batch_size'],
    config['t_context_length'],
    device=config['device']
)

# Create a progress bar to monitor training progress.
pbar = tqdm(range(config['t_train_steps']))
for step in pbar:
    try:
        # Fetch a batch of input and target data.
        xb, yb = next(batch_iterator)

        # Perform a forward pass and compute the loss.
        _, loss = model(xb, yb)

        # Record the loss for tracking.
        losses.append(loss.item())
        pbar.set_description(f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")

        # Backpropagate the loss and update the model parameters.
        optimizer.zero_grad(set_to_none=True)
        accel.backward(loss)
        optimizer.step()

        # Periodically evaluate the model on training and development data.
        if step % config['t_eval_steps'] == 0:
            evaluation_losses = estimate_loss(config['t_eval_iters'])
            train_loss = evaluation_losses['train']
            dev_loss = evaluation_losses['dev']
            print(f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss} ")

        # Decay the learning rate at the specified step.
        if step == config['t_lr_decay_step']:
            print('Decaying learning rate')
            for g in optimizer.param_groups:
                g['lr'] = config['t_lr_decayed']
        
        # Save the model each 20k steps
        if step % 20000 == 0:
            save_model(step)

    except StopIteration:
        # Handle the case where the training data iterator ends early.
        print("Training data iterator finished early.")
        break

save_model()
