import os
import json
import zstandard as zstd
import tiktoken
import h5py
from tqdm import tqdm
import argparse
from typing import Optional

from datasets import load_dataset, ReadInstruction

ds = load_dataset("bastao/VeraCruz_PT-BR", "Brazil (BR)", split="train", streaming=True)
print(ds)

def process_files(input_dir: str, output_file: str, tokenizer_name: str, max_data: Optional[int] = None) -> None:
    """
    Process a specified number of lines from VeraCruz dataset
    and save encoded tokens to an HDF5 file.

    Args:
        input_dir (str): Directory containing input .jsonl.zst files.
        output_file (str): Path to the output HDF5 file.
        tokenizer_name (str): Name of the tiktoken tokenizer to use (e.g., 'r50k_base').
        max_data (int, optional): Maximum number of lines to process from each file.
                                  If None, process all lines.
    """
    # Print processing strategy based on max_data
    if max_data is not None:
        print(f"You have chosen max_data = {max_data}. Processing only the top {max_data} lines.")
    else:
        print("Processing all available lines")

    # Load the tokenizer using the provided tokenizer name
    enc = tiktoken.get_encoding(tokenizer_name)

    # Create an HDF5 file for output
    with h5py.File(output_file, 'w') as out_f:
        # Initialize the dataset for storing tokenized data
        dataset = out_f.create_dataset('tokens', (0,), maxshape=(None,), dtype='i')
        start_index = 0  # Track the starting index for the next batch of tokens

        processed_lines = 0  # Counter for processed lines in the current file

        for line in tqdm(ds, desc=f"Processing...", total=max_data if max_data is not None else None):
            try:
                # Parse the line as JSON
                # data = json.loads(line)
                # text = data.get('text')  # Extract the 'text' field from the JSON object
                text = line['text']

                if text:
                    # Tokenize the text and append an end-of-text token
                    encoded = enc.encode(text + "<|endoftext|>", allowed_special={'<|endoftext|>'})
                    encoded_len = len(encoded)

                    # Resize the dataset to accommodate new tokens
                    end_index = start_index + encoded_len
                    dataset.resize(dataset.shape[0] + encoded_len, axis=0)

                    # Store the encoded tokens in the dataset
                    dataset[start_index:end_index] = encoded
                    start_index = end_index  # Update the start index
                else:
                    # Warn if 'text' key is missing in the JSON object
                    print(f"Warning: 'text' key missing in line ")
            except Exception as e:
                # Handle any other errors
                print(f"An error occurred while processing line   {e}")

            processed_lines += 1
            # Stop processing if max_data limit is reached
            if max_data is not None and processed_lines >= max_data:
                break

def main():
    """
    Main function to parse arguments, validate directories, and process files.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess PILE dataset files and save tokens to HDF5.")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory containing training .jsonl.zst files.")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Directory containing validation .jsonl.zst files.")
    parser.add_argument("--out_train_file", type=str, default="data/train/pile_train_veracruz.h5", help="Path to the output training HDF5 file.")
    parser.add_argument("--out_val_file", type=str, default="data/val/pile_dev_veracruz.h5", help="Path to the output validation HDF5 file.")
    parser.add_argument("--tokenizer_name", type=str, default="r50k_base", help="Name of the tiktoken tokenizer to use.")
    parser.add_argument("--max_data", type=int, default=1000, help="Maximum number of json objects to process from each file in both train and val datasets (default: 1000).")

    args = parser.parse_args()

    # Validate the existence of the training and validation directories
    if not os.path.isdir(args.train_dir):
        print(f"Error: Training directory not found: {args.train_dir}")
        return
    if not os.path.isdir(args.val_dir):
        print(f"Error: Validation directory not found: {args.val_dir}")
        return

    # Process training data
    print("Starting training data preprocessing...")
    process_files(args.train_dir, args.out_train_file, args.tokenizer_name, args.max_data)
    print("Training data preprocessing complete.")

    # Process validation data
    print("Starting validation data preprocessing...")
    process_files(args.val_dir, args.out_val_file, args.tokenizer_name, args.max_data)
    print("Validation data preprocessing complete.")

# Entry point of the script
if __name__ == "__main__":
    main()
