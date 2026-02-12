from datasets import load_from_disk
import pprint
import os

# The dataset was created in the parent directory
# So we construct the path relative to the current script's location
script_dir = os.path.dirname(__file__)
dataset_path = os.path.join(script_dir, "jenny-tts-tags-6h")

print(f"Loading dataset from: {dataset_path}")

try:
    # Load the processed dataset from the disk
    dataset = load_from_disk(dataset_path)

    # Print the dataset structure and features
    print("\n--- Dataset Info ---")
    print(dataset)

    # Get the first split (usually 'train')
    split_name = next(iter(dataset))
    print(f"\n--- First example from '{split_name}' split ---")

    # Pretty-print the first example to see the new columns
    first_example = dataset[split_name][0]
    pprint.pprint(first_example)

    print("\n--- Available columns ---")
    print(dataset[split_name].column_names)

except FileNotFoundError:
    print(f"\nError: Dataset not found at '{dataset_path}'")
    print("Please ensure the path is correct and the dataset directory exists.")

