import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class UnlabeledTextDataset(Dataset):
    def __init__(self, json_path, tokenizer=None, max_length=512):
        """
        Args:
            json_path (str): Path to the JSON file.
            tokenizer (callable, optional): HuggingFace tokenizer (e.g., BertTokenizer).
                                            If None, returns raw text.
            max_length (int): Maximum sequence length for tokenization.
        """
        # Load the data using the orientation specified in your snippet
        self.data = pd.read_json(json_path, orient='index')
        
        # Filter out rows where 'label' is NOT NaN (Keep only unlabeled data for SSL)
        self.data = self.data[self.data['label'].isna()]
        
        # Filter out rows with missing transcriptions if necessary
        self.data = self.data[self.data['transcription'].notna()]
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Get the Filename (Index) and the Row Data
        # We use .iloc because Dataset expects integer indexing, 
        # but your DataFrame index is strings (filenames).
        video_filename = self.data.index[idx]
        row = self.data.iloc[idx]
        
        text = row['transcription']
        chunks = row['timestamped_chunks']
        
        # 2. Prepare the output
        sample = {
            'video_id': video_filename,
            'raw_text': text,
            'chunks': chunks # specific to your data structure
        }

        # 3. Tokenize if a tokenizer is provided (Standard SSL workflow)
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Squeeze removes the batch dimension added by the tokenizer (1, seq_len) -> (seq_len)
            sample['input_ids'] = encoding['input_ids'].squeeze(0)
            sample['attention_mask'] = encoding['attention_mask'].squeeze(0)

        return sample

# --- Usage Example ---
if __name__ == "__main__":
    # Mocking the path for demonstration
    json_path = os.path.join('data', 'text_data_all.json')
    
    # 1. Initialize Dataset
    # Assuming the file exists, otherwise this will error in this script
    try:
        dataset = UnlabeledTextDataset(json_path)
        
        # 2. Test the first item
        print(f"Dataset size: {len(dataset)}")
        sample = dataset[0]
        
        print("\n--- Sample 0 ---")
        print(f"ID: {sample['video_id']}")
        print(f"Text Preview: {sample['raw_text']}...")
        print(f"Chunks Count: {len(sample['chunks'])}")
        
        # 3. Example with DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
    except ValueError as e:
        print("Error: Could not find or parse the JSON file. Make sure 'data/text_data_all.json' exists.")
