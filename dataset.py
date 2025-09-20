# import packages
from datasets import load_dataset
import os
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np



# constants
train_bin_path= "train.bin"
total_batches = 1024



# load dataset and tokenizer
ds = load_dataset("roneneldan/TinyStories")
tokenizer = tiktoken.get_encoding("gpt2")



# function to tokenize each row of the dataset
def tokenize_each_row(row):
  text= row["text"]
  ids = tokenizer.encode_ordinary(text)
  out = {'ids': ids, 'len': len(ids)}
  return out


# custom PyTorch dataset for handling tokenized sequences
class GemmaDataset(Dataset):
    def __init__(self, split: str, context_size: int, stride: int):
        """
        Initializes the dataset.

        Args:
            split (str): name of the data split (e.g., 'train', 'validation')
            context_size (int): length of sequence to feed into the model
            stride (int): step size when moving window across tokenized data
        """
        
        
        self.context_size = context_size
        self.stride = stride
        
        # memory-map the binary token file for efficient access
        self.token_ids = np.memmap(f"{split}.bin", dtype=np.uint16, mode="r")

    def __len__(self):
        # number of samples from stride
        return (len(self.token_ids) - self.context_size) // self.stride

    def __getitem__(self, index):
        """
        Returns a single training sample (x, y) from the dataset.

        Args:
            index (int): index of the sample

        Returns:
            tuple: x (input tokens), y (target tokens shifted by 1)
        """
        
        
        i = index * self.stride
        
        # input sequence
        x = torch.tensor(self.token_ids[i : i+self.context_size].astype(np.int64))
        
        # target sequence (next token prediction)
        y = torch.tensor(self.token_ids[i+1 : i+self.context_size+1].astype(np.int64))
        
        return x, y



# load the data loader from the gemma dataset
def get_dataloader(split: str, pin_memory: bool, context_size: int= 128, stride: int= 128, batch_size: int= 32, shuffle: bool= True):
    """
    Returns a DataLoader for the specified split.

    Args:
        split (str): dataset split name
        pin_memory (bool): whether to pin memory (useful for GPU training)
        context_size (int): length of sequences
        stride (int): stride between sequences
        batch_size (int): number of sequences per batch
        shuffle (bool): whether to shuffle dataset

    Returns:
        DataLoader: iterable PyTorch DataLoader
    """
    
    
    
    # get the dataset
    dataset= GemmaDataset(split, context_size, stride)
    
    # get the dataloader
    dataloader= DataLoader(dataset, batch_size= batch_size, shuffle= shuffle, pin_memory= pin_memory)
    
    return dataloader
    


# main execution
if __name__ == "__main__":
    
    # if train.bin doesn't exist, preprocess the dataset and save tokenized ids
    if not os.path.exists(train_bin_path):
        
        # tokenize the entire dataset
        tokenized = ds.map(
            tokenize_each_row,
            remove_columns=['text'],
            desc="Tokenizing the splits",
            num_proc=8,
        )
        
        # iterate over each split (train, validation, etc.)
        for split, dset in tokenized.items():
            
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = f'{split}.bin'
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

            idx = 0
            
            # write token ids in batches for efficiency
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                
                arr[idx : idx + len(arr_batch)] = arr_batch
                
                idx += len(arr_batch)
                
            arr.flush()
            
            
    
                