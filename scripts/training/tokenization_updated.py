import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
#####
#import os
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import numpy as np
from typing import List
from chronos import ChronosConfig, ChronosTokenizer
from tqdm import tqdm
import pandas as pd 

class ChronosEpochTokenizer:
    """
    Dataset wrapper for tokenizing each 30-second epoch into 512 tokens and including labels.
    """

    def __init__(
        self,
        arrow_file_path: str,
        tokenizer: ChronosTokenizer,
        token_length: int = 512,
        np_dtype=np.float32,
    ) -> None:
        assert os.path.exists(arrow_file_path), f"Arrow file not found: {arrow_file_path}"

        # Open the Arrow file and read all batches
        with pa.memory_map(arrow_file_path, "r") as source:
            reader = ipc.RecordBatchFileReader(source)

            # Load all batches
            tables = []
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                tables.append(batch.to_pandas())

            self.dataset = pd.concat(tables, ignore_index=True)

        self.tokenizer = tokenizer
        self.token_length = token_length
        self.np_dtype = np_dtype


    def preprocess_entry(self, eeg_epochs: List[List[float]], labels: List[str], file_name: str) -> List[dict]:
        """
        Preprocess and tokenize each epoch in eeg_epochs, including labels.
        """
        tokenized_epochs = []

        for epoch, label in zip(eeg_epochs, labels):
            input_ids, attention_mask = self.tokenize_epoch(epoch)
            tokenized_epochs.append({
                "file_name": file_name,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label,  # Add corresponding label
            })

        return tokenized_epochs

    def tokenize_epoch(self, epoch: List[float]) -> tuple:
        """
        Tokenize a single epoch into 512 tokens.
        """
        epoch_tensor = torch.tensor(epoch, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 3000)
        input_ids, attention_mask, _ = self.tokenizer.context_input_transform(epoch_tensor)

        # Ensure tokenization produces the correct number of tokens
        if input_ids.size(1) != self.token_length:
            raise ValueError(
                f"Tokenizer produced {input_ids.size(1)} tokens, expected {self.token_length}"
            )

        return input_ids.squeeze(0), attention_mask.squeeze(0)

    def __iter__(self):
        # Iterate over rows in the dataset
        for _, row in self.dataset.iterrows():
            file_name = row["file_name"]  # Source file name
            eeg_epochs = row["eeg_epochs"]  # EEG epochs
            labels = row.get("labels", ["unknown"] * len(eeg_epochs))  # Labels or default to "unknown"

            # Ensure the number of labels matches the number of epochs
            if len(labels) != len(eeg_epochs):
                print(f"Warning: Mismatch in number of labels and epochs for {file_name}. Padding labels.")
                labels = labels[:len(eeg_epochs)] + ["unknown"] * (len(eeg_epochs) - len(labels))

            yield from self.preprocess_entry(eeg_epochs, labels, file_name)


def main_tokenization():
    # Input and output paths
    arrow_file_path = "/srv/scratch/z5298768/chronos_classification_local/prepare_time_series/nch_sleep_data_18plus.arrow"
    output_dir = "/srv/scratch/z5298768/chronos_classification_local/BERT_tokenization_18plus
    "
    os.makedirs(output_dir, exist_ok=True)
    

    # Tokenizer Configuration
    context_length = 512  # Length of each sequence (512 tokens) #Longformer: 4096
    n_tokens = 512 
    tokenizer_class = "MeanScaleUniformBins"
    # tokenizer_kwargs = {"low_limit": -15.0, "high_limit": 15.0}
    tokenizer_kwargs = {"low_limit": -20.0, "high_limit": 20.0}

    # Required parameters for ChronosConfig
    prediction_length = 0  # Not needed for classification
    n_special_tokens = 2  # Typically PAD and EOS tokens
    pad_token_id = 0  # Padding token ID
    eos_token_id = 1  # End-of-sequence token ID
    use_eos_token = False  # Whether to use EOS tokens
    num_samples = 1  # For stochastic sampling
    temperature = 1.0  # Sampling temperature (not used here)
    top_k = 50  # Top-k sampling (not used here)
    top_p = 1.0  # Top-p sampling (not used here)
    model_type = "classification"  # Task type

    # Initialize tokenizer
    tokenizer = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=use_eos_token,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        context_length=context_length,  # Correctly use context_length
        model_type=model_type,          # Set to "classification"
    ).create_tokenizer()
    
    # if hasattr(tokenizer, "id_to_token"):  
    #     print("Token 3 corresponds to:", tokenizer.id_to_token(3))
    # elif hasattr(tokenizer, "decode"):
    #     print("Token 3 corresponds to:", tokenizer.decode([3]))
    # else:
    #     print("No method found for converting token 3.")


    # Initialize dataset
    dataset = ChronosEpochTokenizer(
        arrow_file_path=arrow_file_path,
        tokenizer=tokenizer,
        token_length=context_length,  # Use context_length here as well
    )


    # Chunked tokenization
    chunk_size = 500_000
    tokenized_data = []
    chunk_index = 0
    total_count = 0

    for i, item in enumerate(tqdm(dataset, desc="Tokenizing epochs", unit="epoch")):
        tokenized_data.append(item)
        total_count += 1

        if len(tokenized_data) >= chunk_size:
            chunk_path = os.path.join(output_dir, f"2hrs_tokenized_chunk_{chunk_index}.pt")
            torch.save(tokenized_data, chunk_path)
            print(f"Saved chunk {chunk_index} with {len(tokenized_data)} entries to {chunk_path}")
            tokenized_data = []
            chunk_index += 1

    # Save remaining data
    if tokenized_data:
        chunk_path = os.path.join(output_dir, f"2hrs_tokenized_chunk_{chunk_index}.pt")
        torch.save(tokenized_data, chunk_path)
        print(f"Saved final chunk {chunk_index} with {len(tokenized_data)} entries to {chunk_path}")

    print(f"Tokenization complete. Total tokenized epochs: {total_count}")


if __name__ == "__main__":
    main_tokenization()
