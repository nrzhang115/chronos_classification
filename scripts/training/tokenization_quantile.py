import os
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import numpy as np
from typing import List
from chronos import ChronosConfig, ChronosTokenizer
from tqdm import tqdm

class ChronosEpochTokenizer:
    """
    Dataset wrapper for tokenizing each 30-second epoch into 512 tokens using quantile-based binning.
    """
    def __init__(
        self,
        arrow_file_path: str,
        tokenizer: ChronosTokenizer,
        num_bins: int = 4096,
    ) -> None:
        assert os.path.exists(arrow_file_path), f"Arrow file not found: {arrow_file_path}"

        # Open the Arrow file using IPC
        with pa.memory_map(arrow_file_path, "r") as source:
            reader = ipc.RecordBatchFileReader(source)
            record_batch = reader.get_batch(0)
            self.dataset = record_batch.to_pandas()  # Load data into a DataFrame

        self.tokenizer = tokenizer
        self.num_bins = num_bins
        
        # Compute quantile-based bin edges from all EEG data
        all_eeg_values = np.concatenate(self.dataset["eeg_epochs"].to_list()).flatten()
        self.bin_edges = self.compute_quantile_bins(all_eeg_values)

    def compute_quantile_bins(self, eeg_data):
        """
        Computes bin edges using quantiles for adaptive binning.
        """
        bin_edges = np.quantile(eeg_data, np.linspace(0, 1, self.num_bins + 1))
        return bin_edges

    def tokenize_epoch(self, epoch: List[float]) -> tuple:
        """
        Tokenize a single epoch using quantile binning.
        """
        token_ids = np.digitize(epoch, self.bin_edges) - 1  # Convert to 0-based token IDs
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)  # All tokens are valid

        return input_ids, attention_mask

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

    def __iter__(self):
        # Iterate over rows in the dataset
        for _, row in self.dataset.iterrows():
            file_name = row["file_name"]  # Source file name
            eeg_epochs = row["eeg_epochs"]  # EEG epochs
            labels = row.get("labels", ["unknown"] * len(eeg_epochs))  # Labels or default to "unknown"
            yield from self.preprocess_entry(eeg_epochs, labels, file_name)


def main_tokenization():
    arrow_file_path = "/srv/scratch/z5298768/chronos_classification/prepare_time_seires/C4-M1_updated/nch_sleep_data_selected.arrow"
    output_dir = "/srv/scratch/z5298768/chronos_classification/tokenization_updated"
    os.makedirs(output_dir, exist_ok=True)
    
    context_length = 512  # Length of each sequence (512 tokens)
    n_tokens = 4096  # Number of quantile bins
    tokenizer_class = "MeanScaleUniformBins"
    num_bins = 4096  # Ensure consistency in quantization
    
    # Initialize tokenizer
    tokenizer = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs={},  # No predefined limits, using quantile binning instead
        n_tokens=n_tokens,
        context_length=context_length,
        model_type="classification",
    ).create_tokenizer()

    # Initialize dataset with quantile binning
    dataset = ChronosEpochTokenizer(
        arrow_file_path=arrow_file_path,
        tokenizer=tokenizer,
        num_bins=num_bins,
    )

    # Tokenize and save
    tokenized_data = []
    for item in tqdm(dataset, desc="Tokenizing epochs", unit="epoch"):
        tokenized_data.append(item)

    tokenized_output_path = os.path.join(output_dir, "tokenized_epochs.pt")
    torch.save(tokenized_data, tokenized_output_path)
    print(f"Tokenized epochs saved at: {tokenized_output_path}")


if __name__ == "__main__":
    main_tokenization()
