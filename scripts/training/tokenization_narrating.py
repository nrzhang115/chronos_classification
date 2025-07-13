import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import numpy as np
from typing import List
from chronos import ChronosConfig, ChronosTokenizer
from tqdm import tqdm
import pandas as pd
import torch.nn as nn  # NEW

# ADD: Feature extractor class
class EEGFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )
        self.pool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):  # x: (B, 1, T)
        feat = torch.cat([
            self.branch3(x),
            self.branch5(x),
            self.branch7(x)
        ], dim=1)
        return self.pool(feat)  # (B, C*3, T//2)

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
        feature_extractor: nn.Module = None  # NEW
    ) -> None:
        assert os.path.exists(arrow_file_path), f"Arrow file not found: {arrow_file_path}"

        # Open the Arrow file and read all batches
        with pa.memory_map(arrow_file_path, "r") as source:
            reader = ipc.RecordBatchFileReader(source)

            tables = [reader.get_batch(i).to_pandas() for i in range(reader.num_record_batches)]
            self.dataset = pd.concat(tables, ignore_index=True)

        self.tokenizer = tokenizer
        self.token_length = token_length
        self.np_dtype = np_dtype
        self.feature_extractor = feature_extractor.eval() if feature_extractor else None  # NEW

    def preprocess_entry(self, eeg_epochs: List[List[float]], labels: List[str], file_name: str) -> List[dict]:
        tokenized_epochs = []
        for epoch, label in zip(eeg_epochs, labels):
            input_ids, attention_mask = self.tokenize_epoch(epoch)
            tokenized_epochs.append({
                "file_name": file_name,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label,
            })
        return tokenized_epochs

    def tokenize_epoch(self, epoch: List[float]) -> tuple:
        epoch_tensor = torch.tensor(epoch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3000)

        if self.feature_extractor:  # NEW
            with torch.no_grad():
                features = self.feature_extractor(epoch_tensor)  # (1, C, T')
                flattened = features.view(1, -1)  # Flatten to (1, L)
        else:
            flattened = epoch_tensor.squeeze(1)  # (1, 3000)

        input_ids, attention_mask, _ = self.tokenizer.context_input_transform(flattened)

        if input_ids.size(1) != self.token_length:
            raise ValueError(f"Tokenizer produced {input_ids.size(1)} tokens, expected {self.token_length}")
        return input_ids.squeeze(0), attention_mask.squeeze(0)

    def __iter__(self):
        for _, row in self.dataset.iterrows():
            file_name = row["file_name"]
            eeg_epochs = row["eeg_epochs"]
            labels = row.get("labels", ["unknown"] * len(eeg_epochs))

            if len(labels) != len(eeg_epochs):
                print(f"Warning: Mismatch in number of labels and epochs for {file_name}. Padding labels.")
                labels = labels[:len(eeg_epochs)] + ["unknown"] * (len(eeg_epochs) - len(labels))

            yield from self.preprocess_entry(eeg_epochs, labels, file_name)


def main_tokenization():
    arrow_file_path = "/srv/scratch/z5298768/chronos_classification_local/prepare_time_series/nch_sleep_data_mid_2hrs_all.arrow"
    output_dir = "/srv/scratch/z5298768/chronos_classification_local/longformer_tokenization_updated_mid_2hrs"
    os.makedirs(output_dir, exist_ok=True)

    context_length = 1024
    n_tokens = 1024
    tokenizer_class = "MeanScaleUniformBins"
    tokenizer_kwargs = {"low_limit": -20.0, "high_limit": 20.0}

    tokenizer = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=False,
        prediction_length=0,
        num_samples=1,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        context_length=context_length,
        model_type="classification",
    ).create_tokenizer()

    # NEW: initialize CNN-based feature extractor
    feature_extractor = EEGFeatureExtractor()

    dataset = ChronosEpochTokenizer(
        arrow_file_path=arrow_file_path,
        tokenizer=tokenizer,
        token_length=context_length,
        feature_extractor=feature_extractor  # NEW
    )

    chunk_size = 500_000
    tokenized_data = []
    chunk_index = 0
    total_count = 0

    for i, item in enumerate(tqdm(dataset, desc="Tokenizing epochs", unit="epoch")):
        tokenized_data.append(item)
        total_count += 1

        if len(tokenized_data) >= chunk_size:
            chunk_path = os.path.join(output_dir, f"tokenized_chunk_{chunk_index}.pt")
            torch.save(tokenized_data, chunk_path)
            print(f"Saved chunk {chunk_index} with {len(tokenized_data)} entries to {chunk_path}")
            tokenized_data = []
            chunk_index += 1

    if tokenized_data:
        chunk_path = os.path.join(output_dir, f"2hrs_tokenized_chunk_{chunk_index}.pt")
        torch.save(tokenized_data, chunk_path)
        print(f"Saved final chunk {chunk_index} with {len(tokenized_data)} entries to {chunk_path}")

    print(f"Tokenization complete. Total tokenized epochs: {total_count}")


if __name__ == "__main__":
    main_tokenization()
