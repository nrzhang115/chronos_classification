import os
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import numpy as np
from typing import List
from chronos import ChronosConfig, ChronosTokenizer
from tqdm import tqdm
from glob import glob

class ChronosEpochTokenizer:
    def __init__(self, arrow_file_path: str, tokenizer: ChronosTokenizer, token_length: int = 512):
        assert os.path.exists(arrow_file_path), f"Arrow file not found: {arrow_file_path}"

        with pa.memory_map(arrow_file_path, "r") as source:
            reader = pa.ipc.open_file(source)
            self.dataset = reader.read_all().to_pandas()

        self.tokenizer = tokenizer
        self.token_length = token_length

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
        epoch_tensor = torch.tensor(epoch, dtype=torch.float32).unsqueeze(0)
        input_ids, attention_mask, _ = self.tokenizer.context_input_transform(epoch_tensor)
        if input_ids.size(1) != self.token_length:
            raise ValueError(f"Expected {self.token_length} tokens, got {input_ids.size(1)}")
        return input_ids.squeeze(0), attention_mask.squeeze(0)

    def __iter__(self):
        for _, row in self.dataset.iterrows():
            file_name = row["file_name"]
            eeg_epochs = row["eeg_epochs"]
            labels = row.get("labels", ["unknown"] * len(eeg_epochs))

            if len(labels) != len(eeg_epochs):
                print(f"Label mismatch in {file_name}, padding labels.")
                labels = labels[:len(eeg_epochs)] + ["unknown"] * (len(eeg_epochs) - len(labels))

            yield from self.preprocess_entry(eeg_epochs, labels, file_name)


def main_tokenize_all_batches():
    input_dir = "/srv/scratch/z5298768/chronos_classification/prepare_time_series/C4-M1_updated_all"
    output_dir = "/srv/scratch/z5298768/chronos_classification/tokenization_updated/tokenized_batches"
    os.makedirs(output_dir, exist_ok=True)

    context_length = 512
    n_tokens = 4096
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

    batch_files = sorted(glob(os.path.join(input_dir, "*.arrow")))

    for path in tqdm(batch_files, desc="Tokenizing batch files"):
        base = os.path.basename(path).replace(".arrow", "")
        output_path = os.path.join(output_dir, f"{base}.pt")

        # Skip already-processed files
        if os.path.exists(output_path):
            print(f"Skipping already tokenized: {output_path}")
            continue

        try:
            dataset = ChronosEpochTokenizer(path, tokenizer, token_length=context_length)
            tokenized_data = [item for item in dataset]
            torch.save(tokenized_data, output_path)
            print(f"Tokenized and saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {path}: {e}")


if __name__ == "__main__":
    main_tokenize_all_batches()
