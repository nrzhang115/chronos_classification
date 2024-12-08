import os
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from typing import List
from chronos import ChronosConfig, ChronosTokenizer


class ChronosEpochTokenizer:
    """
    Dataset wrapper for tokenizing each 30-second epoch into 512 tokens.
    """

    def __init__(
        self,
        arrow_file_path: str,
        tokenizer: ChronosTokenizer,
        token_length: int = 512,
        np_dtype=np.float32,
    ) -> None:
        assert os.path.exists(arrow_file_path), f"Arrow file not found: {arrow_file_path}"
        self.dataset = pq.read_table(arrow_file_path)
        self.tokenizer = tokenizer
        self.token_length = token_length
        self.np_dtype = np_dtype

    def preprocess_entry(self, start, target: List[int]) -> list:
        """
        Preprocess a single dataset row to tokenize every individual label.
        """
        # Generate 512 tokens for each label in the target
        tokenized_epochs = []
        for label in target:
            input_ids, attention_mask = self.tokenize_label(label)
            tokenized_epochs.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label,
            })

        return tokenized_epochs

    def tokenize_label(self, label: int) -> tuple:
        """
        Tokenize a single label into 512 tokens.
        """
        label_tensor = torch.tensor([label]).unsqueeze(0)  # Shape: (1, 1)
        input_ids, attention_mask, _ = self.tokenizer.context_input_transform(
            label_tensor
        )

        # Ensure tokenization produces the correct number of tokens
        if input_ids.size(1) != self.token_length:
            raise ValueError(
                f"Tokenizer produced {input_ids.size(1)} tokens, expected {self.token_length}"
            )

        return input_ids.squeeze(0), attention_mask.squeeze(0)

    def __iter__(self):
        for i in range(self.dataset.num_rows):
            row = self.dataset.slice(i, 1).to_pydict()
            start = row["start"][0]  # Timestamp
            target = row["target"][0]  # List of labels
            yield from self.preprocess_entry(start, target)


def main_tokenization():
    # Input and output paths
    arrow_file_path = "/srv/scratch/z5298768/chronos_classification/prepare_time_seires/C4-M1/nch_sleep_data.arrow"
    output_dir = "/srv/scratch/z5298768/chronos_classification/tokenization_updated"
    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer Configuration
    token_length = 512
    n_tokens = 4096
    tokenizer_class = "MeanScaleUniformBins"
    tokenizer_kwargs = {"low_limit": -15.0, "high_limit": 15.0}

    # Initialize tokenizer
    tokenizer = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        model_type="classification",
        context_length=token_length,
    ).create_tokenizer()

    # Initialize dataset
    dataset = ChronosEpochTokenizer(
        arrow_file_path=arrow_file_path,
        tokenizer=tokenizer,
        token_length=token_length,
    )

    # Tokenize and save
    tokenized_data = []
    for item in dataset:
        tokenized_data.append(item)

    tokenized_output_path = os.path.join(output_dir, "tokenized_epochs.pt")
    torch.save(tokenized_data, tokenized_output_path)

    print(f"Tokenized epochs saved at: {tokenized_output_path}")


if __name__ == "__main__":
    main_tokenization()
