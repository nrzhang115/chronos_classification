# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import logging
import os
import re
import sys
import itertools
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import List, Iterator, Optional, Dict
import traceback

import typer
from typer_config import use_yaml_config
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    T5Config,
    Trainer,
    TrainingArguments,
)
import accelerate
import gluonts
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Map, Filter
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    LeavesMissingValues,
    LastValueImputation,
)

from chronos import ChronosConfig, ChronosTokenizer


app = typer.Typer(pretty_exceptions_enable=False)


def is_main_process() -> bool:
    """
    Check if we're on the main process.
    """
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0


def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    """
    Log the given message using the given logger, if we're on the main process.
    """
    if is_main_process():
        logger.log(log_level, msg)
        


def load_data(arrow_file_path):
    """ Load the NCH dataset from the arrow file. """
    dataset = FileDataset(arrow_file_path, freq="s")
    # # Debug: print the first few entries to inspect their structure
    # for entry in itertools.islice(dataset, 5):  # Inspect first 5 entries
    #     print(f"Entry structure: {entry}")
    #     print(f"Sleep stages (target): {entry.get('target', None)}")
        
    return dataset

def pad_sequence(sequence, context_length):
    """
    Pad a sequence if it's shorter than context_length.
    Pads with -1 to match the required length.
    """
    required_length = context_length
    if len(sequence) < required_length:
        padding_value = -1  # Distinct padding value
        padded_sequence = np.pad(sequence, (0, required_length - len(sequence)), constant_values=padding_value)
        return padded_sequence
    return sequence

# def split_into_chunks(data, context_length, prediction_length):
#     """
#     Split data into chunks of context_length + prediction_length.
#     Each chunk will have length = context_length + prediction_length.
#     """
#     chunk_size = context_length + prediction_length
#     chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
#     # Handle final chunk if it is shorter than the required length to reduce the data losses
#     if len(chunks[-1]) < chunk_size:
#         chunks[-1] = pad_sequence(chunks[-1], context_length, prediction_length)
    
#     return chunks

def map_sleep_stage_to_label(sleep_stage):
    # Map the sleep stages to corresponding labels
    valid_labels = {0, 1, 2, 3, 4}
    if sleep_stage not in valid_labels:
        # print(f"Warning: Encountered unexpected sleep stage '{sleep_stage}', mapping it to None.")
        return None  # Return None for unexpected values

    # Map known sleep stages to corresponding labels
    stage_mapping = {
        0: 0,  # REM
        1: 1,  # Wake
        2: 2,  # N1
        3: 3,  # N2
        4: 4   # N3
    }
    return stage_mapping.get(sleep_stage, -1)  # -1 will be used as padding

def tokenize_data(data, tokenizer, context_length):
    """
    Tokenize the data using Chronos tokenizer for sleep stage classification.
    Handles both short and long sequences by splitting or padding.
    """
    print(f"Tokenizing {len(data)} sleep stages.")
    
    # Pad short sequences and split long ones
    if len(data) < context_length:
        print(f"Padding short sequence with length {len(data)}")
        data = pad_sequence(data, context_length)
    # Convert to tensor with the expected shape of [1, 3000]
    data_tensor = torch.tensor([data], dtype=torch.float32)  # Add batch dimension
    print(f"Shape of data tensor before tokenization: {data_tensor.shape}")  # Debug

    try:
        # Tokenize the padded sequence
        input_ids, attention_mask, _ = tokenizer.context_input_transform(data_tensor)
        print(f"Shape of input_ids after tokenization: {input_ids.shape}")  # Debug

        # Map the padded data to labels
        mapped_labels = [map_sleep_stage_to_label(stage) for stage in data]
        mapped_labels = [label if label is not None else -1 for label in mapped_labels]
        labels_tensor = torch.tensor(mapped_labels).view(1, -1)  # Shape to [1, 3000]
        
        # Set padding labels to ignore index
        labels_tensor[labels_tensor == -1] = -100  # -1 is now -100 for padding ignore index
        tokenized_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor
        }
        return tokenized_data
        
    except Exception as e:
        print(f"Error during tokenization: {e}")
        traceback.print_exc()
        return None  # Explicitly return None if an error occurs

    
            
            

def save_tokenized_data(tokenized_data, output_file):
    """ Save all tokenized data into a single PyTorch .pt file. """
    tokenized_data = [entry for entry in tokenized_data if isinstance(entry, dict)]
    if not tokenized_data:
        print("No tokenized data to save.")
        return
    
    # Prepare a dictionary to store all tokenized entries
    all_tokenized_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    max_label_length = max(entry["labels"].size(1) for entry in tokenized_data)  # Find maximum label length

    # Append each tokenized entry's tensors into the lists
    for tokenized_entry in tokenized_data:
        all_tokenized_data["input_ids"].append(tokenized_entry["input_ids"])
        all_tokenized_data["attention_mask"].append(tokenized_entry["attention_mask"])
        
        # Pad labels to maximum length with -100 for ignore index
        padded_labels = torch.full((1, max_label_length), -100, dtype=torch.long)
        padded_labels[0, :tokenized_entry["labels"].size(1)] = tokenized_entry["labels"]
        all_tokenized_data["labels"].append(padded_labels)

    # Convert lists of tensors into a single stacked tensor (batch format)
    all_tokenized_data["input_ids"] = torch.stack(all_tokenized_data["input_ids"])
    all_tokenized_data["attention_mask"] = torch.stack(all_tokenized_data["attention_mask"])
    all_tokenized_data["labels"] = torch.stack(all_tokenized_data["labels"])

    # Save the entire dataset as a single .pt file
    torch.save(all_tokenized_data, output_file)
    
    print(f"All tokenized data saved to {output_file}")


@app.command()
@use_yaml_config(param_name="config")
def main():
    nch_arrow_path = "/srv/scratch/z5298768/chronos_classification/prepare_time_seires/C4-M1/nch_sleep_data.arrow"
    output_dir = "/srv/scratch/z5298768/chronos_classification/tokenization"
    
    # training_data_paths: str,
    # probability: Optional[str] = None,
    context_length = 3000
    prediction_length = 0
    # min_past: int = 64,
    # max_steps: int = 200_000,
    # save_steps: int = 50_000,
    # log_steps: int = 500,
    # per_device_train_batch_size: int = 32,
    # learning_rate: float = 1e-3,
    # optim: str = "adamw_torch_fused",
    # shuffle_buffer_length: int = 100,
    # gradient_accumulation_steps: int = 2,
    # model_id: str = "google/t5-efficient-tiny",
    model_type = "seq2seq"
    tokenizer_class = "MeanScaleUniformBins"
    tokenizer_kwargs = {'low_limit': -15.0, 'high_limit': 15.0}
    n_tokens = 3000 # Number of tokens for each 30s epoch 
    n_special_tokens = 2
    pad_token_id = 0
    eos_token_id = 1
    # use_eos_token: bool = True,
    # lr_scheduler_type: str = "linear",
    # warmup_ratio: float = 0.0,
    # dataloader_num_workers: int = 1,
    # max_missing_prop: float = 0.9,
    num_samples = 20
    temperature = 1.0
    top_k = 50
    top_p = 1.0

    # log_on_main(f"Using SEED: {seed}", logger)
    log_on_main(f"Loading dataset from {nch_arrow_path}", logger)
    # transformers.set_seed(seed=seed)
    
    # Load the dataset from arrow file
    dataset = load_data(nch_arrow_path)
    # Initialize the tokenizer
    # tokenizer_kwargs = json.loads(tokenizer_kwargs)
    chronos_config = ChronosConfig(
        tokenizer_class=str(tokenizer_class),
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=False, # Set this to False to prevent extra token
        context_length=context_length,  # Required for ChronosConfig
        prediction_length=prediction_length,  # Required for ChronosConfig
        model_type=model_type,  # Required for ChronosConfig
        num_samples=num_samples,  # Required for ChronosConfig
        temperature=temperature,  # Required for ChronosConfig
        top_k=top_k,  # Required for ChronosConfig
        top_p=top_p  # Required for ChronosConfig
    )
    tokenizer = chronos_config.create_tokenizer()

    tokenized_data = []
    
    # Start tokenization and track progress
    log_on_main("Starting tokenization process...", logger)
    for entry_index, entry in enumerate(dataset):
        sleep_stages = entry['target']
        
        
        # Log the current entry being processed
        log_on_main(f"Processing entry {entry_index + 1} with {len(sleep_stages)} sleep stages", logger)
        
        if sleep_stages is None or len(sleep_stages) == 0:
            log_on_main(f"Skipping entry {entry_index + 1} as it has no valid sleep stages", logger)
            continue
        
        # Tokenize the epochs directly
        tokenized_sleep_stages = tokenize_data(sleep_stages, tokenizer, context_length)

        if tokenized_sleep_stages:
            tokenized_data.append(tokenized_sleep_stages)
        
    # Log tokenization result
    log_on_main(f"Tokenization completed with {len(tokenized_data)} tokenized entries", logger)
    # Debug
    log_on_main(f"First 5 tokenized entries: {tokenized_data[:5]}", logger)
    # Save tokenized data
    log_on_main(f"Saving tokenized data to {output_dir}", logger)
    # Save tokenized data
    output_file = os.path.join(output_dir, 'tokenized_data_remapping.pt')
    save_tokenized_data(tokenized_data, output_file)
    


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()
