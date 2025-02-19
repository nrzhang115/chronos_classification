import os
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
import typer
from sklearn.model_selection import train_test_split
from collections import Counter

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# Typer app initialization
app = typer.Typer(pretty_exceptions_enable=False)

# Dataset class for loading tokenized sleep stage data
class SleepStageDataset(Dataset):
    def __init__(self, tokenized_file_path: str):
        logger.info("Loading tokenized data from %s", tokenized_file_path)
        self.data = torch.load(tokenized_file_path)
        
        # self.data = [item for item in self.data if item['label'] != 5]  # Filtering integer label 5
        # logger.info(f"Class distribution after filtering 'unknown': {len(self.data)} samples remaining.")
        
        

        self.eos_token_id = 1  # Assuming EOS token ID is 1

        # Label mapping (if needed)
        self.label_mapping = {
            "W": 0,
            "N1": 1,
            "N2": 2,
            "N3": 3,
            "R": 4,
            "unknown": 5
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Convert string label to integer
        if isinstance(item["label"], str):
            item["label"] = self.label_mapping.get(item["label"], 5)
        # if isinstance(item["label"], str):
        #     if item["label"] not in self.label_mapping:
        #         raise ValueError(f"Invalid label '{item['label']}' encountered in dataset.")
        #     item["label"] = self.label_mapping[item["label"]]


        # Truncate to 511 tokens if necessary
        if len(item["input_ids"]) >= 512:
            item["input_ids"] = item["input_ids"][:511]
            item["attention_mask"] = item["attention_mask"][:511]

        # Append EOS token
        item["input_ids"] = torch.cat([item["input_ids"], torch.tensor([self.eos_token_id])])
        item["attention_mask"] = torch.cat([item["attention_mask"], torch.tensor([1])])
        # Sanity Check
        if idx < 5:  # Just for first few samples
            logger.info(f"Sample {idx}: Label (int) = {item['label']}, Type = {type(item['label'])}")

        # Ensure fields are returned as tensors
        return {
            "input_ids": item["input_ids"].clone().detach().long(),
            "attention_mask": item["attention_mask"].clone().detach().long(),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }



# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

@app.command()
def main(
    tokenized_data_dir: str = "/srv/scratch/z5298768/chronos_classification/tokenization_updated",
    output_dir: str = "/srv/scratch/z5298768/chronos_classification/t5_tiny_output",
    model_id: str = "google/t5-efficient-tiny",
    n_tokens: int = 4096,                # Vocabulary size from Chronos tokenizer
    num_labels: int = 6,                 # Sleep stage classes: W, N1, N2, N3, R, unknown
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_steps: int = 200_000,
    logging_steps: int = 500,
    save_steps: int = 50_000,
    gradient_accumulation_steps: int = 2,
    seed: int = 42,
):
    """
    Train a sleep stage classification model and evaluate using a confusion matrix.
    """
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # # Load tokenized dataset
    # tokenized_data_path = os.path.join(tokenized_data_dir, "tokenized_epochs.pt")
    # train_dataset = SleepStageDataset(tokenized_data_path)
    

    # Load the full dataset
    tokenized_data_path = os.path.join(tokenized_data_dir, "tokenized_epochs.pt")
    full_dataset = SleepStageDataset(tokenized_data_path)

    # Split: 80% training, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Sanity check
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Validate label format after loading the dataset
    sample_item = train_dataset[0]
    print(f"Sample input_ids shape: {sample_item['input_ids'].shape}")
    print(f"Sample label (should be integer): {sample_item['labels']} ({type(sample_item['labels'])})")
    print(f"Last token in sequence: {sample_item['input_ids'][-1].item()} (should be 1 for EOS)")
    print(f"Final sequence length: {len(sample_item['input_ids'])}")  # Should be 512

    logger.info("Loading model %s for sleep stage classification", model_id)

    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    model.resize_token_embeddings(n_tokens)
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = eos_token_id

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        eval_strategy="epoch",
    )

    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Unpack logits from the tuple
        logits = predictions[0] if isinstance(predictions, tuple) else predictions
        preds = logits.argmax(axis=1)

        # Generate classification report
        cm_report = classification_report(labels, preds, output_dict=True)
        logger.info("Classification Report:\n%s", classification_report(labels, preds))

        # Plot and save confusion matrix
        plot_confusion_matrix(labels, preds, ["W", "N1", "N2", "N3", "R", "Unknown"], output_dir)
        return {
            "accuracy": cm_report['accuracy'],
            "precision": cm_report['weighted avg']['precision'],
            "recall": cm_report['weighted avg']['recall'],
            "f1": cm_report['weighted avg']['f1-score']
        }

    
    
    
    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    logger.info("Starting training for sleep stage classification...")
    trainer.train()

    # Save final model
    final_ckpt = output_dir / "checkpoint-final"
    model.save_pretrained(final_ckpt)
    logger.info("Model saved at: %s", final_ckpt)

    # Save training configuration
    training_info = {
        "model_id": model_id,
        "n_tokens": n_tokens,
        "num_labels": num_labels,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "max_steps": max_steps,
        "seed": seed,
    }
    with open(output_dir / "training_info.json", "w") as fp:
        json.dump(training_info, fp, indent=4)
    logger.info("Training configuration saved.")

if __name__ == "__main__":
    app()
