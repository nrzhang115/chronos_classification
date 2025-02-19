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
        # Map labels to integers
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
            item["label"] = self.label_mapping.get(item["label"], 5)  # Default to 'unknown' if missing
        return item


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

    # Load tokenized dataset
    tokenized_data_path = os.path.join(tokenized_data_dir, "tokenized_epochs.pt")
    train_dataset = SleepStageDataset(tokenized_data_path)
    
    # Validate label format after loading the dataset
    sample_item = train_dataset[0]
    print(f"Sample input_ids shape: {sample_item['input_ids'].shape}")
    print(f"Sample label (should be integer): {sample_item['label']} ({type(sample_item['label'])})")

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
        preds = predictions.argmax(axis=1)
        cm_report = classification_report(labels, preds, output_dict=True)
        logger.info("Classification Report:\n%s", classification_report(labels, preds))

        # Plot and save confusion matrix
        plot_confusion_matrix(labels, preds, ["W", "N1", "N2", "N3", "R", "unknown"], output_dir)
        return {"accuracy": cm_report['accuracy']}
    
    
    
    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
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
