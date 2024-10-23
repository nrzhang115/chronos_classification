import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import evaluate
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(p):
    print("Computing metrics...")
    predictions, labels = p
    predictions = torch.argmax(predictions, dim=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    # For each class (sleep stages), calculate precision, recall, and F1 score
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(labels, predictions, average=None)
    
    # Create a dictionary to log all metrics
    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }
    
    # Add metrics for each class (sleep stages 0-5)
    for i in range(6):  # Assuming sleep stages are 0-5
        metrics[f"precision_stage_{i}"] = precision_per_class[i]
        metrics[f"recall_stage_{i}"] = recall_per_class[i]
        metrics[f"f1_stage_{i}"] = f1_per_class[i]
    
    print(f"Metrics: {metrics}")
    return metrics

def save_metrics_to_excel(metrics, output_file):
    print(f"Saving metrics to {output_file}...")
    df = pd.DataFrame([metrics])  # Convert metrics dictionary to DataFrame
    df.to_excel(output_file, index=False)
    print(f"Metrics saved to {output_file}")

class BertForSleepStageClassification(nn.Module):
    def __init__(self, num_labels=6):
        super(BertForSleepStageClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

def load_tokenized_data(file_path):
    print(f"Loading tokenized data from {file_path}...")
    data = torch.load(file_path)
    input_ids = data['input_ids']
    attention_masks = data['attention_mask']
    labels = data['labels']
    print("Data loaded successfully.")
    return input_ids, attention_masks, labels

def main():
    tokenized_data_path = "/srv/scratch/z5298768/chronos_classification/tokenization/tokenized_data.pt"
    output_dir = "/srv/scratch/z5298768/chronos_classification/bert_finetune_output"
    metrics_output_file = "/srv/scratch/z5298768/chronos_classification/bert_finetune_metrics.xlsx"
    
    print("Starting the training process...")
    
    # Load the tokenized data
    input_ids, attention_masks, labels = load_tokenized_data(tokenized_data_path)
    
    # Create a Dataset and DataLoader for both training and validation sets
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    # Split into training and validation datasets (80% training, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print("Data successfully split into training and validation sets.")
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    # Initialize the BERT model for sleep stage classification
    model = BertForSleepStageClassification(num_labels=6)  # Sleep stages 0-5
    print("Model initialized successfully.")
    
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",   # Enables evaluation during training
        eval_steps=500,                # Evaluate every 500 steps
        save_steps=1000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32, # Batch size for evaluation
        num_train_epochs=3,
        logging_dir=f"{output_dir}/logs",
        logging_steps=500,
        save_total_limit=2,            # Keeps only 2 last checkpoints
        load_best_model_at_end=True,   # Load the best model based on evaluation
        metric_for_best_model="f1_macro",  # Specify F1 score as metric for best model
    )

    print("Training arguments set up successfully.")
    
    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,      # Pass the training dataset
        eval_dataset=val_dataset,         # Pass the validation dataset for evaluation
        compute_metrics=compute_metrics   # Function to calculate precision, recall, F1, and accuracy
    )

    # Train the model
    print("Starting model training...")
    trainer.train()

    # Evaluate and save metrics after training
    print("Evaluating the model...")
    metrics = trainer.evaluate()
    save_metrics_to_excel(metrics, metrics_output_file)

    # Save the final model
    print("Saving the model...")
    model.save_pretrained(output_dir)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()