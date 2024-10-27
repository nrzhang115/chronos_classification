import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import evaluate
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import default_data_collator
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    
    # Ensure predictions are a tensor
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    
    predictions = torch.argmax(predictions, dim=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    # For each class (sleep stages), calculate precision, recall, and F1 score
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(labels, predictions, average=None)
    
    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }
    
    for i in range(6):
        metrics[f"precision_stage_{i}"] = precision_per_class[i]
        metrics[f"recall_stage_{i}"] = recall_per_class[i]
        metrics[f"f1_stage_{i}"] = f1_per_class[i]
    
    return metrics

def save_metrics_to_excel(metrics, output_file):
    print(f"Saving metrics to {output_file}...")
    df = pd.DataFrame([metrics])  # Convert metrics dictionary to DataFrame
    df.to_excel(output_file, index=False)
    print(f"Metrics saved to {output_file}")

class BertForSleepStageClassification(nn.Module):
    def __init__(self, num_labels=5):
        super(BertForSleepStageClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

# Function to load your tokenized data
def load_tokenized_data(file_path, bert_max_length=512):
    data = torch.load(file_path)

    # # Ensure input_ids and attention_masks are 2D and truncate to bert_max_length
    # input_ids = data['input_ids'].view(data['input_ids'].size(0), -1)[:, :bert_max_length]  # Reshape to [batch_size, seq_length] and truncate
    # attention_masks = data['attention_mask'].view(data['attention_mask'].size(0), -1)[:, :bert_max_length]  # Same as above
    # Downsample from 3000 tokens to bert_max_length (512 tokens)
    downsample_factor = data['input_ids'].size(1) // bert_max_length  # This should be close to 6 for 3000 tokens
    if downsample_factor > 1:
        # Downsample by averaging over each segment
        input_ids = data['input_ids'].reshape(data['input_ids'].size(0), bert_max_length, downsample_factor).mean(dim=2)
        attention_masks = data['attention_mask'].reshape(data['attention_mask'].size(0), bert_max_length, downsample_factor).mean(dim=2)

        # Convert averaged values to integer tokens
        input_ids = input_ids.long()  # Final shape should be [batch_size, bert_max_length]
        attention_masks = (attention_masks > 0).long()  # Convert to binary
    else:
        # Direct truncation if already within bert_max_length
        input_ids = data['input_ids'][:, :bert_max_length].reshape(data['input_ids'].size(0), bert_max_length)
        attention_masks = data['attention_mask'][:, :bert_max_length].reshape(data['attention_mask'].size(0), bert_max_length)
        
    # Debugging: Check the shape of labels before processing
    print(f"Original labels shape: {data['labels'].shape}")

    # Select only the first column or dimension from labels to make it 1D
    labels = data['labels'][:, 0, 0]  # Assuming you need the first column
    
    # Replace padding label (-1) with ignore_index (-100)
    labels[labels == -1] = -100  # Replace padding with ignore_index
    
    # Check the max and min values in labels for verification
    print(f"Max label value: {labels.max()}")
    print(f"Min label value: {labels.min()}")

    # Ensure labels are valid (excluding padding label)
    num_labels = 5  # Update this according to the number of sleep stages
    assert labels.max() < num_labels and labels.min() >= -100, "Label values are out of range (excluding ignore_index)."


    # Print tensor shapes to verify
    print(f"input_ids shape: {input_ids.shape}")
    print(f"attention_mask shape: {attention_masks.shape}")
    print(f"labels shape: {labels.shape}")
    
    return [{'input_ids': input_id, 'attention_mask': attention_mask, 'labels': label}
            for input_id, attention_mask, label in zip(input_ids, attention_masks, labels)]
    
def main():
    tokenized_data_path = "/srv/scratch/z5298768/chronos_classification/tokenization/tokenized_data_remapping.pt"
    output_dir = "/srv/scratch/z5298768/chronos_classification/bert_finetune_output"
    metrics_output_file = "/srv/scratch/z5298768/chronos_classification/bert_finetune_metrics.xlsx"
    
    print("Starting the training process...")
    
    # Load the tokenized data
    dataset = load_tokenized_data(tokenized_data_path)
    
    print("Data loaded successfully.")
    
    # Split into training and validation datasets (80% training, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print("Data successfully split into training and validation sets.")
    

    # Initialize the BERT model for sleep stage classification
    model = BertForSleepStageClassification(num_labels=5)  # Sleep stages 0-4
    print("Model initialized successfully.")
    
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",   # Enables evaluation during training
        eval_steps=500,                # Evaluate every 500 steps
        save_steps=1000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32, # Batch size for evaluation
        num_train_epochs=3,            # Train the data three times
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
        data_collator=default_data_collator,  # Use the default data collator for dictionary-style batches
        compute_metrics=compute_metrics   # Function to calculate precision, recall, F1, and accuracy
    )

    # Train the model
    print("Starting model training...")
    trainer.train()

    # Evaluate and save metrics after training
    print("Evaluating the model...")
    metrics = trainer.evaluate()
    save_metrics_to_excel(metrics, metrics_output_file)

    # # Save the final model
    # print("Saving the model...")
    # model.save_pretrained(output_dir)
    # print("Model saved successfully.")

if __name__ == "__main__":
    main()