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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Plot and save the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Wake", "N1", "N2", "N3", "REM"], 
                yticklabels=["Wake", "N1", "N2", "N3", "REM"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix for Chronos Sleep Stage Classification")
    plt.savefig("/srv/scratch/z5298768/chronos_classification/confusion_matrix.png")  
    plt.close()
    
    
    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }
    
    for i in range(5):
        metrics[f"precision_stage_{i}"] = precision_per_class[i]
        metrics[f"recall_stage_{i}"] = recall_per_class[i]
        metrics[f"f1_stage_{i}"] = f1_per_class[i]
    
    return metrics

def save_metrics_to_excel(metrics, output_file):
    print(f"Saving metrics to {output_file}...")
    df = pd.DataFrame([metrics])  # Convert metrics dictionary to DataFrame
    df.to_excel(output_file, index=False)
    print(f"Metrics saved to {output_file}")
    
# class BertForSleepStageClassification(nn.Module):
#     def __init__(self, num_labels=5):
#         super(BertForSleepStageClassification, self).__init__()
#         self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         return outputs

# Try FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss

# Update the model to use Focal Loss
class BertForSleepStageClassification(nn.Module):
    def __init__(self, num_labels=5):
        super(BertForSleepStageClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        self.loss_fn = FocalLoss()  # Use Focal Loss here

# Update the model to use Focal Loss
class BertForSleepStageClassification(nn.Module):
    def __init__(self, num_labels=5):
        super(BertForSleepStageClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        self.loss_fn = FocalLoss()  # Use Focal Loss here

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.bert.config.num_labels), labels.view(-1))
            return loss, logits
        return logits
      
# Function to load your tokenized data using sliding window approach
def load_tokenized_data(file_path, bert_max_length=512, window_stride=256):
    data = torch.load(file_path)
    print(f"Shape of loaded input_ids: {data['input_ids'].shape}")
    print(f"Shape of loaded attention_mask: {data['attention_mask'].shape}")

    input_ids = data['input_ids'].squeeze(1)  # Shape becomes [3930, 3000]
    attention_masks = data['attention_mask'].squeeze(1)  # Shape becomes [3930, 3000]
    labels = data['labels'].squeeze(1)  # Shape becomes [3930, 3000]

    # Using a sliding window to create overlapping chunks of 512 tokens
    input_ids_chunks, attention_mask_chunks, label_chunks = [], [], []
    for i in range(input_ids.size(0)):  # Loop over each entry
        for start in range(0, input_ids.size(1) - bert_max_length + 1, window_stride):
            end = start + bert_max_length
            input_ids_chunks.append(input_ids[i, start:end])
            attention_mask_chunks.append(attention_masks[i, start:end])

            # Get the mode of the labels within this chunk for classification
            chunk_labels = labels[i, start:end]
            # Ignore padding labels; check if any valid labels are present
            valid_labels = chunk_labels[chunk_labels != -100]
            if valid_labels.numel() > 0:  # Only proceed if there are valid labels
                label_chunks.append(torch.mode(valid_labels).values)
            else:
                # Use ignore index for chunks with only padding labels
                label_chunks.append(torch.tensor(-100))
    
    # Stack all chunks into tensors
    input_ids = torch.stack(input_ids_chunks)  # Shape: [num_chunks, 512]
    attention_masks = torch.stack(attention_mask_chunks)  # Shape: [num_chunks, 512]
    labels = torch.stack(label_chunks)  # Shape: [num_chunks]

    # Debugging: Print shapes
    print(f"Final input_ids shape: {input_ids.shape}")
    print(f"Final attention_mask shape: {attention_masks.shape}")
    print(f"Final labels shape: {labels.shape}")
    
    return [{'input_ids': input_id, 'attention_mask': attention_mask, 'labels': label}
            for input_id, attention_mask, label in zip(input_ids, attention_masks, labels)]
    
# # Function to load your tokenized data: Truncation
# def load_tokenized_data(file_path, bert_max_length=512):
#     data = torch.load(file_path)
#     print(f"Shape of loaded input_ids: {data['input_ids'].shape}")
#     print(f"Shape of loaded attention_mask: {data['attention_mask'].shape}")

#     # Squeeze to remove the extra dimension in [3930, 1, 3000]
#     input_ids = data['input_ids'].squeeze(1)  # Shape becomes [3930, 3000]
#     attention_masks = data['attention_mask'].squeeze(1)  # Shape becomes [3930, 3000]

#     # Calculate the downsample factor
#     downsample_factor = int(input_ids.size(1) / bert_max_length)  # Expect around 6
#     if downsample_factor > 1:
#         # Reshape and downsample by averaging over each segment
#         input_ids = input_ids[:, :bert_max_length * downsample_factor].reshape(input_ids.size(0), bert_max_length, downsample_factor).float().mean(dim=2).long()
#         attention_masks = attention_masks[:, :bert_max_length * downsample_factor].reshape(attention_masks.size(0), bert_max_length, downsample_factor).float().mean(dim=2).long()
#     else:
#         # Direct truncation if already within bert_max_length
#         input_ids = input_ids[:, :bert_max_length]
#         attention_masks = attention_masks[:, :bert_max_length]
        
#     # For labels, downsample similarly by chunking and taking majority class in each chunk
#     labels = data['labels'].squeeze(1)  # Shape: [3930, 3000]
#     labels = labels[:, :bert_max_length * downsample_factor].reshape(labels.size(0), -1)
#     labels = torch.mode(labels, dim=1).values  # Shape becomes [3930] with majority class per sequence
#     labels[labels == -1] = -100  # Replace padding with ignore_index
    
#     print(f"Final input_ids shape: {input_ids.shape}")
#     print(f"Final attention_mask shape: {attention_masks.shape}")
#     print(f"Final labels shape: {labels.shape}")
    
#     # Debugging:
#     print(f"Sample input_ids: {input_ids[0]}")
#     print(f"Sample attention_mask: {attention_masks[0]}")
#     print(f"Sample label: {labels[0]}")
    
#     return [{'input_ids': input_id, 'attention_mask': attention_mask, 'labels': label}
#             for input_id, attention_mask, label in zip(input_ids, attention_masks, labels)]
    
def main():
    tokenized_data_path = "/srv/scratch/z5298768/chronos_classification/tokenization/tokenized_data_remapping.pt"
    output_dir = "/srv/scratch/z5298768/chronos_classification/bert_finetune_output"
    metrics_output_file = "/srv/scratch/z5298768/chronos_classification/bert_finetune_metrics.xlsx"
    
    print("Starting the training process...")
    
    # Load the tokenized data
    dataset = load_tokenized_data(tokenized_data_path)
    
    print("Data loaded successfully.")
    
    # Check unique labels to ensure correct label distribution
    unique_labels = torch.unique(torch.tensor([item['labels'] for item in dataset]))
    print("Unique labels in dataset:", unique_labels)
    
    # Split into training and validation datasets (80% training, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print("Data successfully split into training and validation sets.")
    
   

    # Initialize the BERT model with class weights
    model = BertForSleepStageClassification(num_labels=5)
    print("Model initialized successfully.")

    # Initialize the BERT model for sleep stage classification
    model = BertForSleepStageClassification(num_labels=5)  # Sleep stages 0-4
    print("Model initialized successfully.")
    
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",   # Enables evaluation during training
        eval_steps=500,                # Evaluate every 500 steps
        save_steps=1000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32, # Batch size for evaluation
        num_train_epochs=10,            # Train the data 10 times
        logging_dir=f"{output_dir}/logs",
        learning_rate=2e-5,          # Lower learning rate for better convergence on minority classes
        logging_steps=500,
        save_total_limit=2,            # Keeps only 2 last checkpoints
        load_best_model_at_end=True,   # Load the best model based on evaluation
        metric_for_best_model="f1_macro",  # Specify F1 score as metric for best model
    )

    print("Training arguments set up successfully.")
    
    # Calculate class weights based on labels in the training dataset
    train_labels = torch.tensor([item['labels'] for item in train_dataset if item['labels'] != -100])
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels.numpy()), y=train_labels.numpy())
    # Convert to a tensor and apply a boost to N2
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    w_label_index = 0 
    n2_label_index = 2  
    boost_factor = 2.0  
    class_weights[w_label_index] *= boost_factor  # Increase the weight for W
    class_weights[n2_label_index] *= boost_factor  # Increase the weight for N2

    # Move class weights to GPU if available
    class_weights = class_weights.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up WeightedRandomSampler for handling class imbalance
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Pass the sampler directly to Trainer by overriding the get_train_dataloader method
    class CustomTrainer(Trainer):
        def get_train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers
            )
    
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