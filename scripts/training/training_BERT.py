import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import torch

if torch.cuda.is_available():
    print(f"Training is running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training is running on CPU only.")


# Define a mapping of sleep stage labels to integers
label_mapping = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
    "unknown": -1  # If "unknown" should be ignored, we will filter it out later
}

# Load the tokenized data
data = torch.load("/srv/scratch/z5298768/chronos_classification/tokenization_updated/tokenized_epochs.pt")

# Convert labels using the mapping
filtered_data = [sample for sample in data if sample["label"] in label_mapping and label_mapping[sample["label"]] != -1]

# Split dataset (80% train, 20% validation)
train_data, val_data = train_test_split(filtered_data, test_size=0.2, random_state=42)

# Process training data
train_input_ids = torch.stack([sample["input_ids"] for sample in train_data])
train_attention_mask = torch.stack([sample["attention_mask"] for sample in train_data])
train_labels = torch.tensor([label_mapping[sample["label"]] for sample in train_data], dtype=torch.long)

# Process validation data
val_input_ids = torch.stack([sample["input_ids"] for sample in val_data])
val_attention_mask = torch.stack([sample["attention_mask"] for sample in val_data])
val_labels = torch.tensor([label_mapping[sample["label"]] for sample in val_data], dtype=torch.long)

# # Print shapes to verify correctness
# print("input_ids shape:", input_ids.shape)       # Expected: (filtered_samples, 512)
# print("attention_mask shape:", attention_mask.shape)  # Expected: (filtered_samples, 512)
# print("labels shape:", labels.shape)            # Expected: (filtered_samples,)


# Define dataset class
class SleepStageDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

# Create train and validation dataloaders
train_dataset = SleepStageDataset(train_input_ids, train_attention_mask, train_labels)
val_dataset = SleepStageDataset(val_input_ids, val_attention_mask, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

# Load pre-trained BERT model
num_classes = len(label_mapping) - 1  # Exclude 'unknown'
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)



# Training loop
epochs = 10
scaler = GradScaler()  # Helps with stable mixed precision training

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast():  # Enables mixed precision
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")
    
# ======= Validation =======
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            val_loss += criterion(logits, labels).item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

    # Compute validation metrics
    report = classification_report(all_labels, all_preds, target_names=["W", "N1", "N2", "N3", "R"])
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("Validation Classification Report:\n", report)
    print("Validation Confusion Matrix:\n", conf_matrix)

    # Save validation results
    with open("validation_results.txt", "w") as f:
        f.write("Validation Classification Report:\n")
        f.write(report + "\n")
        f.write("Validation Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix))


# Save the trained model
torch.save(model.state_dict(), "bert_sleep_classifier.pth")
print("Model training complete and saved.")
