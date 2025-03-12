import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import BertForSequenceClassification, BertConfig, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertConfig
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Training is running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training is running on CPU only.")

# Define sleep stage labels (excluding "unknown")
label_mapping = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
}

# Load the tokenized dataset
data = torch.load("/srv/scratch/z5298768/chronos_classification/tokenization_updated/tokenized_epochs.pt")

# Convert labels using the mapping
filtered_data = [sample for sample in data if sample["label"] in label_mapping]

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

# Create train and validation datasets
train_dataset = SleepStageDataset(train_input_ids, train_attention_mask, train_labels)
val_dataset = SleepStageDataset(val_input_ids, val_attention_mask, val_labels)

# Compute class weights using sklearn (Better Handling of Class Imbalance)
unique_labels = torch.unique(train_labels).cpu().numpy()
class_weights = compute_class_weight(class_weight="balanced", classes=unique_labels, y=train_labels.cpu().numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define a weighted sampler (Alternative to class weights)
class_sample_counts = torch.bincount(train_labels)
weights = 1.0 / class_sample_counts.float()
sample_weights = weights[train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

# Create train and validation dataloaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Load pre-trained BERT model with custom dropout
# Load pre-trained BERT model
num_classes = len(label_mapping)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
model.to(device)

# Define weighted loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define optimizer with weight decay
learning_rate = 2e-5
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Define learning rate scheduler
num_training_steps = len(train_dataloader) * 20  # 20 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training loop with early stopping
epochs = 20
early_stopping_patience = 3
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
        optimizer.step()
        lr_scheduler.step()

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

    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "bert_sleep_allclass_best.pth")  # Save the best model
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered!")
            break

    # Compute validation metrics
    report = classification_report(all_labels, all_preds, target_names=list(label_mapping.keys()))
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("Validation Classification Report:\n", report)
    print("Validation Confusion Matrix:\n", conf_matrix)

    # Save validation results
    with open("validation_results_allclass.txt", "w") as f:
        f.write("Validation Classification Report:\n")
        f.write(report + "\n")
        f.write("Validation Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix))

print("All-class classification model training complete and saved.")
