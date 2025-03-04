import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

# Load tokenized data
data = torch.load("/srv/scratch/z5298768/chronos_classification/tokenization_updated/tokenized_epochs.pt")
print(type(data))  # Should be dict or list
print(len(data))   # If it's a list, check how many items it has
print(data[:3])    # Print the first 3 entries (if it's a list
input_ids = data["input_ids"]  # Shape: (num_samples, seq_length)
attention_mask = data.get("attention_mask", torch.ones_like(input_ids))  # If needed
labels = data["labels"]  # Shape: (num_samples,)

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

# Create dataset and dataloaders
dataset = SleepStageDataset(input_ids, attention_mask, labels)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)



# Load pre-trained BERT model
num_classes = len(torch.unique(labels))  # Number of sleep stages
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 3
scaler = GradScaler()  # Helps with stable mixed precision training

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
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
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "bert_sleep_classifier.pth")
print("Model training complete and saved.")
