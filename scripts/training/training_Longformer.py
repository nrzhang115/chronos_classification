import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerModel, LongformerTokenizer, LongformerForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Define constants
MODEL_NAME = "allenai/longformer-base-4096"
NUM_CLASSES = 2  # Binary classification (W vs. N3)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Longformer tokenizer
tokenizer = LongformerTokenizer.from_pretrained(MODEL_NAME)

# Load the tokenized dataset (assuming it has input_ids, attention_mask, and labels)
data = torch.load("/srv/scratch/z5298768/chronos_classification/tokenization_updated/tokenized_epochs.pt")

# Convert labels to binary (W = 0, N3 = 1)
LABEL_MAPPING = {"W": 0, "N3": 1}

# Prepare data
class SleepDataset(Dataset):
    def __init__(self, data):
        self.samples = [
            {
                "input_ids": sample["input_ids"].clone().detach().to(dtype=torch.long),
                "attention_mask": sample["attention_mask"].clone().detach().to(dtype=torch.long),
                "label": LABEL_MAPPING[sample["label"]]
            }
            for sample in data if sample["label"] in LABEL_MAPPING
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Split data into train/validation (80% train, 20% validation)
np.random.seed(42)
np.random.shuffle(data)

split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

train_dataset = SleepDataset(train_data)
val_dataset = SleepDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Longformer model for classification
model = LongformerForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
model.to(DEVICE)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training function
def train(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Validate model
        validate(model, val_loader)

# Validation function
def validate(model, val_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Print classification results
    print("Validation Classification Report:\n", classification_report(true_labels, predictions))
    print("Validation Confusion Matrix:\n", confusion_matrix(true_labels, predictions))

# Train the model
train(model, train_loader, val_loader, EPOCHS)

# Save the trained model
torch.save(model.state_dict(), "longformer_sleep_classifier.pth")
print("Model training complete and saved.")
