# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from transformers import BertForSequenceClassification
# from peft import LoraConfig, get_peft_model
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from glob import glob
# import numpy as np
# import os

# print("Starting chunk-based training")

# if torch.cuda.is_available():
#     print(f"Training is running on GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("Training is running on CPU only.")

# # ======= Step 1: Label mapping and data loading =======
# label_mapping = { "W": 0, "N3": 1 }
# data_dir = "/srv/scratch/z5298768/chronos_classification/tokenization_updated"
# chunk_files = sorted(glob(os.path.join(data_dir, "tokenized_chunk_*.pt")))

# # Load all filtered W/N3 samples from all chunks
# all_filtered = []
# for chunk_path in chunk_files:
#     print(f"Loading: {chunk_path}")
#     chunk = torch.load(chunk_path)
#     filtered = [s for s in chunk if s["label"] in label_mapping]
#     all_filtered.extend(filtered)
#     print(f"Loaded {len(filtered)} W/N3 samples from {chunk_path}")

# #print(f"Total filtered samples: {len(all_filtered)}")

# # ======= Step 2: Train/val split =======
# train_data, val_data = train_test_split(all_filtered, test_size=0.2, random_state=42)

# def convert_to_tensors(data):
#     input_ids = torch.stack([s["input_ids"] for s in data])
#     attention_mask = torch.stack([s["attention_mask"] for s in data])
#     labels = torch.tensor([label_mapping[s["label"]] for s in data], dtype=torch.long)
#     return input_ids, attention_mask, labels

# train_input_ids, train_attention_mask, train_labels = convert_to_tensors(train_data)
# val_input_ids, val_attention_mask, val_labels = convert_to_tensors(val_data)

# # ======= Step 3: Dataset and Dataloader =======
# class SleepStageDataset(torch.utils.data.Dataset):
#     def __init__(self, input_ids, attention_mask, labels):
#         self.input_ids = input_ids
#         self.attention_mask = attention_mask
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return {
#             "input_ids": self.input_ids[idx],
#             "attention_mask": self.attention_mask[idx],
#             "labels": self.labels[idx],
#         }

# train_dataset = SleepStageDataset(train_input_ids, train_attention_mask, train_labels)
# val_dataset = SleepStageDataset(val_input_ids, val_attention_mask, val_labels)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# # ======= Step 4: Model Setup =======
# num_classes = 2
# base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.1,
#     target_modules=["query", "key", "value"],
#     bias="none",
#     task_type="SEQ_CLS"
# )

# model = get_peft_model(base_model, lora_config)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Class weights
# class_counts = torch.bincount(train_labels)
# weights = 1.0 / class_counts.float()
# weights = weights / weights.sum()
# weights = weights.to(device)

# criterion = nn.CrossEntropyLoss(weight=weights)
# optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# # ======= Step 5: Training Loop =======
# epochs = 5
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         optimizer.zero_grad()
#         outputs = model(input_ids, attention_mask=attention_mask)
#         loss = criterion(outputs.logits, labels)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

#     # ======= Validation =======
#     model.eval()
#     val_loss = 0
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for batch in val_loader:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             outputs = model(input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
#             predictions = torch.argmax(logits, dim=1)

#             val_loss += criterion(logits, labels).item()
#             all_preds.extend(predictions.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     avg_val_loss = val_loss / len(val_loader)
#     print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

#     report = classification_report(all_labels, all_preds, target_names=["W", "N3"])
#     conf_matrix = confusion_matrix(all_labels, all_preds)

#     print("Classification Report:\n", report)
#     print("Confusion Matrix:\n", conf_matrix)

#     # Save checkpoint every 5 epochs
#     if (epoch + 1) % 5 == 0:
#         checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
#         torch.save(model.state_dict(), checkpoint_path)
#         print(f"Saved checkpoint: {checkpoint_path}")

#     # Save validation results
#     with open("validation_results_binary.txt", "a") as f:
#         f.write(f"\nEpoch {epoch+1}\n")
#         f.write("Classification Report:\n" + report + "\n")
#         f.write("Confusion Matrix:\n" + np.array2string(conf_matrix) + "\n")

# # Save final model
# torch.save(model.state_dict(), "bert_sleep_binary_classifier_lora.pth")
# print("Training complete. Final model saved.")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

print("Training on chunk 0 only")

if torch.cuda.is_available():
    print(f"Training is running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training is running on CPU only.")

# ======= Setup =======
label_mapping = { "W": 0, "N3": 1 }
chunk_path = "/srv/scratch/z5298768/chronos_classification/tokenization_updated/merged_tokenized_chunk_0_to_9.pt"

# ======= Load and filter chunk 0 =======
print(f"Loading {chunk_path}")
chunk = torch.load(chunk_path)
filtered_data = [s for s in chunk if s["label"] in label_mapping]
print(f"Loaded {len(filtered_data)} W/N3 samples from chunk 0")

# ======= Split into train and validation =======
train_data, val_data = train_test_split(filtered_data, test_size=0.2, random_state=42)

def convert_to_tensors(data):
    input_ids = torch.stack([s["input_ids"] for s in data])
    attention_mask = torch.stack([s["attention_mask"] for s in data])
    labels = torch.tensor([label_mapping[s["label"]] for s in data], dtype=torch.long)
    return input_ids, attention_mask, labels

train_input_ids, train_attention_mask, train_labels = convert_to_tensors(train_data)
val_input_ids, val_attention_mask, val_labels = convert_to_tensors(val_data)

# ======= Dataset & Dataloader =======
class SleepStageDataset(torch.utils.data.Dataset):
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

train_dataset = SleepStageDataset(train_input_ids, train_attention_mask, train_labels)
val_dataset = SleepStageDataset(val_input_ids, val_attention_mask, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# ======= Model Setup =======
num_classes = 2
base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(base_model, lora_config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ======= Loss, Optimizer, Weights =======
class_counts = torch.bincount(train_labels)
weights = 1.0 / class_counts.float()
weights = weights / weights.sum()
weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# ======= Training Loop =======
epochs = 10  # Lower for test runs or wall time limits
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

    # ======= Validation =======
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            val_loss += criterion(logits, labels).item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

    report = classification_report(all_labels, all_preds, target_names=["W", "N3"])
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)

    with open("validation_results_chunk0.txt", "a") as f:
        f.write(f"\nEpoch {epoch+1}\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + np.array2string(conf_matrix) + "\n")

# ======= Save model =======
torch.save(model.state_dict(), "bert_sleep_binary_chunk0.pth")
print("Chunk 0 training complete. Model saved.")
