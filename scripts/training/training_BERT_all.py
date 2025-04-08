import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sleep stage label mapping (excluding "unknown")
label_mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
id2label = {v: k for k, v in label_mapping.items()}

# Load the large pre-tokenized dataset from disk
dataset_path = "/srv/scratch/z5298768/chronos_classification/tokenization_updated/merged_tokenized_chunk_0_to_9.pt"
data = torch.load(dataset_path, map_location='cpu')

# Filter and remap labels
filtered_data = [sample for sample in data if sample["label"] in label_mapping]

# Custom Dataset
class SleepStageDataset(Dataset):
    def __init__(self, data, label_mapping):
        self.input_ids = torch.stack([sample["input_ids"] for sample in data])
        self.attention_mask = torch.stack([sample["attention_mask"] for sample in data])
        self.labels = torch.tensor([label_mapping[sample["label"]] for sample in data], dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

    def __len__(self):
        return len(self.labels)

# Split data (do this before instantiating Dataset)
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(filtered_data, test_size=0.2, random_state=42)

train_dataset = SleepStageDataset(train_data, label_mapping)
val_dataset = SleepStageDataset(val_data, label_mapping)

# Compute class weights
train_labels = train_dataset.labels
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels.numpy()), y=train_labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_mapping),
    id2label=id2label,
    label2id=label_mapping
).to(device)

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank: usually 4–16
    lora_alpha=16,
    target_modules=["query", "value"],  # attention projection layers
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS  # sequence classification
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # optional: see how few parameters you're tuning

# Move to device
model.to(device)

# Use class weights in loss function
def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fn(logits, labels)
    return (loss, outputs) if return_outputs else loss

# Metrics
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    report = classification_report(p.label_ids, preds, target_names=label_mapping.keys(), zero_division=0, output_dict=False)
    print(report)
    conf_matrix = confusion_matrix(p.label_ids, preds)
    print("Confusion Matrix:\n", conf_matrix)
    return {"accuracy": acc}

# TrainingArguments with checkpointing and resuming
output_dir = "./bert_sleep_allclass"
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{output_dir}/logs",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # Enable mixed precision if possible
    push_to_hub=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    compute_loss=compute_loss,
)

# Resume from checkpoint if exists
last_checkpoint = None
if os.path.isdir(output_dir) and any("checkpoint" in d for d in os.listdir(output_dir)):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if "checkpoint" in d]
    last_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Resuming training from checkpoint: {last_checkpoint}")

# Train model
trainer.train(resume_from_checkpoint=last_checkpoint)

# Save final full model
trainer.save_model(f"{output_dir}/final_model")

# Also save just the LoRA adapters
model.save_pretrained(f"{output_dir}/lora_only")

print("Training complete. Best model saved.")
