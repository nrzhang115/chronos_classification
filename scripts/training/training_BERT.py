import torch
import torch.nn as nn
import torch.optim as optim
from transformers import LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
from transformers import get_scheduler
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
import numpy as np

# Load tokenized dataset
data = torch.load("/srv/scratch/z5298768/chronos_classification/tokenization_updated/tokenized_epochs.pt")

# Convert dataset to Hugging Face Dataset format
def preprocess_data(data):
    input_ids = [sample["input_ids"].clone().detach().long() for sample in data]
    attention_masks = [sample["attention_mask"].clone().detach().long() for sample in data]
    labels = [1 if sample["label"] == "N3" else 0 for sample in data]  # Binary classification (0 = W, 1 = N3)

    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    })

dataset = preprocess_data(data)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

# Load Longformer model with dropout regularization
config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")
config.attention_probs_dropout_prob = 0.2  # Increased dropout
config.hidden_dropout_prob = 0.2  # Increased dropout

model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=2)

# Compute class weights
labels = np.array([sample["labels"] for sample in dataset])
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda")

# Loss function with class weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Training arguments
training_args = TrainingArguments(
    output_dir="./longformer_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Smaller batch size
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
)

# Fix: Define num_training_steps Without train_dataloader**
num_training_steps = (len(train_dataset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs

# Define optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps  
)

# Correct compute_loss Implementation
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]  # Don't pop labels, keep them in inputs
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Implement early stopping
class EarlyStoppingCallback:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0

    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = state.log_history[-1]["eval_loss"]
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True  # Stop training

# Initialize Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, lr_scheduler),  # Fixed optimizer usage
)

early_stopping = EarlyStoppingCallback(patience=3)
trainer.add_callback(early_stopping)

# Train the model
trainer.train()
