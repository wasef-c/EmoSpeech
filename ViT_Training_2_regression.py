import os
import logging
import warnings
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    BertModel,
    AutoTokenizer,
    get_scheduler,
)
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Custom imports from your "functions_older.py" (adjust as needed)
from functions_older import *


warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger().addHandler(logging.NullHandler())

# ----------------------------------------------------------------------
# Configuration and Paths
# ----------------------------------------------------------------------
base_dir = r"C:\Users\Paolo\Documents\carol_emo_rec\MLLM\VIT_BERT\MSP_POD"
output_dir = create_unique_output_dir(base_dir)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 50
LEARNING_RATE = 1e-5
EPOCHS = 20
WEIGHT_DECAY = 0.01
PATIENCE = 5

# Pre-trained model names (ViT + BERT)
image_model_name = "google/vit-base-patch16-224"
bert_model_name = "bert-base-uncased"

# ----------------------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------------------
dataset_name = 'cairocode/MSPP_POD_wav2vec3'
dataset = load_dataset(dataset_name)["train"]

# Example filter for valid transcript
def has_valid_transcript(example):
    return example["transcript"] is not None

dataset = dataset.filter(has_valid_transcript)

# For demonstration, let's define:
column = "valence"  # or "arousal", "score", etc.

# 1) Rename the chosen column to "label"
def rename_to_label(example):
    # Convert to float if necessary
    return {"label": float(example[column])}

# 2) Map the transformation
dataset = dataset.map(rename_to_label)



# If you have a numeric column for regression, map it to "label".
# For example, if your dataset has "valence" in  [0.0, 1.0], do:
# dataset = dataset.map(lambda x: {"label": float(x["valence"])})

# ----------------------------------------------------------------------
# Speaker-based split (Train / Val / Test)
# ----------------------------------------------------------------------
unique_speakers = list(set(dataset["SpkrID"]))
test_speaker_count = int(0.2 * len(unique_speakers))
random.seed(42)
test_speakers = set(random.sample(unique_speakers, test_speaker_count))

test_dataset = dataset.filter(lambda x: x["SpkrID"] in test_speakers)
training_set = dataset.filter(lambda x: x["SpkrID"] not in test_speakers)
split_dataset = training_set.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print("Number of unique speakers:", len(unique_speakers))
print("Test speaker count:", len(test_speakers))
print("Test dataset size:", len(test_dataset))
print("Train dataset size:", len(train_dataset))

# ----------------------------------------------------------------------
# Set Transforms and Prepare Dataloaders
# ----------------------------------------------------------------------
train_dataset.set_transform(train_transforms)
val_dataset.set_transform(val_transforms)
test_dataset.set_transform(val_transforms)

train_loader = DataLoader(
    train_dataset,
    sampler=CustomSampler(train_dataset),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)

# ----------------------------------------------------------------------
# Initialize Image + Text Models
# ----------------------------------------------------------------------
image_processor = AutoImageProcessor.from_pretrained(image_model_name)
base_image_model = ViTForImageClassification.from_pretrained(
    image_model_name,
    ignore_mismatched_sizes=True,  # We'll use the hidden states, ignoring the classifier head
).to(device)

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
base_bert_model = BertModel.from_pretrained(bert_model_name).to(device)

# ----------------------------------------------------------------------
# Initialize the Combined Regression Model
# ----------------------------------------------------------------------
model = CombinedModelsNewRegression(
    image_model=base_image_model,
    bert_model=base_bert_model,
    image_feature_dim=768,   # Feature dimension from ViT
    bert_embedding_dim=768,  # BERT embedding dimension
    combined_dim=1024,       # Combined dimension
    output_dim=1            # Single-value regression output
).to(device)

# ----------------------------------------------------------------------
# Training Setup
# ----------------------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

criterion = nn.MSELoss()

patience_counter = 0
best_val_loss = float("inf")

train_losses = []
val_losses = []
epochs_list = []
best_model_path = os.path.join(output_dir, "best_model.pt")

# ----------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Ensure labels are float for MSE
        labels = batch["labels"].float().to(device)

        # Forward pass
        outputs_dict = model(
            pixel_values=pixel_values,
            bert_input_ids=input_ids,
            bert_attention_mask=attention_mask
        )
        # Your model returns {"outputs": <tensor>} 
        # shape: [batch_size, 1]
        predictions = outputs_dict["outputs"].squeeze(-1)

        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}")
    lr_scheduler.step()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            outputs_dict = model(
                pixel_values=pixel_values,
                bert_input_ids=input_ids,
                bert_attention_mask=attention_mask
            )
            predictions = outputs_dict["outputs"].squeeze(-1)
            loss = criterion(predictions, labels)
            val_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    epochs_list.append(epoch + 1)

    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Optional: compute regression metrics if you have them
    # metrics_dict = compute_regression_metrics(all_predictions, all_labels)
    # print(metrics_dict)

    # Early stopping based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print("Validation loss improved. Saving best model and resetting patience counter.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered. Stopping training.")
            break

# ----------------------------------------------------------------------
# Load Best Model for Final Evaluation
# ----------------------------------------------------------------------
print("Loading best model for final evaluation.")
model.load_state_dict(torch.load(best_model_path))
model.to(device)

# ----------------------------------------------------------------------
# Test Evaluation
# ----------------------------------------------------------------------
print("\nStarting Test Evaluation...")
model.eval()
test_loss = 0.0
all_test_predictions = []
all_test_labels = []

with torch.no_grad():
    test_progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    for batch in test_progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)

        outputs_dict = model(
            pixel_values=pixel_values,
            bert_input_ids=input_ids,
            bert_attention_mask=attention_mask
        )
        predictions = outputs_dict["outputs"].squeeze(-1)

        loss = criterion(predictions, labels)
        test_loss += loss.item()

        all_test_predictions.extend(predictions.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# If you have your own regression metrics function:
# test_metrics = compute_regression_metrics(all_test_predictions, all_test_labels)
# print("Test Metrics:", test_metrics)

# ----------------------------------------------------------------------
# Save Final Results & Metadata
# ----------------------------------------------------------------------
final_metrics_str = f"Test Loss: {avg_test_loss:.4f}"
save_training_metadata(
    output_dir=output_dir,
    pathstr=image_model_name,
    dataset_name=dataset_name,
    model_type="CombinedModelsNewRegression",
    super_loss_params="N/A",
    speaker_disentanglement=True,
    entropy=False,
    column="label",
    metrics=final_metrics_str,
    speakers="N/A",
    angry_weight=None,
    happy_weight=None,
    neutral_weight=None,
    sad_weight=None,
    weight_decay=WEIGHT_DECAY,
    results=final_metrics_str
)

metrics_file = os.path.join(output_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(final_metrics_str + "\n")

print(f"Metrics saved to {metrics_file}")
