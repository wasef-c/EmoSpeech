
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
checkpoint_path = r"C:\Users\Paolo\Documents\carol_emo_rec\MLLM\Currciulum_Models\Speaker\20250107_6\best_model.pt"
base_dir = r"./Curriculum/Regression/Activation"
output_dir = create_unique_output_dir(base_dir)
os.makedirs(output_dir, exist_ok=True)


column =  "EmoDom" # "EmoVal" #"EmoAct"  # or "arousal", "score", etc.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 52
LEARNING_RATE = 1e-5
EPOCHS = 52
WEIGHT_DECAY = 0.01
PATIENCE = 10

# Pre-trained model names (ViT + BERT)
image_model_name = "google/vit-base-patch16-224"
bert_model_name = "bert-base-uncased"

# ----------------------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------------------
dataset_name = 'cairocode/MSPP_Wav2Vec_4'
dataset = load_dataset(dataset_name)["train"]

# Example filter for valid transcript


def has_valid_transcript(example):
    return example["transcript"] is not None


dataset = dataset.filter(has_valid_transcript)

# For demonstration, let's define:

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
image_model = ViTForImageClassification.from_pretrained(
    image_model_name,
    ignore_mismatched_sizes=True,
    num_labels=1,
    problem_type="regression"
).to(device)

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
base_bert_model = BertModel.from_pretrained(bert_model_name).to(device)

# ----------------------------------------------------------------------
# Initialize the Combined Regression Model
# ----------------------------------------------------------------------
model = CombinedModelsBi(
    image_model=image_model,
    bert_model=bert_model,
    image_feature_dim=768,   # Feature dim of ViT
    bert_embedding_dim=768,  # BERT embedding dim
    combined_dim=1024,       # Combined dimension
    num_labels=1
).to(device)



checkpoint = torch.load(checkpoint_path, map_location=device)

# Option 1: Print each key on a separate line
# for key in checkpoint.keys():
#     print(key)


# Define the keywords to include and exclude
include_keyword = "image_model"
exclude_keys = {
    "image_model.classifier.weight",
    "image_model.classifier.bias"
}

# Use dictionary comprehension to filter the keys
filtered_checkpoint = {
    key: value for key, value in checkpoint.items()
    if include_keyword in key and key not in exclude_keys
}

# Optional: Verify the filtered keys
print("Filtered keys to be loaded:")
for key in filtered_checkpoint.keys():
    print(f"- {key}")


# Load the filtered state dict
model.load_state_dict(filtered_checkpoint, strict=False)


# ----------------------------------------------------------------------
# Training Setup
# ----------------------------------------------------------------------
optimizer = optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# import torch
# import torch.nn as nn

class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)
        
        y_true_var = torch.var(y_true, unbiased=False)
        y_pred_var = torch.var(y_pred, unbiased=False)
        
        covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
        
        ccc = (2 * covariance) / (
            y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2 + 1e-8
        )  # Adding epsilon to avoid division by zero
        
        loss = 1 - ccc
        return loss


criterion = CCCLoss()

patience_counter = 0
best_val_loss = 0 # float("inf")

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
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

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
        predictions = outputs_dict["logits"].squeeze(-1)

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
            predictions = outputs_dict["logits"].squeeze(-1)
            loss = criterion(predictions, labels)
            val_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    epochs_list.append(epoch + 1)
    metrics = compute_regression_metrics(all_predictions, all_labels)

    print(f"Validation Loss: {avg_val_loss:.4f} Metrics: {metrics}")

    # Optional: compute regression metrics if you have them
    # metrics_dict = compute_regression_metrics(all_predictions, all_labels)
    # print(metrics_dict)

    # Early stopping based on validation loss
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     patience_counter = 0
    #     torch.save(model.state_dict(), best_model_path)
    #     print("Validation loss improved. Saving best model and resetting patience counter.")
    # else:
    #     patience_counter += 1
    #     if patience_counter >= PATIENCE:
    #         print("Early stopping triggered. Stopping training.")
    #         break
    if metrics['CCC'] > best_val_loss:
        best_val_loss = metrics['CCC']
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
        predictions = outputs_dict["logits"].squeeze(-1)

        loss = criterion(predictions, labels)
        test_loss += loss.item()

        all_test_predictions.extend(predictions.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
metrics = compute_regression_metrics(all_test_predictions, all_test_labels)

print(f"Test Loss: {avg_test_loss:.4f} Metrics: {metrics}")

# If you have your own regression metrics function:
# test_metrics = compute_regression_metrics(all_test_predictions, all_test_labels)
# print("Test Metrics:", test_metrics)

# ----------------------------------------------------------------------
# Save Final Results & Metadata
# ----------------------------------------------------------------------
final_metrics_str = f"Test Loss: {avg_test_loss:.4f} Metrics: {metrics}"
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
    angry_weight="CCC loss fn",
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
