# Standard libraries
import os
import logging
import warnings
import random
from collections import Counter
from datasets import Dataset, DatasetDict

# PyTorch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Hugging Face Transformers
from transformers import (
    AutoImageProcessor,
    DinatForImageClassification,
    TrainingArguments,
    Trainer, 
    get_scheduler,
    BertModel,
    ViTForImageClassification,
    AutoTokenizer,
    AutoModel,
)

# Hugging Face Datasets
from datasets import load_dataset

# Data processing and metrics
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bar
from tqdm.auto import tqdm

# Custom functions (assume these are defined in `functions_older.py`)
from functions_older import * 
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger("natten.functional").setLevel(logging.ERROR)

# Paths and configuration
base_dir = r"C:\Users\Paolo\Documents\carol_emo_rec\MLLM\VIT_BERT\MSP_POD"
output_dir = create_unique_output_dir(base_dir)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 50
bert_embedding_dim = 768
combined_dim = 1024

# Pre-trained model references (ViT + BERT)
image_model_name = "google/vit-base-patch16-224"
bert_model_name = "bert-base-uncased"

# Dataset loading
dataset_name = 'cairocode/MSPP_W2Split_Balanced'
dataset = load_dataset(dataset_name)

# # Filter out rows where "EmoClass" == "X"
# dataset = dataset.filter(lambda example: example["label"] != 0)
# # Decrement labels by 1 for all entries in the dataset
# dataset = dataset.map(lambda example: {"label": example["label"] - 1})

# dataset.push_to_hub('cairocode/MSPP_SPLIT2_wav2vec_FINAL')
train_dataset  = dataset['train']
val_dataset = dataset['validation']
test_dataset = dataset['test']

unique_values = set(train_dataset["label"])
num_labels = len(unique_values)
label_mapping = {
    0: 'C',
    1: 'N',
    2: 'H',
    3: 'S',
    4: 'U',
    5: 'F',
    6: 'A',
    7: 'D'
}


# # Map label strings to integer IDs
# def encode_category(example):
#     example["label"] = label_mapping[example["EmoClass"]]
#     return example

# dataset = dataset.map(encode_category)

print("Mapping of categories to integers:", label_mapping)

# Split by speaker (speaker-disjoint test set)
unique_speakers = list(set(train_dataset["SpkrID"]))
test_speaker_count = int(0.2 * len(unique_speakers))


# Set transforms
train_dataset.set_transform(train_transforms)
val_dataset.set_transform(val_transforms)
test_dataset.set_transform(val_transforms)

# Dataloaders
train_loader = DataLoader(
    train_dataset,
    sampler=CustomSampler(train_dataset),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn
)

# Initialize models
processor = AutoImageProcessor.from_pretrained(image_model_name)
image_model = ViTForImageClassification.from_pretrained(
    image_model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
    problem_type='single_label_classification'
).to(device)

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name).to(device)

# Combined model
# model = CombinedModelsNew(
#     image_model=image_model,
#     bert_model=bert_model,
#     image_feature_dim=768,   # Feature dim of ViT
#     bert_embedding_dim=768,  # BERT embedding dim
#     combined_dim=1024,       # Combined dimension
#     num_labels=num_labels
# ).to(device)

model = CombinedModelsBi(
    image_model=image_model,
    bert_model=bert_model,
    image_feature_dim=768,   # Feature dim of ViT
    bert_embedding_dim=768,  # BERT embedding dim
    combined_dim=1024,       # Combined dimension
    num_labels=num_labels
).to(device)




checkpoint_path = r"C:\Users\Paolo\Documents\carol_emo_rec\MLLM\Curriculum_VIT_Trained\Regression\Domination\20250109_1\best_model.pt"
# checkpoint_path = r"C:\Users\Paolo\Documents\carol_emo_rec\MLLM\VIT_BERT\MSP_POD\20250110_1\best_model.pt"
checkpoint_path = r"C:\Users\Paolo\Documents\carol_emo_rec\MLLM\VIT_BERT\MSP_POD\20250113_2\best_model.pt"
if checkpoint_path != None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # # Option 1: Print each key on a separate line
    # # for key in checkpoint.keys():
    # #     print(key)

    # Define the keywords to include and exclude
    include_keyword = "classifier"
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


# Training configuration
training_args = TrainingArguments(
    output_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=30,
    weight_decay=0.01,
    load_best_model_at_end=True
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=training_args.learning_rate,
    weight_decay=training_args.weight_decay
)

# Learning rate scheduler
num_training_steps = len(train_loader) * training_args.num_train_epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,  # or set training_args.warmup_steps if desired
    num_training_steps=num_training_steps
)

# Focal loss
focal_loss = AdaptiveLearnableFocalLoss2()

# Early stopping
num_epochs = training_args.num_train_epochs
patience = 8
patience_counter = 0
best_val_acc = 0  # We'll track accuracy; adjust if you prefer UAR or F1

# Tracking
train_losses = []
val_losses = []
epochs_list = []

# Directory to save the best model
best_model_path = os.path.join(output_dir, "best_model.pt")

# -------------------- Training Loop --------------------
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        outputs = model(pixel_values=pixel_values,
                        bert_input_ids=input_ids,
                        bert_attention_mask=attention_mask)
        logits = outputs["logits"]

        # Combined (focal) loss
        loss = focal_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix({"combined_loss": loss.item()})

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")
    lr_scheduler.step()

    # -------------------- Validation --------------------
    model.eval()
    val_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values,
                            bert_input_ids=input_ids,
                            bert_attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = focal_loss(logits, labels)
            val_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    uar = recall_score(all_labels, all_predictions, average="macro")
    f1 = f1_score(all_labels, all_predictions, average="macro")

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    epochs_list.append(epoch + 1)

    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, UAR: {uar:.4f}, F1: {f1:.4f}")

    # Early stopping based on accuracy (choose your metric)
    if uar > best_val_acc:
        best_val_acc = uar
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        torch.save(model.image_model.state_dict(), "fine_tuned_image_model.pth")
        print("Validation uar improved. Saving best model and resetting patience counter.")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered. Stopping training.")
        break

# -------------------- Load Best Model --------------------
print("Loading best model for final evaluation.")
model.load_state_dict(torch.load(best_model_path))
model.to(device)

# -------------------- Test Evaluation --------------------
print("\nStarting Test Evaluation...")
model.eval()
test_loss = 0
all_test_predictions = []
all_test_labels = []

with torch.no_grad():
    test_progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    for batch in test_progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values,
                        bert_input_ids=input_ids,
                        bert_attention_mask=attention_mask)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)
        test_loss += loss.item()

        predictions = torch.argmax(logits, dim=-1)
        all_test_predictions.extend(predictions.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_accuracy = accuracy_score(all_test_labels, all_test_predictions)
test_uar = recall_score(all_test_labels, all_test_predictions, average="macro")
test_f1 = f1_score(all_test_labels, all_test_predictions, average="macro")

metrics = (
    f"Test Loss: {avg_test_loss:.4f}, "
    f"Accuracy: {test_accuracy:.4f}, "
    f"UAR: {test_uar:.4f}, "
    f"F1: {test_f1:.4f}"
)
print(metrics)

inv_label_mapping = {v: k for k, v in label_mapping.items()}

# Step 2: Specify an order for the numeric labels
ordered_labels_numeric = [0, 1, 2, 3, 4, 5, 6, 7]
ordered_labels_str = [label_mapping[i] for i in ordered_labels_numeric]

# Suppose these are your test labels and predictions (as numeric):
# all_test_labels = [...]
# all_test_predictions = [...]

# Step 3: Generate the confusion matrix
cm = confusion_matrix(
    y_true=all_test_labels, 
    y_pred=all_test_predictions, 
    # labels=ordered_labels_numeric
)

# Step 4: Plot the confusion matrix with custom axis labels
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=ordered_labels_str,
    yticklabels=ordered_labels_str
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")

# Save and close
save_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Confusion matrix saved to: {save_path}")

# -------------------- Save Metadata --------------------
save_training_metadata(
    output_dir=output_dir,
    pathstr=checkpoint_path,
    dataset_name=dataset_name,
    model_type="CombinedModelBi",
    super_loss_params="N/A",
    speaker_disentanglement=True,
    entropy=False,
    column="label",
    metrics=metrics,
    speakers="N/A",
    angry_weight=label_mapping,  # Adjust if you used weighting per class
    happy_weight=None,  
    neutral_weight=None,
    sad_weight=None,
    weight_decay=training_args.weight_decay,
    results=metrics
)

# Save final metrics to file
metrics_file = os.path.join(output_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(metrics + "\n")

print(f"Metrics saved to {metrics_file}")

