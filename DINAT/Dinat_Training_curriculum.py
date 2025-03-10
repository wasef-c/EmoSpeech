# Standard libraries
import os
import logging
import warnings
import random
from collections import Counter

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
    get_scheduler,
    BertModel,
    AutoTokenizer,
)
# Hugging Face Datasets
from datasets import load_dataset, concatenate_datasets

# Data processing and metrics
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

# Progress bar
from tqdm.auto import tqdm

# Custom functions (from your own module)
from functions_old import *

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
# label_mapping = {
#     0: 'C',
#     1: 'N',
#     2: 'H',
#     3: 'S',
#     4: 'U',
#     5: 'F',
#     6: 'A',
#     7: 'D'
# }


warnings.filterwarnings("ignore", category=UserWarning)
label_mapping = {
    0: 'Male',
    1: 'Female',
    # 2: 'H',
    # 3: 'S',
    # 4: 'U',
    # 5: 'F',
    # 6: 'A',
    # 7: 'D'
}
column  = "Gender"

# Directories and Model Config
base_dir = r"/home/rml/Documents/pythontest/Trained_Models/Curriculum/Gender"
output_dir = create_unique_output_dir(base_dir)
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "shi-labs/dinat-mini-in1k-224"  # For processor loading if needed
checkpoint_path = None #"./DinatCurriculum/Regression/Activation/EmoDom/20250117_1/best_model.pt"
pretrain_model = model_path
bert_model_name = "bert-base-uncased"
BATCH_SIZE = 20

# Load Dataset
dataset_name = "cairocode/MSPP_MEL_6"
dataset = load_dataset(dataset_name)

# Reverse the mapping for efficient lookup
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Define a filtering function
def is_mappable(example):
    return example[column] in reverse_label_mapping

# Filter out rows with unmappable values
filtered_dataset = dataset.filter(is_mappable)

# Map the filtered dataset to add the new column
filtered_dataset = filtered_dataset.map(lambda x: {"label": reverse_label_mapping[x[column]]})


# Define a filtering function
def is_valid_transcript(example):
    return isinstance(example['transcript'], str)


# Filter out non-string transcripts in each split
filtered_dataset = filtered_dataset.filter(is_valid_transcript)


train_test_split = filtered_dataset['train'].train_test_split(test_size=0.2, seed=42)

train_val_split = train_test_split['train'].train_test_split(test_size=0.2, seed=42)

train_dataset = train_val_split['train']
val_dataset = train_val_split['test']
test_dataset = train_test_split['test']
num_labels = len(label_mapping)
print("Test dataset size:", len(test_dataset))
print("Train dataset size:", len(train_dataset))

# Transforms (assumed imported from functions_old)
train_dataset.set_transform(train_transforms)
val_dataset.set_transform(val_transforms)
test_dataset.set_transform(val_transforms)

train_sampler = CustomSampler(train_dataset)

train_loader = DataLoader(
    train_dataset,
    sampler=train_sampler,
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


# Load Models
image_model = DinatForImageClassification.from_pretrained(
    pretrain_model,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
    problem_type="single_label_classification",
).to(device)

processor = DinatForImageClassification.from_pretrained(model_path).to(device)

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name).to(device)


# ----------------------------------------------------------------------
# Initialize the Combined Regression Model
# ----------------------------------------------------------------------
model = CombinedModelsBi(
    image_model=image_model,
    bert_model=bert_model,
    image_feature_dim=512,
    bert_embedding_dim=768,
    combined_dim=512,
    num_labels=1,

).to(device)

if checkpoint_path != None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # # Option 1: Print each key on a separate line
    # for key in checkpoint.keys():
    #     print(key)

    # # Define the keywords to include and exclude
    include_keyword = "model"
    exclude_keys = {
        "image_model.classifier.weight",
        "image_model.classifier.bias",
        "fc3"
    }

    # Use dictionary comprehension to filter the keys
    filtered_checkpoint = {
        key: value for key, value in checkpoint.items()
        if include_keyword in key and key not in exclude_keys
    }

    # # Optional: Verify the filtered keys
    # print("Filtered keys to be loaded:")
    # for key in filtered_checkpoint.keys():
    #     print(f"- {key}")

    # Load the filtered state dict
    model.load_state_dict(filtered_checkpoint, strict=False)


training_args = TrainingArguments(
    output_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True
)

logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger("natten.functional").setLevel(logging.ERROR)

# Optimizer & Scheduler
optimizer = optim.AdamW(
    model.parameters(),
    lr=training_args.learning_rate,
    weight_decay=training_args.weight_decay
)

num_training_steps = len(train_loader) * training_args.num_train_epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps,
)

focal_loss = AdaptiveLearnableFocalLoss()

# Training Variables
num_epochs = training_args.num_train_epochs
patience = 10
best_val_accuracy = 0
patience_counter = 0

train_losses, val_losses, epochs_list = [], [], []

best_model_path = os.path.join(output_dir, "best_model.pt")

##############################################################################
# Training Loop
##############################################################################
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            bert_input_ids=input_ids,
            bert_attention_mask=attention_mask
        )
        logits = outputs["logits"]

        loss = focal_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item()})

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")
    lr_scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                bert_input_ids=input_ids,
                bert_attention_mask=attention_mask
            )
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

    print(
        f"Validation Loss: {avg_val_loss:.4f}, "
        f"Accuracy: {accuracy:.4f}, UAR: {uar:.4f}, F1: {f1:.4f}"
    )

    # Early Stopping based on Accuracy
    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        torch.save(model.image_model.state_dict(), "fine_tuned_image_model.pth")

        print("Validation accuracy improved. Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load Best Model
print("Loading best model for final evaluation.")
model.load_state_dict(torch.load(best_model_path))
model.to(device)

##############################################################################
# Test Evaluation
##############################################################################
print("\nStarting Test Evaluation...")
model.eval()
test_loss = 0
all_test_predictions, all_test_labels = [], []

with torch.no_grad():
    test_progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    for batch in test_progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            bert_input_ids=input_ids,
            bert_attention_mask=attention_mask
        )
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

metrics_str = (
    f"Test Loss: {avg_test_loss:.4f}, "
    f"Accuracy: {test_accuracy:.4f}, "
    f"UAR: {test_uar:.4f}, "
    f"F1: {test_f1:.4f}"
)
print(metrics_str)


# Step 1: Define your label mapping
# label_mapping = {'D': 0, 'H': 1, 'A': 2, 'O': 3, 'N': 4, 'C': 5, 'F': 6, 'S': 7}
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# Step 2: Specify an order for the numeric labels
ordered_labels_numeric = [0, 1, 2, 3, 4, 5, 6, 7]
ordered_labels_str = [label_mapping[i] for i in ordered_labels_numeric]

# Suppose these are your test labels and predictions (as numeric):
# all_test_labels = [...]
# all_test_predictions = [...]


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

# Save Metadata
save_training_metadata(
    output_dir=output_dir,
    pathstr=pretrain_model,
    dataset_name=dataset_name,
    model_type="CombinedModelsDDCA",
    super_loss_params="N/A",
    speaker_disentanglement=True,
    entropy=False,
    column="label",
    metrics=metrics_str,
    weight_decay=training_args.weight_decay,
    results=metrics_str
)

# Overall Metrics (if needed across multiple runs):
# For a single run, these will just match the test metrics.
overall_accuracy = test_accuracy
overall_UAR = test_uar
overall_F1 = test_f1
full_accuracy = test_accuracy

# Save final metrics
output_file = os.path.join(output_dir, "metrics.txt")
with open(output_file, "w") as f:
    f.write(f"Overall F1 Score: {overall_F1:.4f}\n")
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
    f.write(f"Full Accuracy: {full_accuracy:.4f}\n")
    f.write(f"Overall UAR: {overall_UAR:.4f}\n")
    f.write(f" Class Mapping :{label_mapping}\n")

print(f"Metrics saved to {output_file}")



# Step 1: Define your label mapping
# label_mapping = {'D': 0, 'H': 1, 'A': 2, 'O': 3, 'N': 4, 'C': 5, 'F': 6, 'S': 7}
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# Step 2: Specify an order for the numeric labels
ordered_labels_numeric = [0, 1, 2, 3, 4, 5, 6, 7]
ordered_labels_str = [label_mapping[i] for i in ordered_labels_numeric]

# Suppose these are your test labels and predictions (as numeric):
# all_test_labels = [...]
# all_test_predictions = [...]


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

# Save Metadata
save_training_metadata(
    output_dir=output_dir,
    pathstr=pretrain_model,
    dataset_name=dataset_name,
    model_type="CombinedModelsBi",
    super_loss_params="N/A",
    speaker_disentanglement=True,
    entropy=False,
    column="label",
    metrics=metrics_str,
    weight_decay=training_args.weight_decay,
    results=metrics_str
)

# Overall Metrics (if needed across multiple runs):
# For a single run, these will just match the test metrics.
overall_accuracy = test_accuracy
overall_UAR = test_uar
overall_F1 = test_f1
full_accuracy = test_accuracy

# Save final metrics
output_file = os.path.join(output_dir, "metrics.txt")
with open(output_file, "w") as f:
    f.write(f"Overall F1 Score: {overall_F1:.4f}\n")
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
    f.write(f"Full Accuracy: {full_accuracy:.4f}\n")
    f.write(f"Overall UAR: {overall_UAR:.4f}\n")
    f.write(f" Class Mapping :{label_mapping}\n")

print(f"Metrics saved to {output_file}")
