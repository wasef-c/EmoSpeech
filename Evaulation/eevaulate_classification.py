# Standard libraries
import os
import logging
import warnings
import random
from collections import Counter
import pandas as pd
# PyTorch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Hugging Face Transformers
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    get_scheduler,
    BertModel,
    AutoTokenizer,
)
# Hugging Face Datasets
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset

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
from functions_older import *
# mapping = {0: 'C', 1: 'N', 2: 'H', 3: 'S', 4: 'U', 5: 'F', 6: 'A', 7: 'D'}
num_labels = 8
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
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


bert_model_name = "bert-base-uncased"
BATCH_SIZE = 20

# Pre-trained model references (ViT + BERT)
image_model_name = "google/vit-base-patch16-224"
bert_model_name = "bert-base-uncased"

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



model = CombinedModelsBi(
    image_model=image_model,
    bert_model=bert_model,
    image_feature_dim=768,   # Feature dim of ViT
    bert_embedding_dim=768,  # BERT embedding dim
    combined_dim=1024,       # Combined dimension
    num_labels=num_labels
).to(device)
test = True
save_dir = r"C:\Users\Paolo\Documents\carol_emo_rec\MLLM\VIT_BERT\MSP_POD\20250110_3"
# # save_dir = r"C:\Users\Paolo\Documents\carol_emo_rec\MLLM\VIT_BERT\MSP_POD\20250109_9"
# save_dir = r"C:\Users\Paolo\Documents\carol_emo_rec\MLLM\VIT_BERT\MSP_POD\20250106_13"


checkpoint_path = os.path.join(save_dir, "best_model.pt")
# if checkpoint_path != None:
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     print("checkpoint uplaoded successfully")
model.load_state_dict(torch.load(checkpoint_path))
model.to(device)

print("\nStarting Test Evaluation...")
model.eval()
test_loss = 0
all_test_predictions, all_test_files = [], []

if test == True:
    
    dataset_name = 'cairocode/MSPP_test_wav2vec2'
    dataset = load_dataset(dataset_name)
    test_dataset = dataset['test']

    test_dataset.set_transform(train_transforms)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=test_collate_fn
    )

    with torch.no_grad():
        test_progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in test_progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            file_name = batch["file"]
            outputs = model(
                pixel_values=pixel_values,
                bert_input_ids=input_ids,
                bert_attention_mask=attention_mask
            )
            logits = outputs["logits"]

            

            predictions = torch.argmax(logits, dim=-1)
            all_test_predictions.extend(predictions.cpu().numpy())
            all_test_files.extend(file_name)

            # After testing loop completes, create a pandas DataFrame
    df = pd.DataFrame({
        "FileName": all_test_files,
        "EmoClass": all_test_predictions
    })
    # Finally, write DataFrame to CSV
    df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)

    print("Predictions saved to predictions.csv")



if test == False: 

    dataset_name = 'cairocode/MSPP_SPLIT2_wav2vec_FINAL'
    test_dataset = load_dataset(dataset_name)['test']

    # Variables to store results
    all_test_predictions = []
    all_test_labels = []  # To store ground truth labels
    all_test_files = []

    # Use the HF Dataset's map method to add the 'labels' column
    # test_dataset = test_dataset.map(map_labels)
    test_dataset.set_transform(train_transforms)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=training_collate
    )

    # Testing Loop
    with torch.no_grad():
        test_progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in test_progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            file_name = batch["file"]
            labels = batch["labels"]  # Assuming your batch includes ground truth labels

            outputs = model(
                pixel_values=pixel_values,
                bert_input_ids=input_ids,
                bert_attention_mask=attention_mask
            )
            logits = outputs["logits"]

            predictions = torch.argmax(logits, dim=-1)
            all_test_predictions.extend(predictions.cpu().numpy())
            all_test_labels.extend(labels)
            all_test_files.extend(file_name)

    # Create DataFrame for predictions
    df = pd.DataFrame({
        "file": all_test_files,
        "predictions": all_test_predictions
    })
    df.to_csv(os.path.join(save_dir, "training_test_predictions.csv"), index=False)
    print("Predictions saved to predictions.csv")

    # Calculate Accuracy
    accuracy = accuracy_score(all_test_labels, all_test_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Calculate Unweighted Average Recall (UAR)
    uar = recall_score(all_test_labels, all_test_predictions, average='macro')
    print(f"UAR: {uar * 100:.2f}%")


    
    inv_label_mapping = {v: k for k, v in label_mapping.items()}

    # Step 2: Specify an order for the numeric labels
    ordered_labels_numeric = [0, 1, 2, 3, 4, 5, 6, 7]
    ordered_labels_str = [inv_label_mapping[i] for i in ordered_labels_numeric]

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