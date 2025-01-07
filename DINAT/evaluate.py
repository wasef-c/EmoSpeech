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

BEST_MODEL = "dir"
num_labels = 8
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)



DATASET_PATH = "../data"
CHECKPOINT_PATH = "./NLPIMG_Model_001"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "shi-labs/dinat-mini-in1k-224"  # For processor loading if needed
pretrain_model = "/media/carol/Data/Documents/Emo_rec/Trained Models/DINAT/MSPP_PRE/REGRESSION/GSAV/model"

bert_model_name = "bert-base-uncased"
BATCH_SIZE = 20


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

# Combined Model
model = CombinedModelsDCCA(
    image_model=image_model,
    bert_model=bert_model,
    image_feature_dim=512,
    bert_embedding_dim=768,
    num_labels=8,
    latent_dim=16,
).to(device)

# model.load_state_dict(torch.load(os.path.join(BEST_MODEL, "best_model.pt")))


print("\nStarting Test Evaluation...")
model.eval()
test_loss = 0
all_test_predictions, all_test_files = [], []


dataset_name = 'cairocode/MSPP_test_oldfeatures'
dataset = load_dataset(dataset_name)
test_dataset = dataset['train']

test_dataset.set_transform(train_transforms)

def test_collate_fn(examples):
    """
    Custom collate function to handle batching of image data and BERT inputs.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples]).to(device)
    input_ids = torch.stack([example["input_ids"] for example in examples]).to(device)
    attention_mask = torch.stack([example["attention_mask"] for example in examples]).to(device)
    file = torch.tensor([example["file"] for example in examples]).to(device)
    bert_embeddings = torch.stack([example["bert_embeddings"] for example in examples]).to(device)
    


    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "file": file,
        "bert_embeddings":bert_embeddings
    }

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
        file = batch[file].to(device)

        outputs = model(
            pixel_values=pixel_values,
            bert_input_ids=input_ids,
            bert_attention_mask=attention_mask
        )
        logits = outputs["logits"]

        

        predictions = torch.argmax(logits, dim=-1)
        all_test_predictions.extend(predictions.cpu().numpy())
        all_test_files.extend(file.cpu().numpy())

        # After testing loop completes, create a pandas DataFrame
df = pd.DataFrame({
    "file": all_test_files,
    "predictions": all_test_predictions
})

# Finally, write DataFrame to CSV
df.to_csv(os.path.join(BEST_MODEL, "test_predictions.csv"), index=False)

print("Predictions saved to predictions.csv")