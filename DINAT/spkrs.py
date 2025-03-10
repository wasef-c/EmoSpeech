## Standard libraries
#BEST SO FAR: 20250128_14
import os
import logging
import warnings
import random
from collections import Counter
import math

# PyTorch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.utils import resample

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
best_epoch = 0
# Custom functions (from your own module)
from functionsV3 import *
import numpy as np
import os
import logging
import warnings
import random
from collections import Counter, defaultdict
import math

# PyTorch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset

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

from functionsV3 import *

def main():
    ## Standard libraries

    best_epoch = 0
    # Custom functions (from your own module)

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set base_column to SpkrID explicitly
    base_column = "SpkrID"

    # Directories and Model Config
    base_dir = "/home/rml/Documents/pythontest/Trained_Models/curr/spkr"
    output_dir = create_unique_output_dir(base_dir)
    os.makedirs(output_dir, exist_ok=True)

    DATASET_PATH = "../data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "shi-labs/dinat-mini-in1k-224"  # For processor loading if needed
    checkpoint_path = None  # Remove previous checkpoint path
    pretrain_model = model_path
    bert_model_name = "bert-base-uncased"
    BATCH_SIZE = 30

    # Load Dataset
    dataset_name = "cairocode/MSPP_MEL_6"
    dataset = load_dataset(dataset_name)

    unique_labels = set(dataset['train'][base_column])
    print("Unique labels in dataset:", unique_labels)

    random.seed(42)  # For reproducibility
    selected_speakers = set(random.sample(unique_labels, 200))

    # Filter dataset to keep only samples from the selected speakers
    ds1 = dataset['train'].filter(lambda x: x[base_column] in selected_speakers)

    unique_labels = set(ds1[base_column])


    label_mapping = {label: i for i, label in enumerate(sorted(unique_labels))}
    # Reverse the mapping for efficient lookup
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    num_labels = len(label_mapping)

    # Define a filtering function
    def is_mappable(example):
        return example[base_column] in label_mapping

    # column = "label"

    # Filter out rows with unmappable values
    filtered_dataset = ds1.filter(is_mappable)

    # Map the filtered dataset to add the new column
    filtered_dataset = filtered_dataset.map(lambda x: {"label": label_mapping[x[base_column]]})

    # Define a filtering function for transcripts
    def is_valid_transcript(example):
        return isinstance(example['transcript'], str)

    # Filter out non-string transcripts
    filtered_dataset = filtered_dataset.filter(is_valid_transcript)

    inv_label_mapping = {v: k for k, v in label_mapping.items()}

    # Specify an order for displaying labels
    ordered_labels_str = [inv_label_mapping[i] for i in range(num_labels)]

    # MODIFIED: New data splitting approach to ensure each speaker appears in all sets
    # Get all unique speakers
    all_speakers = list(set(filtered_dataset[base_column]))
    print(f"Total unique speakers: {len(all_speakers)}")

    # Create a dictionary to track examples per speaker
    examples_by_speaker = defaultdict(list)
    for i, example in enumerate(filtered_dataset):
        examples_by_speaker[example[base_column]].append(i)

    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    column = 'label'

    # Initialize empty lists for train, val, test indices
    train_indices = []
    val_indices = []
    test_indices = []

    # For each speaker, split their examples into train (70%), val (15%), test (15%)
    for speaker, indices in examples_by_speaker.items():
        random.shuffle(indices)
        
        # Calculate split points
        n_total = len(indices)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        # Split indices
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    # Create the datasets using the indices
    train_dataset = filtered_dataset.select(train_indices)
    val_dataset = filtered_dataset.select(val_indices)
    test_dataset = filtered_dataset.select(test_indices)

    # Print dataset sizes
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    # Verify that each speaker appears in all splits
    train_speakers = set(train_dataset[base_column])
    val_speakers = set(val_dataset[base_column])
    test_speakers = set(test_dataset[base_column])

    print(f"Unique speakers in train: {len(train_speakers)}")
    print(f"Unique speakers in val: {len(val_speakers)}")
    print(f"Unique speakers in test: {len(test_speakers)}")

    # Check the intersection of speakers across splits
    common_speakers = train_speakers.intersection(val_speakers).intersection(test_speakers)
    print(f"Speakers appearing in all splits: {len(common_speakers)} of {len(all_speakers)}")

    # Apply transforms
    train_dataset.set_transform(train_transforms)
    val_dataset.set_transform(val_transforms)
    test_dataset.set_transform(val_transforms)

    # Create data loaders (no custom sampler needed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda examples: collate_fn_reg(examples, column=column),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda examples: collate_fn_reg(examples, column=column),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda examples: collate_fn_reg(examples, column=column),
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

    # Initialize the Combined Model
    unfozen_layers = [10, 11]
    next_layer_to_unfreeze = unfozen_layers[0] - 1

    model = CombinedModelsBi(
        image_model=image_model,
        bert_model=bert_model,
        image_feature_dim=512,
        bert_embedding_dim=768,
        combined_dim=512,
        num_labels=num_labels,
        unfrozen_layers=unfozen_layers
    ).to(device)

    # Configure training
    training_args = TrainingArguments(
        output_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-6,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=50,
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

    # Use standard cross entropy with contrastive loss, but don't apply balanced weighting
    cecc_loss = CrossEntropyWithContrastiveCenterLoss(
        num_classes=num_labels,
        feature_dim=512,
        alpha=1.0,
        beta=0.01
    )

    # Training Variables
    num_epochs = training_args.num_train_epochs
    patience = 12
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
            features = outputs["combined_features"]

            # Use simple cross entropy with contrastive loss
            loss = cecc_loss(logits, features, labels)

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
                features = outputs["combined_features"]

                loss = cecc_loss(logits, features, labels)
                val_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        uar = recall_score(all_labels, all_predictions, average="macro")
        f1 = f1_score(all_labels, all_predictions, average="macro")
        per_class_recall = recall_score(all_labels, all_predictions, average=None)
        uar_std = np.std(per_class_recall)

        # Use simple UAR as comparison metric instead of penalizing by std
        comparison_metric = uar

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs_list.append(epoch + 1)

        print(
            f"Validation Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, UAR: {uar:.4f}, F1: {f1:.4f}, UAR STD: {uar_std} "
            f"\nPer-class Recall: {per_class_recall}"
        )
        
        # Save confusion matrix
        plot_and_save_confusion_matrix(all_labels, all_predictions, ordered_labels_str, output_dir, epoch=epoch)

        # Early Stopping based on UAR
        if comparison_metric > best_val_accuracy:
            best_val_accuracy = comparison_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch
            print("Validation UAR improved. Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Gradually unfreeze BERT layers
        if (epoch + 1) % 3 == 0 and next_layer_to_unfreeze >= 0:
            print(f"Unfreezing BERT layer {next_layer_to_unfreeze}")
            unfreeze_bert_layer(model.bert_model, next_layer_to_unfreeze)
            next_layer_to_unfreeze -= 1

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

    # Generate final confusion matrix
    plot_and_save_confusion_matrix(all_test_labels, all_test_predictions, ordered_labels_str, output_dir, epoch=None)

    # Overall Metrics
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
        f.write(f"Class Mapping: {label_mapping}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Training speakers: {len(train_speakers)}\n")
        f.write(f"Validation speakers: {len(val_speakers)}\n")
        f.write(f"Test speakers: {len(test_speakers)}\n")
        f.write(f"Speakers in all splits: {len(common_speakers)}\n")

    print(f"Metrics saved to {output_file}")
if __name__ == "__main__":
    main()