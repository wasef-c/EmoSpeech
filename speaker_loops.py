# Standard libraries
import os
import logging
import warnings
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
    Trainer,
    get_scheduler,
    BertModel,
    AutoImageProcessor,
    ViTForImageClassification
)

# Hugging Face Datasets
from datasets import load_dataset, concatenate_datasets

# Data processing and metrics
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output

# Progress bar
from tqdm.auto import tqdm

# Custom functions (from your own module)
from functions_older import *

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# old_model = "/media/carol/Data/Documents/Emo_rec/Notebooks/NLP_IMG/Ver1/DinatBert/manual_trained_model.pth"
# pretrain_model = "/media/carol/Data/Documents/Emo_rec/Trained Models/DINAT/MSPP_PRE/REGRESSION/GSAV/model"

# # Load the state_dict
# state_dict = torch.load(old_model)
# model.load_state_dict(state_dict, strict=False)
base_dir = "/media/carol/Data/Documents/Emo_rec/Notebooks/NLP_IMG/Ver1/DinatBert"
output_dir = create_unique_output_dir(base_dir)


# Configuration
DATASET_PATH = "../data"
CHECKPOINT_PATH = "./NLPIMG_Model_001"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'shi-labs/dinat-mini-in1k-224'
pretrain_model = "/media/carol/Data/Documents/Emo_rec/Trained Models/DINAT/MSPP_PRE/REGRESSION/GSAV/model"
# pretrain_model = "/media/carol/Data/Documents/Emo_rec/Trained Models/DINAT/IEMOCAP/Pre_SD_wtd_006/model"

bert_embedding_dim = 768
combined_dim = 1024
num_labels = 4

# Load Dataset
dataset_name = 'cairocode/IEMOCAP_IMG_NLP'
dataset = load_dataset(dataset_name)
dataset = dataset.filter(filter_m_examples)
# train_dataset = dataset['train']
# validation_dataset = dataset['validation']
# test_dataset = dataset['test']

# # Concatenate the datasets
# combined_dataset = concatenate_datasets([train_dataset, validation_dataset, test_dataset])
combined_dataset = dataset
os.makedirs(output_dir, exist_ok=True)


BATCH_SIZE = 20

spkrs = [sample['speakerID'] for sample in combined_dataset]
unique_speakers = list(set(spkrs))

# image_model = DinatForImageClassification.from_pretrained(pretrain_model,num_labels=num_labels,  ignore_mismatched_sizes=True, problem_type = 'single_label_classification').to(device)
# processor = DinatForImageClassification.from_pretrained(model_path).to(device)


image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224")


bert_model_name = "bert-base-uncased"
# bert_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
# Initialize Combined Model
image_feature_dim = 512  # 512
# image_feature_dim = 7 #512

bert_embedding_dim = 768
combined_dim = 1024
num_labels = 4

a = 1
angry_weight = a
happy_weight = 1
neutral_weight = 1.2
sad_weight = a

class_weight_multipliers = {
    0: neutral_weight,
    1: happy_weight,
    2: sad_weight,
    3: angry_weight
}

training_args = TrainingArguments(
    output_dir="./logs",
    evaluation_strategy="epoch",  # Evaluates at the end of each epoch
    save_strategy="epoch",        # Saves the model at the end of each epoch
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True   # Loads the best model based on evaluation metrics
)

logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger("natten.functional").setLevel(logging.ERROR)

num_epochs = training_args.num_train_epochs
patience = 10
best_val_uar = 0
patience_counter = 0

overall_accuracy = 0
overall_UAR = 0
overall_F1 = 0

overall_labels = []
overall_preds = []

for i in range(len(unique_speakers)):

    image_model = DinatForImageClassification.from_pretrained(
        pretrain_model, num_labels=4,  ignore_mismatched_sizes=True, problem_type='single_label_classification').to(device)
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    # Define the combined model
    model = CombinedModelsNew(
        image_model=image_model,
        bert_model=bert_model,
        image_feature_dim=512,  # Match old model dimensions
        bert_embedding_dim=768,
        combined_dim=1024,  # Match the old combined_dim
        num_labels=4,  # Match the old number of labels
    )

    num_epochs = training_args.num_train_epochs
    patience = 5
    best_val_uar = 0
    patience_counter = 0

    speakers = [937+i]  # speakers = [937+i]
    print(f"\n {'#'*120}")
    print(
        f"                                          STARTING SPEAKER {i}                                                      ")
    print(f"\n {'#'*120}")

    new_model_path = os.path.join(output_dir, str(i))
    os.makedirs(new_model_path, exist_ok=True)

    # Create the test split
    test_dataset = combined_dataset.filter(
        lambda x: x['speakerID'] in speakers).filter(filter_m_examples)

    # Create the remaining data
    train_data = combined_dataset.filter(
        lambda x: x['speakerID'] not in speakers).filter(filter_m_examples)

    train_split = train_data.train_test_split(test_size=0.2, seed=42)

    # Get the new splits
    train_dataset = train_split['train']
    val_dataset = train_split['test']

    train_dataset.set_transform(train_transforms)

    val_dataset.set_transform(val_transforms)
    test_dataset.set_transform(val_transforms)
    train_sampler = CustomSampler(train_dataset)

    # DataLoader with Collate Function
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        # shuffle=True
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

    class_weights = calculate_class_weights(
        train_dataset, class_weight_multipliers)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = CombinedModelsNew(
        image_model=image_model,
        bert_model=bert_model,
        image_feature_dim=512,  # Match old model dimensions
        bert_embedding_dim=768,
        combined_dim=1024,  # Match the old combined_dim
        num_labels=4,  # Match the old number of labels
    )

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(
    ), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    # Define the learning rate scheduler
    num_training_steps = len(train_loader) * training_args.num_train_epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps
    )

    class_weights = torch.tensor(class_weights).to(device)

    # focal_loss = FocalLoss(alpha=1, gamma=2, class_weights=class_weights)
    # focal_loss = AdaptiveLearnableFocalLoss(class_weights=class_weights, learnable = False)
    focal_loss = AdaptiveLearnableFocalLoss(class_weights=class_weights)

    # Initialize lists to store loss values
    train_losses = []
    val_losses = []
    epochs_list = []

    # Create the figure and axis for the plot
    fig, ax = plt.subplots()
    train_line, = ax.plot([], [], label="Training Loss", color='blue')
    val_line, = ax.plot([], [], label="Validation Loss", color='orange')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Live Loss Curve")
    ax.legend()
    ax.grid()

    # Directory to save the best model
    best_model_path = os.path.join(new_model_path, "best_model.pt")

    # Training loop with Early Stopping
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        bert_train_loss = 0
        img_train_loss = 0
        all_trained_preds = []

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values,
                            bert_input_ids=input_ids, bert_attention_mask=attention_mask)

            logits = outputs["logits"]

            # Loss for combined features
            combined_loss = focal_loss(logits, labels)

            optimizer_combined = torch.optim.Adam(
                model.parameters(), lr=1e-5, weight_decay=1e-5)

            optimizer_combined.zero_grad()
            combined_loss.backward()
            optimizer_combined.step()

            # Update progress bar
            train_loss += combined_loss.item()

            # , "image_loss": image_loss.item(),"bert_loss": bert_loss.item(),})
            progress_bar.set_postfix({"combined_loss": combined_loss.item()})

            predictions = torch.argmax(logits, dim=-1)
            all_trained_preds.extend(predictions.cpu().numpy())

            # Class distribution every 50 batches
            # if i % 50 == 0:
            #     class_counts = Counter(all_trained_preds)
            #     clean_class_counts = {int(k): v for k, v in class_counts.items()}
            #     print("Predicted class distribution:", clean_class_counts)
            #     all_trained_preds = []

        # Img Loss: {img_train_loss / len(train_loader):.4f}  Bert Loss: {bert_train_loss / len(train_loader):.4f} ")
        print(
            f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss / len(train_loader):.4f}  ")
        lr_scheduler.step()

        # Evaluation on validation set
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
                bert_embeddings = batch["bert_embeddings"].to(device)

                # Forward pass
                # outputs = model(pixel_values=pixel_values, bert_embeddings=bert_embeddings)
                outputs = model(pixel_values=pixel_values,
                                bert_input_ids=input_ids, bert_attention_mask=attention_mask)

                logits = outputs["logits"]

                # Compute Focal Loss
                loss = focal_loss(logits, labels)
                val_loss += loss.item()

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        uar = recall_score(all_labels, all_predictions, average="macro")
        f1 = f1_score(all_labels, all_predictions, average="macro")

        # Append loss values
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(avg_val_loss)
        epochs_list.append(epoch + 1)

        # Update the data for the plot
        train_line.set_data(epochs_list, train_losses)
        val_line.set_data(epochs_list, val_losses)

        # Adjust the axis limits
        ax.relim()
        ax.autoscale_view()

        # Display the updated plot
        # clear_output(wait=True)
        display(fig)

        # Optionally add a pause to control update speed (e.g., 0.1 seconds)
        plt.pause(0.1)
        print(
            f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, UAR: {uar:.4f}, F1: {f1:.4f}")

        # Early stopping logic based on UAR
        if accuracy > best_val_uar:
            best_val_uar = accuracy
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(
                "Validation accuracy improved. Saving best model and resetting patience counter.")
        else:
            patience_counter += 1

        # Check if we should stop early
        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

        plt.show()

    print("Loading best model for final evaluation.")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    model.to(device)

    # Test Loop
    print("\nStarting Test Evaluation...")
    model.eval()
    test_loss = 0
    all_test_predictions = []
    all_test_labels = []

    with torch.no_grad():
        # Progress bar for testing
        test_progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in test_progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            bert_embeddings = batch["bert_embeddings"].to(device)

            outputs = model(pixel_values=pixel_values,
                            bert_input_ids=input_ids, bert_attention_mask=attention_mask)
            logits = outputs["logits"]
            # Compute loss
            loss = F.cross_entropy(logits, labels)
            test_loss += loss.item()

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            all_test_predictions.extend(predictions.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    # Compute test metrics
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = accuracy_score(all_test_labels, all_test_predictions)
    test_uar = recall_score(
        all_test_labels, all_test_predictions, average="macro")
    test_f1 = f1_score(all_test_labels, all_test_predictions, average="macro")

    print(
        f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}, UAR: {test_uar:.4f}, F1: {test_f1:.4f}")

    metrics = f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}, UAR: {test_uar:.4f}, F1: {test_f1:.4f}"

    # Confusion Matrix
    cm = confusion_matrix(all_test_labels, all_test_predictions)
    classes = np.unique(all_test_labels)

    # Plotting Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Save the plot
    save_path = os.path.join(new_model_path, "confusion_matrix.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    # plt.close()  # Close the figure to free up memory
    plt.show()

    print(f"Confusion matrix saved to: {save_path}")

    pathstr = pretrain_model
    model_type = "CombinedModel"
    dataset_name = dataset_name
    speaker_disentanglement = True  # Update this based on your configuration
    entropy = False  # Update based on your configuration
    column = "label"
    # speakers = i
    weight_decay = training_args.weight_decay

    overall_labels.extend(all_test_labels)
    overall_preds.extend(all_test_predictions)

    # Save metadata after training
    save_training_metadata(
        output_dir=new_model_path,
        pathstr=pathstr,
        dataset_name=dataset_name,
        model_type=model_type,
        super_loss_params="N/A",
        speaker_disentanglement=speaker_disentanglement,
        entropy=entropy,
        column=column,
        metrics=metrics,
        speakers=speakers,
        angry_weight=angry_weight,
        happy_weight=happy_weight,
        neutral_weight=neutral_weight,
        sad_weight=sad_weight,
        weight_decay=weight_decay,
        results=metrics

    )
    overall_F1 += test_f1
    overall_accuracy += test_accuracy
    overall_UAR += test_uar

    torch.cuda.empty_cache()
    del model
    del image_model
    del bert_model


# Assuming these variables are already computed in your script
overall_F1 /= len(unique_speakers)
overall_accuracy /= len(unique_speakers)
overall_UAR /= len(unique_speakers)

# Print the metrics
print(f"Overall F1 Score: {overall_F1:.4f}")
print(f"Overall Accuracy: {overall_accuracy:.4f}")
print(f"Overall UAR: {overall_UAR:.4f}")

# File path for saving the metrics
output_file = os.path.join(output_dir, "metrics.txt")

full_accuracy = accuracy_score(overall_labels, overall_preds)

# Save the metrics to a text file
with open(output_file, "w") as f:
    f.write(f"Overall F1 Score: {overall_F1:.4f}\n")
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
    f.write(f"Full Accuracy: {full_accuracy:.4f}\n")

    f.write(f"Overall UAR: {overall_UAR:.4f}\n")

print(f"Metrics saved to {output_file}")
