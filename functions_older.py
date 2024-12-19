import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import accuracy_score, recall_score, f1_score, top_k_accuracy_score
from torch.utils.data import Dataset, DataLoader, Sampler
import os
from datetime import datetime
import logging
from scipy.special import lambertw  # Add this import at the top of your file

from typing import Dict  # Add this import
from transformers import AutoImageProcessor, DinatForImageClassification, TrainingArguments, Trainer, AutoTokenizer, AutoModel
from sklearn.utils.class_weight import compute_class_weight


# Initialize the image processor and BERT tokenizer
image_processor = AutoImageProcessor.from_pretrained("shi-labs/dinat-mini-in1k-224")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global bert_model

# Initialize BERT
bert_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
def filter_m_examples(example):
    return example["label"] != 4 and example["label"] != 5 

# Data Transformations
# Data Transformations
def compute_bert_embeddings(transcripts):
    # Tokenize the transcripts and generate input_ids and attention_mask
    inputs = tokenizer(
        transcripts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        return_attention_mask=True  # Ensure the attention mask is returned
    )
    # Move the inputs to the device (e.g., GPU if available)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get the BERT embeddings without gradient computation
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Extract the CLS token embedding (first token) as BERT embeddings
    bert_embeddings = outputs.last_hidden_state[:, 0, :]

    # Return both the BERT embeddings and the attention mask
    return bert_embeddings, inputs["attention_mask"], inputs["input_ids"]


def get_transforms(new_size=224):
    return Compose([
        Resize((new_size, new_size)),
        ToTensor()
    ])

# Custom Collate Function
def collate_fn(examples):
    """
    Custom collate function to handle batching of image data and BERT inputs.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples]).to(device)
    input_ids = torch.stack([example["input_ids"] for example in examples]).to(device)
    attention_mask = torch.stack([example["attention_mask"] for example in examples]).to(device)
    labels = torch.tensor([example["label"] for example in examples]).to(device)
    bert_embeddings = torch.stack([example["bert_embeddings"] for example in examples]).to(device)


    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "bert_embeddings":bert_embeddings
    }



_test_transforms = Compose(
    [
        Resize((224, 224)),
        # RandomWindowCrop(windows, size),
        # Resize((new_size, new_size)),
        ToTensor(),
        # normalize,
    ]
)


def test_transforms(examples):
    examples['pixel_values'] = [_test_transforms(image.convert("RGB")) for image in examples['image']]
    return examples


# Define Transformations
_train_transforms = Compose([
    Resize((224, 224)),
    ToTensor()
])

_val_transforms = Compose([
    Resize((224, 224)),
    ToTensor()
])

def train_transforms(examples):
    # print("train transform: ", examples)

    # Use the image processor to process the images
    # processed_images = [image_processor(image.convert("RGB"))["pixel_values"].squeeze(0) for image in examples['image']]
    processed_images = [_train_transforms(image.convert("RGB")) for image in examples['image']]

    # Use the BERT tokenizer for transcripts
    transcripts = examples['transcript']
    bert_embeddings, attention_mask, input_ids= compute_bert_embeddings(transcripts)

    examples['pixel_values'] = processed_images
    examples['bert_embeddings'] = bert_embeddings
    examples['attention_mask'] = attention_mask
    examples['input_ids'] = input_ids


    # print("after transform", examples)
    return examples

def val_transforms(examples):
    # Use the image processor to process the images
    # processed_images = [image_processor(image.convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0) for image in examples['image']]
    processed_images = [_train_transforms(image.convert("RGB")) for image in examples['image']]

    # Use the BERT tokenizer for transcripts
    transcripts = examples['transcript']
    bert_embeddings, attention_mask, input_ids= compute_bert_embeddings(transcripts)

    examples['pixel_values'] = processed_images
    examples['bert_embeddings'] = bert_embeddings
    examples['attention_mask'] = attention_mask
    examples['input_ids'] = input_ids
    return examples

# Custom Dataset and Sampler
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    

class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.num_samples = len(self.data_source)

    def _create_group_indices(self, shuffled_indices):
        group_indices = {}
        for idx in shuffled_indices:
            speaker_id = self.data_source[idx]['speakerID']
            if speaker_id not in group_indices:
                group_indices[speaker_id] = []
            group_indices[speaker_id].append(idx)
        return list(group_indices.values())

    def __iter__(self):
        # Shuffle the entire dataset initially
        shuffled_indices = list(range(self.num_samples))
        random.shuffle(shuffled_indices)
        
        # Group the shuffled indices by speakerID
        self.group_indices = self._create_group_indices(shuffled_indices)
        
        # Shuffle the groups
        random.shuffle(self.group_indices)
        
        # Flatten indices after shuffling groups
        final_indices = [idx for group in self.group_indices for idx in group]
        return iter(final_indices)

    def __len__(self):
        return self.num_samples

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
(C)	Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       PLEASE CITE THE ABOVE PAPER IF YOU USE THIS CODE.
'''

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, embed_dim, num_heads, dropout_prob=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_prob,  batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, attention_mask=None):
        # Project query, key, and value to the same embedding dimension
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        # Apply multihead attention
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attention_mask)

        # Residual connection and layer normalization
        output = self.layer_norm(query + self.dropout(attn_output))
        return output

class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Permute to (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        # Apply GeM pooling
        pooled = torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(2, 3)).pow(1.0 / self.p)
        return pooled
    

class CombinedModelsNew(nn.Module):
    def __init__(self, image_model, bert_model, image_feature_dim, bert_embedding_dim, combined_dim, num_labels, dropout_prob=0.1):
        super(CombinedModelsNew, self).__init__()
        self.image_model = image_model
        self.bert_model = bert_model
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.gem_pooling = GeMPooling()


        self.dropout = nn.Dropout(dropout_prob)
        self.cross_attention = CrossAttentionLayer(query_dim=image_feature_dim, embed_dim = bert_embedding_dim, num_heads=4, dropout_prob=dropout_prob)


        # Fully connected layers for combining features
        # self.fc = nn.Linear(image_feature_dim + bert_embedding_dim, combined_dim)
        self.fc1 = nn.Linear(image_feature_dim + bert_embedding_dim, combined_dim)
        self.fc2 = nn.Linear(combined_dim, combined_dim)
        self.fc3 = nn.Linear(combined_dim, num_labels)
        
        self.classifier = nn.Linear(combined_dim, num_labels)

    def forward(self, pixel_values, bert_input_ids, bert_attention_mask, labels=None):
        image_outputs = self.image_model(pixel_values, output_hidden_states=True)

        # image_features = image_outputs.hidden_states[-1].mean(dim=(1, 2))
        image_features = self.gem_pooling(image_outputs.hidden_states[-1])


        bert_outputs = self.bert_model(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            output_hidden_states=True,
        )
        bert_embeddings = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_len, bert_embedding_dim)

        # Prepare image features as queries
        image_features = image_features.unsqueeze(1)  # Shape: (batch_size, 1, image_feature_dim)

        # Apply Cross-Attention (image features attend to text embeddings)
        attended_features = self.cross_attention(query=image_features, key=bert_embeddings, value=bert_embeddings)
        attended_features = attended_features.squeeze(1)  # Shape: (batch_size, bert_embedding_dim)

        # Concatenate attended features and image features
        combined_features = torch.cat([image_features.squeeze(1), attended_features], dim=1)
        combined_features = self.dropout(combined_features)
        # combined_output = F.relu(self.fc(combined_features))

        combined_features = F.relu(self.fc1(combined_features))
        combined_features = F.relu(self.fc2(combined_features) + combined_features)  # Residual connection
        logits = self.fc3(combined_features)
        # logits = self.classifier(combined_output)



        return {"logits": logits}



# Custom Loss Function
class SuperLoss(nn.Module):
    def __init__(self, C, lam, batch_size, class_weights):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam
        self.batch_size = batch_size
        self.class_weights = class_weights

    def forward(self, logits, targets):
        l_i = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights).detach()
        sigma = self.sigma(l_i)
        loss = (F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights) - self.tau) * sigma
        return loss.mean()

    def sigma(self, l_i):
        x = torch.ones_like(l_i) * (-2 / math.exp(1.))
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = torch.clamp(y, min=-1.0, max=10.0) # MIGHT WANNA DELETE IDK

        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = torch.from_numpy(sigma.real.astype(np.float32)).to(l_i.device)
        return sigma

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        # Compute Cross-Entropy Loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights)
        
        # Compute the probability of the true class
        pt = torch.exp(-ce_loss)
        
        # Compute Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Return mean loss
        return focal_loss.mean()



class AdaptiveLearnableFocalLoss(nn.Module):
    def __init__(self, alpha_init=1.0, gamma_init=2.0, learnable=True, class_weights=None):
        super(AdaptiveLearnableFocalLoss, self).__init__()

        # Learnable parameters for alpha and gamma
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, requires_grad=True))
            self.gamma = nn.Parameter(torch.tensor(gamma_init, requires_grad=True))
        else:
            self.alpha = torch.tensor(alpha_init)
            self.gamma = torch.tensor(gamma_init)

        # Class weights (passed as input)
        self.class_weights = class_weights

        # Adaptive weighting factor for focal and class-weighted loss
        self.adaptive_factor = nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def forward(self, logits, targets):
        # Compute Cross-Entropy Loss with class weights
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights.to(logits.device))

        # Compute probability of the true class (pt)
        pt = torch.exp(-ce_loss)

        # Compute Focal Loss with learnable alpha and gamma
        focal_term = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_term * ce_loss

        # Adaptive weighting between focal loss and cross-entropy loss
        combined_loss = self.adaptive_factor * focal_loss + (1 - self.adaptive_factor) * ce_loss

        return combined_loss.mean()


# Metric Calculation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predicted_classes)
    uar = recall_score(labels, predicted_classes, average='macro')
    f1 = f1_score(labels, predicted_classes, average='macro')
    kacc = top_k_accuracy_score(labels, predictions)
    return {'accuracy': accuracy, 'uar': uar, 'f1': f1, 'top_k_acc': kacc}


def calculate_class_weights(train_dataset, class_weight_multipliers):
    labels = [sample['label'] for sample in train_dataset]
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    for class_label, multiplier in class_weight_multipliers.items():
        if class_label in class_weight_dict:
            class_weight_dict[class_label] *= multiplier
    
    return [class_weight_dict[label] for label in unique_classes]


def save_training_metadata(
    output_dir,
    pathstr,
    dataset_name,
    model_type,
    super_loss_params,
    speaker_disentanglement,
    entropy,
    column,
    metrics,
    speakers,
    angry_weight,
    happy_weight,
    neutral_weight,
    sad_weight,
    weight_decay,
    results
    ):
    """
    Save training metadata to a text file in the specified output directory.
    """
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the file path for the metadata file
    file_path = os.path.join(output_dir, 'training_metadata.txt')

    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Write the content to the file
    with open(file_path, 'w') as file:
        file.write(f"Pretrain_file: {pathstr}\n")
        file.write(f"Date: {current_date}\n")
        file.write(f"Dataset Used: {dataset_name}\n")
        file.write(f"Model Type: {model_type}\n")
        file.write(f"Super Loss Parameters: {super_loss_params}\n")
        file.write(f"Speaker Disentanglement: {speaker_disentanglement}\n")
        file.write(f"Entropy Curriculum Training: {entropy}\n")
        file.write(f"Column Trained on: {column}\n")
        file.write(f"Test Results: {metrics}\n")
        file.write(f"Test Speaker IDs: {speakers}\n")
        file.write(f"Angry Weight: {angry_weight}\n")
        file.write(f"Happy Weight: {happy_weight}\n")
        file.write(f"Neutral Weight: {neutral_weight}\n")
        file.write(f"Sad Weight: {sad_weight}\n")
        file.write(f"Weight Decay: {weight_decay}\n")
        file.write(f"Test results {results}\n")

    print(f"Training metadata saved successfully at: {file_path}")


def create_unique_output_dir(base_output_dir: str) -> str:
    """
    Creates a unique output directory appended with the current date and an incremented identifier.
    
    Args:
        base_output_dir (str): The base directory where the new folder should be created.
        
    Returns:
        str: The path of the newly created unique output directory.
    """
    # Get the current date in YYYYMMDD format
    date_str = datetime.now().strftime("%Y%m%d")

    # Get a list of existing directories in the base output directory
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    existing_dirs = [
        d for d in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, d))
    ]

    # Filter for directories that start with the current date string
    matching_dirs = [
        d for d in existing_dirs
        if d.startswith(date_str) and "_" in d and d.split("_")[-1].isdigit()
    ]

    # Determine the next numerical identifier
    if matching_dirs:
        last_num = max(int(d.split("_")[-1]) for d in matching_dirs)
        new_num = last_num + 1
    else:
        new_num = 1

    # Construct the new unique directory path
    unique_output_dir = os.path.join(base_output_dir, f"{date_str}_{new_num}")

    # Create the directory
    os.makedirs(unique_output_dir, exist_ok=True)

    return unique_output_dir
