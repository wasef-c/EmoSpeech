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
from sklearn.cross_decomposition import CCA

from typing import Dict  # Add this import
from transformers import AutoImageProcessor, DinatForImageClassification, TrainingArguments, Trainer, AutoTokenizer, AutoModel
from sklearn.utils.class_weight import compute_class_weight

'''
(C)	Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       PLEASE CITE THE ABOVE PAPER IF YOU USE THIS CODE.
'''
# Initialize the image processor and BERT tokenizer

from sklearn.metrics import r2_score

def concordance_correlation_coefficient(x, y):
    """
    Concordance Correlation Coefficient (CCC):
    https://en.wikipedia.org/wiki/Concordance_correlation_coefficient

    CCC = ( 2 * cov(x, y) ) / ( var(x) + var(y) + (mean(x) - mean(y))^2 )
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)
    y_var = np.var(y)

    # Covariance
    cov_xy = np.mean((x - x_mean) * (y - y_mean))

    numerator = 2 * cov_xy
    denominator = x_var + y_var + (x_mean - y_mean) ** 2

    return numerator / denominator if denominator != 0 else 0.0



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
    labels = torch.tensor([example[column] for example in examples]).to(device)
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
            speaker_id = self.data_source[idx]['SpkrID']
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
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cross_decomposition import CCA

import numpy as np
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np

from sklearn.cross_decomposition import CCA
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class DCCALoss(nn.Module):
    def __init__(self, latent_dim, epsilon=1e-8):
        super(DCCALoss, self).__init__()
        self.latent_dim = latent_dim
        self.epsilon = epsilon

    def forward(self, view1, view2):
        """
        Compute the DCCA loss.
        :param view1: Projected features from modality 1 (batch_size, latent_dim).
        :param view2: Projected features from modality 2 (batch_size, latent_dim).
        :return: Negative canonical correlation.
        """
        batch_size = view1.size(0)

        # Center the features
        view1 -= view1.mean(dim=0)
        view2 -= view2.mean(dim=0)

        # Covariance matrices
        cov_11 = torch.mm(view1.T, view1) / (batch_size - 1) + self.epsilon * torch.eye(self.latent_dim).to(view1.device)
        cov_22 = torch.mm(view2.T, view2) / (batch_size - 1) + self.epsilon * torch.eye(self.latent_dim).to(view2.device)
        cov_12 = torch.mm(view1.T, view2) / (batch_size - 1)

        # Eigen decomposition for canonical correlation
        eigvals_1, eigvecs_1 = torch.linalg.eigh(cov_11)
        eigvals_2, eigvecs_2 = torch.linalg.eigh(cov_22)

        # Whiten the covariance matrices
        cov_11_whitened = eigvecs_1 @ torch.diag(torch.sqrt(1.0 / eigvals_1)) @ eigvecs_1.T
        cov_22_whitened = eigvecs_2 @ torch.diag(torch.sqrt(1.0 / eigvals_2)) @ eigvecs_2.T

        T = cov_11_whitened @ cov_12 @ cov_22_whitened
        _, singular_values, _ = torch.svd(T)

        # Return the negative sum of canonical correlations
        return -torch.sum(singular_values[:self.latent_dim])


class CombinedModelsDCCA(nn.Module):
    def __init__(self, image_model, bert_model, image_feature_dim, bert_embedding_dim, latent_dim, num_labels, dropout_prob=0.1):
        super(CombinedModelsDCCA, self).__init__()
        self.image_model = image_model
        self.bert_model = bert_model
        self.gem_pooling = GeMPooling()

        self.latent_dim = latent_dim

        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(bert_embedding_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

        # Classification layers
        self.fc1 = nn.Linear(latent_dim * 2, 128)
        self.fc2 = nn.Linear(128, num_labels)
        self.dropout = nn.Dropout(dropout_prob)

        # DCCA loss
        self.dcca_loss = DCCALoss(latent_dim)

    def forward(self, pixel_values, bert_input_ids, bert_attention_mask, labels=None):
        # Process image features
        image_outputs = self.image_model(pixel_values, output_hidden_states=True)
        image_features = self.gem_pooling(image_outputs.hidden_states[-1])
        image_features = image_features.view(image_features.size(0), -1)

        # Process text (BERT) features
        bert_outputs = self.bert_model(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            output_hidden_states=True,
        )
        bert_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Project to shared latent space
        image_latent = self.image_projection(image_features)
        text_latent = self.text_projection(bert_embeddings)

        # Calculate DCCA loss if labels are provided
        dcca_loss = None
        if labels is not None:
            dcca_loss = self.dcca_loss(image_latent, text_latent)

        # Concatenate and classify
        combined_latent = torch.cat([image_latent, text_latent], dim=1)
        combined_latent = self.dropout(F.relu(self.fc1(combined_latent)))
        logits = self.fc2(combined_latent)

        return {"logits": logits, "dcca_loss": dcca_loss}


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class CombinedModelsDCCAE(nn.Module):
    def __init__(
        self, image_model, bert_model, image_feature_dim, bert_embedding_dim,
        latent_dim, num_labels, dropout_prob=0.1
    ):
        super(CombinedModelsDCCAE, self).__init__()

        # Backbone models
        self.image_model = image_model
        self.bert_model = bert_model
        self.gem_pooling = GeMPooling()

        # Freeze BERT model
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.latent_dim = latent_dim

        # Image and text projection layers
        self.image_encoder = nn.Sequential(
            nn.Linear(image_feature_dim, 512),
            nn.GELU(),
            # nn.BatchNorm1d(512),  # Add batch normalization
            nn.Linear(512, latent_dim)
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(bert_embedding_dim, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim)
        )

        # Decoders for reconstruction (DCCAE specific)


        self.image_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, image_feature_dim)
        )


        self.text_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, bert_embedding_dim)
        )

        # Classification layers
        self.fc1 = nn.Linear(latent_dim * 2, 128)
        self.fc2 = nn.Linear(128, num_labels)
        self.dropout = nn.Dropout(dropout_prob)

        # DCCA loss
        self.dcca_loss = DCCALoss(latent_dim)

    def forward(self, pixel_values, bert_input_ids, bert_attention_mask, labels=None):
        # Process image features
        image_outputs = self.image_model(pixel_values, output_hidden_states=True)
        image_features = self.gem_pooling(image_outputs.hidden_states[-1])
        print(image_features.mean(), image_features.std())

        image_features = image_features.view(image_features.size(0), -1)


        # Process text (BERT) features
        bert_outputs = self.bert_model(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            output_hidden_states=True,
        )
        bert_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Encode to latent space
        image_latent = self.image_encoder(image_features)
        text_latent = self.text_encoder(bert_embeddings)

        print(image_latent.mean(), image_latent.std())
        print(text_latent.mean(), text_latent.std())


        # Decode for reconstruction
        image_reconstructed = self.image_decoder(image_latent)
        text_reconstructed = self.text_decoder(text_latent)

        # DCCA loss calculation
        dcca_loss = None
        reconstruction_loss = None
        if labels is not None:
            dcca_loss = self.dcca_loss(image_latent, text_latent)
            reconstruction_loss = F.mse_loss(image_reconstructed, image_features) + \
                                 F.mse_loss(text_reconstructed, bert_embeddings)

        # Concatenate and classify
        combined_latent = torch.cat([image_latent, text_latent], dim=1)
        combined_latent = self.dropout(F.relu(self.fc1(combined_latent)))
        logits = self.fc2(combined_latent)

        return {
            "logits": logits,
            "dcca_loss": dcca_loss,
            "reconstruction_loss": reconstruction_loss
        }


class DCCALoss(nn.Module):
    def __init__(self, latent_dim):
        super(DCCALoss, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, image_latent, text_latent):
        # Implementation of DCCA loss (simplified)
        image_latent_centered = image_latent - image_latent.mean(dim=0)
        text_latent_centered = text_latent - text_latent.mean(dim=0)

        covariance = torch.mm(image_latent_centered.T, text_latent_centered)
        u, s, v = torch.svd(covariance)
        loss = -s.sum()

        return loss

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


def calculate_class_weights(train_dataset, class_weight_multipliers=None):
    labels = [sample['label'] for sample in train_dataset]
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    
    class_weight_dict = dict(zip(unique_classes, class_weights))
    if class_weight_multipliers != None:
        for class_label, multiplier in class_weight_multipliers.items():
            if class_label in class_weight_dict:
                class_weight_dict[class_label] *= multiplier
    
    return [class_weight_dict[label] for label in unique_classes]


def save_training_metadata(
    output_dir,
    Pretrain_file,
    dataset_name,
    model_type,
    results, 
    column,
    speaker_disentanglement,
    metrics,
    weight_decay,
    class_weights=None,
    speakers = None,


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
        file.write(f"Pretrain_file: {Pretrain_file}\n")
        file.write(f"Date: {current_date}\n")
        file.write(f"Dataset Used: {dataset_name}\n")
        file.write(f"Model Type: {model_type}\n")
        file.write(f"Speaker Disentanglement: {speaker_disentanglement}\n")
        file.write(f"Column Trained on: {column}\n")
        file.write(f"Test Results: {metrics}\n")
        file.write(f"Test Speaker IDs: {speakers}\n")
        file.write(f"Class Weights: {class_weights}\n")
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





class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, embed_dim, num_heads, dropout_prob=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_prob,  batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, attention_mask=None):
        # Project query, key, and value to the same embedding dimension
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        # Apply multihead attention
        attn_output, _ = self.multihead_attn(
            query, key, value, attn_mask=attention_mask)

        # Residual connection and layer normalization
        output = self.layer_norm(query + self.dropout(attn_output))
        return output



class BiCrossAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention:
      Pass 1: x1 attends to x2
      Pass 2: x2 attends to x1

    Returns updated x1 and x2.
    """
    def __init__(self, dim1, dim2, embed_dim, num_heads, dropout_prob=0.1):
        super(BiCrossAttentionLayer, self).__init__()
        # 1->2 cross attention (unchanged)
        self.query_proj_12 = nn.Linear(dim1, embed_dim)
        self.key_proj_12   = nn.Linear(dim2, embed_dim)
        self.value_proj_12 = nn.Linear(dim2, embed_dim)
        self.attn_12 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_prob, batch_first=True)
        self.layer_norm_12 = nn.LayerNorm(embed_dim)
        self.dropout_12 = nn.Dropout(dropout_prob)

        # 2->1 cross attention
        self.query_proj_21 = nn.Linear(dim2, embed_dim)  # x2 (dim2 -> embed_dim)
        self.key_proj_21   = nn.Linear(embed_dim, embed_dim)  # Updated x1 (embed_dim -> embed_dim)
        self.value_proj_21 = nn.Linear(embed_dim, embed_dim)  # Updated x1 (embed_dim -> embed_dim)
        self.attn_21 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_prob, batch_first=True)
        self.layer_norm_21 = nn.LayerNorm(embed_dim)
        self.dropout_21 = nn.Dropout(dropout_prob)

    def forward(self, 
                x1,   # (batch, seq_len1, dim1)
                x2,   # (batch, seq_len2, dim2)
                mask1=None,  # optional attention mask for pass 1
                mask2=None): # optional attention mask for pass 2
        
        # --------------------------------------------------
        # 1) x1 -> x2 cross-attention
        #    - Query = x1
        #    - Key/Value = x2
        # --------------------------------------------------
        # Project each
        q_12 = self.query_proj_12(x1)   # shape: (B, seq_len1, embed_dim)
        k_12 = self.key_proj_12(x2)     # shape: (B, seq_len2, embed_dim)
        v_12 = self.value_proj_12(x2)   # shape: (B, seq_len2, embed_dim)

        attn_out_12, _ = self.attn_12(q_12, k_12, v_12, attn_mask=mask1)
        # Residual + LayerNorm
        updated_x1 = self.layer_norm_12(q_12 + self.dropout_12(attn_out_12))
        # print("q_12 in_features:", x1.shape[-1], "vs. query_proj_12 weight shape:", self.query_proj_12.weight.shape)

        # --------------------------------------------------
        # 2) x2 -> x1 cross-attention
        #    - Query = x2
        #    - Key/Value = x1 (specifically the updated_x1?)
        # --------------------------------------------------
        # Usually, we let x2 attend to the *original* x1 or the updated_x1.
        # Common approach: use updated_x1 as K/V so that x2 sees the refined representation.
        # If you prefer the original x1, just pass x1. 
        q_21 = self.query_proj_21(x2)
        k_21 = self.key_proj_21(updated_x1)
        v_21 = self.value_proj_21(updated_x1)

        attn_out_21, _ = self.attn_21(q_21, k_21, v_21, attn_mask=mask2)
        updated_x2 = self.layer_norm_21(q_21 + self.dropout_21(attn_out_21))

        return updated_x1, updated_x2

class CombinedModelsBi(nn.Module):
    def __init__(
        self,
        image_model,
        bert_model,
        image_feature_dim,
        bert_embedding_dim,
        combined_dim,
        num_labels,
        dropout_prob=0.3
    ):
        super(CombinedModelsBi, self).__init__()
        self.image_model = image_model
        self.bert_model = bert_model
        for name, param in self.bert_model.named_parameters():
            # Allow gradients only for the parameters in the last layer
            # if "encoder.layer.11" not in name and "pooler" not in name:
            #     param.requires_grad = False
            if "encoder.layer.9" not in name and "encoder.layer.10" not in name and "encoder.layer.11" not in name and "pooler" not in name:
                param.requires_grad = False


        self.gem_pooling = GeMPooling()

        self.dropout = nn.Dropout(dropout_prob)

        # -------------------------
        # Bidirectional Cross-Attention
        # -------------------------
        # We internally project image_feature_dim -> embed_dim
        # and bert_embedding_dim -> embed_dim
        # so both are 512-dimensional (or whatever embed_dim you choose).
        self.bi_cross_attn = BiCrossAttentionLayer(
            dim1=image_feature_dim,
            dim2=bert_embedding_dim,
            embed_dim=512,   # internal dimension for cross-attention
            num_heads=4,
            dropout_prob=0.1
        )

        # -------------------------
        # Fully connected layers
        # -------------------------
        # IMPORTANT: after cross-attention, we will produce
        # (batch_size, 512) for image, (batch_size, 512) for text
        # -> concatenated => (batch_size, 1024).
        # So we set in_features = 1024 for fc1 below:
        self.fc1 = nn.Sequential(
            nn.Linear(2 * 512, combined_dim),  # 512 + 512 = 1024
            nn.LayerNorm(combined_dim),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.LayerNorm(combined_dim)
        )

        self.fc3 = nn.Linear(combined_dim, num_labels)

        # You can remove or keep this if you like:
        # self.classifier = nn.Linear(combined_dim, num_labels)

    def forward(self, pixel_values, bert_input_ids, bert_attention_mask, labels=None):
        # --------------------------------------------
        # 1) Image Model Forward
        # --------------------------------------------
        # E.g., a CNN or ViT that returns hidden states
        image_outputs = self.image_model(pixel_values, output_hidden_states=True)
        # We'll do GeM pooling on the last hidden state:
        # shape (batch_size, C, H, W) or (batch_size, patch_seq_len, feat_dim)
        image_features = self.gem_pooling(image_outputs.hidden_states[-1])
        # Now shape is (batch_size, image_feature_dim)

        # Make it a "sequence" of length 1: (batch_size, 1, image_feature_dim)
        image_features = image_features.unsqueeze(1)

        # --------------------------------------------
        # 2) BERT Forward
        # --------------------------------------------
        bert_outputs = self.bert_model(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            output_hidden_states=True
        )
        bert_embeddings = bert_outputs.last_hidden_state
        # shape: (batch_size, seq_len, bert_embedding_dim)

        # --------------------------------------------
        # 3) Bidirectional Cross-Attention
        #    updated_image: (B, 1, embed_dim)
        #    updated_text:  (B, seq_len, embed_dim)
        # --------------------------------------------
       
        updated_image, updated_text = self.bi_cross_attn(
            x1=image_features,      # shape (B, 1, image_feature_dim)
            x2=bert_embeddings      # shape (B, seq_len, bert_embedding_dim)
        )

        # --------------------------------------------
        # 4) Pool / flatten the updated representations
        # --------------------------------------------
        # For the updated image, we only have seq_len=1 => squeeze(1)
        # => (batch_size, embed_dim), e.g. (B, 512)
        image_vector = updated_image.squeeze(1)

        # For text, let's do a simple average pooling across seq_len
        # => (batch_size, embed_dim), e.g. (B, 512)
        text_vector = updated_text.mean(dim=1)

        # --------------------------------------------
        # 5) Combine and feed into your MLP
        # --------------------------------------------
        # Concatenate => shape (B, 512 + 512 = 1024)
        combined_features = torch.cat([image_vector, text_vector], dim=1)
        combined_features = self.dropout(combined_features)

        # Pass through fc layers with residual
        combined_features = self.fc1(combined_features)  # LN + ReLU inside
        residual = combined_features
        combined_features = self.fc2(combined_features)  # LN inside
        combined_features = F.relu(combined_features + residual)

        logits = self.fc3(combined_features)

        # Return the logits (and optionally loss if needed)
        return {"logits": logits}


class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Permute to (batch_size, channels, height, width)
        if x.dim() == 3:
            patch_dim = int((x.size(1) - 1) ** 0.5)  # Calculate grid size
            x = x[:, 1:, :]  # Remove classification token if present
            # Reshape to [batch_size, height, width, channels]
            x = x.view(x.size(0), patch_dim, patch_dim, x.size(2))

        x = x.permute(0, 3, 1, 2)
        # Apply GeM pooling
        pooled = torch.mean(x.clamp(min=self.eps).pow(
            self.p), dim=(2, 3)).pow(1.0 / self.p)
        return pooled



class AdaptiveLearnableFocalLoss(nn.Module):
    def __init__(self, alpha_init=1.0, gamma_init=2.0, learnable=True, class_weights=None):
        super(AdaptiveLearnableFocalLoss, self).__init__()

        # Learnable parameters for alpha and gamma
        if learnable:
            self.alpha = nn.Parameter(
                torch.tensor(alpha_init, requires_grad=True))
            self.gamma = nn.Parameter(
                torch.tensor(gamma_init, requires_grad=True))
        else:
            self.alpha = torch.tensor(alpha_init)
            self.gamma = torch.tensor(gamma_init)

        # Class weights (passed as input)
        self.class_weights = class_weights

        # Adaptive weighting factor for focal and class-weighted loss
        self.adaptive_factor = nn.Parameter(
            torch.tensor(0.5, requires_grad=True))

    def forward(self, logits, targets):
        # Compute Cross-Entropy Loss with class weights
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none', weight=self.class_weights.to(logits.device))

        # Compute probability of the true class (pt)
        pt = torch.exp(-ce_loss)

        # Compute Focal Loss with learnable alpha and gamma
        focal_term = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_term * ce_loss

        # Adaptive weighting between focal loss and cross-entropy loss
        combined_loss = self.adaptive_factor * \
            focal_loss + (1 - self.adaptive_factor) * ce_loss

        return combined_loss.mean()

