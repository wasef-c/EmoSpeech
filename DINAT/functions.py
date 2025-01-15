import os
import numpy as np
import random
import math
import json
from functools import partial
from PIL import Image
import pandas as pd
from transformers import (
    AutoImageProcessor, ViTForImageClassification, ViTHybridForImageClassification,
    BeitForImageClassification, DinatForImageClassification, ViTImageProcessor,
    ConvNextV2ForImageClassification, EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, top_k_accuracy_score,
    mean_squared_error, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Sampler, Dataset
from torchvision import transforms
from transformers import TrainingArguments, Trainer, SchedulerType
from datasets import load_dataset, concatenate_datasets
from scipy.special import lambertw
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict
import logging
from datetime import datetime
from collections import defaultdict
import re
import numpy as np

def get_next_model_path(base_path, model_prefix):
    # Get all existing directories
    existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith(model_prefix)]

    # Extract numbers from existing directories
    numbers = [int(re.search(r'\d+', d).group()) for d in existing_dirs if re.search(r'\d+', d)]

    # Find the highest number
    highest_number = max(numbers) if numbers else 0

    # Increment the highest number
    new_number = highest_number + 1

    # Create the new model path
    new_model_path = os.path.join(base_path, f'{model_prefix}{new_number:03d}')

    return new_model_path

# Constants
EMOTIONS = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry'}
Map2Num = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3}

# Utility functions
def filter_m_examples(example):
    return example["label"] not in [4, 5]

def get_random_crop_size(min_crop=0.5, max_crop=1.0):
    return random.uniform(min_crop, max_crop)

# Transform classes and functions
class RandomWindowCrop:
    def __init__(self, windows, output_size):
        self.windows = windows
        self.output_size = output_size

    def __call__(self, img):
        window = random.choice(self.windows)
        if window is not None:
            cropped_img = img.crop(window)
        else:
            cropped_img = img
        return cropped_img.resize((self.output_size, self.output_size), Image.BILINEAR)

import random
from PIL import Image
import torchvision.transforms as transforms

def get_random_crop_size(min_crop=0.5, max_crop=1.0):
    return random.uniform(min_crop, max_crop)

class RandomWindowCrop:
    def __init__(self, windows, output_size):
        self.windows = windows
        self.output_size = output_size
    
    def __call__(self, img):
        window = random.choice(self.windows)
        if window is not None:
            cropped_img = img.crop(window)
        else:
            cropped_img = img
        return cropped_img.resize((self.output_size, self.output_size), Image.BILINEAR)

# Define default values for transforms
DEFAULT_NEW_SIZE = 224
DEFAULT_SIZE = 224
DEFAULT_WINDOWS =  [
    (0, 0, 112, 147),
    (0, 0, 112, 75),
    (112, 0, 224, 75),
    (0, 75, 112, 147),
    (112, 75, 224, 147),
    (0, 149, 112, 224),
    (112, 149, 224, 224),
    None,
    None,
    None
]

# Create the transform functions
def create_train_transforms(new_size=DEFAULT_NEW_SIZE, size=DEFAULT_SIZE, windows=DEFAULT_WINDOWS):
    def transform_fn(examples):
        # Apply image transformations
        pixel_values = [
            transforms.Compose([
                RandomWindowCrop(windows, size),
                # transforms.RandomApply([
                #     transforms.RandomResizedCrop(
                #         size=224,
                #         scale=(0.3, 1.0),
                #         ratio=(0.75, 1.3333)
                #     )
                # ], p=0.4),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=(0.5, 3),
                        contrast=(0.5, 3),
                        saturation=(0.5, 3),
                    )
                ], p=0.8),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(image.convert("RGB")) for image in examples['image']
        ]
        
        # Create a new dictionary with all original keys plus the transformed 'pixel_values'
        transformed_examples = {
            'pixel_values': pixel_values,
            'speakerID': examples['speakerID'],
            'label': examples['label']
            # 'valence': examples['valence'],
            # 'arousal': examples['arousal'],
            # 'arousal_norm': examples['arousal_norm'],
            # 'valence_norm': examples['valence_norm']
        }
        
        return transformed_examples
    
    return transform_fn

def create_val_transforms(new_size=DEFAULT_NEW_SIZE, size=DEFAULT_SIZE, windows=DEFAULT_WINDOWS):
    def transform_fn(examples):
        # Apply image transformations
        pixel_values = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(image.convert("RGB")) for image in examples['image']
        ]
        
        # Create a new dictionary with all original keys plus the transformed 'pixel_values'
        transformed_examples = {
            'pixel_values': pixel_values,
            'speakerID': examples['speakerID'],
            'label': examples['label']
            # 'valence': examples['valence'],
            # 'arousal': examples['arousal'],
            # 'arousal_norm': examples['arousal_norm'],
            # 'valence_norm': examples['valence_norm']
        }
        
        return transformed_examples
    
    return transform_fn


# Create the actual transform objects
train_transforms = create_train_transforms()
val_transforms = create_val_transforms()

# The test_transforms can use the same transform as val_transforms
test_transforms = val_transforms

# Dataset and Sampler classes
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
        shuffled_indices = list(range(self.num_samples))
        random.shuffle(shuffled_indices)
        self.group_indices = self._create_group_indices(shuffled_indices)
        random.shuffle(self.group_indices)
        final_indices = [idx for group in self.group_indices for idx in group]
        return iter(final_indices)

    def __len__(self):
        return self.num_samples

# Loss function
class SuperLoss(nn.Module):
    def __init__(self, C=10, lam=1, batch_size=128, class_weights=None):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam
        self.batch_size = batch_size
        self.class_weights = class_weights

    def forward(self, logits, targets):
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]
        else:
            sample_weights = torch.ones_like(targets, dtype=torch.float)
        
        l_i = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights).detach()
        sigma = self.sigma(l_i)
        loss = (F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights) - self.tau) * sigma + self.lam * (torch.log(sigma)**2)
        loss = (loss * sample_weights).sum() / self.batch_size
        return loss

    def sigma(self, l_i):
        x = torch.ones(l_i.size()) * (-2 / math.exp(1.))
        x = x.cuda()
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma

    def set_lambda(self, new_lambda):
        self.lam = new_lambda
# Trainer classes
class SuperTrainer(Trainer):
    def __init__(self, *args, super_loss_params=None, lambda_adjust_interval=100, **kwargs):
        super().__init__(*args, **kwargs)
        if super_loss_params is None:
            super_loss_params = {'C': 10, 'lam': 1, 'batch_size': self.args.train_batch_size}
        self.super_loss = SuperLoss(**super_loss_params)
        self.initial_lambda = super_loss_params['lam']
        self.current_lambda = self.initial_lambda
        self.lambda_adjust_interval = lambda_adjust_interval
        self.current_step = 0
        self.loss_history = []
        logging.getLogger().addHandler(logging.NullHandler())
        logging.getLogger("natten.functional").setLevel(logging.ERROR)

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     outputs = model(**inputs)
    #     logits = outputs.get('logits')
    #     labels = inputs.get('labels')
    #     loss = self.super_loss(logits, labels)
    #     return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        filtered_logs = {k: v for k, v in logs.items() if "natten.functional" not in str(k)}
        super().log(filtered_logs)

    def training_step(self, model, inputs):
        self.current_step += 1
        loss = super().training_step(model, inputs)
        self.loss_history.append(loss.item())
        
        if self.current_step % self.lambda_adjust_interval == 0:
            self.adjust_lambda()
        
        return loss

    def adjust_lambda(self):
        if len(self.loss_history) >= self.lambda_adjust_interval:
            recent_losses = self.loss_history[-self.lambda_adjust_interval:]
            avg_loss = np.mean(recent_losses)
            loss_trend = np.mean(np.diff(recent_losses))

            if loss_trend > 0:  # Loss is increasing
                self.current_lambda *= 0.9  # Decrease lambda
            elif loss_trend < 0:  # Loss is decreasing
                self.current_lambda *= 1.1  # Increase lambda
            
            self.current_lambda = max(0.01, min(10, self.current_lambda))  # Keep lambda in a reasonable range
            self.super_loss.set_lambda(self.current_lambda)
            
            self.log({"lambda": self.current_lambda, "avg_loss": avg_loss})
            self.loss_history = []  # Reset loss history

class CustomTrainer(SuperTrainer):
    def __init__(self, *args, custom_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_sampler = custom_sampler

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        custom_sampler = CustomSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            sampler=custom_sampler,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

# Metric functions
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predicted_classes = np.argmax(predictions, axis=1)
    weights = np.ones_like(labels)
    accuracy = accuracy_score(labels, predicted_classes)
    uar = recall_score(labels, predicted_classes, average='macro')
    f1 = f1_score(labels, predicted_classes, average='macro')
    kacc = top_k_accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'uar': uar,
        'f1': f1,
        'top_k_acc': kacc,
    }

def calculate_class_weights(train_dataset, class_weight_multipliers):
    labels = [sample['label'] for sample in train_dataset]
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    class_weight_dict = dict(zip(unique_classes, class_weights))
    for class_label, multiplier in class_weight_multipliers.items():
        if class_label in class_weight_dict:
            class_weight_dict[class_label] *= multiplier
    print(class_weight_dict)
    return [class_weight_dict[label] for label in unique_classes]

# Saving functions
def save_model_header(new_model_path, model_info):
    os.makedirs(new_model_path, exist_ok=True)
    file_path = os.path.join(new_model_path, 'header.txt')
    current_date = datetime.now().strftime("%Y-%m-%d")
    with open(file_path, 'w') as file:
        file.write(f"Date: {current_date}\n")
        for key, value in model_info.items():
            file.write(f"{key}: {value}\n")
    print(f"File saved successfully at: {file_path}")
    return file_path

def save_confusion_matrix(outputs, dataset_train, new_model_path):
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Map2Num)
    disp.plot(ax=ax, xticks_rotation=45, cmap='PuBuGn')
    plt.title('Confusion Matrix')
    accuracy = outputs.metrics['test_accuracy'] * 100
    uar = outputs.metrics['test_uar'] * 100
    filename = f"{os.path.split(dataset_train)[1]}_accuracy_{accuracy:.2f}_UAR_{uar:.2f}.png"
    save_path = os.path.join(new_model_path, 'results')
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    plt.tight_layout()
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    # plt.close(fig)
    print(f"Confusion matrix saved to: {full_path}")
    return full_path

# Collate function
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

import torch
import torch.nn as nn
from transformers import Trainer
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

class ModelEnsemble(nn.Module):
    def __init__(self, models):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, **inputs):
        outputs = [model(**inputs) for model in self.models]
        logits = torch.stack([output.logits for output in outputs])
        return logits.mean(dim=0)

def create_ensemble_trainer(ensemble, compute_metrics):
    class EnsembleTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            return (loss, outputs) if return_outputs else loss

    return EnsembleTrainer(
        model=ensemble,
        compute_metrics=compute_metrics,
    )

def evaluate_ensemble(model1, model2, eval_dataset):
    # Create the ensemble
    ensemble = ModelEnsemble([model1, model2])
    
    # Create the ensemble trainer
    trainer = create_ensemble_trainer(ensemble, compute_metrics)
    
    # Evaluate
    results = trainer.evaluate(eval_dataset)
    
    return results

