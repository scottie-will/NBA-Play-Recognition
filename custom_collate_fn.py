from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import F1Score, Accuracy
from torchmetrics import Accuracy, Precision, Recall
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from torchmetrics import Accuracy, Precision, Recall
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelRecall
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Subset
import random
from torch.utils.data import DataLoader
caption_mapping = {
    'Shot': 0,
    'Rebound': 1,
    'Turnover': 2,
    'Foul': 3,
    'Free_Throw': 4,
    'Jump_Ball': 5,
    'Violation': 6,
    'Timeout-Regular': 7,
    'period-start': 8,
    'period-end': 9,
    'Ejection-Other': 10
}
num_labels = len(caption_mapping)  # Should be 11 
num_classes = 11

def custom_collate_fn(batch):
    collated_data = {
        'frames': [],
        # 'frame_details': [],
        'attention_maps': [],
        'captions': []
    }
    
    for item in batch:
        # print(item['frames'].shape)
        # print(item['attention_maps'].shape)
        
        for key in collated_data:
            if key == 'captions':
                # Create a zero tensor for each caption in the batch
                one_hot_captions = torch.zeros(num_classes, dtype=torch.float32)
                # Set the appropriate indices to 1 for each caption present
                for caption in item[key]:
                    if caption in caption_mapping:
                        one_hot_captions[caption_mapping[caption]] = 1
                collated_data[key].append(one_hot_captions)
            else:
                collated_data[key].append(item[key])

    # Use default_collate for other data types
    for key in ['frames','attention_maps']:
        collated_data[key] = torch.utils.data.dataloader.default_collate(collated_data[key])
    
    # Directly stack the list of one-hot encoded caption tensors
    collated_data['captions'] = torch.stack(collated_data['captions'])

    return collated_data