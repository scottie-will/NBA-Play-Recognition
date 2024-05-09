import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import EarlyStopping
import multiprocessing
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchmetrics import F1Score, Accuracy
from torchmetrics import Accuracy, Precision, Recall
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18,R2Plus1D_18_Weights 
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
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
import gc
from dataset import VideoCaptionDataset
from custom_collate_fn import custom_collate_fn
from memory_profiler import profile
import json
from collections import Counter
from tqdm import tqdm


def main():
    
    mp.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()  # Clear unused memory

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
    num_labels = len(caption_mapping)  # Should be 11 in this case

    class VideoClassifier(pl.LightningModule):
        def __init__(self, num_classes):
            super().__init__()
            self.save_hyperparameters()
            self.model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1) # Load a pretrained R(2+1)D model
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            # Metrics
            self.train_accuracy = MultilabelAccuracy(num_labels=num_labels, average='weighted')
            self.val_accuracy = MultilabelAccuracy(num_labels=num_labels, average='weighted')
            self.precision = MultilabelPrecision(num_labels=num_labels, average='weighted')
            self.recall = MultilabelRecall(num_labels=num_labels, average='weighted')
            self.f1_score = F1Score(task="multilabel",num_labels=num_labels, average='weighted')
            #loss
            self.loss_function = torch.nn.BCEWithLogitsLoss()

        def report_memory(self):
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"Allocated: {allocated:.2f} GB")
            print(f"Cached: {cached:.2f} GB")

        def forward(self, x):
            return torch.sigmoid(self.model(x))  # Use sigmoid for multi-label classification

        def training_step(self, batch, batch_idx):
            #Memory Issue debugging
            # if batch_idx % 50 == 0:
            #     self.report_memory()

            frames, attention_maps, labels = batch['frames'], batch['attention_maps'], batch['captions']
            frames = frames.permute(0, 2, 1, 3, 4)  # New shape: [batch_size, 3, num_frames, height, width
            attention_maps = attention_maps.permute(0, 2, 1, 3, 4)
            frames *= attention_maps  # Element-wise multiplication

            outputs = self(frames)

            #loss
            loss = self.loss_function(outputs, labels) 

            acc = self.train_accuracy(outputs, labels)
            train_prec = self.precision(outputs, labels)
            
            
            train_recall = self.recall(outputs, labels)
            
            self.log('train_loss', loss,on_step=True, on_epoch=True, logger=True, prog_bar=True)
            self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)
            self.log('train_recall',  train_recall, on_step=True, on_epoch=True, logger=True, prog_bar=True)
            self.log('train_precision',train_prec , on_step=True, on_epoch=True, logger=True, prog_bar=True)
            

#             if batch_idx % 50 == 0:
#                 self.report_memory()
#                 gc.collect()
#                 torch.cuda.empty_cache()
#                 self.report_memory()
                
#                 print(f"Batch {batch_idx}: Train Loss: {loss.item()}, Train Accuracy: {acc.item()}")

            return loss

        def validation_step(self, batch, batch_idx):
            frames, attention_maps, labels = batch['frames'], batch['attention_maps'], batch['captions']
            frames = frames.permute(0, 2, 1, 3, 4)  # New shape: [batch_size, 3, num_frames, height, width
            attention_maps = attention_maps.permute(0, 2, 1, 3, 4)

            frames *= attention_maps  # Element-wise multiplication
            outputs = self(frames)

            #loss
            loss = self.loss_function(outputs, labels)
            val_acc = self.val_accuracy(outputs, labels)
            val_prec = self.precision(outputs, labels)
            val_recall = self.recall(outputs, labels)
            #val_f1 = self.f1_score(outputs, labels)
            
            self.log('val_loss', loss,on_step=True, on_epoch=True, prog_bar=True,logger=True)
            self.log('val_acc', val_acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)
            self.log('val_precision',val_prec , on_step=True, on_epoch=True, logger=True, prog_bar=True)
            self.log('val_recall',  val_recall, on_step=True, on_epoch=True, logger=True, prog_bar=True)
            #self.log('val_f1', val_f1, on_step=True, on_epoch=True, logger=True, prog_bar=True)
            # if batch_idx % 50 == 0:
            #     print(f"Batch {batch_idx}: Val Loss: {loss.item()}, Val Accuracy: {val_acc.item()}").

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

    frame_transforms = Compose([Resize(224),  
                                ToTensor(),  # Convert the image to a PyTorch tensor
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet's mean and std
                               ])
    # Defining dataset
    dataset = VideoCaptionDataset(json_file='coarse_actions_paths.json', dataset_type='train', total_samples=20, transform=frame_transforms)
    val_dataset = VideoCaptionDataset(json_file='coarse_actions_paths.json', dataset_type='val', total_samples=20, transform=frame_transforms)


    batch_size = 2
    num_workers = 4

    video_classifier = VideoClassifier(num_classes=11)

    def get_class_distribution_from_json(json_file, dataset_type='train'):
        """Calculate the distribution of classes directly from a JSON file."""
        # Load the JSON data
        with open(json_file, 'r') as file:
            data = json.load(file)[dataset_type]

        class_counts = Counter()

        # Iterate over each video and its corresponding actions
        for video_path, actions in tqdm(data.items(), desc=f'Processing {dataset_type} data'):
            # Update the count for each action in the list
            class_counts.update(actions)

        return class_counts

    import numpy as np
    from torch.utils.data import DataLoader, Subset

    import json

    def get_video_data(json_file_path):
        """Load video data from a JSON file."""
        with open(json_file_path, 'r') as file:
            return json.load(file)

    json_file_path = 'coarse_actions_paths.json'
    video_data = get_video_data(json_file_path)


    def stratify_data(class_distribution, dataset, num_samples):
        """ Create stratified sample indices for the dataset. """
        indices_per_class = {action: set() for action in class_distribution}
        for idx, (video_path, actions) in enumerate(dataset.items()):
            for action in actions:
                indices_per_class[action].add(idx)

        # Compute the number of samples we need from each class
        num_samples_per_class = {action: int(np.round(num_samples * (count / sum(class_distribution.values()))))
                                 for action, count in class_distribution.items()}

        selected_indices = set()
        for action, indices in indices_per_class.items():
            if indices:
                selected_indices.update(np.random.choice(list(indices), min(len(indices), num_samples_per_class[action]), replace=False))

        return list(selected_indices)

    # Calculate the distribution of classes from the JSON file
    def get_class_distribution(data):
        class_labels = []
        for _, actions in tqdm(data.items()):
            class_labels.extend(actions)

        return Counter(class_labels)

    # Get the class distribution for training set
    


    train_class_distribution = get_class_distribution_from_json(json_file_path, 'train')
    val_class_distribution = get_class_distribution_from_json(json_file_path, 'val')

    #Use the stratify_data function to get stratified indices for both train and val sets.
    train_indices = stratify_data(train_class_distribution, video_data['train'], 2000)
    val_indices = stratify_data(val_class_distribution, video_data['val'], 400)



    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Create the DataLoaders with the stratified subsets
    train_loader_50 = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,num_workers=num_workers, persistent_workers = True)
    val_loader_50 = DataLoader(val_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers, persistent_workers = True)



    # Logger
    logger = CSVLogger("logs_PLATINUM_10k", name="video_classifier", flush_logs_every_n_steps=100)

    # Checkpoint
    checkpoint_callback = ModelCheckpoint(
                                          monitor='val_loss', 
                                          dirpath='checkpoints_PLATINUM_10k', 
                                          # filename='video_classifier-{epoch:02d}-{val_loss:.2f}', 
                                          filename='platinum-{epoch:02d}', 
                                          every_n_epochs=1,        # Save a checkpoint after every epoch
                                          verbose=True             # Print out messages when saving checkpoints.s
    )

    # Trainer
    trainer = pl.Trainer(max_epochs=20, 
                         callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=3)],
                         precision=16,
                         logger=logger,
                        )


    trainer.fit(video_classifier, train_loader_50, val_loader_50)

if __name__ == '__main__':
    main()
