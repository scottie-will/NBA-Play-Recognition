import torch
import json
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import copy


def visualize_attention(frame_tensor, attention_map):
    # Convert the frame tensor from PyTorch to NumPy and adjust for image display
    frame = frame_tensor.cpu().numpy().transpose(1, 2, 0)  # Change from CxHxW to HxWxC
    frame = (frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))  # Inverse normalize
    frame = np.clip(frame, 0, 1)  # Clip to valid [0,1] range for image data

    # Process the attention map
    attention = attention_map.cpu().numpy().squeeze(0)  # Squeeze out the channel dimension if it's 1xHxW

    # Create a figure with subplots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title("Original Frame")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(frame)
    plt.imshow(attention, alpha=0.4, cmap='rainbow')  # Overlay the attention map with transparency
    plt.title("Frame with Attention Overlay")
    plt.axis('off')

    plt.show()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frame_transforms = Compose([
    Resize(224),  # Resize smaller edge to 256 pixels
    ToTensor(),  # Convert the image to a PyTorch tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet's mean and std
])
class VideoCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, dataset_type='train', transform=None, total_samples=5):
        with open(json_file, 'r') as f:
            self.video_data = json.load(f)[dataset_type]
        self.transform = transform
        self.total_samples = total_samples
        # print(f"Loaded {len(self.video_data)} videos for {dataset_type}.")

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_path, captions = list(self.video_data.items())[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // self.total_samples)

        # print(f"Loading video from path: {video_path}")
        frames = []
        frame_id = 0
        sample_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or sample_count >= self.total_samples:
                break
            if frame_id % frame_interval == 0:
                frame_data = self.process_frame(frame)
                frames.append(frame_data)
                sample_count += 1

            frame_id += 1

        
        cap.release()
        # Clear CUDA cache to free unused memory from GPU
        torch.cuda.empty_cache()
        
        frames_tensor = torch.stack([f['frame'] for f in frames]) if frames else torch.tensor([], dtype=torch.float32)
        attention_maps = torch.stack([f['attention_map'] for f in frames]) if frames else torch.tensor([], dtype=torch.float32)

        video_data = {
            'frames': frames_tensor,
            #'frame_details': frames,  # Each frame's data including detections and attention maps
            'attention_maps': attention_maps,  # Stacked attention maps for each frame
            'captions': captions
        }
        return video_data


    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(rgb_frame)
        frame_tensor = self.transform(frame_pil) if self.transform else frame_pil
        frame_tensor = frame_tensor.to(device)
        
        custom_model_path = 'normalized_comprehensive_basketball_detector.pth'  # Path to fine-tuned model
        ball_detector = self.load_custom_model(custom_model_path, num_classes=4)
        ball_detector.eval()
        
        ball_detector.to(device)
        with torch.no_grad():
            basketball_det_out = ball_detector([frame_tensor])[0]
        # ball_detector.to('cpu')
        framecopy = frame_tensor.cpu()
        detections = self.process_detections(basketball_det_out, 0.4)
        
        del frame_tensor
        del ball_detector
        torch.cuda.empty_cache()
        
        attention_map = self.generate_attention_maps(detections, framecopy)
    
        return {
            'frame': framecopy,
            'detections': detections,
            'attention_map': attention_map
        }
    
    def load_custom_model(self, model_path, num_classes):
        # Create an instance of the model architecture with no pre-trained weights
        model = fasterrcnn_resnet50_fpn(weights=None)

        # Replace the classifier and box predictor head with a new one that matches the number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Load the state dictionary from the file
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Remove 'model.' prefix from the state dictionary keys if present
        new_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}

        # Load the modified state dictionary into the model
        model.load_state_dict(new_state_dict)

        return model


    def generate_attention_maps(self, detections, frame_tensor):
        _, height, width = frame_tensor.shape  # Assuming frame_tensor is CxHxW

        # Initialize the attention map tensor to zeros
        attention_map = torch.zeros((3, height, width), dtype=torch.float32)

        for box, label in zip(detections['boxes'], detections['labels']):
            # Define weights based on class labels
            if label == 2:  # Assuming label 2 is the 'ball'
                weight = 2.0  # Highest weight for balls
            elif label == 3:  # Assuming label 3 is the 'rim'
                weight = 1.5  # Middle weight for rims
            else:
                weight = 0.5  # Lower weight for players and others

            x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)

            if (x2 > x1) and (y2 > y1):
                attention_map[0, y1:y2, x1:x2] = weight

        # Normalize the attention map to range between 0 and 1
        max_val = torch.max(attention_map)
        if max_val > 0:
            attention_map /= max_val
            
        # visualize_attention(frame_tensor, attention_map)
        att_map = attention_map.cpu()
        del attention_map
        torch.cuda.empty_cache()
        
        return att_map


    def process_detections(self, detection_output, threshold=.6):
        scores = detection_output['scores']
        high_confidence_indices = scores > threshold
        processed_output = {k: v[high_confidence_indices].cpu() for k, v in detection_output.items()}
        return processed_output

    def process_segmentations(self, segmentation_output):
        thresholded_output = segmentation_output > 0.8
        return thresholded_output.squeeze(1).cpu().byte()
