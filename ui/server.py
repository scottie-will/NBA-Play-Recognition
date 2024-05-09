from flask import Flask, request, jsonify, render_template
import torch
from torch.utils.data import Dataset
import json
from torchvision.transforms import Compose, Resize, ToTensor
import cv2
from PIL import Image
import anthropic

import re
import os

def extract_text_in_quotes(text):
    # Regular expression to find text enclosed in double quotes
    pattern = r'"(.*?)"'
    
    # Find all occurrences of the pattern
    results = re.findall(pattern, text)
    
    return results

app = Flask(__name__)
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
# Define the dataset class to handle video processing
class VideoCaptionDataset(Dataset):
    def __init__(self, json_file, dataset_type='train', transform=None, frame_interval=5):
        with open(json_file, 'r') as f:
            self.video_data = json.load(f)[dataset_type]
        self.transform = transform
        self.frame_interval = frame_interval

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_path, captions = list(self.video_data.items())[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % self.frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            frame_id += 1
        cap.release()
        frames_tensor = torch.stack(frames) if frames else torch.tensor([], dtype=torch.float32)
        return frames_tensor, captions

def generate_text_with_claude(prompt):
    try:
        if not prompt:
            return "No prompt provided", False

        response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
        )
        return extract_text_in_quotes(response.content[0].text), True

    except Exception as e:
        return str(e), False

transform = Compose([
    Resize((112, 112)),
    ToTensor()
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected video'}), 400
    video_path = './videos/' + video.filename
    video.save(video_path)
    # This is the output for testing
    output = '[Shot, Rebound]'
    prompt = f"Create a short voiceover script in the style of Mike Breen, the sports commentator. Here are the captions generated from our model - add some flair: {output}. Make output to be readable in 15s. Don't include context, just commentary. Don't specify a particular player. Just provide the script in the output as if the commentator was reading it. Add quotes for the script."
    generated_text, success = generate_text_with_claude(prompt)
    if success:
        print("Generated Text:", generated_text)
    else:
        print("Error:", generated_text)
    return jsonify({'result': 'Processed', 'captions': str(generated_text)})

if __name__ == '__main__':
    app.run(debug=True)
