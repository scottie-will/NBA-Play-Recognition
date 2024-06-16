# NBA Play Recognition
*Aidana Imangozhina*<sup>1*</sup>, 
*Shree Phadke*<sup>1*</sup>, 
*Scottie Williamson*<sup>1*</sup>, 

<sup>1</sup>Columbia University &nbsp;&nbsp;

<span>*</span> Equal Contribution - to see the timeline, refer to the **Meeting Progress and Code Timeline PDF**.

 ## Approach outline
 <div align="center">
<img src="img/appraoch_outline.png" width=750px></img>
</div>
 Our approach relies on first extracting positional features using a finetuned Fastrcnn object detector. The model is finetuned to detecting 3 classes of objects: player, basket, and ball. Using those detections an attention map is created for each frame. The attention map has varying intensities for each class of object with the ball, and basket receiving the most emphasis. This attention map and video frames are then fed into a pre-trained R(2+1)D CNN for finetuning. 
 <div align="center">
<img src="img/frames_comparison.png" width=750px></img>
</div>

## Training the Detection Model

Our detection model is a finetuned Faster R-CNN, pre-trained on the ResNet50 dataset. We leveraged a dataset from [Roboflow](https://universe.roboflow.com/cv-8scak/cv-cnfd4/model/1) that contains labeled data for players, the rim, and the ball. This dataset was instrumental in fine-tuning our model to detect these categories accurately.

To replicate our training process, you can use the original notebook provided in the repository. This notebook, `fine_tune_fast_rcnn_4_cats.ipynb`, outlines the step-by-step procedure to train the detection model. For convenience, this notebook can be run in Google Colab, where we initially performed our training.
 
 ## Detection Demo
You can access demo footage of our finetuned detection model by clicking the image below.
[<img src="img/thumbnail.png" width="50%">](https://youtu.be/TV_bLXzXce8?si=pvAcY3qM2YwcDHZ5 )

## Video downloading tools
To download the raw mp4 files from NBA.com use the video collector tool
```
cd nsva_data
cd NSVA_project
cd tools
python collect_videos.py
```
Files download to .pbp_video folder. 
Download tools taken from [Sports Video Analysis on Large-Scale Data](https://github.com/jackwu502/NSVA)

## Model Training
To train the model run
```
python training.py

```

## Results and Performance Metrics
### Faster R-CNN
| **Metric**  | **Score**  |
| ------------------------------------| ---------- |
| **Instersection Over Union (IoU)**  | **0.971**  |

### R(2+1)D CNN 
Evaluated using Multilabel Metrics

| **Metric**  | **Score**  |
| ----------------| ---------- |
| **Accuracy**  | **0.59**  |
| **Precision**  | **0.42**  |
| **Recall**  | **0.50**  |

## Generating MP4 With Detected Features
Our development [notebook](/Restarted_Approach_with_object_detection_and_attention_mapping.ipynb) contain the function 'process_video' which can be used to generate MP4 files of full length clips visualizing our postional features (object detection boxes, and segmentations)
## Acknowledgement
This approach used in this project is largely based on [Sports Video Analysis on Large-Scale Data](https://github.com/jackwu502/NSVA)


