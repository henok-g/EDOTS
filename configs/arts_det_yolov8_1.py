import sys
sys.path.append('EDOTS')
from utils.dataVisUtil import dataVis
from datasets.artsDataset import artsDataset,getTrainTestValSplit
from trainer.CustomYoloTrainer import CustomYoloTrainer
from ultralytics import YOLO
import torch


if __name__ == '__main__':
    # loading in pre-trained model weights for yolov8 architecture
    model = 'yolov8n.pt'

    # number of epochs 
    epochs = 50
    
    # batch size
    # Define the parameters for training/evaluating the model
    params = {
        'model'     : model,
        'epochs'    : epochs,
        'device'    : 0 if torch.cuda.is_available() else 'cpu',
        'batch'     : 16,
        'workers'   : 0,
    }
    
    trainer = CustomYoloTrainer(overrides=params)
    trainer.train()
