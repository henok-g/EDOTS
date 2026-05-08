import sys
sys.path.append('EDOTS')
from utils.dataVisUtil import dataVis
from datasets.artsDataset import artsDataset,getTrainTestValSplit
from trainer.BaseTrainer import BaseTrainer
from ultralytics import YOLO


if __name__ == '__main__':
    # loading in pre-trained model weights for yolov8 architecture
    model = YOLO('yolov8n.pt')
    
    # determine the train/test/val split
    train_split,test_split,val_split = getTrainTestValSplit()
        
    # load in the easy arts dataset
    rootDir = "/mnt/data/arts"
    train_dataset = artsDataset(rootDir,ids=train_split)
    test_dataset = artsDataset(rootDir,ids=test_split)
    val_dataset = artsDataset(rootDir,ids=val_split)
    
    # MuSGD is the default optimizer for YOLOV8
    optimizer = "MuSGD"
    # number of epochs 
    epochs = 10
    
    # Define the parameters for training/evaluating the model
    params = {
        'model' : model,
        'optimizer' : optimizer,
        'epochs' : epochs,
        'train_data': train_dataset,
        'test_data': test_dataset,
        'val_data': val_dataset,
    }
    
    # 
    trainer = BaseTrainer(**params)
     