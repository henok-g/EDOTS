import os
from datasets.baseDetectionDataset import baseTrafficSignDataset
import glob
import pandas as pd
from xml.etree import ElementTree as ET
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
import warnings
import torch

sys.path.append('EDOTS')
from utils.dataVisUtil import dataVis
from trainer.label_map import inverted_dict



class artsDataset(baseTrafficSignDataset):
    '''
    Docstring for artsDataset
    
        __init__(self, root, difficulty="easy"): Initializes the dataset with the root directory and difficulty level.
            root (str) - The root directory of the dataset.
            
            difficulty (str) - The difficulty level of the dataset options are: "easy","challenging"
        
            images_sub_dir (str) - TODO: Fill this in 
            
            annotations_subdir (str) - TODO: Fill this in
            
        load_images_as_numpy: Method to load images from the dataset
        
        parse_annotation
    
    
    
    '''
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% __init__
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def __init__(self, root,difficulty="easy", images_subdir="JPEGImages", annotations_subdir="Annotations",ids:list=None):
        self.difficulty = difficulty
        self.ids = ids
        super().__init__(root,images_subdir,annotations_subdir)
        
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% load_images_as_numpy
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def load_images_as_numpy(self, image_files):
        images = []
        for img_file in tqdm(image_files,desc="Loading Images"):
            img = Image.open(img_file)
            img = img.resize((640, 640), Image.BILINEAR)
            img_array = np.array(img)
            images.append({os.path.basename(img_file[:-4]): img_array})
        return images
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% parse_annotation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def parse_annotation(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename").text
        id = filename[:-4]

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        cls = []; bboxes = []


        for obj in root.findall("object"):
            className = obj.find("name").text

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            
            cx = (xmin + xmax) / (2*width)
            cy = (ymin + ymax) / (2*height)
            
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            cls.append([inverted_dict[className]])
            bboxes.append([cx, cy, w, h])

        return {
            "id"            : id,
            "im_file"       : filename,
            "img_shape"     : (width,height),
            "ori_shape"     : (1920, 1080),
            "cls"           : np.array(cls),
            "bboxes"        : np.array(bboxes),
            "normalized"    : False
        }
        
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% load_images
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def load_images(self, root_dir, images_subdir):
        image_files = sorted(glob.glob(os.path.join(root_dir, self.difficulty, images_subdir,"*.jpg")))
        if self.ids:
            # if ids are defined, only retain the images that are specified in ids
            image_files = [img for img in image_files if os.path.basename(img).rsplit('.')[0]  in self.ids]
        return self.load_images_as_numpy(image_files) # TODO: request more memory from aws, g4dn.2xlarge not enough

    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% load_annotations
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def load_annotations(self, root_dir, annotations_subdir):
        annotation_files = sorted(glob.glob(os.path.join(root_dir, self.difficulty, annotations_subdir,"*.xml")))
        if self.ids:
            #if ids are defined, only retain the images that are specified in ids
            annotation_files = [label for label in annotation_files if os.path.basename(label).rsplit('.')[0] in self.ids]
        return [self.parse_annotation(ann_file) for ann_file in tqdm(annotation_files,desc="Loading Annotations")]
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% __getitem__
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def __getitem__(self, index):
        # Implementation for getting an item by index specific to artsDataset
        
        # check if the id of the image and annotation matches before returning data/label pair
        img_id = list(self.images[index].keys())[0]
        ann_id = self.labels[index]['id']
 
        if img_id == ann_id:    
            return self.images[index][img_id],self.labels[index]
        else:
            warnings.warn("The id of the image and annotation does not match")


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% __len__
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def __len__(self):
        # Implementation for getting the length of the dataset specific to artsDataset
        return len(self.images)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% collate_fn
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        
        # (H,W,C) -> (B,C,H,W)
        imgs        = torch.stack([torch.as_tensor(img,dtype=torch.float32).permute(2,0,1) / 255.0 for img in imgs])
        batch_idx   = torch.cat([torch.full((len(l['cls']),),i,dtype=torch.long) for i,l in enumerate(labels)])
        cls         = torch.cat([torch.as_tensor(l['cls'],dtype=torch.long).reshape(-1, 1) for l in labels])
        bboxes      = torch.cat([torch.as_tensor(l['bboxes'],dtype=torch.float32) for l in labels])
        im_files    = [l['im_file'] for l in labels]

        return {
            'im_file'   : im_files,
            'img'       : imgs,
            'cls'       : cls,
            'bboxes'    : bboxes,
            'batch_idx' : batch_idx,
            'ori_shape': [(1920, 1080)] * len(imgs),
            'ratio_pad': [((640/1920, 640/1080), (0, 0))] * len(imgs)
        }    
    
def getTrainTestValSplit(rootDir="/mnt/data/arts", difficulty="easy"):
    with open(os.path.join(rootDir,difficulty,r'ImageSets/Main/train.txt'),'r') as f:
        train_split = f.read().splitlines()
        
    with open(os.path.join(rootDir,difficulty,r'ImageSets/Main/test.txt'),'r') as f:
        test_split = f.read().splitlines()
        
    with open(os.path.join(rootDir,difficulty,r'ImageSets/Main/val.txt'),'r') as f:
        val_split = f.read().splitlines()
        
    return train_split,test_split,val_split

if __name__ == "__main__":
    dataset = artsDataset("/mnt/data/arts", difficulty="easy", images_subdir="JPEGImages", annotations_subdir="Annotations")
        
        
    # Generate visualizations
    for idx,data in enumerate(dataset):
        img,annotation = data
        dataVis(img, annotation)
    
    # print the length of the dataset
    print(f"Dataset length: {len(dataset)}")