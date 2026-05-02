import os
from baseDetectionDataset import baseTrafficSignDataset
import glob
import pandas as pd
from xml.etree import ElementTree as ET
from PIL import Image
import numpy as np
from tqdm import tqdm


class artsDataset(baseTrafficSignDataset):
    '''
    Docstring for artsDataset
    
    __init__(self, root, difficulty="easy"): Initializes the dataset with the root directory and difficulty level.
        root (str) - The root directory of the dataset.
        
        difficulty (str) - The difficulty level of the dataset options are: "easy","challenging"
    
        images_sub_dir (str) - TODO: Fill this in 
        
        annotations_subdir (str) - TODO: Fill this in
        
    load_images(self): Method to load images from the dataset
    
    '''
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% __init__
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def __init__(self, root,difficulty="easy", images_subdir="JPEGImages", annotations_subdir="Annotations"):
        self.difficulty = difficulty
        super().__init__(root)
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% load_images_as_numpy
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def load_images_as_numpy(self, image_files):
        images = {}
        for img_file in tqdm(image_files,desc="Loading Images"):
            img = Image.open(img_file)
            img_array = np.array(img)
            images[os.path.basename(img_file[:-4])] = img_array
        return images
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% parse_annotation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def parse_annotation(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename").text

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        objects = []

        for obj in root.findall("object"):
            label = obj.find("name").text

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            objects.append({
                "label": label,
                "bbox": [xmin, ymin, xmax, ymax]
            })

        return {
            "filename": filename,
            "width": width,
            "height": height,
            "objects": objects
        }
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% load_images
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def load_images(self, root_dir, images_subdir):
        image_files = glob.glob(os.path.join(root_dir, self.difficulty, images_subdir,"*.jpg"))
        return self.load_images_as_numpy(image_files[:3000]) # TODO: request more memory from aws, g4dn.2xlarge not enough

    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% load_annotations
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def load_annotations(self, root_dir, annotations_subdir):
        annotation_files = glob.glob(os.path.join(root_dir, self.difficulty, annotations_subdir,"*.xml"))
        return [self.parse_annotation(ann_file) for ann_file in tqdm(annotation_files,desc="Loading Annotations")]
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% __getitem__
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def __getitem__(self, index):
        # Implementation for getting an item by index specific to artsDataset
        pass    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% __len__
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def __len__(self):
        # Implementation for getting the length of the dataset specific to artsDataset
        pass
    

if __name__ == "__main__":
    dataset = artsDataset("/mnt/data/arts", difficulty="easy", images_subdir="JPEGImages", annotations_subdir="Annotations")
    
    # validate that annotations are sequential with images, currently can only load 3000 images from arts dataset
    
    # Generate visualizations

    
    # come up with models
    
    # determine metrics
    
    # perform classification
    
    # scale up
    
    # augmentation
    print(f"Dataset length: {len(dataset)}")