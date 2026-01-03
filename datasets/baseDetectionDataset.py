from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class baseTrafficSignDataset(Dataset, ABC):
    '''
    Docstring for baseTrafficSignDataset
    
    Detection dataset contract:

    - load_image: Method to load images from the dataset
        returns:
            list of tuples: Each tuple contains (image_id, image_data)

            
    - load_annotations: Method to load annotations from the dataset 
        returns:
            dict: A dictionary where keys are image_ids and values are lists of annotations

            
    - __getitem__: Method to get an item by index
        returns:
            tuple: (image_data, annotations) for the given index

            
    - __len__: Method to get the length of the dataset
        returns:
            int: Total number of items in the dataset


    '''

    def __init__(self, root_dir):
        self.root_dir       = root_dir
        self.annotations    = self.load_annotations(root_dir)
        self.images         = self.load_image(root_dir)
        

    @abstractmethod
    def load_image(self, root_dir):
        pass

    @abstractmethod
    def load_annotations(self, root_dir):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def preprocess(self, data):
        pass

