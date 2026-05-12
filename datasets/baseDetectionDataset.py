from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class baseTrafficSignDataset(Dataset, ABC):
    '''
    Docstring for baseTrafficSignDataset
    
    Detection dataset contract:

    - load_images: Method to load images from the dataset
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

    def __init__(self, root_dir, images_subdir="JPEGImages", annotations_subdir="Annotations"):
        self.root_dir       = root_dir
        self.images_subdir = images_subdir
        self.annotations_subdir = annotations_subdir
        self.labels         = self.load_annotations(root_dir,annotations_subdir)
        self.images         = self.load_images(root_dir,images_subdir)
        self.num_samples    = self.__len__()
        

    @abstractmethod
    def load_images(self):
        pass

    @abstractmethod
    def load_annotations(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


