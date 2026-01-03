from baseDetectionDataset import baseDetectionDataset

class mapsDataset(baseDetectionDataset):
    def __init__(self, root):
        super().__init__(root)

    def load_image(self):
        # Implementation for loading images specific to mapsDataset
        pass

    def load_annotations(self):
        # Implementation for loading annotations specific to mapsDataset
        pass

    def __getitem__(self, index):
        # Implementation for getting an item by index specific to mapsDataset
        pass    

    def __len__(self):
        # Implementation for getting the length of the dataset specific to mapsDataset
        pass