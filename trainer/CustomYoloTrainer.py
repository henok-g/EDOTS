import sys
sys.path.append('EDOTS')
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import strip_optimizer
from ultralytics.utils import RANK
from datasets.artsDataset import artsDataset,getTrainTestValSplit
from trainer.label_map import label_mapping


class CustomYoloTrainer(DetectionTrainer):
    def final_eval(self):
        if RANK in {-1, 0}:
            if self.last.exists():
                strip_optimizer(self.last)
            if self.best.exists():
                strip_optimizer(self.best)
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        self.metrics = self.validator(self)
        self.metrics.pop("fitness", None)
        self.run_callbacks("on_fit_epoch_end")

    def get_dataset(self):
        return {
            "path"      : "/mnt/data/arts",
            "train"     : "/mnt/data/arts",
            "val"       : "/mnt/data/arts",
            "test"      : "/mnt/data/arts",
            "channels"  : 3,
            "nc"        : 78,
            "names"     : label_mapping[0]  
        }
    
    def build_dataset(self, img_path, mode='train',batch=None):
        # determine the train/test/val split
        train_split,test_split,val_split = getTrainTestValSplit(img_path)
        
        if mode == 'train':
            return artsDataset(img_path,ids=train_split)
        elif mode == 'val':
            return artsDataset(img_path,ids=val_split)
        elif mode == 'test':
            return artsDataset(img_path,ids=test_split)
        
        return super().build_dataset(img_path,mode,batch)
    

if __name__ == '__main__':
    trainer = CustomYoloTrainer()
    