import torch

class BaseTrainer:
    '''
    This is just the skeleton code for the trainer function. Will implement functions once model, optimizer, 
    loss function have been determined,
    '''
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% __init__
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def __init__(self, model, optimizer,criterion=None,device=None,train_data=None,test_data=None,val_data=None,epochs=None):
        self.model      = model.to(device) if device else model
        self.optimizer  = optimizer if optimizer else torch.optim.SGD
        self.criterion  = criterion if criterion else torch.nn.BCELoss
        self.device     = device if device else 'cpu'
        self.train_data = train_data
        self.test_data  = test_data
        self.val        = val_data
        self.epochs     = epochs if epochs else 10
        
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% train_epoch
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def train_epoch(self,dataloader):
        '''
        Function for training model for an epoch
        TODO: Implement the train epoch function
        '''
        
        self.model.train()
        pass
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% evaluate
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def evaluate(self,dataloader):
        '''
        Function for running validation set through model
        TODO: Implement the evaluate function
        '''
        
        self.model.eval()
        
        with torch.no_grad():
            pass
        

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% save_checkpoint
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def save_checkpoint(self):
        '''
        Save out the model checkpoint for the current epoch
        TODO: implement the save checkpoint function
        '''
        
        pass
    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% fit
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def fit(self, epochs):
        '''
        Simple training loop that trains model each epoch and evaluates at the end of epoch
        TODO: implement the fit function
        '''
        
        for epoch in range(epochs):
            # train the model
            self.train_epoch(self.train_data)
            
            self.evaluate(self.test_data)
            
            
            
if __name__ == '__main__':
    import torch
    import torchvision
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    
    
    # initialize the arguments for the FasterRCNN model
    # TODO Verify that this works, this follows the example on FasterRCNN docs but they use pretrained weights
    # I removed the pretrained weights so that I can test with my own dataset
    
    backbone = torchvision.models.mobilenet_v2().features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    
    # define the model
    model = FasterRCNN(num_classes=2,rpn_anchor_generator=anchor_generator,
                  box_roi_pool=roi_pooler)
    
    # define the optimizer
    opt = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    
    # define the loss function
    criterion = torch.nn.modules.loss.CrossEntropyLoss()
    
    # define the train and test data
    # TODO pass this through a dataloader
    train_data = None
    test_data = None
    
    epochs = 10
    
    trainer_params = {
        'model'     : model,
        'opt'       : opt,
        'criterion' : criterion,
        'train_data': train_data,
        'test_data' : test_data,
        'epochs'    : epochs
    }
    
    trainer = BaseTrainer(**trainer_params)