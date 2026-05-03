import matplotlib.pyplot as plt
import matplotlib.patches as patches

def dataVis(image,annotation:dict):
    # determine the id for the current data sample
    id = annotation['id']
    
    # Plot the image
    fig,ax = plt.subplots()
    ax.imshow(image)
    
    # Loop through each label, determine where the object is located, and plot a rectangle around it
    for obj in annotation['objects']:
        # determine where upper left (UL) and lower right (LR) corners of the bounding box are located
        bbox = obj['bbox']
        ul = bbox[:2]
        lr = bbox[2:]
        
        width = lr[0] - ul[0]
        height = lr[1] - ul[1]
        
        # generate rect around the object within the image
        rect = patches.Rectangle(ul,width,height,linewidth=1,edgecolor='r',alpha=0.3)
        ax.add_patch(rect)
        
        # add the label of the object to the image
        ax.text(ul[0]-30,ul[1]-30, obj['label'],fontsize=6,fontweight='bold',bbox=dict(facecolor='lime', alpha=0.3, edgecolor='none'))
    
    # save the figure TODO: Determine if I should use absolute paths or not
    plt.savefig(fr'EDOTS/figures/labeled_{id}.png')