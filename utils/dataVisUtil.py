import matplotlib.pyplot as plt
import matplotlib.patches as patches

def dataVis(image,annotation:dict):
    id = annotation['id']
    
    fig,ax = plt.subplots()
    ax.imshow(image)
    
    for obj in annotation['objects']:
        bbox = obj['bbox']
        upper_left = bbox[:2]
        lower_right = bbox[2:]
        
        width = lower_right[0] - upper_left[0]
        height = lower_right[1] - upper_left[1]
        
        rect = patches.Rectangle(upper_left,width,height,linewidth=1,edgecolor='r',alpha=0.3)
        ax.add_patch(rect)
        
        ax.text(upper_left[0]-30,upper_left[1]-30, obj['label'],fontsize=6,fontweight='bold',bbox=dict(facecolor='lime', alpha=0.3, edgecolor='none'))
    
    
    plt.savefig(fr'EDOTS/figures/labeled_{id}.png')