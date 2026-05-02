import matplotlib.pyplot as plt

def dataVis(image,annotation:dict):
    # need to figure out how to port forward so that we can display the images over ssh
    plt.imshow(image)
    plt.show()