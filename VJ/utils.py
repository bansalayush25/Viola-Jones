import numpy as np
import pandas as pd
from scipy import misc
import glob

def load(path):

    images = []
    for image_path in glob.glob(path):
        image = misc.imread(image_path)
        images.append(image)
        
    return images