from skimage import segmentation 
import numpy as np

img = np.zeros((21, 21))
img[:5, :5] = 0.2
_ = segmentation.felzenszwalb(img)
