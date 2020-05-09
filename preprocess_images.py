# =============================================================================
# resizing, normalization and adaptive histogram equalization to images
# =============================================================================
import skimage.io as io
from skimage.transform import  rescale,resize
from skimage.util import img_as_uint,img_as_ubyte
from skimage.color import rgb2gray
from skimage import exposure
import os
import numpy as np
# =============================================================================
# source and destination dirs
# =============================================================================
class_name='normal'#'covid' or 'normal' or 'pneumonia'
source_dir='dataset/original_images/'+class_name
destination_dir='dataset/original_images_preprocessed/'+class_name
# =============================================================================
# list images list from source dir
# =============================================================================
image_list=os.listdir(source_dir)# list of images
# =============================================================================
# Normalization and adaptive histogram equalization to each single image
# =============================================================================
for img_name in image_list:
    img=io.imread(os.path.join(source_dir,img_name))
    img_gray = rgb2gray(img)
    img_resized = resize(img_gray, (512, 512))#convert image size to 512*512
    img_rescaled=(img_resized-np.min(img_resized))/(np.max(img_resized)-np.min(img_resized))#min-max normalization 
    img_enhanced=exposure.equalize_adapthist(img_rescaled)#adapt hist
    img_resized_8bit=img_as_ubyte(img_enhanced)
    io.imsave(os.path.join(destination_dir,img_name),img_resized_8bit)#save enhanced image to destination dir
   

