import numpy as np
from scipy import ndimage

# This function uses tensorflow implementation of tri-linear interpolation to resample image
# input:
#   image: input image, [H, W, D, C]
#   seg: segmentation, [H, W, D]
#   res: 0-1, relative resolution of output to original image
# outputs:
#   image_resize, seg_resize: image, segmentation that get resampled
def resample_by_resolution(image, seg, res):
    image_resize = ndimage.zoom(image, (res, res, res, 1.0), order = 1)
    seg_resize = ndimage.zoom(seg, (res, res, res), order = 0)
    
    return image_resize, seg_resize

# This functio resampled img and seg to 1x1x1.
# inputs: 
#   img: the image, [H, W, D, C]
#   seg: the segmentation, [H, W, D]
#   spacing: tuple, voxel spacing in 3D before resampled
#   new_spacing: tuple, voxel spacing in 3D after resampled
# outputs:
#   image_resize, seg_resize: image, segmentation that get resampled
def resample_by_spacing(image, seg, spacing, new_spacing = (1.0, 1.0, 1.0)):
    # calculate resize factor
    rf = np.array(spacing) / np.array(new_spacing)
    
    image_resize = ndimage.zoom(image, (rf[0], rf[1], rf[2], 1.0), order = 1)
    seg_resize = ndimage.zoom(seg, rf, order = 0)
    
    return image_resize, seg_resize