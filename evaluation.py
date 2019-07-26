import numpy as np
from sklearn.metrics import jaccard_score
from PIL import Image


gt_color_to_label = {
    (128,   0,   0) : 0,  # building
    (128, 128, 128) : 1,  # sky
    (128,  64, 128) : 2,  # road
    (128, 128,   0) : 3,  # vegetation
    (  0,   0, 192) : 4,  # sidewalk
    ( 64,   0, 128) : 5,  # car
    ( 64,  64,   0) : 6,  # pedestrain
    (  0, 128, 192) : 7,  # cyclist
    (192, 128, 128) : 8,  # signate
    ( 64,  64, 128) : 9,  # fence
    (192, 192, 128) : 10, # pose
    (  0,   0,   0) : 255 # invalid
}


# Read images
img_bgk = Image.open('/home/ganlu/reproj_label_maps.png')
img_bgk = np.array(img_bgk)
img_crf = Image.open('/home/ganlu/la3dm_ws/src/semantic_3d_mapping/grid_sensor/data_kitti/crf_3d_reproj/000000_bw.png')
img_crf = np.array(img_crf)
img_gt = Image.open('/home/ganlu/000000.png')
img_gt = np.array(img_gt)

# Convert rgb to label
rows, cols, _ = img_gt.shape
gt = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        gt[i, j] = gt_color_to_label[tuple(img_gt[i, j, :])]
        
# Ignore label 255
mask = (gt == 255)
img_bgk[mask] = 255
img_crf[mask] = 255
mask = (img_bgk == 255)
gt[mask] = 255

# Flatten
img_bgk = img_bgk.flatten()
img_crf = img_crf.flatten()
gt = gt.flatten()

# IoU for each class
print( np.unique(np.concatenate((img_bgk, gt), axis=0)) )
jaccard_score(gt, img_bgk, average=None)
