import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

dataset_size = 4000
attributes = ['hair', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'nose', 'mouth', 'skin', 'l_ear', 'r_ear', 'u_lip', 'l_lip']
mask_attributes = ['hair','l_ear', 'r_ear', 'skin']
img_dir = '/kaggle/input/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img'
mask_dir = '/kaggle/input/celebamaskhq/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
output_dir = 'dataset/edge2face'

for i in range(dataset_size):
    print(i)
    mask_dir_index = int(i/2000)
    img = cv2.imread(os.path.join(img_dir, f'{i}.jpg'))
    img = cv2.resize(img, (256,256))
    
    combined_mask = np.zeros(img.shape[:2], dtype='uint8') 
    combined_edge = np.zeros(img.shape[:2], dtype='uint8') 
    for attribute in attributes:
        path = os.path.join(mask_dir, str(mask_dir_index), f'{i:05}_{attribute}.png')
        if os.path.exists(path):
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img.shape[:2])
            
            edge = cv2.Canny(mask, 100, 200)
            combined_edge = cv2.bitwise_or(combined_edge, edge)
            
            if attribute in mask_attributes:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    masked_img = cv2.bitwise_and(img, img, mask=combined_mask)
    combined_mask = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    masked_img[combined_mask==0] = 255
    combined_edge = 255 - cv2.cvtColor(combined_edge, cv2.COLOR_GRAY2BGR)
    
    pair = cv2.hconcat([combined_edge, masked_img])
    cv2.imwrite(os.path.join(output_dir, f'{i}.png'), pair)