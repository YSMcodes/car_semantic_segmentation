import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pycocotools import mask as mask_utils
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def pipeline(image):

    resized = read_resize_img((512,264),image)
    segmented = segment_image(resized)
    displayed = display_image(segmented)

    return displayed

def read_resize_img(size, image):

  color_corrected = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  resized = cv2.resize(color_corrected, size)

  return resized

def segment_image(image):
    
    model_path = "./sam_vit_b_01ec64.pth"
    torch_available = torch.cuda.is_available()
    sam = None 
    if torch_available:
        sam = sam_model_registry["vit_b"](checkpoint=model_path).to("cuda")
    else:
        sam = sam_model_registry["vit_b"](checkpoint=model_path)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    return masks

def display_image(masks):

    height, width = masks[0]['segmentation'].shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Set random seed for consistent coloring
    np.random.seed(42)

    # Loop through all masks
    for i, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        mask_uint8 = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_utils.encode(mask_uint8)
        decoded_mask = mask_utils.decode(rle)

        # Generate a random color
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)

        # Overlay color on canvas where mask is True
        for c in range(3):  # RGB channels
            canvas[:, :, c] = np.where(decoded_mask == 1, color[c], canvas[:, :, c])

    return canvas

