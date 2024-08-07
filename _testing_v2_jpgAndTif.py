"""
Date: 2024/07/23
@author: OAO
Procedure: 1. Read the tiff file as the image stack (can deal with jpg and tif files)
           2. Normalize the image stack 
           3. Resize the image stack
           4. Load into the model and save the predicted results
Notes:     1. Remember to revise the name of "FILENAME", "INPUT_FOLDER", and "OUTPUT_FOLDER"
           2. Check the name of "FILENAME" if your data is jpg or tif file
"""
# Load the plugins
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
# from queue import Queue
# from threading import Thread
# import warnings
import numpy as np
# import h5py as h5
# from datetime import datetime
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Add, Input, Concatenate, Lambda
# from tensorflow.keras.layers import Conv2D, BatchNormalization
# from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, LeakyReLU
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
from PIL import Image
import cv2
import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Parameters setting
print('-'*30)
model = load_model('HiLo220511_fromYH_v2.h5', compile=False) # Load the model
INPUT_FOLDER = "test_Data_beads" # "test_Data_beads", "test_Data_Lung_deconv", "LungTissue_from230901"
OUTPUT_FOLDER = "test"  # "240723_secondTry", "240724_thirdTry", "240724_Lung_RL_results", "240724_Lung_results_data5"
FILENAME = "data62to68_from280830_BF_LS_6um.tif" # "data62to68_from280830_BF_LS_6um.tif"
# image stack size (width, height, number of images)
IMG_WIDTH = 256 #477
IMG_HEIGHT = 256 #588
print('-'*30)

input_path = os.path.join(os.getcwd(), INPUT_FOLDER)
output_path = os.path.join(os.getcwd(), OUTPUT_FOLDER)

def show_information(np_array, name):
    """
    Show the information of the image stack(numpy array)
    """
    print(f"The shape of {name} is {np_array.shape}")  # Check the size
    print(f"The data type of {name} is {np_array.dtype}")  # Check the data type
    print(f"The max value of {name} is {np_array.max()}")  # Check the max value
    print(f"The min value of {name} is {np_array.min()}")  # Check the min value

def read_tiff_stack(file_path):
    """
    Read the tiff file and use numpy to store the data (3D matrix)
    """
    with Image.open(file_path) as img: # Open the file
        img.seek(0) # Go to the first frame
        frames = [] # Create an empty list to store the frames
        try:
            while True:  # Read the next frame and add it to the list
                current_frame = np.array(img)
                frames.append(current_frame)
                img.seek(img.tell() + 1)
        except EOFError:  # If there are no more frames, stop reading
            pass
        stack = np.stack(frames, axis=0) # Convert the list to a 3D matrix
    return stack

def read_jpg_stack(file_path):
    pass

def normalize_images(images):
    """
    Normalize each image in the stack
    """
    normalized_images = [] # Create an empty list to store normalized images
    for image in images: # Process each image in the stack
        image_float = image.astype(np.float32) # Convert to float32
        normalized_image = image_float / image_float.max() # Normalize the image
        normalized_images.append(normalized_image) # Append the processed image to the list
    # Optionally, convert list back to a numpy array with dtype float32
    normalized_images_array = np.array(normalized_images, dtype=np.float32)
    return normalized_images_array

def resize_images(images, save_or_not, saved_path):
    """
    Resize each image in the stack
    """
    resized_images = [] # Create an empty list to store resized images
    for image in images: # Process each image in the stack
        resized_image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        resized_images.append(resized_image) # Append the processed image to the list
    resized_images_array = np.array(resized_images, dtype=np.float32)

    if save_or_not:
        print(f"Save the input image stack")
        if not os.path.exists(saved_path): # if the directory does not exist, create it
            os.makedirs(saved_path)
        for index, image in enumerate(resized_images_array):
            input_img = (image * 65535).astype(np.uint16)
            filename = f'Input_{index}.tiff'
            cv2.imwrite(saved_path +"/"+filename,input_img)
    return resized_images_array

def model_predict_save(images, saved_path):
    """
    Input the resized images and store the predicted results
    """
    if not os.path.exists(saved_path): # if the directory does not exist, create it
        os.makedirs(saved_path)
    for index, image in enumerate(images):
        input_img = image.reshape((1, IMG_HEIGHT, IMG_WIDTH, 1))
        pred = model.predict(input_img)
        pred_img = pred.reshape((IMG_HEIGHT, IMG_WIDTH))
        # plt.figure(1)
        # plt.imshow(predimg,cmap='gray')
        output = (pred_img * 65535).astype(np.uint16)
        filename = f'HiLo_pred_{index}.tiff'
        cv2.imwrite(saved_path +"/"+filename, output)

def main():
    if FILENAME.endswith('.tif') or FILENAME.endswith('.tiff'):
        image_stack = read_tiff_stack(input_path+"/"+FILENAME)
    else:
        # Assuming if not tiff, then it's a folder of jpg and FILENAME should be ignored
        image_stack = read_jpg_stack(INPUT_FOLDER)
    show_information(image_stack, "image_stack")
    print("-"*30)
    normalized_stack = normalize_images(image_stack)
    show_information(normalized_stack, "normalized_stack")
    print("-"*30)
    resized_stack = resize_images(normalized_stack, True, output_path)
    show_information(resized_stack, "resized_stack")
    print("-"*30)
    model_predict_save(resized_stack, output_path)

if __name__ == "__main__":
    main()