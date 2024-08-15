"""
Date: 2024/08/07
@author: OAO
Procedure: 0. Load dataset and run the model to obtain the predicted results
           1. Read the tiff file as the image stack
           2. Normalize the image stack 
           3. Resize the image stack
           4. Load into the model and save the predicted results
Notes:     1. Remember to revise "LOAD_WHOLE_DATASET","LOAD_SINGLE_IMAGE",
              "FILENAME","INPUT_FOLDER",and"OUTPUT_FOLDER"
           2. Check the "FILENAME" if you need to load single image
           3. Check whether you want to run whole dataset or single dataset
"""
# Load the plugins
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time

# Parameters setting
print('-'*30)
model = load_model('HiLo220511_fromYH_v2.h5', compile=False) # Load the model
LOAD_WHOLE_DATASET = True # Whether you want to run whole dataset
IMG_WIDTH = 256 # image width
IMG_HEIGHT = 256 # image height

if LOAD_WHOLE_DATASET: # Whether you want to run whole dataset
    INPUT_FOLDER = ["test_Data_beads", "test_Data_Lung_deconv",
                    "LungTissue_from230901", "deconvoluted_6um_0.5um_10iter_from230803"]
    DATE = "240806/output_"
    OUTPUT_FOLDER = [f"{DATE}{folder_name}" for folder_name in INPUT_FOLDER]
    FILENAME = "" # For whole dataset, it is empty
else: # Just run single dataset
    INPUT_FOLDER = "deconvoluted_6um_0.5um_10iter_from230803" 
    OUTPUT_FOLDER = "240806/output_deconvoluted_6um_0.5um_10iter_from230803"  
    FILENAME = "Final Display of RL.tif" # "data62to68_from280830_BF_LS_6um.tif"
print('-'*30)



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

def resize_images(images, save_or_not, saved_path, file_name):
    """
    Resize each image in the stack
    """
    resized_images = [] # Create an empty list to store resized images
    for image in images: # Process each image in the stack
        resized_image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        resized_images.append(resized_image) # Append the processed image to the list
    resized_images_array = np.array(resized_images, dtype=np.float32)

    if save_or_not:
        print("Save the input image stack")
        if not os.path.exists(saved_path): # if the directory does not exist, create it
            os.makedirs(saved_path)
        for index, image in enumerate(resized_images_array):
            input_img = (image * 65535).astype(np.uint16)
            filename = f'Input_{file_name}_{index}.tiff'
            cv2.imwrite(saved_path +"/"+filename,input_img)
    return resized_images_array

def model_predict_save(images, saved_path, file_name):
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
        filename = f'HiLo_pred_{file_name}_{index}.tiff'
        cv2.imwrite(saved_path +"/"+filename, output)

def main():
    """
    Read single or whole dataset and run the model
    """
    if LOAD_WHOLE_DATASET:
        for input_folder, output_folder in zip(INPUT_FOLDER, OUTPUT_FOLDER):
            input_path = os.path.join(os.getcwd(), input_folder)
            output_path = os.path.join(os.getcwd(), output_folder)

            for filename in os.listdir(input_path):
                if filename.endswith('.tif') or filename.endswith('.tiff'):
                    full_path = os.path.join(input_path, filename)
                    base_filename, _ = os.path.splitext(filename)
                    image_stack = read_tiff_stack(full_path)
                    show_information(image_stack, f"{filename} image_stack")
                    print("-"*30)
                    normalized_stack = normalize_images(image_stack)
                    show_information(normalized_stack, f"{filename} normalized_stack")
                    print("-"*30)
                    resized_stack = resize_images(normalized_stack, True, output_path, base_filename)
                    show_information(resized_stack, f"{filename} resized_stack")
                    print("-"*30)
                    model_predict_save(resized_stack, output_path, base_filename)
    else:
        input_path = os.path.join(os.getcwd(), INPUT_FOLDER)
        output_path = os.path.join(os.getcwd(), OUTPUT_FOLDER)
        image_stack = read_tiff_stack(input_path+"/"+FILENAME)
        show_information(image_stack, "image_stack")
        print("-"*30)
        normalized_stack = normalize_images(image_stack)
        show_information(normalized_stack, "normalized_stack")
        print("-"*30)
        resized_stack = resize_images(normalized_stack, True, output_path, FILENAME)
        show_information(resized_stack, "resized_stack")
        print("-"*30)
        model_predict_save(resized_stack, output_path, FILENAME)

if __name__ == "__main__":
    # Record the time of running the model
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
