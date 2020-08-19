import os
import os.path
import constant
import cv2
import numpy as np
from medpy.io import load, save

main_path = constant.MAIN_PATH
data_folder = constant.DATA_PATH
ori_training_folder = constant.RAW_TRAINING_DATA
ori_testing_folder = constant.RAW_TESTING_DATA
training_folder = constant.PROCESSED_TRAINING_DATA
testing_folder = constant.PROCESSED_TESTING_DATA
liver_voxel_folder = constant.LIVER_VOXEL_DATA
bounding_box_folder = constant.LIVER_BOX_DATA


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                 #
#   1. Preprocessing data with HU clipping, normalization and enhance contrast    #
#   2. Generate a list of voxel labeled as liver from liver mask                  #
#   3. Generate a bounding box for liver                                          #
#                                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Verify data folder
def verify_data_folder():
    print("*** Verifying folder creation ***")
    if not os.path.exists(main_path):
        print("Creating H-DenseUNet folder on Google Drive")
        os.mkdir(main_path)
    if not os.path.exists(data_folder):
        print("Creating data folder for saving training and testing data")
        os.mkdir(data_folder)
    if not os.path.exists(ori_training_folder):
        print("Create raw training data folder")
        os.mkdir(ori_training_folder)
    if not os.path.exists(ori_testing_folder):
        print("Create raw testing data folder")
        os.mkdir(ori_testing_folder)
    if not os.path.exists(training_folder):
        print("Create processed training data folder")
        os.mkdir(training_folder)
    if not os.path.exists(testing_folder):
        print("Create processed testing data folder")
        os.mkdir(testing_folder)
    if not os.path.exists(bounding_box_folder):
        print("Create liver bounding box data folder")
        os.mkdir(bounding_box_folder)
    if not os.path.exists(liver_voxel_folder):
        print("Create liver voxel data folder")
        os.mkdir(liver_voxel_folder)


# Load training data to data folder; Users modify folder path of training data if needed.
# This function is only used for notebook
"""
def load_data_to_folders():
    print("Moving training data to data folder")
    file_list = os.listdir("/content/drive/My Drive/LITS Challenge/Training Batch 1")
    for item in file_list:
        %cp -av /content/drive/My\ Drive/LITS\ Challenge/Training\ Batch\ 1/{item} /content/drive/My\ Drive/H-DenseUNet/data/TrainingData
    file_list = os.listdir("/content/drive/My Drive/LITS Challenge/Training Batch 2")
    for item in file_list:
        %cp -av /content/drive/My\ Drive/LITS\ Challenge/Training\ Batch\ 2/{item} /content/drive/My\ Drive/H-DenseUNet/data/TrainingData
    print("Moving testing data to test folder")
    file_list = os.listdir("/content/drive/My Drive/LITS-Challenge-Test-Data")
    for item in file_list:
        %cp -av /content/drive/My\ Drive/LITS-Challenge-Test-Data/{item} /content/drive/My\ Drive/H-DenseUNet/data/TestData     
"""


def normalize_image(img):
    min_value, max_value = float(np.min(img)), float(np.max(img))
    return (img - min_value) / (max_value - min_value)


def apply_clahe(img):
    # Adaptive histogram equalization
    # Get the number of slices of a CT scan
    slice_num = img.shape[2]

    for i in range(slice_num):
        # Image must be normalize and convert to unit8 type to be applied with CLAHE
        norm_img = np.uint8(cv2.normalize(img[:, :, i], None, 0, 255, cv2.NORM_MINMAX))
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(norm_img)
        # Replace original slice with histogram slice
        img[:, :, i] = cl1

    return img


# HU value of CT scans is set to range [-200:250] in order to eliminate unnecessary information
def truncate_hu_value(image_path, saved_folder):
    print("*** Truncating HU value to eliminate superfluous information ***")
    file_list = os.listdir(image_path)
    volume_list = []
    # Only volume data need preprocessing, liver mask is not needed.
    for item in file_list:
        if 'volume' in item:
            volume_list.append(item)

    for item in volume_list:
        img, img_header = load(image_path + '/' + item)
        img = np.clip(img, -200, 250)
        normalize_image(img)
        apply_clahe(img)
        normalize_image(img)
        img = np.array(img, dtype='int16')
        print('Saving image ' + item)
        save(img, saved_folder + '/' + item)


# Create liver text file to save location of liver in the CT slices
def generate_liver_voxel(image_path, saved_folder):
    print("*** Getting pixel-wise locations of liver in the CT slices ***")

    for i in range(0, 131):
        print("Loading segmentation-" + str(i))
        liver_pixel, header = load(image_path + "/segmentation-" + str(i) + ".nii")
        f = open(saved_folder + "/" + "liver_" + str(i) + ".txt", "w")
        # Liver is labeled as 1
        index = np.where(liver_pixel == 1)
        x = index[0]
        y = index[1]
        z = index[2]
        print("Saving liver_" + str(i) + ".txt")
        # Save the location with (x,y,z) format
        np.savetxt(f, np.c_[x, y, z], fmt="%d")

        f.write("\n")
        f.close()


# Create liver bounding box for quicker finding
def generate_bounding_box(image_path, saved_folder):
    print("*** Getting min and max pixels of liver in the CT slices ***")

    for i in range(0, 131):
        print("Loading liver_" + str(i) + ".txt")
        values = np.loadtxt(image_path + "/liver_" + str(i) + ".txt", delimiter=" ", usecols=[0, 1, 2])
        a = np.min(values, axis=0)
        b = np.max(values, axis=0)
        box = np.append(a, b, axis=0)
        print("Saving box_" + str(i) + ".txt")
        np.savetxt(saved_folder + "/box_" + str(i) + ".txt", box, fmt="%d")


print("-" * 15 + " Beginning to preprocess data " + "-" * 15)
verify_data_folder()
# load_data_to_folders()
truncate_hu_value(image_path=ori_training_folder, saved_folder=training_folder)
truncate_hu_value(image_path=ori_testing_folder, saved_folder=testing_folder)
generate_liver_voxel(image_path=ori_training_folder, saved_folder=liver_voxel_folder)
generate_bounding_box(image_path=ori_training_folder, saved_folder=bounding_box_folder)
