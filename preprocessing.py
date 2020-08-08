import os
import os.path
import numpy as np
from medpy.io import load, save

google_drive_path = "/content/drive/My Drive/H-DenseUNet"


# Verify data folder
def verify_data_folder():
    print("*** Verifying folder creation ***")

    if not os.path.exists(google_drive_path):
        print("Creating H-DenseUNet folder on Google Drive")
        os.mkdir(google_drive_path)
    if not os.path.exists(google_drive_path + "/data"):
        print("Creating data folder for saving training and testing data")
        os.mkdir(google_drive_path + "/data")
    if not os.path.exists(google_drive_path + "/data/TrainingData"):
        print("Create training data folder")
        os.mkdir(google_drive_path + "/data/TrainingData")
    if not os.path.exists(google_drive_path + "/data/TestData"):
        print("Create testing data folder")
        os.mkdir(google_drive_path + "/data/TestData")


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


# HU value of CT scans is set to range [200:250] in order to eliminate unnecessary information
def truncate_hu_value(image_path, saved_folder):
    print("*** Truncating HU value to eliminate superfluous information ***")

    if not os.path.exists(google_drive_path + "/data" + saved_folder):
        print("Creating " + saved_folder + " directory")
        os.mkdir(google_drive_path + "/data" + saved_folder)
    file_list = os.listdir(image_path)
    volume_list = []
    for item in file_list:
        if 'volume' in item:
            volume_list.append(item)

    for item in volume_list:
        img, img_header = load(image_path + "/" + item)
        img[img < -200] = -200
        img[img > 250] = 250
        img = np.array(img, dtype='float32')
        print("Saving image " + item)
        save(img, google_drive_path + "/data" + saved_folder + item)


# Create liver text file to save location of liver in the CT slices
def generate_liver_text(image_path, saved_folder):
    print("*** Getting pixel-wise locations of liver in the CT slices ***")

    if not os.path.exists(google_drive_path + "/data" + saved_folder):
        print("Creating folder to save liver and tumor texts")
        os.mkdir(google_drive_path + "/data" + saved_folder)

    if not os.path.exists(google_drive_path + "/data" + saved_folder + "/LiverPixels"):
        print("Creating liver text folder")
        os.mkdir(google_drive_path + "/data" + saved_folder + "/LiverPixels")

    for i in range(0, 131):
        # Load using Nibabel
        print("Loading segmentation-" + str(i))
        liver_pixel, header = load(google_drive_path + image_path + "/segmentation-" + str(i) + ".nii")
        f = open(google_drive_path + "/data" + saved_folder + "/LiverPixels/liver_" + str(i) + ".txt", "w")
        # Liver is labeled as 1
        index = np.where(liver_pixel == 1)
        x = index[0]
        y = index[1]
        z = index[2]
        print("Saving liver_" + str(i) + ".txt")
        np.savetxt(f, np.c_[x, y, z], fmt="%d")

        f.write("\n")
        f.close()


# Create tumor text file to save location of tumor in the CT slices
def generate_tumor_text(image_path, saved_folder):
    print("*** Getting pixel-wise locations of tumor in the CT slices ***")

    if not os.path.exists(google_drive_path + "/data" + saved_folder):
        print("Creating folder to save liver and tumor texts")
        os.mkdir(google_drive_path + "/data" + saved_folder)

    if not os.path.exists(google_drive_path + "/data" + saved_folder + "/TumorPixels"):
        print("Creating liver text folder")
        os.mkdir(google_drive_path + "/data" + saved_folder + "/TumorPixels")

    for i in range(0, 131):
        print("Loading segmentation-" + str(i))
        tumor_pixel, header = load(google_drive_path + image_path + "/segmentation-" + str(i) + ".nii")
        f = open(google_drive_path + "/data" + saved_folder + "/TumorPixels/tumor_" + str(i) + ".txt", "w")
        # Tumor is labeled as 2
        index = np.where(tumor_pixel == 2)
        x = index[0]
        y = index[1]
        z = index[2]
        print("Saving tumor_" + str(i) + ".txt")
        np.savetxt(f, np.c_[x, y, z], fmt="%d")

        f.write("\n")
        f.close()


# Create liver bounding box for quicker finding
def generate_box_txt(image_path, saved_folder):
    print("*** Getting min and max pixels of liver in the CT slices ***")

    if not os.path.exists(google_drive_path + "/data" + saved_folder):
        print("Creating folder to save liver and tumor texts")
        os.mkdir(google_drive_path + "/data" + saved_folder)

    if not os.path.exists(google_drive_path + "/data" + saved_folder + "/LiverBox"):
        print("Creating liver box folder")
        os.mkdir(google_drive_path + "/data" + saved_folder + "/LiverBox")

    for i in range(0, 131):
        # Load up the pixel values of the liver
        print("Loading liver_" + str(i) + ".txt")
        values = np.loadtxt(google_drive_path + '/data/myTrainingDataTxt/LiverPixels/liver_' + str(i) + '.txt', delimiter=' ', usecols=[0, 1, 2])
        a = np.min(values, axis=0)
        b = np.max(values, axis=0)
        box = np.append(a, b, axis=0)
        print("Saving box_" + str(i) + ".txt")
        np.savetxt(google_drive_path + '/data/myTrainingDataTxt/LiverBox/box_' + str(i) + '.txt', box, fmt='%d')


print("-" * 15 + " Beginning to preprocess data " + "-" * 15)
verify_data_folder()
# load_data_to_folders()
truncate_hu_value(image_path=google_drive_path + "/data/TrainingData/", saved_folder="/myTrainingData/")
truncate_hu_value(image_path=google_drive_path + "/data/TestData", saved_folder="/myTestData/")
generate_liver_text(image_path="/data/TrainingData", saved_folder="/myTrainingDataTxt")
generate_tumor_text(image_path="/data/TrainingData", saved_folder="/myTrainingDataTxt")
generate_box_txt(image_path="/data/TrainingData", saved_folder="/myTrainingDataTxt")
