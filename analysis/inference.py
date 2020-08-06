'''
Script to iterate over directories and output specimen/mosaic-level probabilities distribution
'''
import os
import sys
import time
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
from keras.models import load_model
from pandas import DataFrame
from collections import defaultdict

def import_dicom(image_dir):
    """Function that will import SRH image strips.
    image_dir must contain only the image strips for a single mosaic
    First file = CH2
    Second file = CH3
    """
    files = [file for file in os.listdir(image_dir) if ("dcm" in file)]
    image_num = []
    for file in files:
        _ ,_ , num_str = file.split(sep=("_"))
        image_num.append(int(num_str.split(sep=".")[0]))

    # sort the strips based on the image image number, NOTE: Assumes CH2 is the first channel captured
    sorted_files = [name for name, _ in sorted(zip(files, image_num), key=lambda pair: pair[1])]

    # read in every other file
    CH2_files = sorted_files[::2]
    CH3_files = sorted_files[1::2]

    # import the first image to get dicom specs
    CH3 = dicom.read_file(os.path.join(image_dir, CH3_files[0])).pixel_array.astype(float)
    CH2 = dicom.read_file(os.path.join(image_dir, CH2_files[0])).pixel_array.astype(float)

    def import_array(filelist, first_strip):
        """Iteratively concatenate each strip columnwise
        """
        for file in filelist[1:]: 
            strip = dicom.read_file(os.path.join(image_dir, file)).pixel_array.astype(float)
            first_strip = np.concatenate((first_strip, strip), axis = 1)
        return first_strip

    CH3 = import_array(CH3_files, CH3)
    CH2 = import_array(CH2_files, CH2)

    subtracted_array = np.subtract(CH3, CH2)
    subtracted_array[subtracted_array < 0] = 0.0 # negative values set to zero

    dcm_stack = np.zeros((CH2.shape[0], CH2.shape[1], 3), dtype=float)
    dcm_stack[:, :, 0] = subtracted_array
    dcm_stack[:, :, 1] = CH2
    dcm_stack[:, :, 2] = CH3

    return dcm_stack

def cnn_preprocessing(image):
    """
    Subtract training set channel mean
    """
    image[:,:,0] -= 102.1
    image[:,:,1] -= 91.0
    image[:,:,2] -= 101.5
    return(image)

def return_channels(array):
    """
    Helper function to return channels
    """
    return array[:,:,0], array[:,:,1], array[:,:,2]

def percentile_rescaling(array):
    """
    Pixel clipping by percentile and rescaling
    """
    p_low, p_high = np.percentile(array, (3, 97))
    array = array.clip(min = p_low, max = p_high)
    img = (array - p_low)/(p_high - p_low)
    return img

def channel_preprocessing(array):
    """
    Function to rescale each individual patch channels
    """
    CH3minusCH2, CH2, CH3 = return_channels(array)
    img = np.empty((array.shape[0], array.shape[1], 3), dtype=float)
    img[:,:,0] = percentile_rescaling(CH3minusCH2)
    img[:,:,1] = percentile_rescaling(CH2)
    img[:,:,2] = percentile_rescaling(CH3)
    img *= 255
    return img

def patch_generation(array, step_size = 300):
    """Accepts a square 2-channel numpy array that should be the result of stitching together SRS strips

    CH2 <- first channel (Green)
    CH3 <- second channel (Blue)

    Returns a (patch_num X 300 X 300 X 3) numpy array for feed forward CNN pass
    """

    side = array.shape[1]
    starts = list(np.arange(0, side - step_size, step_size)) # arange function excludes the last value, so much add a step_size
    patch_array = np.zeros((len(starts) ** 2, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), dtype=float)
    patch_count = 0

    for x in starts:
        for y in starts: # iterate in the x and y direction
            patch_array[patch_count,:,:,:] = channel_preprocessing(array[x:x + IMAGE_SIZE, y:y + IMAGE_SIZE, :])
            patch_count += 1
    assert patch_array.shape[0] == len(starts) ** 2, "Incorrect number of patches in patch_array"
    return patch_array

def feedforward(patch_array, model):
    """Function to perform a forward pass, with preprocessing on all the patches generated above, outputs 
    """
    num_patches = patch_array.shape[0]

    softmax_output = np.zeros((1,TOTAL_CLASSES), dtype=float)
    nondiag_count = 0

    for i in range(num_patches):
        patch = cnn_preprocessing(patch_array[i,:,:,:])
        pred = model.predict(patch[None,:,:,:], batch_size = 1)
        softmax_output += pred # unnormalized probabability distribution

    return softmax_output.reshape(TOTAL_CLASSES)

def prediction(softmax_output):
    """Implementation of inference algorithm for patient-level probability distribution
    """
    renorm_dist = softmax_output/softmax_output.sum()
    # np.testing.assert_allclose(sum(renorm_dist), 1.0, rtol=0.01, err_msg='Probability not properly normalized.')  
    # if (renorm_dist[2] + renorm_dist[11] + renorm_dist[13]) > 0.9: # nonneoplastic classes
    #     return renorm_dist
    # else:
    #     renorm_dist[[2, 11, 13]] = 0 # set nonneoplastic classes to zero
    #     return renorm_dist/renorm_dist.sum()
    return renorm_dist

def plotting(renorm_dist):
    """Saves a .dicom file with diagnosis and bar plot of probabilities for each mosaic.
    """
    # dist =  np.bincount(np.random.randint(0, 14, 1000))
    # dist = list(dist/dist.sum())

    sorted_classes = [name for name, _ in sorted(zip(class_names, renorm_dist), key=lambda pair: pair[1], reverse = True)]

    plt.rcParams["figure.figsize"] = (10,10)
    plt.bar(x = sorted_classes, height = sorted(renorm_dist, reverse=True))
    plt.xticks(rotation = 90)
    plt.title(str(class_dict[np.argmax(renorm_dist)]) + " (probability = " + str(np.round(np.max(renorm_dist), decimals=3)) + ")", fontsize=24)
    plt.subplots_adjust(bottom = 0.25)
    plt.savefig('gui_image.png', dpi = 500)
    print("Figure saved to working directory.")


def directory_iterator(root):
    """Iterator through a directory that contains:
        1) individual subdirectories that contain SRH strips in alterating order
        2) Bijection of specimens to directories
    """

    def remove_nondiag(norm_dist):
        norm_dist[0] = 0 # set nondiagnostic class to zero
        return norm_dist/norm_dist.sum() # renormalize the distribution

    pred_dict = defaultdict(list)
    for dirpath, dirname, files in os.walk(root): 
        if "NIO" in dirpath:
            print(dirpath)
            mosaic = import_dicom(dirpath)
            patches = patch_generation(mosaic)
            normalized_dist = prediction(feedforward(patches, model))

            pred_dict["specimen"].append(dirpath.split("/")[-1]) # select only the filename, remove root
            pred_dict["nondiagnostic"].append(normalized_dist[0])
            pred_dict["pseudoprogression"].append(normalized_dist[1])
            pred_dict["recurrence"].append(normalized_dist[2])

            # probabilities when renormalizing without nondiagnostic
            rm_nondiag = remove_nondiag(normalized_dist)
            pred_dict["pseudo_renorm"].append(rm_nondiag[1])
            pred_dict["recurrence_renorm"].append(rm_nondiag[2])

    return pred_dict


if __name__ == '__main__':

    # constants
    IMAGE_SIZE, IMAGE_CHANNELS = 300, 3
    TOTAL_CLASSES = 3

    # output specifications
    class_names = ['nondiagnostic', 'pseudoprogression', 'recurrence']
    class_dict = dict(zip(range(TOTAL_CLASSES), class_names))

    # load model    
    model = load_model("/home/todd/Desktop/RecPseudo_project/patches/cv_round2/recurmodel_kthfold_0.hdf5")

    # iterate through the directories with 
    root_dir = "" # root directory with each specimen in own directory 
    pred_dict = directory_iterator(root=root_dir)

    # save to results to excel spreadsheet
    df = DataFrame(pred_dict)
    df.to_excel("predict.xlsx")

    mosaic = import_dicom("")
    patches = patch_dictionary(mosaic)
