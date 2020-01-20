

# standard python
import os
import sys
from collections import defaultdict, OrderedDict
# data science
import numpy as np
import pandas as pd
from scipy import stats
# plotting
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_SIZE, IMAGE_CHANNELS = 300, 3
TOTAL_CLASSES = 3

def plot(array):
	'''
	function for quick plotting
	'''
	plt.imshow(array)
	plt.show()

def return_channels(array):
    """
    Helper function
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
    Function to rescale each individual patch
    """
    CH3minusCH2, CH2, CH3 = return_channels(array)
    img = np.empty((array.shape[0], array.shape[1], 3), dtype=float)
    img[:,:,0] = percentile_rescaling(CH3minusCH2)
    img[:,:,1] = percentile_rescaling(CH2)
    img[:,:,2] = percentile_rescaling(CH3)
    img *= 255
    return img

def cnn_preprocessing(image):
    """
    Subtract training set channel mean
    """
    image[:,:,0] -= 102.1
    image[:,:,1] -= 91.0
    image[:,:,2] -= 101.5
    return(image)

class Patch(object):
	def __init__(self, patch_num, num_classes):
		self.patch_num = patch_num
		self.tile_indices = 0
		self.classes = num_classes
		self.softmax = np.zeros((num_classes))

	def __str__(self):
		return("patch_" + str(self.patch_num))

	def __repr__(self):
		return("patch_" + str(self.patch_num))

	def set_indices(self, index):
		self.tile_indices = index

	def get_indices(self):
		return self.tile_indices

	def set_softmax(self, softmax):
		self.softmax = softmax

	def get_softmax(self):
		return(self.softmax)

def indices_map(array, step_size = 100): 
	"""
	This is an index map the same size as the input image that used to identify which heatmap pixels overlap with each patch
	"""
	# define the number of values needed in your matrix if indices
	array_values = np.arange(array.shape[0]/step_size * array.shape[1]/step_size).astype(int)
	
	# initialize an empty matrix
	init_array = np.empty(shape=(array.shape[0], array.shape[1]))
	
	# generate starting points for for-loop
	starts = np.arange(array.shape[0], step_size) # this assumes a square matrix 
	counter = 0 # counter for indexing into array_values
	for y in starts:
		for x in starts:
			fill_array = np.zeros((step_size, step_size))
			fill_array.fill(array_values[counter]) # fill with index value
			init_array[y:y + step_size, x:x + step_size] = fill_array
			counter += 1 
	
	return init_array

def patch_dictionary(image, model, indices_map, step_size):
	"""
	Function that generates a set of patches as value
	key = patch_number
	value = patch_object
	"""
	assert image.shape[0] == indices_map.shape[0], "Image and indices_map are different dimensions"
	assert step_size <= IMAGE_SIZE, "Step size is too large. Must be less than 300."

	starts = np.arange(image.shape[0]-(IMAGE_SIZE - step_size), step = step_size) # subtract step size to include the last 300 X 300 patch
	patch_dict = {}
	counter = 0
	for y in starts:
		for x in starts:
			patch_dict[counter] = Patch(patch_num = counter, num_classes = TOTAL_CLASSES)

			# preprocess patch
			patch = image[y:y + IMAGE_SIZE, x:x + IMAGE_SIZE, :]
			patch = channel_preprocessing(patch)
			patch = cnn_preprocessing(patch)
			# forward pass
			pred = model.predict(patch[None,:,:,:])

			# update patch_object
			patch_dict[counter].set_softmax(pred)
			patch_dict[counter].set_indices(np.unique(indices_map[y:y + IMAGE_SIZE, x:x + IMAGE_SIZE]))
			counter += 1
			print(counter)

	return patch_dict


def srh_heatmap(patch_dict, image_size, step_size):
	
	heatmap_pixels = int(np.square(image_size/step_size))
	heatmap_dict = {}

	for pixel in range(heatmap_pixels):
		heatmap_dict[pixel] = np.zeros((TOTAL_CLASSES))

	for pixel in range(heatmap_pixels):
		print(pixel)
		for patch, patch_object in patch_dict.items():
			if pixel in patch_object.get_indices():
				heatmap_dict[pixel] += patch_object.get_softmax().reshape((TOTAL_CLASSES))

	flattened_image = np.zeros((heatmap_pixels, TOTAL_CLASSES))
	for pixel, unormed_dist in heatmap_dict.items():
		flattened_image[pixel, :] = unormed_dist/unormed_dist.sum()
		
	height_width = int(np.sqrt(heatmap_pixels))
	heatmap = flattened_image.reshape((height_width, height_width, TOTAL_CLASSES))
	return heatmap
		

if __name__ == '__main__':


	tiles_mosaic = indices_map(test_array)
	patch_dict = patch_dictionary(test_array, tiles_mosaic)
	print(patch_dict)

	for num, patch in patch_dict.items():
		print(patch.get_indices())
