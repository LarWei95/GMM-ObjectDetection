from PIL import Image
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from depthimagetools import DepthImageTools
from datatools import DataTools

def get_depth_image_paths (directory):
	depth_images = [
			osp.join(path, file)
			for file in os.listdir(directory)
		]
	return depth_images

def load_image (path, remove_extremes=True):
	im = np.array(Image.open(path), dtype=np.uint8)[:,:,0]
	
	if remove_extremes:
		selector = ~(np.equal(im, 0) | np.equal(im, 255))
		im = im[selector]	
	else:
		im = im.flatten()
	
	return im

def simple_statistics (path):
	depth_image_paths = get_depth_image_paths(path)
	count = len(depth_image_paths)
	
	means = np.empty(count, dtype=np.float)
	stds = np.empty(count, dtype=np.float)
	mins = np.empty(count, dtype=np.float)
	maxs = np.empty(count, dtype=np.float)
	
	for i, path in enumerate(depth_image_paths):
		im = load_image(path)
		
		means[i] = np.mean(im)
		stds[i] = np.std(im)
		mins[i] = np.min(im)
		maxs[i] = np.max(im)
		
	means = np.mean(means)
	stds = np.mean(stds)
	mins = np.min(mins)
	maxs = np.max(maxs)
	
	# 66.2765903478926 10.309150914167866 1.0 88.0
	print(means, stds, mins, maxs)

def distribution_statistics (path):
	depth_image_paths = get_depth_image_paths(path)
	count = len(depth_image_paths)
	
	value_counts = np.full(256, 0, dtype=np.int)
	
	for path in depth_image_paths:
		im = 255 - load_image(path, False)
		im, unique_counts = np.unique(im, return_counts=True)
		value_counts[im] += unique_counts
		
	plt.bar(np.arange(0, 256), value_counts)
	plt.yscale("log")
	plt.title("Verteilung von rohen Distanzen (invertiert)\n0: Am nähesten | 255: Am weitesten")
	plt.show()
	
if __name__ == '__main__':
	path = "/home/rosvm/data/3D/img/"
	distribution_statistics(path)
		