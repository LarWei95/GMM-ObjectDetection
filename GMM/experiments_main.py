from PIL import Image
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from depthimagetools import DepthImageTools
from datatools import DataTools, DataConverter
from sklearn import mixture
import pandas as pd

FIGURE_SIZE = (7, 6)
FONT_SIZE = 11.5

def plot_heightmap():
	depth_image = np.array(Image.open('/home/rosvm/data/3D/img/24223.jpg'), dtype=np.uint8)[:,:,0]
	
	distances = DepthImageTools.get_distance_for_pixel(depth_image)	
	
	plt.imshow(distances)
	plt.show()
	
	ypos = np.repeat(np.arange(0, 512)[:,np.newaxis], 512, axis=1)
	
	alphas = DepthImageTools.get_alpha(ypos)	
	
	plt.imshow(alphas)
	plt.show()
	
	heights = DepthImageTools.get_height_above_ground(alphas, distances)
	
	plt.imshow(heights)
	plt.show()

def default():
	df = DataTools.load_dataframe()
	group_df = DataTools.split_dataframes_into_groups(df)
	
	print(len(group_df["pedestrian"]), len(group_df["cars"]))
	
	group_df = group_df["pedestrian"]
	
	group_df = group_df["height_above_ground"]
	
	print(np.min(group_df), np.max(group_df))
	
	plot_heightmap()
	
def get_binned_distances_0_1 (distances, bars):
	bins = np.linspace(0.0, 1.0, num=bars + 1)
	counts = np.full(bars, 0, dtype=np.int)
	
	for i in range(0, bars):
		bin = bins[i]
		nbin = bins[i+1]
		counts[i] = np.sum((distances >= bin) & (distances < nbin))
	
	return bins, counts

def get_binned (distances, bars):
	bins = np.linspace(np.min(distances), np.max(distances), num=bars + 1)
	counts = np.full(bars, 0, dtype=np.int)
	
	for i in range(0, bars):
		bin = bins[i]
		nbin = bins[i+1]
		counts[i] = np.sum((distances >= bin) & (distances < nbin))
	
	return bins, counts
	
def plot_distance_histogram (bins, counts, logspace=False):
	plt.bar(bins[:-1], counts, 1 / (len(bins)))
	
	if logspace:
		title = "Verteilung Dist. (log.)"
		plt.yscale("log")
	else:
		title = "Verteilung Dist. (lin.)"
		
	plt.grid(True)
	plt.title(title)
	
def plot_height_histogram (bins, counts, logspace=False):
	plt.bar(bins[:-1], counts, 1 / (len(bins)))
	
	if logspace:
		title = "Verteilung Hoehe (log.)"
		plt.yscale("log")
	else:
		title = "Verteilung Hoehe (lin.)"
		
	plt.grid(True)
	plt.title(title)
	
	
def plot_configuration (depth_image, distances, suptitle, filename):
	hag = DepthImageTools.get_height_above_ground(
		DepthImageTools.get_alpha(
			np.repeat(np.arange(0, 512)[:,np.newaxis], 512, axis=1)
		),
		distances
	)
	
	plt.subplot(2, 4, 1)
	plt.title("Tiefenbild")
	plt.imshow(depth_image)
	
	plt.subplot(2, 4, 2)
	plt.title("Distanzen")
	plt.imshow(distances)
	
	plt.subplot(2, 4, 3)
	plt.title("Hoehen ueber dem Boden")
	plt.imshow(hag)
	
	# Distanzhistos
	bins, counts = get_binned_distances_0_1(distances, 50)
	
	plt.subplot(2, 4, 4)
	plot_distance_histogram (bins, counts, False)
	
	plt.subplot(2, 4, 5)
	plot_distance_histogram (bins, counts, True)
	
	# Hoehenhistos
	bins, counts = get_binned(hag, 50)
	
	plt.subplot(2, 4, 6)
	plot_height_histogram (bins, counts, False)
	
	plt.subplot(2, 4, 7)
	plot_height_histogram (bins, counts, True)
	
	plt.suptitle(suptitle)
	
	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename, dpi=200)
	plt.close("all")
	
def plot_xy_distance_scores (depth_image, class_df, suptitle, filename_base):
	class_df = class_df[["midy", "midx", "dm_avg_dist"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	scores = np.empty(depth_image.shape, dtype=np.float)
	heights = np.arange(0, depth_image.shape[0])
	widths = np.arange(0, depth_image.shape[1])
	
	for h in heights:
		for w in widths:
			distance = depth_image[h, w]
			score = m.score([[h, w, distance]])
			scores[h, w] = score
			
	plt.title("X-Y-Distanz: "+suptitle)
	plt.imshow(scores)
	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (XYDist).png", dpi=200)
	plt.close("all")
	
def plot_xy_height_scores (depth_image, class_df, suptitle, filename_base):
	class_df = class_df[["midy", "midx", "height_above_ground"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	height_image = DepthImageTools.get_height_above_ground(
			DepthImageTools.get_alpha(np.repeat(np.arange(0, 512)[:,np.newaxis], 512, axis=1)),
			depth_image
		)
	
	scores = np.empty(height_image.shape, dtype=np.float)
	heights = np.arange(0, height_image.shape[0])
	widths = np.arange(0, height_image.shape[1])
	
	for h in heights:
		for w in widths:
			hag = height_image[h, w]
			score = m.score([[h, w, hag]])
			scores[h, w] = score
			
	plt.title("X-Y-Hoehe: "+suptitle)
	plt.imshow(scores)
	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (XYHeight).png", dpi=200)
	plt.close("all")
	
def get_box_distance_ticks (width, height, widthsplit, heightsplit, widths, heights, distances):
	xticks = []
	box_labels = []
	
	for i in range(width):
		if i % widthsplit == 0:
			xticks.append(i)
			box_labels.append(str(np.round(widths[i], 1))+":"+str(np.round(heights[i], 1)))
	
	yticks = []
	dist_labels = []
	
	for i in range(height):
		if i % heightsplit == 0:
			yticks.append(i)
			dist_labels.append(str(np.round(distances[i], 3)))
			
	return xticks, box_labels, yticks, dist_labels
	
def plot_width_height_distance_scores (class_df, suptitle, filename_base):
	mindist = class_df["dm_avg_dist"].min()
	maxdist = class_df["dm_avg_dist"].max()
	
	minwidth = class_df["width"].min()
	maxwidth = class_df["width"].max()
	
	minheight = class_df["height"].min()
	maxheight = class_df["height"].max()
	
	class_df = class_df[["dm_avg_dist", "width", "height"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	width, height = 250, 250
	widthsplit = np.round(width / 5).astype(int)
	heightsplit = np.round(height / 5).astype(int)
	
	widths = np.linspace(minwidth, maxwidth, width)
	heights = np.linspace(minheight, maxheight, width)
	distances = np.linspace(maxdist, mindist, height)
	
	xticks, box_labels, yticks, dist_labels = get_box_distance_ticks (width, height, widthsplit, 
																	  heightsplit, widths, heights, distances)
	
	scores = np.empty((height, width), dtype=np.float)

	for i in range(height):
		c_dist = distances[i]
		
		for j in range(width):
			c_width = widths[j]
			c_height = heights[j]
			
			score = m.score([[c_dist, c_width, c_height]])
			scores[i, j] = score
			
	avg_score = np.mean(scores)
			
	min_score = np.min(scores)
	max_score = np.max(scores)
	dif_score = max_score - min_score
	scores = (scores - min_score) / dif_score
	
	plt.imshow(scores)
	plt.xlabel("Box (({:.1f}:{:.1f}) - ({:.1f}:{:.1f}))".format(
			np.round(minwidth, 1), 
			np.round(minheight, 1),	
			np.round(maxwidth, 1),
			np.round(maxheight, 1)
		)
	  )
	plt.ylabel("Distanz ({:.1f} - {:.1f})".format(mindist, np.round(maxdist, 1)))
	
	plt.xticks(xticks, box_labels)
	plt.yticks(yticks, dist_labels)
	
	plt.suptitle("GMM {:s}: Distanz - Box".format(suptitle))
	plt.title("Durchschn. Score: {:.5f}".format(avg_score))
	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (WHDist).png", dpi=200)
	plt.close("all")
	
def plot_width_height_distance_hag_scores (class_df, suptitle, filename_base):
	mindist = class_df["dm_avg_dist"].min()
	maxdist = class_df["dm_avg_dist"].max()
	
	minwidth = class_df["width"].min()
	maxwidth = class_df["width"].max()
	
	minheight = class_df["height"].min()
	maxheight = class_df["height"].max()
	
	minhag = class_df["height_above_ground"].min()
	maxhag = class_df["height_above_ground"].max()
	
	class_df = class_df[["height_above_ground", "dm_avg_dist", "width", "height"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	width, height = 100, 100
	widthsplit = np.round(width / 3).astype(int)
	heightsplit = np.round(height / 3).astype(int)
	
	widths = np.linspace(minwidth, maxwidth, width)
	heights = np.linspace(minheight, maxheight, width)
	distances = np.linspace(maxdist, mindist, height)
	
	xticks, box_labels, yticks, dist_labels = get_box_distance_ticks (width, height, widthsplit, 
																	  heightsplit, widths, heights, distances)
	
	hags = np.linspace(minhag, maxhag, 6)
	
	for i, c_hag in enumerate(hags):
		plt.subplot(2, 3, 1+i)
		scores = np.empty((height, width), dtype=np.float)

		for i in range(height):
			c_dist = distances[i]

			for j in range(width):
				c_width = widths[j]
				c_height = heights[j]

				score = m.score([[c_hag, c_dist, c_width, c_height]])
				scores[i, j] = score
		
		avg_score = np.mean(scores)
		
		min_score = np.min(scores)
		max_score = np.max(scores)
		dif_score = max_score - min_score
		scores = (scores - min_score) / dif_score
		
		plt.title("HAG: "+str(np.round(c_hag, 3))+" | Score: {:.5f}".format(avg_score))
		plt.imshow(scores)
		plt.xlabel("Box (({:.1f}:{:.1f}) - ({:.1f}:{:.1f}))".format(
				np.round(minwidth, 1), 
				np.round(minheight, 1),	
				np.round(maxwidth, 1),
				np.round(maxheight, 1)
			)
		  )
		plt.ylabel("Distanz ({:.1f} - {:.1f})".format(mindist, np.round(maxdist, 1)))

		plt.xticks(xticks, box_labels)
		plt.yticks(yticks, dist_labels)
	
	plt.suptitle("GMM {:s}: Distanz - Box".format(suptitle))
	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (WHDistHag).png", dpi=200)
	plt.close("all")
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
'''
RELEVANT STUFF HERE
RELEVANT STUFF HERE
RELEVANT STUFF HERE
RELEVANT STUFF HERE
RELEVANT STUFF HERE
RELEVANT STUFF HERE
'''
def set_ticks_font_size ():
	plt.xticks(fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.tight_layout(pad=1.00)
	
def plot_width_height_distance_scores_example (depth_image, class_df, raster, suptitle, filename_base):
	class_df = class_df[["width", "height", "dm_avg_dist"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	flattened_depth_image = np.reshape(depth_image, depth_image.shape[0] * depth_image.shape[1])
	
	width, height = 512, 512
	
	plt.suptitle("BoxPlot GMM (WHD): "+suptitle)
	
	for i in range(6):
		c_raster = raster[i]
		c_boxwidth = c_raster[0]
		c_boxheight = c_raster[1]
		
		plt.subplot(2, 3, i+1)
		
		x1 = np.array([[c_boxwidth, c_boxheight]])
		x2 = flattened_depth_image
		
		c = np.concatenate([
			np.repeat(x1, len(x2), axis=0),
			np.tile(x2, len(x1))[:,np.newaxis]
		], axis=1)
		
		scores = m.score_samples(c)
		
		scores = np.reshape(scores, (height, width))
		
		avg_score = np.mean(scores)
		
		set_ticks_font_size ()
		plt.title(str(c_boxwidth)+"x"+str(c_boxheight)+"\nScore: {:.3f}".format(avg_score))
		plt.imshow(scores)
		
	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (BoxPlot WHD).png", dpi=200)
	plt.close("all")
	
def plot_width_height_hag_scores_example (depth_image, class_df, raster, suptitle, filename_base):
	class_df = class_df[["width", "height", "height_above_ground"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	# flattened_depth_image = np.reshape(depth_image, depth_image.shape[0] * depth_image.shape[1])
	
	width, height = 512, 512
	
	# (Height, Width)
	hags = DepthImageTools.get_height_above_ground(
			DepthImageTools.get_alpha(np.repeat(np.arange(0, height)[:,np.newaxis], width, axis=1)),
			depth_image
		)
	hags = np.reshape(hags, height * width)
	
	plt.suptitle("BoxPlot GMM (WHHag): "+suptitle)
	
	for i in range(6):
		c_raster = raster[i]
		c_boxwidth = c_raster[0]
		c_boxheight = c_raster[1]
		
		plt.subplot(2, 3, i+1)
		
		x1 = np.array([[c_boxwidth, c_boxheight]])
		
		c = np.concatenate([
			np.repeat(x1, len(hags), axis=0),
			hags[:,np.newaxis]
		], axis=1)
		
		scores = m.score_samples(c)
		
		scores = np.reshape(scores, (height, width))
		
		avg_score = np.mean(scores)
		
		set_ticks_font_size ()
		plt.title(str(c_boxwidth)+"x"+str(c_boxheight)+"\nScore: {:.3f}".format(avg_score))
		plt.imshow(scores)
		
	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (BoxPlot WHHag).png", dpi=200)
	plt.close("all")
	
def plot_width_height_xy_scores_example (depth_image, class_df, raster, suptitle, filename_base):
	class_df = class_df[["width", "height", "midx", "midy"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	width, height = 512, 512
	
	xys = np.concatenate([
			np.tile(np.arange(0, width), height)[:,np.newaxis],
			np.repeat(np.arange(0, height), width)[:,np.newaxis]
		], axis=1)
	
	plt.suptitle("BoxPlot GMM (WHXY): "+suptitle)
	
	for i in range(6):
		c_raster = raster[i]
		c_boxwidth = c_raster[0]
		c_boxheight = c_raster[1]
		
		plt.subplot(2, 3, i+1)
		
		x1 = np.array([[c_boxwidth, c_boxheight]])
		
		c = np.concatenate([
			np.repeat(x1, len(xys), axis=0),
			xys
		], axis=1)
		
		scores = m.score_samples(c)
		
		scores = np.reshape(scores, (height, width))
		
		avg_score = np.mean(scores)
		
		set_ticks_font_size ()
		plt.title(str(c_boxwidth)+"x"+str(c_boxheight)+"\nScore: {:.3f}".format(avg_score))
		plt.imshow(scores)
		
	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (BoxPlot WHXY).png", dpi=200)
	plt.close("all")
		
def plot_width_height_xy_distance_hag_dynamic_scores_example (depth_image, class_df, raster, suptitle, filename_base):
	class_df = class_df[["width", "height", "midx", "midy", "dm_avg_dist", "height_above_ground"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	hags = DepthImageTools.get_height_above_ground(
			DepthImageTools.get_alpha(np.repeat(np.arange(0, 512)[:,np.newaxis], 512, axis=1)),
			depth_image
		)
	width, height = 512, 512
	
	xys = np.concatenate([
			np.tile(np.arange(0, width), height)[:,np.newaxis],
			np.repeat(np.arange(0, height), width)[:,np.newaxis]
		], axis=1)
	
	combined = np.concatenate([
		xys,
		np.reshape(depth_image, depth_image.shape[0] * depth_image.shape[1])[:,np.newaxis],
		np.reshape(hags, depth_image.shape[0] * depth_image.shape[1])[:,np.newaxis]
	], axis=1)
	
	
	
	plt.suptitle("BoxPlot GMM (WHXYDHag): "+suptitle)

	for i in range(6):
		c_raster = raster[i]
		c_boxwidth = c_raster[0]
		c_boxheight = c_raster[1]

		plt.subplot(2, 3, i+1)

		x1 = np.array([[c_boxwidth, c_boxheight]])
		x2 = combined

		c = np.concatenate([
			np.repeat(x1, len(x2), axis=0),
			x2
		], axis=1)

		scores = m.score_samples(c)
		scores = np.reshape(scores, (height, width))

		avg_score = np.mean(scores)
		
		set_ticks_font_size ()
		plt.title(str(c_boxwidth)+"x"+str(c_boxheight)+"\nScore: {:.3f}".format(avg_score))
		plt.imshow(scores)

	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (BoxPlot WHXYDHag).png", dpi=200)
	plt.close("all")
	
def plot_width_height_y_distance_hag_dynamic_scores_example (depth_image, class_df, raster, suptitle, filename_base):
	class_df = class_df[["width", "height", "midy", "dm_avg_dist", "height_above_ground"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	hags = DepthImageTools.get_height_above_ground(
			DepthImageTools.get_alpha(np.repeat(np.arange(0, 512)[:,np.newaxis], 512, axis=1)),
			depth_image
		)
	width, height = 512, 512
	
	ys = np.repeat(np.arange(0, height), width)[:,np.newaxis]
	
	combined = np.concatenate([
		ys,
		np.reshape(depth_image, depth_image.shape[0] * depth_image.shape[1])[:,np.newaxis],
		np.reshape(hags, depth_image.shape[0] * depth_image.shape[1])[:,np.newaxis]
	], axis=1)
	
	plt.suptitle("BoxPlot GMM (WHYDHag): "+suptitle)

	for i in range(6):
		c_raster = raster[i]
		c_boxwidth = c_raster[0]
		c_boxheight = c_raster[1]

		plt.subplot(2, 3, i+1)

		x1 = np.array([[c_boxwidth, c_boxheight]])
		x2 = combined

		c = np.concatenate([
			np.repeat(x1, len(x2), axis=0),
			x2
		], axis=1)

		scores = m.score_samples(c)
		scores = np.reshape(scores, (height, width))

		avg_score = np.mean(scores)
		
		set_ticks_font_size ()
		plt.title(str(c_boxwidth)+"x"+str(c_boxheight)+"\nScore: {:.3f}".format(avg_score))
		plt.imshow(scores)

	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (BoxPlot WHYDHag).png", dpi=200)
	plt.close("all")
	
def plot_width_height_xy_distance_dynamic_scores_example (depth_image, class_df, raster, suptitle, filename_base):
	class_df = class_df[["width", "height", "midx", "midy", "dm_avg_dist"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	width, height = 512, 512
	
	xys = np.concatenate([
			np.tile(np.arange(0, width), height)[:,np.newaxis],
			np.repeat(np.arange(0, height), width)[:,np.newaxis]
		], axis=1)
	
	combined = np.concatenate([
		xys,
		np.reshape(depth_image, depth_image.shape[0] * depth_image.shape[1])[:,np.newaxis]
	], axis=1)
	
	plt.suptitle("BoxPlot GMM (WHXYD): "+suptitle)

	for i in range(6):
		c_raster = raster[i]
		c_boxwidth = c_raster[0]
		c_boxheight = c_raster[1]

		plt.subplot(2, 3, i+1)

		x1 = np.array([[c_boxwidth, c_boxheight]])
		x2 = combined

		c = np.concatenate([
			np.repeat(x1, len(x2), axis=0),
			x2
		], axis=1)

		scores = m.score_samples(c)
		scores = np.reshape(scores, (height, width))

		avg_score = np.mean(scores)

		set_ticks_font_size ()
		plt.title(str(c_boxwidth)+"x"+str(c_boxheight)+"\nScore: {:.3f}".format(avg_score))
		plt.imshow(scores)

	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (BoxPlot WHXYD).png", dpi=200)
	plt.close("all")
	
def plot_width_height_y_distance_dynamic_scores_example (depth_image, class_df, raster, suptitle, filename_base):
	class_df = class_df[["width", "height", "midy", "dm_avg_dist"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	width, height = 512, 512
	
	ys = np.repeat(np.arange(0, height), width)[:,np.newaxis]
	
	combined = np.concatenate([
		ys,
		np.reshape(depth_image, depth_image.shape[0] * depth_image.shape[1])[:,np.newaxis]
	], axis=1)
	
	plt.suptitle("BoxPlot GMM (WHYD): "+suptitle)

	for i in range(6):
		c_raster = raster[i]
		c_boxwidth = c_raster[0]
		c_boxheight = c_raster[1]

		plt.subplot(2, 3, i+1)

		x1 = np.array([[c_boxwidth, c_boxheight]])
		x2 = combined

		c = np.concatenate([
			np.repeat(x1, len(x2), axis=0),
			x2
		], axis=1)

		scores = m.score_samples(c)
		scores = np.reshape(scores, (height, width))

		avg_score = np.mean(scores)
		
		set_ticks_font_size ()
		plt.title(str(c_boxwidth)+"x"+str(c_boxheight)+"\nScore: {:.3f}".format(avg_score))
		plt.imshow(scores)

	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (BoxPlot WHYD).png", dpi=200)
	plt.close("all")
	
def plot_width_height_xy_hag_dynamic_scores_example (depth_image, class_df, raster, suptitle, filename_base):
	class_df = class_df[["width", "height", "midx", "midy", "height_above_ground"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	hags = DepthImageTools.get_height_above_ground(
			DepthImageTools.get_alpha(np.repeat(np.arange(0, 512)[:,np.newaxis], 512, axis=1)),
			depth_image
		)
	width, height = 512, 512
	
	xys = np.concatenate([
			np.tile(np.arange(0, width), height)[:,np.newaxis],
			np.repeat(np.arange(0, height), width)[:,np.newaxis]
		], axis=1)
	
	combined = np.concatenate([
		xys,
		np.reshape(hags, depth_image.shape[0] * depth_image.shape[1])[:,np.newaxis]
	], axis=1)
	
	plt.suptitle("BoxPlot GMM (WHXYHag): "+suptitle)

	for i in range(6):
		c_raster = raster[i]
		c_boxwidth = c_raster[0]
		c_boxheight = c_raster[1]

		plt.subplot(2, 3, i+1)

		x1 = np.array([[c_boxwidth, c_boxheight]])
		x2 = combined

		c = np.concatenate([
			np.repeat(x1, len(x2), axis=0),
			x2
		], axis=1)

		scores = m.score_samples(c)
		scores = np.reshape(scores, (height, width))

		avg_score = np.mean(scores)

		set_ticks_font_size ()
		plt.title(str(c_boxwidth)+"x"+str(c_boxheight)+"\nScore: {:.3f}".format(avg_score))
		plt.imshow(scores)

	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (BoxPlot WHXYHag).png", dpi=200)
	plt.close("all")
	
def plot_width_height_y_hag_dynamic_scores_example (depth_image, class_df, raster, suptitle, filename_base):
	class_df = class_df[["width", "height", "midy", "height_above_ground"]].values
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000).fit(class_df)
	
	hags = DepthImageTools.get_height_above_ground(
			DepthImageTools.get_alpha(np.repeat(np.arange(0, 512)[:,np.newaxis], 512, axis=1)),
			depth_image
		)
	width, height = 512, 512
	
	ys = np.repeat(np.arange(0, height), width)[:,np.newaxis]
	
	combined = np.concatenate([
		ys,
		np.reshape(hags, depth_image.shape[0] * depth_image.shape[1])[:,np.newaxis]
	], axis=1)
	
	plt.suptitle("BoxPlot GMM (WHYHag): "+suptitle)

	for i in range(6):
		c_raster = raster[i]
		c_boxwidth = c_raster[0]
		c_boxheight = c_raster[1]

		plt.subplot(2, 3, i+1)

		x1 = np.array([[c_boxwidth, c_boxheight]])
		x2 = combined

		c = np.concatenate([
			np.repeat(x1, len(x2), axis=0),
			x2
		], axis=1)

		scores = m.score_samples(c)
		scores = np.reshape(scores, (height, width))

		avg_score = np.mean(scores)
		
		set_ticks_font_size ()
		plt.title(str(c_boxwidth)+"x"+str(c_boxheight)+"\nScore: {:.3f}".format(avg_score))
		plt.imshow(scores)

	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(filename_base+" (BoxPlot WHYHag).png", dpi=200)
	plt.close("all")
	
def plot_scores_for_category (depth_image, split_df, category, raster, suptitle, filename_base):
	suptitle = suptitle+" ("+category+")"
	filename_base = filename_base+" "+category
	
	# plot_xy_distance_scores(depth_image, split_df[category], suptitle, filename_base)
	# plot_xy_height_scores(depth_image, split_df[category], suptitle, filename_base)
	# plot_width_height_distance_scores(split_df[category], suptitle, filename_base)
	# plot_width_height_distance_hag_scores(split_df[category], suptitle, filename_base)
	
	
	plot_width_height_distance_scores_example(depth_image, split_df[category], raster, suptitle, filename_base)
	plot_width_height_xy_scores_example(depth_image, split_df[category], raster, suptitle, filename_base)
	plot_width_height_hag_scores_example(depth_image, split_df[category], raster, suptitle, filename_base)
	plot_width_height_xy_distance_hag_dynamic_scores_example(depth_image, split_df[category], raster, suptitle, filename_base)
	plot_width_height_y_distance_hag_dynamic_scores_example(depth_image, split_df[category], raster, suptitle, filename_base)
	plot_width_height_xy_distance_dynamic_scores_example(depth_image, split_df[category], raster, suptitle, filename_base)
	plot_width_height_y_distance_dynamic_scores_example(depth_image, split_df[category], raster, suptitle, filename_base)
	plot_width_height_xy_hag_dynamic_scores_example(depth_image, split_df[category], raster, suptitle, filename_base)
	plot_width_height_y_hag_dynamic_scores_example(depth_image, split_df[category], raster, suptitle, filename_base)
	
def output_results (depth_image, suptitle, filename_base, distance_op, extra_parameters):
	'''
	Erstellt die Ergebnisse fuer das gegebene Tiefenbild mit der gegebenen
	Distanz-Funktion und den gegebenen Extra-Parametern fuer dieselbe
	Distanzfunktion.
	'''
	
	distances = distance_op(depth_image, *extra_parameters)
	# Plottet das Beispielbild mit Tiefenbild, Hoehenbild und Histogrammen
	plot_configuration(depth_image, distances, suptitle,
					  filename_base+".png")
	
	df = DataConverter.get_gmm_dataframe(distance_op, extra_parameters)
	split_df = DataTools.split_dataframes_into_groups(df)
	
	cars_raster = [
		[22.570, 16.891], # Most likely (486)
		[121.167, 126.756], # 11th most likely (260)
		[50.177, 35.201], # 34th most likely (113)
		[73.840, 29.098], # 200th most likely (16)
		[58.065, 16.891], # 499th most likely (5) 
		[200.044, 93.186], # 773th most likely (1)
	]
	ped_raster = [
		[6.069, 18.390], # Most likely (37)
		[13.653, 38.910], # 5th most likely (22)
		[7.080, 21.321], # 19th most likely (15)
		[9.103, 25.230], # 38th most likely (7)
		[16.687, 39.887], # 142th most likely (2)
		[23.766, 54.544] # 270th most likely (1)
	]
	
	plot_scores_for_category(distances, split_df, "cars", cars_raster, suptitle, filename_base)
	plot_scores_for_category(distances, split_df, "pedestrian", ped_raster, suptitle, filename_base)
	
	
	
def plot_distance_calculation_statistics ():
	outputdir = "./Analysis"
	os.makedirs(outputdir, exist_ok=True)
	depth_image = np.array(Image.open('/home/rosvm/data/3D/img/24223.jpg'), dtype=np.uint8)[:,:,0]
	# Ohne Bias
	
	mod_depth_image = depth_image
	
	'''
	output_results(mod_depth_image,
				   "255-0 -> 1-0",
				   osp.join(outputdir, "255-0 - 1-0"),
				   DepthImageTools.normalize_value_switch,
				   tuple([False])
			)
	
	output_results(mod_depth_image,
				   "255-0 -> 0-1",
				   osp.join(outputdir, "255-0 - 0-1"),
				   DepthImageTools.normalize_value_switch,
				   tuple([True])
			)
	
	output_results(mod_depth_image,
				   "1-0 -> 1 / x",
				   osp.join(outputdir, "1-0 - 1 div x"),
				   DepthImageTools.get_distance_inverted,
				   tuple([False])
			)
	'''
	output_results(mod_depth_image,
				   "0-1 -> 1 / x",
				   osp.join(outputdir, "0-1 - 1 div x"),
				   DepthImageTools.get_distance_inverted,
				   tuple([True])
			)
	'''
	# Mit Bias
	mod_depth_image = DepthImageTools.remove_empty_space_from_raw(depth_image)
	
	output_results(mod_depth_image,
				   "Biased 255-0 -> 1-0",
				   osp.join(outputdir, "Biased 255-0 - 1-0"),
				   DepthImageTools.normalize_value_switch,
				   tuple([False])
			)
	
	output_results(mod_depth_image,
				   "Biased 255-0 -> 0-1",
				   osp.join(outputdir, "Biased 255-0 - 0-1"),
				   DepthImageTools.normalize_value_switch,
				   tuple([True])
			)
	
	output_results(mod_depth_image,
				   "Biased 1-0 -> 1 / x",
				   osp.join(outputdir, "Biased 1-0 - 1 div x"),
				   DepthImageTools.get_distance_inverted,
				   tuple([False])
			)
	
	output_results(mod_depth_image,
				   "Biased 0-1 -> 1 / x",
				   osp.join(outputdir, "Biased 0-1 - 1 div x"),
				   DepthImageTools.get_distance_inverted,
				   tuple([True])
			)
	'''
	
def boxes_scatterplot ():
	df = DataConverter.get_gmm_dataframe(DepthImageTools.get_distance_inverted, tuple([True]))
	split_df = DataTools.split_dataframes_into_groups(df)
	
	cars_boxes = split_df["cars"][["width", "height"]].values
	pedestrian_boxes = split_df["pedestrian"][["width", "height"]].values
	
	plt.scatter(cars_boxes[:,0], cars_boxes[:,1], s=1, label="Cars")
	plt.scatter(pedestrian_boxes[:,0], pedestrian_boxes[:,1], s=1, label="Pedestrians")
	plt.grid(True, which='both')
	plt.legend()
	
	plt.xlabel("Width", fontsize=FONT_SIZE)
	plt.ylabel("Height", fontsize=FONT_SIZE)
	
	plt.title("Box sizes of classes", fontsize=FONT_SIZE)
	
	
	set_ticks_font_size ()
	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig("box_sizes.png", dpi=200)
	plt.close("all")
	
def statistics_for_width_height_boxes (boxes, count, class_name, save_name):
	minwidth = np.min(boxes[:,0])
	maxwidth = np.max(boxes[:,0])
	minheight = np.min(boxes[:,1])
	maxheight = np.max(boxes[:,1])
	
	widthrange = np.linspace(minwidth, maxwidth, count+1)
	heightrange = np.linspace(minheight, maxheight, count+1)
	
	countimage = np.full((count, count), 0, dtype=np.int)
	
	indx = []
	counts = []
	
	for i in range(count):
		minh = heightrange[i]
		maxh = heightrange[i+1]
		
		medh = np.mean([minh, maxh])
		
		for j in range(count):
			minw = widthrange[j]
			maxw = widthrange[j + 1]
			medw = np.mean([minw, maxw])
			
			c_count = (boxes[:,0] >= minw) & (boxes[:,0] < maxw)
			c_count &= (boxes[:,1] >= minh) & (boxes[:,1] < maxh)
			c_count = np.sum(c_count)
			
			countimage[i, j] = c_count
			
			indx.append("{:.3f}:{:.3f}".format(medw, medh))
			counts.append(c_count)
			
	count_df = pd.DataFrame({"Counts" : counts}, index=indx)
	count_df.sort_values(by="Counts", ascending=False, inplace=True)
	count_df.to_csv(save_name+".csv")
	
	plt.imshow(countimage)
	plt.yticks(np.arange(0, count), np.round(heightrange[:-1]).astype(int))
	plt.xticks(np.arange(0, count), np.round(widthrange[:-1]).astype(int))
	plt.gca().invert_yaxis()
	plt.suptitle("Heatmap for box distributions")
	plt.title("Class: "+class_name)
	plt.gcf().set_size_inches(*FIGURE_SIZE)
	plt.savefig(save_name)
	
	
	
def boxes_statistics ():
	df = DataConverter.get_gmm_dataframe(DepthImageTools.get_distance_inverted, tuple([True]))
	split_df = DataTools.split_dataframes_into_groups(df)
	
	cars_boxes = split_df["cars"][["width", "height"]].values
	pedestrian_boxes = split_df["pedestrian"][["width", "height"]].values
	
	statistics_for_width_height_boxes (cars_boxes, 75, "Cars", "cars_distribution.png")
	statistics_for_width_height_boxes (pedestrian_boxes, 75, "Pedestrians", "pedestrians_distribution.png")
	
if __name__ == '__main__':
	# plot_distance_calculation_statistics()
	boxes_scatterplot()
	# boxes_statistics()
	
	