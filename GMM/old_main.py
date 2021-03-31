import numpy as np
import pandas as pd
from sklearn import mixture
import pickle as p
import matplotlib.pyplot as plt
from PIL import Image
from datatools import DataTools
from plottools import PlotTools
from depthimagetools import DepthImageTools

def get_numerics (df):
	df = df[["width", "height", "dm_avg_dist"]]
	# df = df[["midx", "midy"]]
	return df

def train_gmm (values):
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000)
	m = m.fit(values)
	
	return m

def inference_model (values, model, width, height):
	n_components = model.weights_.shape[0]
	
	#probas = model.predict_proba(values)
	#print(probas)
	
	xvals = np.arange(0, width)
	yvals = np.arange(0, height)
	
	pixels = np.concatenate([
			np.repeat(xvals, height)[:,np.newaxis],
			np.tile(yvals, width)[:,np.newaxis]			
		], axis=1)
	
	
	probas = model.predict_proba(pixels)
	print(probas)
	print(np.min(probas))
	
def heat_map (values, width, height):
	values = np.round(values).astype(int)
	sel = (values[:,0] >= 0) & (values[:,0] < width)
	sel &= (values[:,1] >= 0) & (values[:,1] < height)
	values = values[sel]
	print(len(values))
	
	heat_map = np.full((height, width), 0, dtype=np.int)
	
	for v in values:
		heat_map[v[1], v[0]] += 1
		
	plt.imshow(heat_map)
	plt.show()

def default_test (df):
	splitted_dfs = DataTools.split_dataframes_into_groups(df)
	
	group_models = {}
	
	for group in splitted_dfs:
		group_df = splitted_dfs[group]
		group_df = get_numerics(group_df).values
		print("Count for group "+group+": "+str(len(group_df)))
		
		m = train_gmm(group_df)
		inference_model(group_df, m, 512, 512)
		group_models[group] = m
		
		heat_map(group_df, 512, 512)
		
	with open("models.p", "wb") as f:
		p.dump(group_models, f, protocol=-1)


def train_gmms_with_proba (df):
	group_df = DataTools.split_dataframes_into_groups(df)
	
	fussgaenger = group_df["pedestrian"]
	# fussgaenger_values = fussgaenger[["midy", "midx", "width", "height", "dm_avg_dist", "height_above_ground"]]
	fussgaenger_values = fussgaenger[["width", "height", "dm_avg_dist"]]
	
	m = train_gmm(fussgaenger_values)
	
	heat_map_model(m)

	
def test_sample (x, y, width, height, m, relevant_indices):
	p = np.array([x, y, width, height])
	
	print("Testing: {:s}".format(str(p)))

	test2 = m.score([p])
	
	print("T1:\n"+str(test1)+" "+str(test1.max())+" "+str(np.argmax(test1)))
	print("T2:\n"+str(test2))
	print("Relevant: "+str(np.any(relevant_indices == p_cls)))

def heat_map_model (m):
	'''
	raster = [
		[50,50],
		[70,40],
		[60,30],
		[60,40],
		[40,40],
		[80,50]
	]
	
	
	'''
	raster = [
		[5,10],
		[6,12],
		[7,14],
		[8,16],
		[4,8],
		[3,6]
	]
	
	
	# plt.suptitle("Fahrzeuge - GMM")
	plt.suptitle("Fußgänger - GMM")
	
	depth_image = np.array(Image.open('/home/rosvm/data/3D/img/16022.jpg'), dtype=np.uint8)[:,:,0]
	
	for i in range(len(raster)):
		plt.subplot(2, 3, i+1)
		
		scores, avg_score = calculate_scores_depth_image(m,raster[i][0],raster[i][1], depth_image)
		avg_score = np.round(avg_score, 3)
		
		title_base = str(raster[i][0])+"x"+str(raster[i][1])
		title_base += " (Score: "+str(avg_score)+")"
		
		plt.title(title_base)
		
		plt.imshow(scores)
		print(str(i+1)+"/"+str(len(raster)));
		
		
	plt.show()
	
def calculate_scores (m, box_width, box_height, distance):
	
	width, height = 512, 512
	scores = []
	
	for w in range(width):
		for h in range(height):
			scores.append(m.score([[h,w,box_width,box_height, distance]]))
	
	scores = np.reshape(np.array(scores), (height, width))
	avg_score = np.mean(scores)
	
	min_score = np.min(scores)
	max_score = np.max(scores)
	dif_score = max_score - min_score
	scores = (scores - min_score) / dif_score
	
	return scores, avg_score

def calculate_scores_depth_image (m, box_width, box_height, depth_image):
	
	width, height = 512, 512
	scores = []
	
	for h in range(height):
		for w in range(width):
			pixel_value = depth_image[h,w]
			
			distance = DepthImageTools.get_distance_for_pixel(pixel_value, 1)
			# hag = DepthImageTools.get_height_above_ground(DepthImageTools.get_alpha(h), distance)
			
			# scores.append(m.score([[h,w,box_width,box_height, distance, hag]]))
			scores.append(m.score([[box_width,box_height, distance]]))
	
	scores = np.reshape(np.array(scores), (height, width))
	avg_score = np.mean(scores)
	
	min_score = np.min(scores)
	max_score = np.max(scores)
	dif_score = max_score - min_score
	scores = (scores - min_score) / dif_score
	
	return scores, avg_score
	
def run_test_for_image ():
	# 16022
	df = DataTools.load_dataframe()
	group_df = DataTools.split_dataframes_into_groups(df)
	
	cars = group_df["cars"]
	cars_values = cars[["midy", "midx", "width", "height", "dm_avg_dist"]]
	
	m = train_gmm(cars_values)
	
	sample = cars[cars.index.get_level_values("filename") == "16022.jpg"]
	sample_values = get_numerics(sample)
	
	im = np.array(Image.open('/home/rosvm/data/img/16022.jpg'), dtype=np.uint8)
	
	
	display_model_in_image(im, sample, m)
	
def display_model_in_image (image, sample_values, model):
	sample_values_np = sample_values[["width", "height", "dm_avg_dist"]].values
	print(sample_values_np)
	im_count = len(sample_values_np)
	
	
	fig, axes = plt.subplots(im_count, 2)
	plt.suptitle("Image: 16022")
	
	for i in range(im_count):
		print(str(i+1)+"/"+str(im_count))
		detection = sample_values.iloc[i]
		
		ax_im = axes[i, 0]
		ax_model = axes[i, 1]
		
		plot_with_box(image, detection, ax_im)
		
		
		
		scores, avg_score = calculate_scores(model, sample_values_np[i, 0], sample_values_np[i, 1], sample_values_np[i, 2])
		avg_score = np.round(avg_score, 2)
		
		title_base = str(np.round(detection["width"], 2))+"x"+str(np.round(detection["height"], 2))+" -> "+str(np.round(detection["dm_avg_dist"],2))
		title_base += " (Score: "+str(avg_score)+")"
		
		ax_model.set_title(title_base)
		
		ax_model.imshow(scores)
		
		
	plt.show()
	
def plot_with_box (image, detection, axis):
	axis.imshow(image)
	patch = PlotTools.box_patch_for_detection_series(detection)
	axis.add_patch(patch)
	
def plot_distance_width_for_class (df, class_name):
	mindist = df["dm_avg_dist"].min()
	maxdist = df["dm_avg_dist"].max() * 1.25
	
	minwidth = df["width"].min()
	maxwidth = df["width"].max()
	
	minheight = df["height"].min()
	maxheight = df["height"].max()
	
	
	df = df[["dm_avg_dist", "width", "height"]].values
	
	m = mixture.GaussianMixture(covariance_type='diag', n_components=100, max_iter=10000)
	m = m.fit(df)	
	
	width, height = 100, 100
	
	scores = np.empty((height, width), dtype=np.float)
	widths = np.linspace(minwidth, maxwidth, width)
	heights = np.linspace(minheight, maxheight, width)
	distances = np.linspace(maxdist, mindist, height)
	
	xticks = []
	box_labels = []
	
	for i in range(width):
		if i % 100 == 0:
			xticks.append(i)
			box_labels.append(str(np.round(widths[i], 1))+":"+str(np.round(heights[i], 1)))
	
	yticks = []
	dist_labels = []
	
	for i in range(height, -1, -1):
		if i % 100 == 0:
			yticks.append(i)
			dist_labels.append(str(np.round(heights[i])))
	
	
	for i in range(height):
		c_dist = distances[i]
		
		for j in range(width):
			c_width = widths[j]
			c_height = heights[j]
			
			score = m.score([[c_dist, c_width, c_height]])
			scores[i, j] = score
			
	min_score = np.min(scores)
	max_score = np.max(scores)
	dif_score = max_score - min_score
	scores = (scores - min_score) / dif_score
	
	plt.imshow(scores)
	plt.xlabel("Box (({:f}:{:f}) - ({:f}:{:f}))".format(
			np.round(minwidth, 1), 
			np.round(minheight, 1),	
			np.round(maxwidth, 1),
			np.round(maxheight, 1)
		)
	  )
	plt.ylabel("Distanz ({:f} - {:f})".format(mindist, np.round(maxdist, 6)))
	
	plt.xticks(np.arange(0, width), box_labels)
	plt.yticks(np.arange(0, height), dist_labels)
	
	
	plt.title("GMM {:s}: Distanz - Box".format(class_name))
	
def train_distance_width_height (df):
	df = DataTools.split_dataframes_into_groups(df)
	
	
	plt.subplot(1, 2, 1)
	current = df["cars"]
	plot_distance_width_for_class(current, "Autos")
	
	plt.subplot(1, 2, 2)
	current = df["pedestrian"]
	plot_distance_width_for_class(current, "Fußgänger")
	
	plt.show()
	
	
	
	
if __name__ == '__main__':
	df = DataTools.load_dataframe()
	train_distance_width_height(df)
	print("Loaded data count: "+str(len(df)))
	
	# train_gmms_with_proba(df)
	
	
	# 50 / 100 Components Raster-Plots (19x19, 10x10, 5x5, 3x3, 2x2 und 1x1)
	
	'''
	Öffnungswinkel
	Distanzberechnung nicht richtig, KORRIGIEREN
	
	Ein GMM nur mit Width Height Entfernung
	X Y mal rauslassen --> starker Peak bei Fussgaengern
		
	'''
	
	