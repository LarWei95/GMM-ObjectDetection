import pandas as pd
import numpy as np
import os.path as osp
import os
from PIL import Image
from depthimagetools import DepthImageTools

class DataTools ():
	CAR_CLASSES = np.array([
			"BoxTruck",
			"Hatchback",
			"Jeep",
			"SUV",
			"Sedan",
			"SchoolBus"
		])
	PEDESTRIAN_CLASSES = np.array([
			"Pedestrian"
		])

	CARS_GROUP = "cars"
	PEDESTRIAN_GROUP = "pedestrian"
	UNUSED_GROUP = "unused"

	GROUPS = {
			CARS_GROUP : CAR_CLASSES,
			PEDESTRIAN_GROUP : PEDESTRIAN_CLASSES
		}
	
	
	@classmethod
	def load_dataframe (cls, path="gmm_data.csv"):
		df = pd.read_csv(path, index_col=["filename", "id"])
		return df
	
	@classmethod
	def split_dataframes_into_groups (cls, df):
		splitted = {}

		classes = df["class"].values

		used = np.full(len(df), False, dtype=np.bool)

		for group in cls.GROUPS:
			group_list = cls.GROUPS[group]

			group_selector = np.isin(classes, group_list)
			group_df = df[group_selector]

			splitted[group] = group_df
			used |= group_selector

		if not np.all(used):
			group_selector = ~used
			group_df = df[group_selector]
			print("Unused classes: "+str(np.unique(group_df["class"])))

			splitted[cls.UNUSED_GROUP] = group_df

		return splitted
	
class DataConverter ():
	IMAGE_WIDTH = 512
	IMAGE_HEIGHT = 512
	
	@classmethod
	def filename_multiindex_to_basename (cls, indx):
		new_index = []

		for i in range(len(indx)):
			bn = osp.basename(indx[i][0])
			indx_id = indx[i][1]

			new_index.append((bn, indx_id))

		new_index = pd.MultiIndex.from_tuples(new_index, names=["filename", "id"])

		return new_index
	
	@classmethod
	def create2D_points (cls, df2d):
		# xmin,ymin,xmax,ymax
		xmins = df2d["xmin"].values
		ymins = df2d["ymin"].values
		xmaxs = df2d["xmax"].values
		ymaxs = df2d["ymax"].values

		xmedians = xmins + (xmaxs - xmins) / 2
		ymedians = ymins + (ymaxs - ymins) / 2

		points = np.concatenate([
			xmedians[:,np.newaxis],
			ymedians[:,np.newaxis]
		], axis=1)

		return points
	
	@classmethod
	def get_2d_dataframe(cls, path="/home/rosvm/data/train.csv"):
		df2d = pd.read_csv(path, index_col=["filename", "id"])
		d2_points = cls.create2D_points(df2d)
		
		height = (df2d['ymax']-df2d['ymin']).values
		width = (df2d['xmax']-df2d['xmin']).values
		
		indx = cls.filename_multiindex_to_basename(df2d.index)
		
		df2d = pd.DataFrame({
			"class" : df2d["class"].values,
			"midx" : d2_points[:,0],
			"midy" : d2_points[:,1],
			"height" : height,
			"width" : width
		}, index=indx)
		return df2d
	
	@classmethod
	def get_3d_dataframe (cls, path="/home/rosvm/data/3D/train.csv"):
		df3d = pd.read_csv(path, index_col=["filename", "id"])

		d3_points = np.concatenate(
			[
				df3d['pos_x'].values[:,np.newaxis],
				df3d['pos_y'].values[:,np.newaxis],
				df3d['pos_z'].values[:,np.newaxis]
			],axis=1)

		distance = np.linalg.norm(d3_points, axis=1)

		indx = cls.filename_multiindex_to_basename(df3d.index)

		new_dataframe = pd.DataFrame({
			"dist3d" : distance
		}, index=indx)

		return new_dataframe
	
	@classmethod
	def get_depthmap_paths (cls, path="/home/rosvm/data/3D/img"):
		subs = [
			osp.join(path, x)
			for x in os.listdir(path)
		]
		picture_ids = np.array([
			osp.splitext(osp.basename(x))[0]
			for x in subs
		]).astype(int)

		dm_paths = pd.DataFrame({
			"path" : subs
		}, index=picture_ids)
		print(dm_paths)
		return dm_paths
	
	@classmethod
	def get_values_for_depth_image (cls, depth_image, value, distance_function, extra_parameters):
		xmin = np.round(value["xmin"]).astype(int)
		ymin = np.round(value["ymin"]).astype(int)

		xmax = np.round(value["xmax"]).astype(int)
		ymax = np.round(value["ymax"]).astype(int)

		xcombo = np.array([xmin, xmax])
		ycombo = np.array([ymin, ymax])

		if np.any([xcombo < 0, xcombo > cls.IMAGE_WIDTH]) or np.any([ycombo < 0, ycombo > cls.IMAGE_HEIGHT]):
			min_distance = np.nan
			avg_distance = np.nan
			hag = np.nan
		else:
			depth_image = depth_image[ymin : ymax, xmin : xmax]

			depth_image = distance_function(depth_image, *extra_parameters)
			ymid = np.mean([value["ymin"], value["ymax"]])

			min_distance = np.max(depth_image)
			avg_distance = np.mean(depth_image)

			alpha = DepthImageTools.get_alpha(ymid)
			hag = DepthImageTools.get_height_above_ground(alpha, avg_distance)
			
		return min_distance, avg_distance, hag
	
	@classmethod
	def create_depthmap_dataframe (cls, path="/home/rosvm/data/train.csv", distance_function=None, extra_parameters=None):
		if distance_function is None:
			distance_function = DepthImageTools.get_distance_for_pixel
			
		if extra_parameters is None:
			extra_parameters = tuple([])
		else:
			extra_parameters = tuple(extra_parameters)
		
		df2d = pd.read_csv(path, index_col=["filename", "id"])

		picture_ids = df2d.index.values
		picture_ids = np.array([
			osp.splitext(osp.basename(x[0]))[0]
			for x in picture_ids
		]).astype(int)

		df2d["PictureId"] = picture_ids

		df2d.set_index('PictureId', append=True, inplace=True)

		dm_paths_df = cls.get_depthmap_paths()
		dm_paths_df = df2d.join(dm_paths_df, on='PictureId', how='inner')
		
		dm_min_distances = []
		dm_avg_distances = []
		height_above_grounds = []
		
		# Entries of index: ('/home/rosvm/data/3D/img/12491.jpg', 1075, 12491)
		# (Path to Depth image, Detection Id, Image Id)
		# Sorted by first, second, third!
		
		last_depth_image = None
		last_image_id = None
		
		for entry in dm_paths_df.index:
			value = dm_paths_df.loc[entry]
			image_id = value[2]
			
			if image_id == last_image_id:
				depth_image = last_depth_image
			else:
				path = value["path"]
				depth_image = np.array(Image.open(path), dtype=np.uint8)[:,:,0]
				last_depth_image = depth_image
				last_image_id = image_id
			
			min_distance, avg_distance, hag = cls.get_values_for_depth_image(depth_image, value, distance_function, extra_parameters)

			dm_min_distances.append(min_distance)
			dm_avg_distances.append(avg_distance)
			height_above_grounds.append(hag)

		new_index = dm_paths_df.index.droplevel(2)
		new_index = cls.filename_multiindex_to_basename(new_index)

		new_df = pd.DataFrame({
			"dm_min_dist" : np.array(dm_min_distances),
			"dm_avg_dist" : np.array(dm_avg_distances),
			"height_above_ground" : np.array(height_above_grounds)
		}, index=new_index)

		selector = ~(np.isnan(new_df["dm_min_dist"]) | np.isnan(new_df["dm_avg_dist"]))
		new_df = new_df[selector]

		return new_df
		
	@classmethod
	def combine_dataframes (cls, df_2d, df_3d, df_dm):
		joined_df = df_2d.join(df_3d, how='inner')
		joined_df = joined_df.join(df_dm, how='inner')
		return joined_df
	
	@classmethod
	def get_gmm_dataframe (cls, distance_function=None, extra_parameters=None):
		df_2d = DataConverter.get_2d_dataframe()
		df_3d = DataConverter.get_3d_dataframe()
		df_dm = DataConverter.create_depthmap_dataframe(distance_function=distance_function, extra_parameters=extra_parameters)
		return cls.combine_dataframes(df_2d, df_3d, df_dm)