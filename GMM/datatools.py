import pandas as pd
import numpy as np
import os.path as osp
import os
from PIL import Image
from depthimagetools import DepthImageTools
import re

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

    @classmethod
    def image_shape_to_coordinates (self, shape):
        y = np.repeat(np.arange(shape[0])[:,np.newaxis], shape[1], axis=1)
        x = np.repeat(np.arange(shape[1])[np.newaxis], shape[0], axis=0)

        return x, y
    
class DataConverter ():
    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512

    NUMBER_EXPR = "[0-9]+"
    NUMBER_RE = re.compile(NUMBER_EXPR)
    
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
    def filename_multiindex_to_picture_id (cls, indx):
        new_index = []

        for i in range(len(indx)):
            bn = osp.basename(indx[i][0])
            pi_match = cls.NUMBER_RE.search(bn)
            pi = int(bn[pi_match.start() : pi_match.end()])
            indx_id = indx[i][1]

            new_index.append((pi, indx_id))

        new_index = pd.MultiIndex.from_tuples(new_index, names=["PictureId", "Id"])
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
        
        indx = cls.filename_multiindex_to_picture_id(df2d.index)
        
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

        indx = cls.filename_multiindex_to_picture_id(df3d.index)

        new_dataframe = pd.DataFrame({
        	"dist3d" : distance
        }, index=indx)

        return new_dataframe
    
    @classmethod
    def get_id_paths (cls, path, ending):
        subfiles = [
                    x
                    for x in os.listdir(path)
                    if x.endswith(ending)
                ]

        picture_ids = []
        full_paths = []

        for subfile in subfiles:
            match = cls.NUMBER_RE.search(subfile)

            if match is not None:
                picture_id = int(subfile[match.start() : match.end()])
                full_path = osp.join(path, subfile)

                picture_ids.append(picture_id)
                full_paths.append(full_path)

        indx = pd.Index(picture_ids, name="PictureId")
        id_paths = pd.DataFrame({
                "path" : full_paths
            }, index=indx)
        return id_paths

    @classmethod
    def get_depthmap_paths (cls, path="/home/rosvm/data/3D/img"):
        subs = [
        	osp.join(path, x)
        	for x in os.listdir(path)
        ]
        '''
        picture_ids = np.array([
        	osp.splitext(osp.basename(x))[0]
        	for x in subs
        ]).astype(int)
        '''
        picture_ids = np.array([
                osp.splitext(osp.splitext(osp.basename(x).replace("hog", ""))[0])[0]
                for x in subs
                ]).astype(int)

        dm_paths = pd.DataFrame({
        	"path" : subs
        }, index=picture_ids)
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
    def get_image_section (cls, image, xmin, ymin, xmax, ymax):
        xmin = np.round(xmin).astype(int)
        ymin = np.round(ymin).astype(int)

        xmax = np.round(xmax).astype(int)
        ymax = np.round(ymax).astype(int)

        xcombo = np.array([xmin, xmax])
        ycombo = np.array([ymin, ymax])

        if np.any([xcombo < 0, xcombo > cls.IMAGE_WIDTH]) or np.any([ycombo < 0, ycombo > cls.IMAGE_HEIGHT]):
            image_section = None
        else:
            image_section = image[ymin : ymax, xmin : xmax]

        return image_section 

    @classmethod
    def process_image_values (cls, image, xmin, ymin, xmax, ymax):
        image_section = cls.get_image_section(image, xmin, 
                                              ymin, xmax, ymax)

        if image_section is not None:
            ymid = image_section.shape[0] // 2
            xmid = image_section.shape[1] // 2

            mid_value = image_section[ymid, xmid]
            mean_value = np.mean(image_section)
            median_value = np.median(image_section)
        else:
            mid_value = np.nan
            mean_value = np.nan
            median_value = np.nan

        return mid_value, mean_value, median_value
    
    @classmethod
    def process_image (cls, image, value):
        xmin = value["xmin"]
        ymin = value["ymin"]

        xmax = value["xmax"]
        ymax = value["ymax"]

        return cls.process_image_values(image, xmin, ymin,
                                        xmax, ymax) 


    @classmethod
    def get_depths_for_depth_image (cls, depth_image, value):
        depth_image = DepthImageTools.get_distance(depth_image)
        return cls.process_image(depth_image, value)


    @classmethod
    def create_depthmap_dataframe (cls, df_path, img_path):
        df2d = pd.read_csv(df_path, index_col=["filename", "id"])
        df2d.index = cls.filename_multiindex_to_picture_id(df2d.index)
     
        dm_paths_df = cls.get_id_paths(img_path, ending="jpg")
        dm_paths_df = df2d.join(dm_paths_df, on='PictureId', how='inner')
        dm_paths_df = dm_paths_df.sort_index(level="PictureId")
        

        mid_dists = []
        mean_dists = []
        med_dists = []
        
        # Entries of index: ('/home/rosvm/data/3D/img/12491.jpg', 1075, 12491)
        # (Path to Depth image, Detection Id, Image Id)
        # Sorted by first, second, third!
        
        last_depth_image = None
        last_image_id = None
        
        for entry in dm_paths_df.index:
            value = dm_paths_df.loc[entry]
            image_id = entry[0]
        	
            if image_id == last_image_id:
                depth_image = last_depth_image
            else:
                path = value["path"]
                depth_image = np.array(Image.open(path), dtype=np.uint8)[:,:,0]
                last_depth_image = depth_image
                last_image_id = image_id
        	
            mid_dist, mean_dist, med_dist = cls.get_depths_for_depth_image(depth_image, value)

            mid_dists.append(mid_dist)
            mean_dists.append(mean_dist)
            med_dists.append(med_dist)

        new_df = pd.DataFrame({
        	"mid_dist" : np.array(mid_dists),
        	"mean_dist" : np.array(mean_dists),
        	"med_dist" : np.array(med_dists)
        }, index=dm_paths_df.index)

        selector = ~(np.isnan(new_df["mid_dist"]) | np.isnan(new_df["mean_dist"]) | np.isnan(new_df["med_dist"]))
        new_df = new_df[selector]

        return new_df
       

    @classmethod
    def new_get_values_for_depth_image (cls, depth_image, value):
        xmin = np.round(value["xmin"]).astype(int)
        ymin = np.round(value["ymin"]).astype(int)

        xmax = np.round(value["xmax"]).astype(int)
        ymax = np.round(value["ymax"]).astype(int)

        xcombo = np.array([xmin, xmax])
        ycombo = np.array([ymin, ymax])

        if np.any([xcombo < 0, xcombo > cls.IMAGE_WIDTH]) or np.any([ycombo < 0, ycombo > cls.IMAGE_HEIGHT]):
            mid_hag = np.nan
            mean_hag = np.nan
            median_hag = np.nan
        else:
            sel_depth_image = depth_image[ymin : ymax, xmin : xmax]
            xmid = np.round(np.mean([value["xmin"], value["xmax"]])).astype(int)
            ymid = np.round(np.mean([value["ymin"], value["ymax"]])).astype(int)
            
            mid_hag = depth_image[ymid, xmid]
            mean_hag = np.mean(sel_depth_image)
            median_hag = np.median(sel_depth_image)

        return mid_hag, mean_hag, median_hag

    @classmethod
    def create_hag_dataframe (cls, df_path, img_path):
        df2d = pd.read_csv(df_path, index_col=["filename", "id"])
        df2d.index = cls.filename_multiindex_to_picture_id(df2d.index)

        dm_paths_df = cls.get_id_paths(img_path, ending="jpg.npy")
        dm_paths_df = df2d.join(dm_paths_df, on='PictureId', how='inner')
        dm_paths_df = dm_paths_df.sort_index(level="PictureId")

        last_image_id = None
        last_depth_image = None
        
        mid_hags = []
        mean_hags = []
        median_hags = []

        for i, entry in enumerate(dm_paths_df.index):
            value = dm_paths_df.loc[entry]
            image_id = entry[0]
            
            if image_id == last_image_id:
                depth_image = last_depth_image
            else:
                path = value["path"]
                print((i / len(dm_paths_df) * 100), image_id, path)
                depth_image = np.load(path)
                
                last_image_id = image_id
                last_depth_image = depth_image
            
            mid_hag, mean_hag, median_hag = cls.process_image(depth_image, value)

            mid_hags.append(mid_hag)
            mean_hags.append(mean_hag)
            median_hags.append(median_hag)

        new_df = pd.DataFrame({
        	"mid_hag" : np.array(mid_hags),
        	"mean_hag" : np.array(mean_hags),
        	"median_hag" : np.array(median_hags)
        }, index=dm_paths_df.index)

        selector = ~(np.isnan(new_df["mid_hag"]) | np.isnan(new_df["median_hag"]) | np.isnan(new_df["mean_hag"])) 
        new_df = new_df[selector]

        return new_df


    @classmethod
    def combine_dataframes (cls, df_2d, df_3d, df_dm, df_hag):
        joined_df = df_2d.join(df_3d, how='inner')
        joined_df = joined_df.join(df_dm, how='inner')
        joined_df = joined_df.join(df_hag, how='inner')
        return joined_df
    
    @classmethod
    def old_get_gmm_dataframe (cls, distance_function=None, extra_parameters=None):
        df_2d = DataConverter.get_2d_dataframe()
        df_3d = DataConverter.get_3d_dataframe()
        df_dm = DataConverter.old_create_depthmap_dataframe(distance_function=distance_function, extra_parameters=extra_parameters)
        return cls.combine_dataframes(df_2d, df_3d, df_dm)

    @classmethod
    def get_gmm_dataframe (cls, path_2d, path_2d_img, 
                                path_3d, path_3d_img,
                                path_hag_img, pool):
        
        df_2d = pool.apply_async(DataConverter.get_2d_dataframe, (path_2d,))
        df_3d = pool.apply_async(DataConverter.get_3d_dataframe, (path_3d,))
        df_dm = pool.apply_async(DataConverter.create_depthmap_dataframe, (path_2d, path_3d_img))
        df_hag = pool.apply_async(DataConverter.create_hag_dataframe, (path_2d, path_hag_img))

        # df_2d = DataConverter.get_2d_dataframe(path=path_2d)
        # df_3d = DataConverter.get_3d_dataframe(path=path_3d)
        # df_dm = DataConverter.create_depthmap_dataframe(df_path=path_2d, img_path=path_3d_img)
        # df_hag = DataConverter.create_hag_dataframe(df_path=path_2d, img_path=path_hag_img)
        
        df_2d = df_2d.get()
        df_3d = df_3d.get()
        df_dm = df_dm.get()
        df_hag = df_hag.get()

        return cls.combine_dataframes(df_2d, df_3d, df_dm, df_hag)

class DataManager ():
    def __init__ (self, df_path_2d, df_path_3d,
                  img_path_2d, img_path_3d, img_path_hag,
                  pool):
        self.df_path_2d = df_path_2d
        self.df_path_3d = df_path_3d
        self.img_path_2d = img_path_2d
        self.img_path_3d = img_path_3d
        self.img_path_hag = img_path_hag
        self.pool = pool
    
        self.dataframe = None

    def load_dataframe (self, path=None):
        if path is None:
            self.dataframe = DataConverter.get_gmm_dataframe(
                        self.df_path_2d, self.img_path_2d,
                        self.df_path_3d, self.img_path_3d,
                        self.img_path_hag, self.pool
                    )
        else:
            self.dataframe = pd.read_csv(path, sep=";", header=0, index_col=[0, 1])

        return self.dataframe

    def load_rgb_image (self, picture_id):
        paths_df = DataConverter.get_id_paths(self.img_path_2d, ending="jpg")
        path = paths_df["path"].loc[picture_id]
        image = np.array(Image.open(path), dtype=np.uint8)
        return image

    def load_depth_image (self, picture_id):
        paths_df = DataConverter.get_id_paths(self.img_path_3d, ending="jpg")
        path = paths_df["path"].loc[picture_id]
        image = np.array(Image.open(path), dtype=np.uint8)[:,:,0]
        image = DepthImageTools.get_distance(image)
        return image

    def load_height_above_ground (self, picture_id):
        paths_df = DataConverter.get_id_paths(self.img_path_hag, ending="jpg.npy")
        path = paths_df["path"].loc[picture_id]
        hag = np.load(path)
        return hag


