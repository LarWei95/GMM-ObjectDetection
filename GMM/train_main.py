import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from datatools import DataTools, DataConverter, DataManager
import multiprocessing as mp
import pickle as p
from collections import defaultdict
import os.path as osp
import os
import gc

class ModelSet ():
    def __init__(self):
        # LabelSet -> Columns -> Model
        self.models = defaultdict(dict)

    def store(self, label_set, column_set, model):
        self.models[label_set][column_set] = model

    def has(self, label_set, column_set):
        return column_set in self.models[label_set]

    def save(self, path):
        with open(path, "wb") as f:
            p.dump(self, f)

    @classmethod
    def new(cls, path):
        if osp.isfile(path):
            with open(path, "rb") as f:
                ms = p.load(f)
        else:
            ms = ModelSet()

        return ms

class Visualizer():
    def __init__(self, output_folder):
        output_folder = osp.abspath(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        self._output_folder = output_folder
    
    @classmethod
    def analyse_detection_boxes(cls, model_set, label_group, bincount=9):
        model = model_set.models[label_group][("height", "width")]
        samples, clusters = model.sample(100000)
        probas = model.score_samples(samples)
        mean_proba = np.mean(probas)

        sorted_indices = np.argsort(probas)[::-1]
        samples = samples[sorted_indices]
        probas = probas[sorted_indices]
        
        stop_index = np.where(probas <= mean_proba)[0][0]
        samples = samples[:stop_index]
        probas = probas[:stop_index]
        
        rnd = np.random.randint(len(samples))

        picks = [samples[rnd]]
        samples = np.delete(samples, rnd, axis=0)
        
        for i in range(1, bincount):
            cpicks = np.array(picks)
            all_dists = np.empty(len(samples))

            for j, sample in enumerate(samples):
                sample = np.repeat(sample[np.newaxis], len(cpicks), axis=0) 
                
                distances = cpicks - sample
                distances = np.mean(np.sqrt(distances[:,0] ** 2 + distances[:,1] ** 2))
                all_dists[j] = distances

            maxindx = np.argmax(all_dists)
            selected_sample = samples[maxindx]
            picks.append(samples[maxindx])
            samples = np.delete(samples, maxindx, axis=0)
        
        picks = np.array(picks)
        sorted_indices = np.argsort(np.sum(picks, axis=1))
        picks = picks[sorted_indices]
        return picks

    def set_detection_boxes(self, model_set, label_groups):
        self._detection_boxes = {
                    label_group : Visualizer.analyse_detection_boxes(model_set, label_group, bincount=16)
                    for label_group in label_groups
                }


        

    def image_to_data(self, data_manager, model_set, picture_id, columns_tup,
                      label_group):
        img_depth = data_manager.load_depth_image(picture_id)
        img_hag = data_manager.load_height_above_ground(picture_id)
        img_shape = img_depth.shape
        pixel_count = img_shape[0] * img_shape[1]
        x, y = DataTools.image_shape_to_coordinates(img_shape)
        
        if "height" not in columns_tup and "width" not in columns_tup:
            # datas: ImageData

            if columns_tup == ("midy", "midx"):
                datas = np.stack([y, x], axis=2)
            elif columns_tup == ("midy", "midx", "mid_dist"):
                datas = np.stack([y, x, img_depth], axis=2)
            elif columns_tup == ("midy", "midx", "mid_hag"):
                datas = np.stack([y, x, img_hag], axis=2)
            elif columns_tup == ("midy", "midx", "mid_dist", "mid_hag"):
                datas = np.stack([y, x, img_depth, img_hag], axis=2)
            else:
                raise ValueError(str(columns_tup))
        else:
            # hws = cls.analyse_detection_boxes(model_set, label_group, bincount=16)
            hws = self._detection_boxes[label_group]
            # datas: {(Height, Width) : ImageData}
            datas = {}

            for height, width in hws:
                hw_tup = (height, width)
                
                hrep = np.reshape(np.repeat(height, pixel_count), img_shape)
                wrep = np.reshape(np.repeat(width, pixel_count), img_shape)

                if columns_tup == ("midy", "midx", "height", "width"):
                    data = np.stack([y, x, hrep, wrep], axis=2)
                elif columns_tup == ("midy", "midx", "height", "width", "mid_dist"):
                    data = np.stack([y, x, hrep, wrep, img_depth], axis=2)
                elif columns_tup == ("midy", "midx", "height", "width", "mid_hag"):
                    data = np.stack([y, x, hrep, wrep, img_hag], axis=2)
                elif columns_tup == ("midy", "midx", "height", "width", "mid_dist", "mid_hag"):
                    data = np.stack([y, x, hrep, wrep, img_depth, img_hag], axis=2)
                else:
                    raise ValueError(str(columns_tup))

                datas[hw_tup] = data

        return datas
            
    @classmethod
    def _plot_data(cls, img_data, title, save_path):
        plt.rcParams.update({"font.size" : 32})
        plt.imshow(img_data)
        plt.title(title)
        
        plt.gcf().set_size_inches(10.0, 10.0)
        plt.subplots_adjust(left=0.06, right=0.99, bottom=0.05)
        plt.savefig(save_path)
        plt.close("all")
    
    @classmethod
    def _plot_basics(cls, picture_id, rgb, depth, hag, target_directory):
        # Passt
        plt.rcParams.update({"font.size" : 32})
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(rgb)
        ax1.set_title("RGB image")

        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(depth)
        ax2.set_title("Depth image")

        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(hag)
        ax3.set_title("Height above ground")

        plt.suptitle(f"Basic information: {picture_id}")
        
        target_name = f"{picture_id}_basics.png"
        target = osp.join(target_directory, target_name)
        
        plt.gcf().set_size_inches(20.0, 8.0)
        plt.subplots_adjust(left=0.05, right=0.99, bottom=0.01)

        plt.savefig(target)
        plt.close("all")

    def plot_simple(self, data_manager, label_group, picture_id, columns, img_data, model):
        save_folder = osp.join(self._output_folder, label_group, str(picture_id))
        os.makedirs(save_folder, exist_ok=True)
        
        img_rgb = data_manager.load_rgb_image(picture_id)
        img_depth = data_manager.load_depth_image(picture_id)
        img_hag = data_manager.load_height_above_ground(picture_id)
        
        Visualizer._plot_basics(picture_id, img_rgb, img_depth, img_hag,
                                save_folder)
     
        img_shape = img_data.shape
        new_shape = (img_shape[0] * img_shape[1], img_shape[2])
        backshape = img_shape[:2]

        reshaped_data = np.reshape(img_data, new_shape)

        log_likelihoods = model.score_samples(reshaped_data)
        log_likelihoods = np.reshape(log_likelihoods, backshape)
        base_likelihoods = np.exp(log_likelihoods)
        likelihoods = base_likelihoods ** (1/40)
        
        col_descr = "-".join(columns)
        title = f"Log likelihoods: {picture_id}\n({label_group})"
        path = osp.join(save_folder, f"{picture_id}-{label_group}-{col_descr}-log.png")
        Visualizer._plot_data(log_likelihoods, title, path) 

        title = f"Probabilities: {picture_id}\n({label_group})"
        path = osp.join(save_folder, f"{picture_id}-{label_group}-{col_descr}-mod.png")
        Visualizer._plot_data(likelihoods, title, path) 



    def plot_multi(self, data_manager, label_group, picture_id, columns, img_data_dict, model):
        save_folder = osp.join(self._output_folder, label_group, str(picture_id))
        os.makedirs(save_folder, exist_ok=True)
        
        img_rgb = data_manager.load_rgb_image(picture_id)
        img_depth = data_manager.load_depth_image(picture_id)
        img_hag = data_manager.load_height_above_ground(picture_id)
        
        Visualizer._plot_basics(picture_id, img_rgb, img_depth, img_hag,
                                save_folder)

        col_descr = "-".join(columns)

        for height, width in img_data_dict:
            box_descr = "{:.1f}x{:.1f}".format(height, width)

            img_data = img_data_dict[(height, width)]
            img_shape = img_data.shape
            new_shape = (img_shape[0] * img_shape[1], img_shape[2])
            backshape = img_shape[:2]

            reshaped_data = np.reshape(img_data, new_shape)

            log_likelihoods = model.score_samples(reshaped_data)
            log_likelihoods = np.reshape(log_likelihoods, backshape)
            base_likelihoods = np.exp(log_likelihoods)
            likelihoods = base_likelihoods ** (1/40)

            title = f"Log likelihoods: {picture_id}\n({label_group}, {box_descr})"
            path = osp.join(save_folder, f"{picture_id}-{label_group}-{col_descr}-{box_descr}-log.png")
            Visualizer._plot_data(log_likelihoods, title, path) 

            title = f"Probabilities: {picture_id}\n({label_group}, {box_descr})"
            path = osp.join(save_folder, f"{picture_id}-{label_group}-{col_descr}-{box_descr}-mod.png")
            Visualizer._plot_data(likelihoods, title, path)             

    def visualize(self, data_manager, model_set, picture_ids, 
            column_sets):
        splitted_df = DataTools.split_dataframes_into_groups(data_manager.dataframe)
        print("Setting detection boxes.")
        self.set_detection_boxes(model_set, splitted_df.keys())

        for columns in column_sets:
            columns_tup = tuple(columns)
            
            for label_group in splitted_df:
                group_df = splitted_df[label_group]
                model = model_set.models[label_group][columns_tup]
                
                for picture_id in picture_ids:
                    print(columns, label_group, picture_id)
                    img_data = self.image_to_data(data_manager, model_set, picture_id,
                                                 columns_tup, label_group)

                    if isinstance(img_data, dict):
                        # Multiple images with various boxes
                        self.plot_multi(data_manager, label_group, picture_id, columns_tup, img_data, model)
                    else:
                        self.plot_simple(data_manager, label_group, picture_id, columns, img_data, model)

                    gc.collect()
                        

def create_datamanager(pool):
    base_path = "/media/bsbsuser/New_Seagate/Eclipse-Projekte/GMM-ObjectDetection"

    path_2d = osp.join(base_path, "data/train.csv") 
    path_3d = osp.join(base_path, "data/3D/train.csv") 
    img_path_2d = osp.join(base_path, "data/img")
    img_path_3d = osp.join(base_path, "data/3D/img")
    img_path_hag = osp.join(base_path, "data/hag")

    df_path = osp.join(base_path, "alldata.csv")

    dm = DataManager(path_2d, path_3d,
                     img_path_2d, img_path_3d, img_path_hag,
                     pool)
    dm.load_dataframe(df_path)
    return dm

def train(values):
    print("Training a model.")
    model = GaussianMixture(n_components=50, n_init=5, tol=1e-6, max_iter=100000).fit(values)
    print("Done.")
    return model


def create_model_set(dm, column_sets, pool):
    df = dm.dataframe
    splitted = DataTools.split_dataframes_into_groups(df)
    ms = ModelSet.new("ModelSet.pkl")

    for label_group in splitted:
        group_df = splitted[label_group]
        
        for column_set in column_sets:
            tupled = tuple(column_set)
            
            if not ms.has(label_group, tupled):
                print(f"Training {column_set} for {label_group}")

                values = group_df[column_set].values
                result = train(values)
                
                ms.store(label_group, tupled, result)
                ms.save("ModelSet.pkl")

    return ms

def main():
    output_path = "./Results"

    pool = mp.Pool()
    dm = create_datamanager(pool)

    ref_ids = [
                11849, # Only SchoolBus
                13404, # Multiple cars, strong angle
                14767, # Multiple cars, no angle
                16314, # Less cars, one pedestrian rigt
                20098, # Only a pedestrian mid, strong angle
                25347 # Multiple cars to left
            ]

    column_sets = [
                ["height", "width"],

                ["midy", "midx"],
                ["midy", "midx", "mid_dist"],
                ["midy", "midx", "mid_hag"],
                ["midy", "midx", "mid_dist", "mid_hag"],
                
                ["midy", "midx", "height", "width"],
                ["midy", "midx", "height", "width", "mid_dist"],
                ["midy", "midx", "height", "width", "mid_hag"],
                ["midy", "midx", "height", "width", "mid_dist", "mid_hag"]
            ]

    ms = create_model_set(dm, column_sets, pool)
   
    visualizer = Visualizer(output_path)
    eval_column_sets = column_sets[1:]
    
    visualizer.visualize(dm, ms, ref_ids, eval_column_sets)

if __name__ == '__main__':
    main()
