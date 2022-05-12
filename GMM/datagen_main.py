from PIL import Image
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from depthimagetools import DepthImageTools
from datatools import DataTools, DataConverter
from sklearn import mixture
import pandas as pd
import multiprocessing as mp
    
if __name__ == '__main__':
    # plot_distance_calculation_statistics()
    # boxes_scatterplot()
    # boxes_statistics()
    base_path = "/home/mt21/GMM-ObjectDetection"
    path_2d = osp.join(base_path, "data/train.csv")
    path_2d_img = osp.join(base_path, "data/img")
    path_3d = osp.join(base_path, "data/3D/train.csv")
    path_3d_img = osp.join(base_path, "data/3D/img")
    path_hag_img = osp.join(base_path, "data/hag")
    out_path = osp.join(base_path, "alldata.csv")
    pool = mp.Pool()

    df = DataConverter.get_gmm_dataframe(path_2d, path_2d_img, 
                                         path_3d, path_3d_img, 
                                         path_hag_img, pool)
    print(df)
    df.to_csv(out_path, sep=";")
