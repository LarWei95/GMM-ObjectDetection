import cv2, os ;
import matplotlib.pyplot as plt ;
import numpy as np ;
from process3d import *
from argparse import ArgumentParser ;
import multiprocessing as mp
import os.path as osp

train_csv = "../train.csv" ;

#import open3d ;
# todo: open3d RANSAC to find ground plabe from accepted pixels!!
# 

def process (imgF, cgt, rgb_dir, depth_dir):
    target_name = f"hog{imgF}.npy"
    
    if osp.isfile(target_name):
        print(f"{target_name} already exists.")
        return

    img = cv2.imread(rgb_dir+"/"+imgF) ;
    print ("IMG SH=", img.shape)

    # initial pos mask just filters for image positions: between  280 and 450, to be adjusted
    pos_arr = (np.arange(0,512)[:,np.newaxis] * np.ones([512,512])) ;
    pos_mask = ((pos_arr >280) * (pos_arr < 450)).astype("float32") ;
    
    # now: remove everything inside GT boxes
    for gt in cgt:
        ulx = int(gt[0])
        uly = int(gt[1])
        lrx = int(gt[2])
        lry = int(gt[3])
        pos_mask[uly:lry+1,ulx:lrx+1] = 0 ;

    # color mask to eliminate mainly cars that are not in GT boxes because they are partially occluded
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    color_mask = (hsv[:,:,0] < 30).astype("int32") ;  # thresholding L, but can be saturation as well

    raw_mask = (pos_mask*color_mask) ;

    raw_depth = cv2.imread(depth_dir+"/"+imgF).sum(axis=2) ;
    
    try:
        plane_params, x, y, z, new_depth = extract_ground_plane(raw_depth=raw_depth, raw_mask=raw_mask)

        """
        # obsolete
        hog = x*plane_params[0] + y*plane_params[1] + z*plane_params[2] + plane_params[3] ;
        hog2 = hog - hog.min() ;
        hog2 /= hog2.max() ;
        #hog[0:255] = 0
        hog -= hog.min()
        print (hog.min(), hog.max()) ;
        """

        if False:
            f = plt.figure() ;
            f.add_subplot(1,2,1) ;
            plt.imshow(new_depth) ;
            f.add_subplot(1,2,2) ;
            plt.imshow(raw_mask) ;
            plt.show() ;

        # cv2.imwrite("hog"+imgF, new_depth) ;
        np.save(target_name, new_depth)
    except ValueError as e:
        print(f"Shapeerror:\n{e}")
 

if __name__ == "__main__":
    pool = mp.Pool()
    parser = ArgumentParser() ;
    parser.add_argument("--csv", type=str, default="./train.csv") ;
    parser.add_argument("--rgb_dir", type=str, default="./img/") ;
    parser.add_argument("--depth_dir", type=str, default="./3D/img") ;
    FLAGS = parser.parse_args(sys.argv[1:]) 

    train_csv = FLAGS.csv ;
    rgb_dir = FLAGS.rgb_dir ;
    depth_dir = FLAGS.depth_dir ;
  
    gt_raw = [l.strip().split(",") for l in open(train_csv, "r").readlines()[1:]] ;
    gt_dict = {} ;
    for i,l in enumerate(gt_raw):
        key = l[0].split("/")[-1] ;
        val = gt_dict.get(key,[]) ;
        val.append((float(l[4]),float(l[5]), float(l[6]), float(l[7]))) ;
        gt_dict[key] = val ;

    imagefiles = [f for f in os.listdir(rgb_dir) if f.find(".jpg") != -1 and f.find("mask") == -1] ;

    results = []

    for imgF in imagefiles:
        cgt = gt_dict[imgF]
        result = pool.apply_async(process, (imgF, cgt, rgb_dir, depth_dir))
        results.append(result)

    for result in results:
        result.get()

