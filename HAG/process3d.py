
import numpy as np, imageio, sys, math ;
import matplotlib.pyplot as plt ;
import open3d as o3d;
import open3d.visualization  as vis ;
import cv2 ;

vis = o3d.visualization.Visualizer()


def corrImage(imG):
  #return 84. - imG ;
  #return 100. / (imG + 0.01)
  #return imG+0.1
  #return 255. / (imG + 0.001)
  #return ((255. / (imG + 1.)) + 1)*1.
  return 255.-imG ;


def extract_ground_plane(fov=60., img_pix_x=512, img_pix_y=512, raw_depth=None, raw_mask=None):
  fov = fov  # offnungswinkel
  fx =  (img_pix_x - img_pix_x//2) / math.sin(0.5*fov*3.14159265 / 180) ;
  fy =  (img_pix_y - img_pix_y//2) / math.sin(0.5*fov*3.14159265 / 180) ;

  raw_depth = corrImage(raw_depth) ;
  full_depth=o3d.geometry.Image((raw_depth).astype("float32")) ;
  full_pc = o3d.geometry.PointCloud.create_from_depth_image(full_depth,
            o3d.camera.PinholeCameraIntrinsic(img_pix_x,img_pix_y,fx,fy, img_pix_x//2, img_pix_y//2), depth_scale=1)

  mask = (raw_mask  - 0.5) / 0.5 ;
  filtered_depth=o3d.geometry.Image((raw_depth*mask).astype("float32")) ;
  
  filtered_pc = o3d.geometry.PointCloud.create_from_depth_image(filtered_depth,
     o3d.camera.PinholeCameraIntrinsic(img_pix_x,img_pix_y,fx,fy, img_pix_x//2, img_pix_y//2), depth_scale=1)
  print ("ALT PC has ", np.asarray(filtered_pc.points).shape[0]) ;



  # extract ground plane from PC
  plane_model, inliers = filtered_pc.segment_plane(distance_threshold=0.5,
                                             ransac_n=3,
                                             num_iterations=1000)
  nr_inliers = len(inliers)
  print ("nr_inliers=", nr_inliers) ;
  [a, b, c, d] = plane_model
  print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
  
  if False:
    print("Displaying pointcloud with planar points in red ...")
	  
    vis.create_window()
    vis.clear_geometries()
    inlier_cloud = filtered_pc.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = filtered_pc.select_by_index(inliers, invert=True)
    vis.add_geometry(inlier_cloud)
    vis.add_geometry(outlier_cloud)
    vis.run()
    #for i in range(0,1000):
    #  print(i)
    #  vis.update_renderer()
  
  #if True:
  #  o3d.visualization.draw([inlier_cloud, outlier_cloud])
  
  y = None ; 
  pts =  np.asarray(full_pc.points) ;  
  mode = "no_rot" ;
  if mode == "no_rot":
    print (pts.shape,"SH") ;
    normal_vec = np.array([a,b,c]).reshape(1,3) ;
    plane_dists = ((pts * normal_vec) ).sum(axis=1)+d ;
    y = -plane_dists ;
    
  
    
  else:
    # compute euler angles to rotate points --> plane notmal points upwards
    angle_xy = -math.atan(a/b) ;
    angle_yz = -math.atan(c/b) ;
    print ("Angless: XY, YZ=", angle_xy, angle_yz) ;
    rotM = filtered_pc.get_rotation_matrix_from_xyz(np.array([angle_yz,0,angle_xy]))

    # check whether plane normal now points upwards, after rot.
    filtered_rotated_pc = filtered_pc.rotate(rotM)
    plane_model2, inliers2 = filtered_rotated_pc.segment_plane(distance_threshold=0.5,
                                             ransac_n=3,
                                             num_iterations=100)
    print("after rot=", plane_model2) ;


    # now transform/rotate full PC
    full_pc = full_pc.rotate(rotM)  ;
    pts =  np.asarray(full_pc.points) ;  
    # obtain h.o.g. by extracting y component
    d = plane_model2[3] ;
    y = (pts[:,1])- math.fabs(d)

  
  
  y_clipped = y #* 255./20. ;
  
  
  # possible since pc points are still ordered as the original depth image was
  new_depth = y_clipped.reshape(512,512)
  """
  u = (pts[:,0] * fx / pts[:,2]).astype("int32") + img_pix_x //2  ;
  v = (pts[:,1] * fy / pts[:,2]).astype("int32") + img_pix_y // 2;
  u = np.clip(u, 0,img_pix_x-1) ;
  v = np.clip(u, 0,img_pix_y-1) ;
  d = (pts**2).sum(axis=1) ;
  new_depth = np.zeros([img_pix_y,img_pix_x],dtype=np.float32)  ;
  new_depth[v,u][:] = d ;
  #new_depth = d.reshape(img_pix_y, img_pix_x) ;
  """
  

  

  #return plane_model, coords_x, coords_y, coords_z, new_depth
  return plane_model, None, None, None, new_depth
  #o3d.visualization.draw([inlier_cloud, outlier_cloud])










