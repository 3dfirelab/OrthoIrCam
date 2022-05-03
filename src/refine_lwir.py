from __future__ import print_function
from __future__ import division
from builtins import input
from builtins import zip
from builtins import range
from past.utils import old_div
import cv2
import numpy as np
import socket 
import matplotlib as mpl
if 'ibo' in socket.gethostname(): mpl.use('Agg')
if 'moritz' in socket.gethostname(): mpl.use('Agg')
if 'kraken' in socket.gethostname(): mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import io, ndimage
import asciitable
import datetime 
import os
import sys
import glob 
import argparse
import importlib
import shutil 
from osgeo import gdal,osr,ogr
import pdb
import itertools
import skimage 
import img_scale
import scipy 
import pandas 
import math 
import warnings

'''
import keras
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .7
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
keras.__version__
K.clear_session()
#import keras
#import tensorflow
from keras.models import load_model
'''

from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import importlib

#homebrewed
import optris
import flir

import tools
import camera_tools
import spectralTools
import warp_lwir_mir
#import estimate_image_georef_error
import hist_matching
#import tools_cnn 
import get_mask_to_remove_missed_burn_area
#sys.path.append('../../Ros/src/')
#import map_georefImage


#reload(estimate_image_georef_error)
#reload(map_georefImage)

########################################################
def runThreshold(temp,temp_threshold,plotmask,frame_time, arrivalTime):
   
    mask_fire  = np.where((temp >= temp_threshold),
                          np.ones_like(temp), np.zeros_like(temp))

    mask_now  = np.where( (mask_fire==1) & (plotmask==2) & (arrivalTime<0),
                          np.ones_like(temp), np.zeros_like(temp))
  
    arrivalTime = np.where(mask_now>0, frame_time*np.ones_like(temp), arrivalTime)

    #remove detached flame
    s = [[1,1,1], \
         [1,1,1], \
         [1,1,1]] # for diagonal
    labeled, nbre_cluster = scipy.ndimage.label(np.array(np.where(arrivalTime>0,1,0),np.uint8), structure=s )
    if nbre_cluster > 0: 
        size_clus = np.zeros(nbre_cluster)
        for iclus in range(nbre_cluster):
            idx_ = np.where(labeled == iclus+1)
            size_clus[iclus] = idx_[0].shape[0]

        kernel_ = 11
        if size_clus.max()>kernel_*5:
            
            for iclus in range(nbre_cluster):
                idx_ = np.where(labeled == iclus+1)
                if  idx_[0].shape[0] < kernel_ : 
                    arrivalTime[idx_] = -999
            
            #print('apply flame filter in arrival time map') 
            #mask_noFlame =cv2.morphologyEx(255*np.array(np.where(arrivalTime>0,1,0),np.uint8), cv2.MORPH_OPEN, np.ones((kernel_,kernel_),np.uint8) )
            #arrivalTime[np.where(mask_noFlame==255)] = -999

    return np.where((arrivalTime<=frame_time)&(arrivalTime>=0),1,0), arrivalTime


#########################################
def get_radgradmask(params_lwir_camera, georef_temp, georef_mask, flag_keepPlotAeraOnly, tree, burnplot, plotmask, plotmask_large,
                    arrivalTime, time_igni, ssim2dF_mean, dist_behind=30, dist_ahead=20, 
                    temp_fire=370, time_behind_input=120, ssim2D_thres=None, ssim2dF_filter=False, diskSize=10):
    
    mask = np.where( (np.array(georef_mask,dtype=np.float32) ==1) , 1, 0) 
    kernel = np.ones((3,3),np.uint8)
    img_ = np.array(np.where(mask==1,1,0),dtype=np.uint8)*255
    tmp_= cv2.erode(img_, kernel, iterations = 1)    
    mask = np.where(tmp_==255,1,0)

    mask_fire, arrivalTime = runThreshold(georef_temp, temp_fire, plotmask, time_igni, arrivalTime)
    
    time_behind  = max( [0, time_igni-time_behind_input] )
    time_aheadOf = 0 if 'time_aheadOf_refined' not in list(params_lwir_camera.keys()) else params_lwir_camera['time_aheadOf_refined']

    if time_igni <= time_aheadOf: 
        time_aheadOf_here = time_aheadOf
    else:
        time_aheadOf_here = 0

    flag_temperature_check = np.zeros_like(ssim2dF_mean)
    flag_temperature_check = np.where((plotmask==2)      &(georef_temp<540), np.ones_like(ssim2dF_mean), flag_temperature_check)
    flag_temperature_check = np.where((plotmask_large==2)&(georef_temp<350), np.ones_like(ssim2dF_mean), flag_temperature_check)

  
    if ssim2dF_mean.max()>.8: 
        if ssim2D_thres is None:
            mask_output = np.array(np.where( ((plotmask_large==2) & (plotmask!=2) & (ssim2dF_mean>.66)          & (georef_mask==1)  ) | ( (arrivalTime>=time_aheadOf_here)&(arrivalTime<=time_behind) ), 1, 0),dtype=np.uint8)   # .66 for tmp3
        else:
            mask_output = np.array(np.where( (((plotmask_large==2) & (plotmask!=2) & (ssim2dF_mean>ssim2D_thres) & (georef_mask==1)  ) |\
                                              ((arrivalTime>=time_aheadOf_here) & (arrivalTime<=time_behind)                         ))   &\
                                             (flag_temperature_check==1)
                                             , 1, 0), dtype=np.uint8 ) 
            #mask_output = np.array(np.where( ((plotmask_large==2) &                  (ssim2dF_mean>ssim2D_thres) & (georef_mask==1)  ) | ( (arrivalTime>=time_aheadOf_here)&(arrivalTime<=time_behind) ), 1, 0),dtype=np.uint8)
    else: 
        mask_output     = np.array(np.where( ((plotmask_large==2) & (plotmask!=2)                               & (georef_mask==1)  ) | ( (arrivalTime>=time_aheadOf_here)&(arrivalTime<=time_behind) ), 1, 0),dtype=np.uint8)

    #remove high temperature with potentially active fire
    georef_temp_thresholdActiveFire = 390 if 'tempThresholdActiveFire_refined' not in list(params_lwir_camera.keys()) \
                                          else params_lwir_camera['tempThresholdActiveFire_refined']

    mask_output = np.where( (burnplot.mask==2) &                       (georef_temp>temp_fire                      ), 0, mask_output) # inside  plot 
    mask_output = np.where( (burnplot.mask!=2) & (plotmask_large==2) & (georef_temp>georef_temp_thresholdActiveFire),       0, mask_output) # outside plot

    #patch_ngarkt = np.zeros_like(burnplot.mask)
    #patch_ngarkt[80:210,60:135] = 1
    #mask_output = np.where(patch_ngarkt==1, 0, mask_output)

    
    #remove zone with no clear gradient
    temp1_     = local_normalization(georef_temp    , mask_output,     diskSize=diskSize)
    gtemp1_ = tools.get_gradient(temp1_)[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H = skimage.filters.rank.entropy( skimage.img_as_uint( old_div(gtemp1_,(max([abs(gtemp1_.min()),abs(gtemp1_.max())])))), skimage.morphology.disk(3), mask=mask_output)
    
    mask_lowH = np.where(H<4.4,1,0)
    mask_output = np.where( mask_lowH==1, 0, mask_output)
   

    #close small hole
    if ssim2dF_filter:
        #mask_output_open= old_div(cv2.morphologyEx(255*mask_output, cv2.MORPH_OPEN, np.ones((11,11),np.uint8) ),255)
        mask_output_open= old_div(cv2.morphologyEx(255*mask_output, cv2.MORPH_OPEN, np.ones((7,7),np.uint8) ),255)
        mask_output = np.where(mask_output_open==1, mask_output,0 )

    temp = georef_temp
    gtemp = np.array(tools.get_gradient(temp)[0],dtype=np.float32)

    #if mask_output.max() == 0: pdb.set_trace()

    return np.array(temp, dtype=np.float32),\
           np.array(georef_mask, dtype=np.uint8), \
           gtemp, \
           np.array(mask_output,dtype=np.uint8), \
           arrivalTime

###############################################################
def get_matching_feature_opticalFlow(rad, mask, rad_ref, qualityLevel=0.3 , temp_range=[None,None], blockSize=41, relative_err = 0.1, maxLevel=7):
    
    nx,ny = rad.shape
    feature_params = dict( maxCorners = 5000,
                           qualityLevel = qualityLevel, #lwir
                           minDistance = old_div(blockSize,2),
                           blockSize = blockSize )
    lk_params = dict( winSize  = (blockSize,blockSize),
                      maxLevel = maxLevel,  
                      criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1000, 0.01))

    if (temp_range[0] is None) | (temp_range[1] is None):
        temp1_arr = np.arange(290,320,5) #295 
        temp2_arr = np.arange(320,370,5) #340 #min( [input_temp.max(), input_temp_ref.max()] )

        nbre_feature = []
        temps=[]
        for (temp1, temp2) in itertools.product(temp1_arr, temp2_arr):
            input     = camera.convert_2_uint8(rad,     [temp1,temp2]) 
            input_ref = camera.convert_2_uint8(rad_ref, [temp1,temp2]) 
           
            p0 = cv2.goodFeaturesToTrack(input, mask = mask, **feature_params)
            if p0 is None: continue
            nbre_feature.append(p0.shape[0])    
            temps.append([temp1,temp2])

        temp1,temp2 = temps[np.array(nbre_feature).argmax()]
    
    else:
        #temp1,temp2 = rad[np.where(rad>0)].min(),temp_range[1]
        temp1,temp2 = temp_range[0],temp_range[1]

    if  relative_err is not None: 
        input     = camera.convert_2_uint8(rad,     [temp1,temp2]) 
        input_ref = camera.convert_2_uint8(rad_ref, [temp1,temp2]) 
    else:
        input     = camera.convert_2_uint8(rad,     [temp1,temp2], flag_sqrtScale=False) 
        input_ref = camera.convert_2_uint8(rad_ref, [temp1,temp2], flag_sqrtScale=False) 

    p0 = cv2.goodFeaturesToTrack(input, mask = mask, **feature_params)
    if p0 is None:
        return np.zeros([0,2]), np.zeros([0,2]), 0, 0, input

    # calculate optical flow
    try:
        p1,  st, err = cv2.calcOpticalFlowPyrLK(input     , input_ref, p0, None, **lk_params)
    except: 
        pdb.set_trace()
    p0r, st, err = cv2.calcOpticalFlowPyrLK(input_ref, input, p1, None, **lk_params)
    
    d = np.sqrt( ((p0[:,0,:]-p0r[:,0,:])**2).sum(-1) )

    good = np.where( (d < 1)                              &\
                     (p1[:,0,0]>=50) & (p1[:,0,0]<=ny-50) &\
                     (p1[:,0,1]>=50) & (p1[:,0,1]<=nx-50), \
                     np.ones(d.shape,dtype=bool), np.zeros(d.shape,dtype=bool) )
    
    nbrept_badLoc = len(np.where( good == False)[0])

    if relative_err is not None:
        #average temperature at each good point on both side
        sbox=5
        idx_t1 = (np.array(np.round(p1[good,0,1],0),dtype=int),np.array(np.round(p1[good,0,0],0,),dtype=int))
        t1 = []
        for i,j in zip(idx_t1[0],idx_t1[1]):
            i_l = max([0,i-sbox]); i_r = min([i+sbox,nx])
            j_l = max([0,j-sbox]); j_r = min([j+sbox,nx])
            t1.append(rad_ref[i_l:i_r,j_l:j_r].mean())

        t0 = []
        idx_t0 = (np.array(np.round(p0[good,0,1],0),dtype=int),np.array(np.round(p0[good,0,0],0,),dtype=int))
        for i,j in zip(idx_t0[0],idx_t0[1]):
            i_l = max([0,i-sbox]); i_r = min([i+sbox,nx])
            j_l = max([0,j-sbox]); j_r = min([j+sbox,nx])
            t0.append(rad[i_l:i_r,j_l:j_r].mean())

        idx_temp_diff = np.where( np.abs( old_div((np.array(t0)-np.array(t1)),np.array(t0)) ) > relative_err )
        nbrept_badTemp2 = len(idx_temp_diff[0])
        good[idx_temp_diff] = False
    else:
        nbrept_badTemp2 = 0


    # Select good points
    p1_good = p1[good,:,:] 
    p0_good = p0[good,:,:] 
    
    if False: #temp2 > 500: 
        plt.clf()
        ax = plt.subplot(121)
        ax.imshow(input.T,origin='lower',cmap=mpl.cm.Greys_r,interpolation='nearest')
        ax.scatter(p0[:,0,1],p0[:,0,0],c='g',s=80)
        ax.scatter(p0_good[:,0,1],p0_good[:,0,0],c='r')
        ax = plt.subplot(122)
        ax.imshow(input_ref.T,origin='lower',cmap=mpl.cm.Greys_r,interpolation='nearest')
        ax.scatter(p1_good[:,0,1],p1_good[:,0,0],c='r')

        plt.show()
        pdb.set_trace()
    

    if p0_good.size == 0 :
        return np.zeros([0,2]), np.zeros([0,2]), 0, 0, input
    else: 
        return p0_good[:,0,:], p1_good[:,0,:], nbrept_badLoc, nbrept_badTemp2, input
    

###############################################################
def get_matching_feature_SIFT(input_temp, mask, input_temp_ref, mask_ref ):
    
    MIN_MATCH_COUNT = 10


    nbre_feature_to_select = 2000
    nbre_match_pt_1 = 0
    nbre_call_sift = 1

    temp1 = 295 
    temp2 = 350 #min( [input_temp.max(), input_temp_ref.max()] )

    input     = camera.convert_2_uint8(input_temp,     [temp1,temp2]) 
    input_ref = camera.convert_2_uint8(input_temp_ref, [temp1,temp2]) 

    while nbre_match_pt_1 < 200: 

        sift = cv2.xfeatures2d.SIFT_create(nbre_call_sift*nbre_feature_to_select) # limit number of feature
       
        # find the keypoints and descriptors with SIFT
        kp1_vis, des1_vis = sift.detectAndCompute(input,    mask    )
        kp2_vis, des2_vis = sift.detectAndCompute(input_ref,mask_ref)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)  # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(des1_vis,des2_vis,k=2)

        good_vis = []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.6*n.distance:
                good_vis.append(m)

        kp1_all  = [kp1_vis]*len(good_vis) 
        kp2_all  = [kp2_vis]*len(good_vis) 

        #kp1_all2  = [kp1_vis]*len(good_vis2) 
        #kp2_all2  = [kp2_vis]*len(good_vis2) 
        
        nbre_match_pt_1 = len(good_vis)
        #nbre_match_pt_2 = len(good_vis2)
   
        #for next loop
        nbre_call_sift += 1

        if nbre_call_sift > 4: 
            break
    
    #print '  nbre match Point = ', len(good)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m,kp1 in zip(good_vis,kp1_all) ]).reshape(-1,1,2) 
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m,kp2 in zip(good_vis,kp2_all) ]).reshape(-1,1,2) 

    #ax = plt.subplot(121)
    #ax.imshow(input.T,origin='lower',cmap=mpl.cm.Greys_r)
    #ax.scatter(src_pts[:,0,1], src_pts[:,0,0],c='r' )
    #ax = plt.subplot(122)
    #ax.imshow(input_ref.T,origin='lower',cmap=mpl.cm.Greys_r)
    #ax.scatter(dst_pts[:,0,1], dst_pts[:,0,0],c='r' )
    #plt.show()
    #pdb.set_trace()

  


    #remove point to close from the helico legs
    nbrept_remove_helico = 0
    idx_helico_src = np.where(getattr(frame,    input_mask)==0)
    idx_helico_dst = np.where(getattr(frame_ref,input_mask)==0)
    
    if (len(idx_helico_src[0]) != 0) | (len(idx_helico_dst[0]) != 0) : 
        tree_neighbour_src    = scipy.spatial.cKDTree(list(zip(idx_helico_src[1],idx_helico_src[0]))) # all point tree
        tree_neighbour_dst    = scipy.spatial.cKDTree(list(zip(idx_helico_dst[1],idx_helico_dst[0]))) # all point tree
        flag_pt_ok = np.zeros(src_pts.shape[0])
        for i_pt in range(src_pts.shape[0]):
            pt_src = src_pts[i_pt,0,:]
            pt_dst = dst_pts[i_pt,0,:]
            
            d_src, inds_src = tree_neighbour_src.query(pt_src, k = 3)
            d_dst, inds_dst = tree_neighbour_dst.query(pt_dst, k = 3)

            if (min(d_src) < 5) | (min(d_dst) < 5): # point too close to mask
                flag_pt_ok[i_pt] = 1

        idx_helico_ok = np.where(flag_pt_ok==0)[0]
        nbrept_remove_helico = src_pts.shape[0]-len(idx_helico_ok)
        src_pts = src_pts[idx_helico_ok,:,:]
        dst_pts = dst_pts[idx_helico_ok,:,:]
    
    if src_pts.shape[0] >MIN_MATCH_COUNT:

        #src_pts2 = np.float32([ kp2[m.queryIdx].pt for m,kp2 in zip(good_vis2,kp2_all2) ]).reshape(-1,1,2)
        #dst_pts2 = np.float32([ kp1[m.trainIdx].pt for m,kp1 in zip(good_vis2,kp1_all2) ]).reshape(-1,1,2)

        '''
        #only keep matching point
        idx_to_keep = []
        pt_to_keep = []
        for ii, pt in enumerate(zip(src_pts[:,0,0],src_pts[:,0,1])):
        
            #remove duplicate
            if pt in pt_to_keep:
                continue

            pts2 = dst_pts2[:,0,:]
            dist1 = np.sqrt(np.sum( (pt-pts2)**2,axis=1))
            idx = np.where(dist1 <= 1)
            if len(idx[0]) > 0:
                idx_ = np.where(dist1 == dist1.min())[0]

                dist2 = np.sqrt(np.sum( (dst_pts[ii,0,:]-src_pts2[idx_,0,:])**2) )
                
                if dist2 <= 1:
                    idx_to_keep.append(ii)
                    pt_to_keep.append(pt)


        new_nbre_pt = len(idx_to_keep)
        src_pts = src_pts[idx_to_keep,:,:]
        dst_pts = dst_pts[idx_to_keep,:,:]
        '''
        new_nbre_pt=nbre_match_pt_1
        '''
        ax = plt.subplot(121)
        ax.imshow(getattr(frame,input).T,origin='lower',cmap=mpl.cm.Greys_r)
        ax.scatter(src_pts[:,0,:][:,1],src_pts[:,0,:][:,0],marker='o',s=15,facecolors='none',edgecolors='r')
        ax = plt.subplot(122)
        ax.imshow(getattr(frame_ref,input).T,origin='lower',cmap=mpl.cm.Greys_r)
        ax.scatter(dst_pts[:,0,:][:,1],dst_pts[:,0,:][:,0],marker='o',s=15,facecolors='none',edgecolors='r')
        plt.show()
        pdb.set_trace()
        '''

        if flag == 'use img':
            dst_pts_refFrame00 = cv2.perspectiveTransform(dst_pts,frame_ref.H2Ref)
            return src_pts[:,0,:], dst_pts_refFrame00[:,0,:], nbre_call_sift-1, nbrept_remove_helico
        elif (flag == 'use warp') | (flag =='use new trange'):
            return src_pts[:,0,:], dst_pts[:,0,:], nbre_call_sift-1, nbrept_remove_helico
        elif flag == 'use for Wt':
            return src_pts[:,0,:],dst_pts[:,0,:], nbre_call_sift-1, nbrept_remove_helico

    else:
        return None, None, None, None 

#################################################
def convert_2_uint16(x,trange=None):

    x = np.array(x,dtype=float)
    if  trange is None: 
        xmin, xmax = x.min(), x.max()
    else:
        xmin, xmax = trange
    x = np.where(x<xmin,xmin,x)
    x = np.where(x>xmax,xmax,x)

    m = xmax-xmin
    p = xmin
    x_01 = img_scale.sqrt( old_div((x-p),m)  , scale_min=0, scale_max=1)
    
    return np.array(np.round(x_01*65535,0),dtype=np.uint16)


#########################################
def local_normalization(input_float, mask, diskSize=30, trange=[300,600]):

    #idx = np.where(mask==1)
    #trange_ = [input_float[idx].min(), input_float[idx].max()]
    #img = convert_2_uint16(input_float, trange_ )
   
    input_float = np.where(input_float<trange[0],trange[0],input_float)
    input_float = np.where(input_float>trange[1],trange[1],input_float)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.img_as_uint(old_div((input_float-trange[0]),(trange[1]-trange[0])),force_copy=True)

        selem = skimage.morphology.disk(diskSize)
        img_eq = skimage.filters.rank.equalize(img, selem=selem, mask=mask)


    return np.float32(old_div(img_eq,6.5535e4) )

    '''
    plt.imshow(img_eq.T,origin='lower'); plt.show()

    pdb.set_trace()

    float_gray = input_.astype(np.float32) / 300

    blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
    num = float_gray - blur

    blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
    den = cv2.pow(blur, 0.5)

    gray = num / den

    cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    #plt.imshow(gray.T,origin='lower')
    #plt.show()
    #pdb.set_trace()

    return gray
    '''

##########################################
def get_gridded_error(feature_on_frame_all, feature_on_prev_, rad, mask_grid, reso):

    error_pts = np.linalg.norm(feature_on_frame_all-feature_on_prev_,axis=1)

    outlier_flag=np.zeros_like(error_pts)
    if False: 
        idx_error_sorted = np.argsort(error_pts)
        #filter outlier
        idx_good = []
        tree = NearestNeighbors(n_neighbors=min([10,feature_on_frame_all.shape[0]]), algorithm='ball_tree').fit(feature_on_frame_all)
        for ipt, pt in enumerate(feature_on_frame_all[idx_error_sorted,:]):
            try:
                distances, indices = tree.kneighbors([pt])
            except: 
                pdb.set_trace()
            idx_ok = np.where( (distances<41) & (distances>0) & (outlier_flag[indices]==0) )
            
            if idx_ok[0].shape[0]<3: continue 
            if error_pts[ipt] < ( error_pts[indices[idx_ok]].mean() + 3*error_pts[indices[idx_ok]].std() ) :
                idx_good.append(ipt)
            else:
                outlier_flag[ipt] = 1
            
        feature_on_frame_ = feature_on_frame_all[idx_good,:] 
        error_pts         = error_pts[idx_good]
    else: 
        feature_on_frame_ = feature_on_frame_all

    #grid data on grid with resolution x nn
    nn = int( old_div(np.round(old_div(1.*rad.shape[0],8),0), reso))
    grid_x, grid_y = np.mgrid[0:rad.shape[0]-nn:old_div(rad.shape[0],nn)*1j, 0:rad.shape[1]-nn:old_div(rad.shape[0],nn)*1j]
    error_gridded = np.zeros(grid_x.shape).flatten()-999
    flag_pts = np.zeros_like(error_pts)
    mm = 0
    for ii, idx_ in enumerate(zip(grid_x.flatten(),grid_y.flatten())):
        idx_pts = np.where( (feature_on_frame_[:,1] >= idx_[0]) & (feature_on_frame_[:,1] < idx_[0]+nn) & \
                            (feature_on_frame_[:,0] >= idx_[1]) & (feature_on_frame_[:,0] < idx_[1]+nn)   )
        if idx_pts[0].shape[0]<2:
            continue
            #error_gridded[ii] = error_pts[idx_pts].max()
            #flag_pts[idx_pts] = 1
            #mm+= idx_pts[0].shape[0]
        elif idx_pts[0].shape[0]==2:
            error_gridded[ii] = error_pts[idx_pts].mean()
            flag_pts[idx_pts] = 1
            mm+= idx_pts[0].shape[0]
        else:
            error_gridded[ii] = np.percentile(error_pts[idx_pts],80)
            flag_pts[idx_pts] = 1
            mm+= idx_pts[0].shape[0]

        #if np.percentile(error_pts[idx_pts],80) > 2:
        #    plt.imshow(error_gridded.reshape(grid_x.shape).T,origin='lower',extent=(0,grid_x.shape[0],0,grid_x.shape[1]))
        #    plt.scatter(feature_on_frame_[:,1]/nn,feature_on_frame_[:,0]/nn,c='k')
        #    plt.scatter(feature_on_frame_[idx_pts[0],1]/nn,feature_on_frame_[idx_pts[0],0]/nn,c='r')
        #    plt.show()
        #    pdb.set_trace()

    error_gridded = error_gridded.reshape(grid_x.shape)
    nbre_pixel_lowRes = np.where(error_gridded>0)[0].shape[0]

    #pdb.set_trace()
    #fillup
    #pts  = np.dstack((np.where(error_gridded>0)[0],np.where(error_gridded>0)[1]))[0]
    #vals = error_gridded[np.where(error_gridded>0)]
    #grid_x, grid_y = np.mgrid[0:error_gridded.shape[0]-1:error_gridded.shape[0]*1j, 0:error_gridded.shape[1]-1:error_gridded.shape[1]*1j]
    #try:
    #    error_gridded = scipy.interpolate.griddata(pts, vals, (grid_x, grid_y), method='linear',fill_value=-999)
    #except:
    #    error_gridded = error_gridded
    #    #pdb.set_trace()
    
    #back to full resolution    
    error_gridded_z = np.zeros(rad.shape) #ndimage.zoom(error_gridded, nn, order=0, cval=-999)
    
    #pdb.set_trace()
    #for i,j in itertools.product(range(nn),range(nn)): error_gridded_z[i::nn,j::nn] = error_gridded
    for i,j in itertools.product(list(range(error_gridded.shape[0])),list(range(error_gridded.shape[1]))): 
        error_gridded_z[ int(grid_x[i,j]):int(grid_x[i,j])+nn+1, int(grid_y[i,j]):int(grid_y[i,j])+nn+1] = error_gridded[i,j]

    #plt.imshow(error_gridded_z.T,origin='lower',extent=(0,error_gridded_z.shape[0],0,error_gridded_z.shape[1]))
    #plt.scatter(feature_on_frame_[:,1],feature_on_frame_[:,0],c='k')
    #plt.scatter(feature_on_frame_[np.where(outlier_flag==1)[0],1],feature_on_frame_[np.where(outlier_flag==1)[0],0],c='r')
    #plt.show()
    #pdb.set_trace()
    '''pdb.set_trace()
    #correct for eventual padding
    error_gridded = np.zeros(mask_grid.shape)
    nnx_, nny_ = nn*(mask_grid.shape[0]/nn),nn*(mask_grid.shape[1]/nn)
    idx=np.where(np.logical_not(np.isnan(error_gridded_)))
    error_gridded[:nnx_,:nny_][idx] = error_gridded_[idx]
    '''
    #plt.imshow(np.ma.masked_where(plot_mask_enlarged==0,error_gridded).T, origin='lower')
    #plt.scatter(feature_on_frame_[:,1],feature_on_frame_[:,0],c='r',alpha=.5,s=5)
    #plt.show()
    #pdb.set_trace()

    return error_gridded_z, outlier_flag, nbre_pixel_lowRes


#############################################
def load_georefnpy(filename, frame):
    tmp_     = np.load(filename, allow_pickle=True)
    if len(tmp_) == 7: #input from refined run 
        info_,                           \
             homogra_mat_ori,                     \
             georef_radiance,            \
             georef_mask, georef_maskfull,                \
             georef_temp, arrivalTime_here, = tmp_
        frame_info = info_[2]
        frame_time_igni = frame_info[1]
    else:  #input from georef run
        homogra_mat_ori = frame.H2Grid
        frame_time_igni = frame.time_igni
        frame_info,                           \
             georef_img,                      \
             georef_maskfull,                     \
             georef_temp,                     \
             georef_radiance                = tmp_

    return frame_info, homogra_mat_ori, frame_time_igni,  georef_radiance, georef_temp, georef_maskfull


#########################################
def get_local_EP08_from_img(img, img_ref, inputMask=None, inputMask_ref=None, inputMask_local=None):
    
    if inputMask     is None: inputMask     = np.zeros_like(img)
    if inputMask_ref is None: inputMask_ref = np.zeros_like(img)

    idx_mask_ = np.where( (inputMask==1) & (inputMask_ref==1) )

    if idx_mask_[0].shape[0] == 0: 
        print('get_EP08_from_img: no mask intersection')
        return -999.

    if inputMask_local is None:
        img_mean     = img[idx_mask_].mean()
        img_ref_mean = img_ref[idx_mask_].mean()

        iw = np.array(     img[idx_mask_].flatten() - img_mean,  dtype=float)
        ir = np.array( img_ref[idx_mask_].flatten() - img_ref_mean,  dtype=float)
        #print '**',  img_mean,img_ref_mean, np.linalg.norm(iw), np.linalg.norm(ir)
        ep08 =  old_div(np.dot(ir,iw),(np.linalg.norm(ir)*np.linalg.norm(iw)))
        ep08 = ep08 if (not(np.isnan(ep08))) else -999.
        return ep08

    else: 
        ep082d = np.zeros(img.shape,dtype=float)
        for idx_local_ in np.unique(inputMask_local):
            
            idx_mask_ = np.where( (inputMask==1) & (inputMask_ref==1) & (inputMask_local==idx_local_) )

            if idx_mask_[0].shape[0] <= 3: 
                ep082d[np.where(inputMask_local==idx_local_)] = -999.
                continue 

            img_mean     = img[idx_mask_].mean()
            img_ref_mean = img_ref[idx_mask_].mean()

            iw = np.array(     img[idx_mask_].flatten() - img_mean,  dtype=float)
            ir = np.array( img_ref[idx_mask_].flatten() - img_ref_mean,  dtype=float)
            #print '**',  img_mean,img_ref_mean, np.linalg.norm(iw), np.linalg.norm(ir)
            ep08_ =  old_div(np.dot(ir,iw),(np.linalg.norm(ir)*np.linalg.norm(iw)))
            ep082d[idx_mask_] =ep08_ if (not(np.isnan(ep08_))) else -999.
        
        return ep082d



##########################################################
def filter_badLwirFrame(flag_parallel, mir_info, dir_out_mir_frame, dir_out_mir_georef_npy, plotname, burnplot, mask_burnNoburn, arrivalTime_lwir, window_ssimNeighbors=5, flag_save_ssim_npy=False):
    
    ncfilenames       = sorted(glob.glob(dir_out_mir_frame+'frameMIR*.nc'))
    ncfilenames_id    = [int(os.path.basename(xx).split('MIR')[1].split('.')[0]) for xx in ncfilenames] 
    ncfilenames_idref = [ [-999] for ii in range(len(ncfilenames))] 
    
    nbrePixelLowSsim       = np.zeros(len(ncfilenames)) - 999
    limit_nbrePixelLowSsim = np.zeros(len(ncfilenames)) - 999

    badMirId = []
    Idtime_firstWarpMir = get_initialIdTime(ifile, ncfilenames, dir_out_mir_georef_npy, georefMode)

    if flag_save_ssim_npy:
        tools.ensure_dir(dir_out_mir_georef_npy.replace('npy','npy_ssim'))
        
    iloop = 0 
    while (True):
        #ep08_arr = []; mirId_arr = []; ssim_80Percentile_arr = []
        ncfilenames_id_new    = [int(os.path.basename(xx).split('MIR')[1].split('.')[0]) for xx in ncfilenames] 
            
        params = []
        results00 = []
        for inc, ncfilename in enumerate(ncfilenames): 
            frameMir = camera_mir.load_existing_file(ncfilename)
            
            idref_ = ncfilenames_id_new[max([0,inc-window_ssimNeighbors]):min([len(ncfilenames)-1,inc+window_ssimNeighbors]) ]
            
            idx = np.where(frameMir.id == ncfilenames_id)[0][0]
            if (frameMir.id in mir_info.id) & (lists_identical(idref_, ncfilenames_idref[idx]) == False):
                params.append([inc, ncfilenames, dir_out_mir_georef_npy, arrivalTime_lwir, georefMode, window_ssimNeighbors,flag_save_ssim_npy])
            elif frameMir.id not in mir_info.id:
                continue
            else:
                results00.append(results_prev[idx])
        
        flag_parallel_ = flag_parallel 
        if flag_parallel_:
            # set up a pool to run the parallel processing
            cpus = tools.cpu_count()
            pool = multiprocessing.Pool(processes=cpus)

            # then the map method of pool actually does the parallelisation  
            results11 = pool.map(star_get_ssim_percentile_to_neighbor, params)
            pool.close()
            pool.join()
           
        else:
            results11 = []
            for param in params:
                print(os.path.basename(param[1][param[0]]))
                sys.stdout.flush()
                results11.append(star_get_ssim_percentile_to_neighbor(param))
        
        results = results11[:] + results00[:]

        mirId_arr              = ([xx[0] for xx in results])
        nbrePixelLowSsim       = ([xx[1] for xx in results])
        limit_nbrePixelLowSsim = ([xx[2] for xx in results])
        
        mirId_arr              = np.array(mirId_arr)
        nbrePixelLowSsim       = np.array(nbrePixelLowSsim)
        limit_nbrePixelLowSsim = np.array(limit_nbrePixelLowSsim)
       
        idx_sorted = np.argsort(mirId_arr)
        mirId_arr              = mirId_arr[idx_sorted]
        nbrePixelLowSsim       = nbrePixelLowSsim[idx_sorted]
        limit_nbrePixelLowSsim = limit_nbrePixelLowSsim[idx_sorted]
        results_copy = results[:]
        results = [results_copy[ii] for ii in idx_sorted]

        window_size = 40
        idx_outlier, [diff_, limit_] = hampel_filter_forloop(nbrePixelLowSsim, limit_nbrePixelLowSsim, 0.5, mirId_arr, window_size, n_sigmas=3., centeredData=True)
        #idx_outlier += idx_outlier_00
        idx_ok      = np.setdiff1d(np.arange(len(mirId_arr)),idx_outlier)
        idx_outlier = [np.array(idx_outlier),]

        fig=plt.figure()
        plt.plot(mirId_arr,diff_)
        plt.plot(mirId_arr,limit_)
        fig.savefig(dir_out_mir_georef_npy+plotname+'_hampel_filter_output_{:03d}.png'.format(iloop))
        plt.close(fig)
        
        np.save(dir_out_mir_georef_npy+plotname+'_hampel_filter_output_{:03d}.npy'.format(iloop),[mirId_arr, nbrePixelLowSsim, limit_nbrePixelLowSsim, diff_, limit_, idx_outlier])
        iloop += 1

        #remove groupe of badid to keep only first one.
        ncfilenames_arr = np.array( [ dir_out_mir_frame+'frameMIR{:06d}.nc'.format(ii) for ii in mirId_arr ] )
        ncfilenames_id    = [int(os.path.basename(xx).split('MIR')[1].split('.')[0]) for xx in ncfilenames_arr] 
        ncfilenames_idref = [ ncfilenames_id[max([0,ii-window_ssimNeighbors]):min([len(ncfilenames)-1,ii+window_ssimNeighbors]) ] for ii in range(len(ncfilenames_arr))] 
        results_prev = results[:]

        if len(idx_outlier[0])!=0:
            outlier = np.zeros_like(nbrePixelLowSsim); outlier[idx_outlier]=1
            outlier = np.ma.array(outlier,mask=1-outlier)
            idx_outlier2 = []
            groups_outlier = tools.group_consecutives(outlier.data, vals_mask=outlier.mask, step=0, flag_output='idx')
            for group_outlier in groups_outlier:
                if len(group_outlier) != 1: 
                    idx_outlier2.append(group_outlier[nbrePixelLowSsim[group_outlier].argmax()])
                else:
                    idx_outlier2.append(group_outlier[0])

            [badMirId.append(mirId_arr[ii]) for ii in idx_outlier2]
            print(badMirId)
            print('----')
            
            idx_ = np.setdiff1d(np.arange(len(ncfilenames_arr)), idx_outlier2) 
            ncfilenames = (ncfilenames_arr[idx_]).tolist()

        else: 
            break
        
    Idtime_firstWarpMir = np.array(Idtime_firstWarpMir)

    badMirId = sorted(badMirId)
    goodMirId = np.setdiff1d(Idtime_firstWarpMir[0,:], badMirId) 
    
    goodMirIdxs = []; [ goodMirIdxs.append(np.where( Idtime_firstWarpMir[0,:] == goodMirId_)[0][0]) for goodMirId_ in goodMirId ]
    badMirIdxs  = []; [ badMirIdxs.append( np.where( Idtime_firstWarpMir[0,:] == badMirId_)[0][0] ) for badMirId_  in badMirId ]

    goodMirTime = Idtime_firstWarpMir[1,:][[np.array(goodMirIdxs),]]
    badMirTime  = Idtime_firstWarpMir[1,:][[np.array(badMirIdxs),] ] if np.array(badMirIdxs).shape[0] > 0 else []




    return [badMirId,badMirTime],[goodMirId,goodMirTime]


#########################################
if __name__ == '__main__':
#########################################
    importlib.reload(camera_tools)
    importlib.reload(tools)
    importlib.reload(spectralTools)
    importlib.reload(warp_lwir_mir) 
    importlib.reload(get_mask_to_remove_missed_burn_area)

    time_start_run = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='this is the driver of the GeorefCam Algo.')
    parser.add_argument('-i','--input', help='Input run name',required=True)
    parser.add_argument('-s','--newStart',  help='True then it uses existing data',required=False)
    parser.add_argument('-cnnV','--cnnV',  help='cnn version, is None use threshold',required=False)
    args = parser.parse_args()

    #define Input
    if args.input.isdigit():
        if args.input == '1':
            runName = 'test'
        else:
            print('number not defined')
            sys.exit()
    else:
        runName = args.input
    cnnV=args.cnnV

    modelType = 'cnn' if (cnnV is not None) else 'thr'

    #inputConfig = imp.load_source('config_'+runName,os.getcwd()+'/../input_config/config_'+runName+'.py')
    inputConfig = importlib.machinery.SourceFileLoader('config_'+runName,os.getcwd()+'/../input_config/config_'+runName+'.py').load_module()

    if socket.gethostname() == 'coreff':
        path_ = 'goulven/data/' # 'Kerlouan/'
        #path_ = 'Kerlouan/data/' # 'Kerlouan/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/paugam/data/','/media/paugam/'+path_)
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/paugam/data/','/media/paugam/'+path_)
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/paugam/data/','/media/paugam/'+path_)
    

    if socket.gethostname() == 'ibo':
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/paugam/data/','/space/paugam/data/')
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/paugam/data/','/space/paugam/data/')
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/paugam/data/','/space/paugam/data/')

    if not os.path.isdir(inputConfig.params_rawData['root']): 
        print('###########')
        print('data are missing, stop here.')
        print('###########')
        sys.exit()

    # input parameters
    params_grid       = inputConfig.params_grid
    params_gps        = inputConfig.params_gps 
    params_lwir_camera     = inputConfig.params_lwir_camera
    params_rawData    = inputConfig.params_rawData
    params_georef     = inputConfig.params_georef

    if 'agema' in params_lwir_camera['camera_name']: 
        import flir as camera

    elif 'optris' in params_lwir_camera['camera_name']:
        import optris as camera
    importlib.reload(camera)

    # control flag
    flag_georef_mode = inputConfig.params_flag['flag_georef_mode']
    flag_parallel = inputConfig.params_flag['flag_parallel']
    flag_restart  = inputConfig.params_flag['flag_restart']
    if flag_georef_mode   == 'WithTerrain'     : 
        georefMode = 'WT'
        params_lwir_camera['dir_input'] = params_lwir_camera['dir_input'][:-1] + '_WT/'
    elif flag_georef_mode == 'SimpleHomography': 
        georefMode = 'SH'
    if args.newStart is not None:
        flag_restart = tools.string_2_bool(args.newStart)

    plotname          = params_rawData['plotname']
    root_postproc     = params_rawData['root_postproc']

    reso = params_lwir_camera['reso_refined'] # 3
    kernel_small_ring_ = 61 if 'ringPlot_refined' not in list(params_lwir_camera.keys()) else params_lwir_camera['ringPlot_refined'] #121# 61#default
    kernel_small_ring = np.ones((old_div(old_div(kernel_small_ring_,reso),2)*2+1,old_div(old_div(kernel_small_ring_,reso),2)*2+1),np.uint8)
    kernel_small_ring00 = np.ones((kernel_small_ring_,kernel_small_ring_),np.uint8) 
    dist_behind00 = 50
    dist_behind   = old_div(dist_behind00,reso) 
    flag_call_mask = 'full'
    #temp_fire  = params_lwir_camera['temp_fire_refined']     # 400#370#400 for sha1     it was 340
    time_behind= params_lwir_camera['time_ff_behind_refined']#15 #40#60#120          it was 40 for sha1 and sku6 aparently
    limit_EP08_i=.99
    
    limit_EP08_F = .99 #limit_EP08_i
    

    if 'temp_threshFire_refined' in list(params_lwir_camera.keys()):
        temp_threshFire = params_lwir_camera['temp_threshFire_refined'] #175 #200  #150 for reso=03 sha3


    diskSize = old_div(params_lwir_camera['diskSize_refined'],reso) 
    dir_out_frame      = root_postproc + params_lwir_camera['dir_input'] + 'Frames/'

    dir_out_refine = root_postproc + params_lwir_camera['dir_input'] + 'Georef_refined_reso{:02d}_SH/'.format(int(reso))   
    dir_out_refine11 = root_postproc + params_lwir_camera['dir_input'] + 'Georef_refined_SH/'   # native resolution # MERDEOLI
  
    
    #00 variable are for raw data. not corrected by first georef loop
    #1  variables are the final state of the variables at reso resolution
    #11 variables are the final state of the variables at full resolution
    
    dir_out_refine_png = dir_out_refine+'/png/'
    dir_out_refine_npy = dir_out_refine+'/npy/'
    dir_out_refine11_png = dir_out_refine11+'/png/'
    dir_out_refine11_npy = dir_out_refine11+'/npy/'
    

    Id2Skip = [] 
    dir_out_georef_npy = root_postproc + params_lwir_camera['dir_input'] + 'Georef_{:s}/npy/'.format(georefMode)
         
    print('##########')
    print('  output dir is set to :', dir_out_refine_npy)
    print('  followind frame id will be skipped: ', Id2Skip)
    print('##########')
    
    if not(flag_restart): 
        res = 'na'
        while res not in {'y','n'}:
            res = input('are you sure you want to delete the existing data in \n \t {:s} \n \t {:s} \nanswer (y/n): '.format(dir_out_refine,dir_out_refine11))
        if res == 'n':
            print('stopped here')
            sys.exit()

    if not(flag_restart):
        print('clean existing output dir')
        if os.path.isdir(dir_out_refine_npy): shutil.rmtree(dir_out_refine_npy)
        if os.path.isdir(dir_out_refine_png): shutil.rmtree(dir_out_refine_png)
        if os.path.isdir(dir_out_refine11_npy): shutil.rmtree(dir_out_refine11_npy)
        if os.path.isdir(dir_out_refine11_png): shutil.rmtree(dir_out_refine11_png)
        if os.path.isfile(dir_out_refine11+'EP08_F.npy'): os.remove(dir_out_refine11+'EP08_F.npy')
        if os.path.isfile(dir_out_refine11+'arrivalTime_lowRes.npy'): os.remove(dir_out_refine11+'arrivalTime_lowRes.npy')

    tools.ensure_dir(dir_out_refine_png)
    tools.ensure_dir(dir_out_refine_npy)
    tools.ensure_dir(dir_out_refine11_png)
    tools.ensure_dir(dir_out_refine11_npy)   

    #read ignition time
    ############
    file_ignition_time = params_rawData['root_data'] + 'ignition_time.dat'
    f = open(file_ignition_time,'r')
    lines = f.readlines()
    ignitionTime = datetime.datetime.strptime(params_rawData['fire_date']+'_'+lines[0].rstrip(), "%Y-%m-%d_%H:%M:%S")
    endTime = datetime.datetime.strptime(params_rawData['fire_date']+'_'+lines[1].rstrip(), "%Y-%m-%d_%H:%M:%S")
    fire_durationTime = (endTime-ignitionTime).total_seconds()


    #load grid 
    ###########################################
    burnplot = np.load(root_postproc+'grid_'+plotname+'.npy')
    burnplot = burnplot.view(np.recarray)
    grid_e, grid_n = burnplot.grid_e, burnplot.grid_n

    #dilate plot mask when image blured as in sha3 
    #kernel = np.ones((5,5),np.uint8)
    #img_ = np.array(np.where(burnplot.mask==2,1,0),dtype=np.uint8)*255
    #mask_ = cv2.dilate(img_, kernel, iterations = 1)    
    #burnplot.mask = np.where(mask_==255,2,0)
    
    burnplot00 = np.copy(burnplot)
    burnplot00 = burnplot00.view(np.recarray)
    

    if reso > 1: 
        diag_res_cte_shape = (int(math.ceil(old_div(1.*burnplot.shape[0],reso))), int(math.ceil(old_div(1.*burnplot.shape[1],reso))))
        burnplot = np.zeros(diag_res_cte_shape,dtype=burnplot00.dtype)
        burnplot = burnplot.view(np.recarray)
        burnplot.grid_e = tools.downgrade_resolution_4nadir(burnplot00.grid_e, diag_res_cte_shape, flag_interpolation='min')
        burnplot.grid_n = tools.downgrade_resolution_4nadir(burnplot00.grid_n, diag_res_cte_shape, flag_interpolation='min')
        burnplot.mask   = np.where( old_div(tools.downgrade_resolution_4nadir(burnplot00.mask  , diag_res_cte_shape, flag_interpolation='average'),2) > .3, 2, 0)

    '''
    kernel_largePlotMask = 111

    kernel = np.ones((old_div(old_div(kernel_largePlotMask,reso),2)*2+1,old_div(old_div(kernel_largePlotMask,reso),2)*2+1),np.uint8)
    img_ = np.array(np.where(burnplot.mask==2,1,0),dtype=np.uint8)*255
    mask_ = cv2.dilate(img_, kernel, iterations = 1)    
    plot_mask_enlarged = np.where(mask_==255,2,0)
    '''
    plot_mask = np.where(burnplot.mask==2,2,0)
    
    img_ = np.array(np.where(burnplot.mask==2,1,0),dtype=np.uint8)*255
    mask_ = cv2.dilate(img_, kernel_small_ring, iterations = 1)    
    plot_mask_enlarged_small = np.where(mask_==255,2,0)
   
    #same for original resolution
    '''
    kernel = np.ones((kernel_largePlotMask,kernel_largePlotMask),np.uint8)
    img_ = np.array(np.where(burnplot00.mask==2,1,0),dtype=np.uint8)*255
    mask_ = cv2.dilate(img_, kernel, iterations = 1)    
    plot_mask_enlarged00 = np.where(mask_==255,2,0)
    '''
    plot_mask00 = np.where(burnplot00.mask==2,2,0)
    
    img_ = np.array(np.where(burnplot00.mask==2,1,0),dtype=np.uint8)*255
    mask_ = cv2.dilate(img_, kernel_small_ring00, iterations = 1)    
    plot_mask_enlarged_small00 = np.where(mask_==255,2,0)

    '''
    mask_localSquare = np.zeros_like(plot_mask)
    boxS = old_div(80,reso)
    nxS, nyS = mask_localSquare[::boxS,::boxS].shape
    arange = np.arange(nxS*nyS).reshape(nxS,nyS)
    for i,j in itertools.product(list(range(boxS)),list(range(boxS))):
        nxS_, nyS_ = mask_localSquare[i::boxS,j::boxS].shape
        mask_localSquare[i::boxS,j::boxS] = arange[:nxS_,:nyS_]
    '''

    #load lwir_info
    ################
    lwir_info = np.load(root_postproc + params_lwir_camera['dir_input'] + params_lwir_camera['dir_img_input']+'filename_time.npy')
    lwir_info = lwir_info.view(np.recarray)

    if   (not os.path.isfile(dir_out_frame+'frame_time_info.npy') )  \
       | (not inputConfig.params_flag['flag_restart']):
        lwir_frame_names = sorted(glob.glob(params_rawData['root_postproc']+params_lwir_camera['dir_input']+'Frames/*.nc' ))
        lwirFrame_info = np.array([(0,0,0,'mm')]*len(lwir_frame_names),dtype=np.dtype([('time_igni',float),('ep08',float),('ssim',float),('filename','U600')]))
        lwirFrame_info = lwirFrame_info.view(np.recarray)    
        for ifile, lwir_frame_name in enumerate(lwir_frame_names):
            frameLwir = camera.load_existing_file(params_lwir_camera, lwir_frame_name)
            lwirFrame_info.time_igni[ifile] = frameLwir.time_igni
            lwirFrame_info.ep08[ifile] = frameLwir.corr_ref00
            lwirFrame_info.ssim[ifile] = frameLwir.ssim
            lwirFrame_info.filename[ifile] = lwir_frame_name
        np.save(dir_out_frame+'frame_time_info',lwirFrame_info)
    else:
        lwirFrame_info =  np.load(dir_out_frame+'frame_time_info.npy')
        lwirFrame_info = lwirFrame_info.view(np.recarray)

    #add id
    lwir_info2 = np.array(lwir_info.shape[0]*[('mm',0,-999)], dtype=np.dtype([('name','U100'),('time',float),('id',int)]))
    lwir_info2 = lwir_info2.view(np.recarray)
    for i, name in enumerate(lwir_info.name): 
        lwir_info2[i].name = lwir_info[i].name
        lwir_info2[i].time = lwir_info[i].time
        if np.where(lwirFrame_info.time_igni == lwir_info2[i].time)[0].shape[0] != 0:
            lwir_info2[i].id   = int(os.path.basename(lwirFrame_info.filename[np.where(lwirFrame_info.time_igni == lwir_info2[i].time)][0]).split('me')[1].split('.')[0])
    lwir_info = lwir_info2[np.where(lwir_info2.id>=0)]


    #load npy georef 
    ###########################################    
    lwir_geo_names = sorted(glob.glob(dir_out_georef_npy+'*SH.npy'))
    
    mask_type_ = inputConfig.params_lwir_camera['mask_burnNobun_type_refined']
    mask_type_val = inputConfig.params_lwir_camera['mask_burnNobun_type_val_refined']

    if os.path.isfile(os.path.dirname(lwir_geo_names[0])+'/mask_nodata_{:s}_georef.npy'.format(plotname)): 
        mask_burnNoburn00 = np.load(os.path.dirname(lwir_geo_names[0])+'/mask_nodata_{:s}_georef.npy'.format(plotname))
    else:
        print('#### mask_burnNoburn not found. now generated with mask type:', mask_type_)
        mask_burnNoburn00 =  get_mask_to_remove_missed_burn_area.get_mask(inputConfig, burnplot00, georefMode, 
                                                                        dir_out_georef_npy,
                                                                        mask_type_, mask_type_val, flag_plot=False)
    if mask_burnNoburn00.size != plot_mask.size: 
        diag_res_cte_shape = (int(math.ceil(old_div(1.*mask_burnNoburn00.shape[0],reso))), int(math.ceil(old_div(1.*mask_burnNoburn00.shape[1],reso))))
        mask_burnNoburn   = np.where( (tools.downgrade_resolution_4nadir(mask_burnNoburn00  , diag_res_cte_shape, flag_interpolation='average') > .3), 1, 0)
    else:
        mask_burnNoburn = mask_burnNoburn00

    if mask_burnNoburn.max()> 0: 
        plot_mask = np.where(mask_burnNoburn==1,0,plot_mask)
        plot_mask00 = np.where(mask_burnNoburn00==1,0,plot_mask00)


    #plot perimeter
    img_, contours,hierarchy  = cv2.findContours(np.array(np.where(plot_mask==2,1,0), dtype=np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    plotLocation = np.zeros_like(burnplot.mask)
    for ic, contour in enumerate(contours):
        #if  hierarchy[0,ic,-1] != -1: continue
        polygon =[ tuple( pt[0] ) for pt in contour ]
        if len(contour)>3:
            img = Image.new('L', burnplot.shape , 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=0)
            plotLocation = np.where( plotLocation!=1, plotLocation+np.copy(img), plotLocation )
    idx_plot = np.dstack(np.where(plotLocation==1))[0]
    tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(idx_plot)
    
    img_, contours,hierarchy  = cv2.findContours(np.array(np.where(plot_mask00==2,1,0), dtype=np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    plotLocation = np.zeros_like(burnplot00.mask)
    for ic, contour in enumerate(contours):
        #if  hierarchy[0,ic,-1] != -1: continue
        polygon =[ tuple( pt[0] ) for pt in contour ]
        if len(contour)>3:
            img = Image.new('L', burnplot00.shape , 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=0)
            plotLocation = np.where( plotLocation!=1, plotLocation+np.copy(img), plotLocation )
    idx_plot = np.dstack(np.where(plotLocation==1))[0]
    tree00 = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(idx_plot)


    srf_file = '../data_static/Camera/'+inputConfig.params_lwir_camera['camera_name'].split('_')[0]+'/SpectralResponseFunction/'+inputConfig.params_lwir_camera['camera_name'].split('_')[0]+'.txt'
    wavelength_resolution = 0.01
    param_set_radiance = [srf_file, wavelength_resolution]
    param_set_temperature = spectralTools.get_tabulated_TT_Rad(srf_file, wavelength_resolution)

    lk_params = dict( winSize  = (21,21),
                      maxLevel = 7,  
                      criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1000, 0.01))


    flag_plot_diagram = False 
    if flag_plot_diagram:
        igeo_plot = 650                
        dir_out_img_diagram = root_postproc + params_lwir_camera['dir_input'] + 'ImageDiagram/frame{:06d}/'.format(igeo_plot)
        tools.ensure_dir(dir_out_img_diagram)


    ############################################
    # wrap lwir frame on previous
    ############################################
    if (not(flag_restart)) | (not(os.path.isfile(dir_out_refine11+'EP08_F.npy'))):
        
        arrivalTime = np.zeros_like(burnplot.mask)-999    ; arrivalTime[np.where(mask_burnNoburn==1)]=0

        EP_id = []; EP_id_ref = []; EP_plotUncovered = []; EP08_0_prev = []; EP08_1_prev = []; EP08_2_prev = []; correc_ecc = []; EP08_F_arr=[]; status_arr=[]; pixErr_F_arr=[]
        #test_ok_metric_arr = []
        id_ref_arr = []; id_prev_arr = [] 
        length_tail_ringCorr = 5   # tail of ssim2F_mean = 2xlength_tail_ringCorr  and loop on iref is length_tail_ringCorr long
        tailStart = 2 # no refine before  igeostart+tailStart
        tailRef = 20  # tail to select ref  from -2*tailRef_here to tailRef_here/2
        flag_again = False
        ii_geo=0
        iprev =0
        iprev_ssim =0
        ii_lag_ssim = 4 # TO DEFINE IN INPUTCONFIG
        iref = 0
        igeostart = inputConfig.params_lwir_camera['igeostart_refined'] if ('igeostart_refined' in list(inputConfig.params_lwir_camera.keys()) ) \
                                                                        else 0
        time_aheadOf = 0 if 'time_aheadOf_refined' not in list(params_lwir_camera.keys()) else params_lwir_camera['time_aheadOf_refined']
        ssim2dF_mean_arr = [[],[]] 
        ssim2dF_mean = np.zeros_like(arrivalTime); ssim2dF_mean[np.where(plot_mask_enlarged_small==2)] = 1
        ssim2dF = np.copy(ssim2dF_mean)
        #ii_ssim2dF = 0
        ssim2D_thres   = None  if 'ssim2D_thres_refined'   not in list(inputConfig.params_lwir_camera.keys()) else inputConfig.params_lwir_camera['ssim2D_thres_refined']
        ssim2dF_filter = False if 'ssim2dF_filter_refined' not in list(inputConfig.params_lwir_camera.keys()) else inputConfig.params_lwir_camera['ssim2dF_filter_refined']
        flag_change_iref = False
        EP08_F_00 = 1
        while (ii_geo < len(lwir_geo_names)): #MERDEOLI

            if (len(EP08_0_prev) != len(EP08_1_prev)) | (len(EP08_0_prev) != len(EP08_F_arr)) | (len(EP08_1_prev) != len(EP08_F_arr)) :
                print('## pb in array lenght')
                sys.exit()

            igeo = int(os.path.basename(lwir_geo_names[ii_geo]).split('_')[-2])
            
            if igeo in Id2Skip : 
                ii_geo += 1
                continue 
            print(igeo, end=' ') 
            
            #load georef and frame
            frame = camera.load_existing_file(params_lwir_camera, dir_out_frame+'frame{:06d}.nc'.format(igeo))
            frame_info, homogra_mat_ori, frame_time_igni,  rad, temp, maskfull = load_georefnpy(lwir_geo_names[ii_geo], frame)
            time_igni = frame_info[1] 
            temp = np.array(temp,dtype=np.float32)
          
            radiance00   = camera.return_radiance(frame, srf_file, wavelength_resolution=0.01) 
            maskfull00  = frame.mask_img 
            temp00      = frame.temp
            scaling_mat = np.identity(3)
            
            #load temp_prev_ssim
            if igeo != igeostart: 
                if iprev_ssim in Id2Skip:  
                    iprev_ssim__ = iprev_ssim -1
                    flag_loaded_ = False
                    while not(flag_loaded_): 
                        try: 
                            _ , _, _, mask_prev_ssim, _, temp_prev_ssim, _, _ = \
                                      np.load( dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,iprev_ssim__,georefMode), allow_pickle=True)
                            iprev_ssim = iprev_ssim__
                            flag_loaded_ = True 
                        except: 
                            iprev_ssim__ -= 1
                else:
                    _ , _, _, mask_prev_ssim, _, temp_prev_ssim, _, _ = \
                              np.load( dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,iprev_ssim,georefMode), allow_pickle=True)

            #time in seconds before the one the inside mask is not used. this is to avoid being corrupted by slow moving backfire that are lighted at the start of the burn.
            if time_igni <= time_aheadOf: 
                maskfull00  = np.where( cv2.warpPerspective( mask_burnNoburn00, np.linalg.inv(homogra_mat_ori), maskfull00.shape[::-1], flags=cv2.INTER_NEAREST)==1,0,maskfull00)


            if temp.size != plot_mask.size: 
                diag_res_cte_shape = (int(math.ceil(old_div(1.*rad.shape[0],reso))), int(math.ceil(old_div(1.*rad.shape[1],reso))))
                rad = tools.downgrade_resolution_4nadir(rad, diag_res_cte_shape, flag_interpolation='conservative')
                maskfull     = np.array(np.round(
                                              tools.downgrade_resolution_4nadir(np.array(maskfull,dtype=float), diag_res_cte_shape, flag_interpolation='average')
                                              ,0),
                                           dtype=np.uint8)
                rad = np.where(rad<param_set_temperature.radiance.min(),0,rad)
                temp     = spectralTools.conv_Rad2Temp(rad, param_set_temperature)
                scaling_mat[0,0] = reso; scaling_mat[1,1] = reso
                homogra_mat_ori = np.linalg.inv(scaling_mat).dot(homogra_mat_ori)
           
           
            #check that the plot is in the field of the view and not masked
            ratio_uncovered_plot_mask = old_div(1.*np.where(maskfull[np.where(plot_mask==2)]==0)[0].shape[0],\
                                           np.where(maskfull[np.where(plot_mask==2)]==1)[0].shape[0])
            
            
            if time_igni <= time_aheadOf: 
                maskfull    = np.where(mask_burnNoburn==1,0,maskfull)

            maskfull_    = maskfull   
            temperature_ = temp     
            temp, maskfull, gtemp, mask, arrivalTime_here = get_radgradmask(params_lwir_camera, temp, maskfull, True, tree, burnplot, plot_mask, plot_mask_enlarged_small, 
                                                                         arrivalTime, frame_time_igni, ssim2dF_mean, dist_behind=dist_behind,
                                                                         temp_fire=temp_threshFire,time_behind_input=time_behind, 
                                                                         ssim2D_thres=ssim2D_thres, ssim2dF_filter=ssim2dF_filter, diskSize=diskSize)
            #define ref at fisrt iteration
            if ii_geo == 0: 
                rad_ref, temp_ref, maskfull_ref, mask_ref, gtemp_ref = rad, temp, maskfull, mask, gtemp 


            #-------------------  
            if igeo < igeostart: 
            #-------------------  
                #if restart
                if flag_restart & os.path.isfile(dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,igeo,georefMode)): 
                    [iref, iprev, frame_info, corr_status, EP08_0, EP08_1, correc_ecc_, EP08_F, pixErr_F], _, rad1, mask1, maskfull1, temp1, arrivalTime, ssim2dF = \
                            np.load( dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,igeo,georefMode))
                    _ , _, rad_prev, mask_prev, maskfull_prev, temp_prev,   _, _ = np.load( dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,iprev,georefMode))
                    frame_prev = camera.load_existing_file(params_lwir_camera, dir_out_frame+'frame{:06d}.nc'.format(iprev))
                
                    iprev    = igeo
                    iprev_ssim = max([  int(os.path.basename(lwir_geo_names[max([ii_geo-ii_lag_ssim,0])]).split('_')[-2]), 0])
                    ii_geo += 1 
                    print('< igeostart={:d} loaded'.format(igeostart))
                    continue
                
                mask_fire,arrivalTime = runThreshold(temp, temp_threshFire, plot_mask, time_igni, arrivalTime)
                
                #save native resolution
                shape_     = plot_mask00.shape
                rad11      = cv2.warpPerspective( radiance00, np.linalg.inv(scaling_mat).dot(homogra_mat_ori), shape_, flags=cv2.INTER_LINEAR)
                rad11      = np.where(rad11<param_set_temperature.radiance.min(),0,rad11)
                
                maskfull11 = cv2.warpPerspective( maskfull00, np.linalg.inv(scaling_mat).dot(homogra_mat_ori), shape_, flags=cv2.INTER_NEAREST)
                
                temp11 = spectralTools.conv_Rad2Temp( rad11, param_set_temperature )
                
                ssim2dF_mean00 = ndimage.zoom(ssim2dF_mean, reso, order=0, cval=0)[:temp11.shape[0],:temp11.shape[1]]
                arrivalTime00  = ndimage.zoom(arrivalTime, reso, order=0, cval=0)[:temp11.shape[0],:temp11.shape[1]]
                
                #temp11, maskfull11, gtemp11, mask11, arrivalTime11_here = get_radgradmask(params_lwir_camera, temp11, maskfull11, True, tree00, burnplot00, plot_mask00, plot_mask_enlarged_small00,
                #                                                                           arrivalTime00, frame_time_igni, ssim2dF_mean00,
                #                                                                           dist_behind=dist_behind, 
                #                                                                           temp_fire=temp_threshFire, time_behind_input=time_behind, 
                #                                                                           ssim2D_thres=ssim2D_thres, ssim2dF_filter=ssim2dF_filter, diskSize=diskSize) 
                
                #np.save(dir_out_refine11_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),  \
                #                                       np.array([frame_info, np.linalg.inv(scaling_mat).dot(homogra_mat_ori), 
                #                                                 maskfull11, temp11, rad11, mask11], dtype=object) )
                
                np.save(dir_out_refine11_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),  \
                                                       np.array([frame_info, np.linalg.inv(scaling_mat).dot(homogra_mat_ori), 
                                                                 maskfull11, temp11, rad11,], dtype=object) )

                corr_status = '<igeostart'
                EP08_0_prev.append(tools.get_EP08_from_img(temp, temp_ref, inputMask=mask, inputMask_ref=mask_ref))
                EP08_1_prev.append(EP08_0_prev[-1])
                EP08_F_arr.append(EP08_0_prev[-1])
                np.save(dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),\
                        np.array([[iref,iprev,frame_info,
                                  corr_status,EP08_0_prev[-1],EP08_1_prev[-1],0,EP08_F_arr[-1],-999],
                                 np.linalg.inv(scaling_mat).dot(homogra_mat_ori),rad, mask, maskfull, temp, arrivalTime, ssim2dF,
                        ], dtype=object))

                #plot png native reso
                mpl.rcdefaults()
                mpl.rcParams['text.usetex'] = True
                mpl.rcParams['font.size'] = 16.
                mpl.rcParams['axes.linewidth'] = 1
                mpl.rcParams['axes.labelsize'] = 14.
                mpl.rcParams['xtick.labelsize'] = 14.
                mpl.rcParams['ytick.labelsize'] = 14.
                mpl.rcParams['figure.subplot.left'] = .0
                mpl.rcParams['figure.subplot.right'] = 1.
                mpl.rcParams['figure.subplot.top'] = 1.
                mpl.rcParams['figure.subplot.bottom'] = 0.
                mpl.rcParams['figure.subplot.hspace'] = 0.
                mpl.rcParams['figure.subplot.wspace'] = 0.0
                fig = plt.figure(figsize=(8.,8))
                ax = plt.subplot(111)
                idx_plot =np.where(burnplot00.mask==2)
                buff=20
                xmin,xmax,ymin,ymax= max([0,idx_plot[0].min()-buff]), min([temp11.shape[0]-1,idx_plot[0].max()+buff]), max([0,idx_plot[1].min()-buff]), min([temp11.shape[1]-1,idx_plot[1].max()+buff])
                ax.imshow(np.ma.masked_where(maskfull11==0,temp11)[xmin:xmax,ymin:ymax].T,origin='lower',vmin=290,vmax=500)
                fig.savefig(dir_out_refine11_png + '{:s}_georef2nd_{:06d}_{:s}.png'.format(plotname,igeo,georefMode) )
                plt.close(fig) 

                iprev    = igeo
                iprev_ssim = max([  int(os.path.basename(lwir_geo_names[max([ii_geo-ii_lag_ssim,0])]).split('_')[-2]), 0])
                ii_geo += 1 
                print('< igeostart={:d}'.format(igeostart))
                continue

            #-------------------  
            # check if file already exists
            #-------------------  
            if flag_restart & os.path.isfile(dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,igeo,georefMode)): 
                
                [iref_new, iprev_new, frame_info, corr_status, EP08_0, EP08_1, correc_ecc_, EP08_F, pixErr_F], _, rad1, mask1, maskfull1, temp1, arrivalTime_here, ssim2dF= \
                        np.load( dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,igeo,georefMode), allow_pickle=True)
                info_, _, rad_ref, mask_ref, maskfull_ref, temp_ref, _, _ = np.load( dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,iref_new,georefMode), allow_pickle=True)
                rad_ref  = np.array(rad_ref, dtype=np.float32)
                temp_ref = np.array(temp_ref,dtype=np.float32)
                EP08_F_00 = info_[-2]
                
                iref = iref_new

                EP_id.append(igeo)
                EP_id_ref.append(iref)
                EP_plotUncovered.append(ratio_uncovered_plot_mask)  
                EP08_0_prev.append(EP08_0)

                EP08_1_prev.append(EP08_1)
                status_arr.append(corr_status)
                correc_ecc.append(correc_ecc_)
                EP08_F_arr.append(EP08_F)
                pixErr_F_arr.append(pixErr_F)

                iprev = iprev_new
               
                #update ssim2dF_mean
                ssim2dF_mean_arr[0].append(ssim2dF); ssim2dF_mean_arr[1].append(maskfull1) 
                if len(ssim2dF_mean_arr[0]) > 2*length_tail_ringCorr:
                    ssim2dF_mean_arr[0].pop(0); ssim2dF_mean_arr[1].pop(0)
               

                ssim2dF_nan = np.where(np.array(ssim2dF_mean_arr[1])==1, np.array(ssim2dF_mean_arr[0]), np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ssim2dF_mean = np.nanmean(ssim2dF_nan, axis=0)
                ssim2dF_mean = np.ma.array(np.where(np.isnan(ssim2dF_mean),1.e-6,ssim2dF_mean), mask=np.where(np.isnan(ssim2dF_mean),True, False))

                if iprev_new not in id_prev_arr: id_prev_arr.append(iprev_new)
               
                if (iprev_new == igeo-1) | (igeo==0):
                    maskfull_prev = maskfull1
                    rad_prev      = rad1
                    mask_prev     = mask1
                    temp_prev      = temp1
                    frame_prev    = frame
                    time_igni_prev = time_igni
                else: 
                    _ , _, rad_prev, mask_prev, maskfull_prev, temp_prev, _, _ = np.load( dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,iprev,georefMode), allow_pickle=True)
                    frame_prev = camera.load_existing_file(params_lwir_camera, dir_out_frame+'frame{:06d}.nc'.format(iprev))
              

                if len(EP08_F_arr)<=1:                                                           arrivalTime = arrivalTime_here
                elif  (old_div(np.abs(EP08_F_arr[-1] - EP08_F_arr[-2]), EP08_F_arr[-2]) ) < .1 : arrivalTime = arrivalTime_here
                
                if (flag_plot_diagram) :
                    if (igeo == igeo_plot):
                        #plot for flowChart
                        
                        # temp1
                        #---------------------
                        nx, ny = temp1.shape 
                        mpl.rcdefaults()
                        mpl.rcParams['text.usetex'] = True
                        mpl.rcParams['font.size'] = 38
                        mpl.rcParams['axes.linewidth'] = 1
                        mpl.rcParams['axes.labelsize'] = 14.
                        mpl.rcParams['xtick.labelsize'] = 38.
                        mpl.rcParams['ytick.labelsize'] = 14.
                        mpl.rcParams['figure.subplot.left'] = .0
                        mpl.rcParams['figure.subplot.right'] = 1.
                        mpl.rcParams['figure.subplot.top'] = 1.
                        mpl.rcParams['figure.subplot.bottom'] = .0
                        mpl.rcParams['figure.subplot.hspace'] = 0.02
                        mpl.rcParams['figure.subplot.wspace'] = 0.02
                        fig= plt.figure(figsize=(8,8./(old_div(1.*nx,ny)))) 
                        ax = plt.subplot(111)
                        im = ax.imshow(temp1.T, origin='lower',vmin=300,vmax=750)
                        cax = fig.add_axes([0.066, 0.9, 0.75, 0.05])
                        
                        cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(300,750,5))
                        cb.ax.set_xticklabels( [ r'${:3.0f}$'.format(xx) for xx in np.linspace(300,750,5)], color='white')
                        cb.ax.xaxis.set_tick_params(color='white')
                        cb.outline.set_edgecolor('white')
                        
                        
                        #cb = fig.colorbar(im, cax=cax, orientation='horizontal')
                        #cb.ax.xaxis.set_tick_params(color='white',labelsize=48)
                        #cb.ax.locator_params(nbins=5)
                        #cb.outline.set_edgecolor('white')
                        #plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='white')
                        
                        fig.text(0.84, 0.9,u'$T (K)$',color='white')
                        ax.set_axis_off()
                        fig.savefig(dir_out_img_diagram+'temp.png')
                        plt.close(fig)


                        #mask1
                        #---------------------
                        mpl.rcdefaults()
                        mpl.rcParams['text.usetex'] = True
                        mpl.rcParams['font.size'] = 16.
                        mpl.rcParams['axes.linewidth'] = 1
                        mpl.rcParams['axes.labelsize'] = 14.
                        mpl.rcParams['xtick.labelsize'] = 14.
                        mpl.rcParams['ytick.labelsize'] = 14.
                        mpl.rcParams['figure.subplot.left'] = .0
                        mpl.rcParams['figure.subplot.right'] = 1.
                        mpl.rcParams['figure.subplot.top'] = 1.
                        mpl.rcParams['figure.subplot.bottom'] = .0
                        mpl.rcParams['figure.subplot.hspace'] = 0.02
                        mpl.rcParams['figure.subplot.wspace'] = 0.02
                        fig= plt.figure(figsize=(8,8./(old_div(1.*nx,ny)))) 
                        ax = plt.subplot(111)
                        im = ax.imshow( (1-mask1).T, origin='lower', cmap=mpl.cm.Greys) 
                        ax.set_axis_off()
                        
                        fig.savefig(dir_out_img_diagram+'grid_mask.png')
                        plt.close(fig) 
                       

                        #arrvial time
                        #---------------------
                        mpl.rcdefaults()
                        mpl.rcParams['text.usetex'] = True
                        mpl.rcParams['font.size'] = 38.
                        mpl.rcParams['legend.fontsize'] = 38
                        mpl.rcParams['axes.linewidth'] = 1
                        mpl.rcParams['axes.labelsize'] = 14.
                        mpl.rcParams['xtick.labelsize'] = 38.
                        mpl.rcParams['ytick.labelsize'] = 14.
                        mpl.rcParams['figure.subplot.left'] = .0
                        mpl.rcParams['figure.subplot.right'] = 1.
                        mpl.rcParams['figure.subplot.top'] = 1.
                        mpl.rcParams['figure.subplot.bottom'] = .0
                        mpl.rcParams['figure.subplot.hspace'] = 0.02
                        mpl.rcParams['figure.subplot.wspace'] = 0.02
                        fig= plt.figure(figsize=(8,8./(old_div(1.*nx,ny)))) 
                        ax = plt.subplot(111)
                        idx_ok = np.where((plot_mask==2)&(arrivalTime>=0))
                        im = ax.imshow( np.ma.masked_where((plot_mask!=2)|(arrivalTime<0),arrivalTime).T, origin='lower',cmap=mpl.cm.jet,vmin=arrivalTime[idx_ok].min(),vmax=arrivalTime[idx_ok].max())
                        ax.set_axis_off()
                        
                        cax = fig.add_axes([0.065, 0.9, 0.75, 0.05])
                        cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(arrivalTime[idx_ok].min(),arrivalTime[idx_ok].max(),5))
                        cb.ax.set_xticklabels( [ r'${:3.0f}$'.format(xx) for xx in np.linspace(arrivalTime[idx_ok].min(),arrivalTime[idx_ok].max(),5)] )
                        #cb.ax.xaxis.set_tick_params(color='k')
                        #cb.ax.locator_params(tight=True, nbins=4)
                        
                        #cb.outline.set_edgecolor('k')
                        #plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='k')
                        fig.text(0.87, 0.9,u'$t (s)$',color='k')
                        
                        fig.savefig(dir_out_img_diagram+'arrivalTime.png')
                        plt.close(fig)

                        
                        #local normalized temp with mask
                        #---------------------
                        mpl.rcdefaults()
                        mpl.rcParams['text.usetex'] = True
                        mpl.rcParams['font.size'] = 38.
                        mpl.rcParams['legend.fontsize'] = 38
                        mpl.rcParams['axes.linewidth'] = 1
                        mpl.rcParams['axes.labelsize'] = 14.
                        mpl.rcParams['xtick.labelsize'] = 38.
                        mpl.rcParams['ytick.labelsize'] = 14.
                        mpl.rcParams['figure.subplot.left'] = .0
                        mpl.rcParams['figure.subplot.right'] = 1.
                        mpl.rcParams['figure.subplot.top'] = 1.
                        mpl.rcParams['figure.subplot.bottom'] = .0
                        mpl.rcParams['figure.subplot.hspace'] = 0.02
                        mpl.rcParams['figure.subplot.wspace'] = 0.02
                        fig= plt.figure(figsize=(8,8./(old_div(1.*nx,ny)))) 
                        ax = plt.subplot(111)

                        temp1_     = local_normalization(temp1    , mask1,    diskSize=diskSize)
                        im = ax.imshow(np.ma.masked_where(mask1==0,temp1_).T,origin='lower')
                        ax.set_axis_off()
                        
                        vmin = np.ma.masked_where(mask1==0,temp1_).min()
                        vmax = np.ma.masked_where(mask1==0,temp1_).max()

                        cax = fig.add_axes([0.063, 0.9, 0.7, 0.05])
                        cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(vmin,vmax,5))
                        cb.ax.set_xticklabels( [ r'${:3.1f}$'.format(xx) for xx in np.linspace(vmin,vmax,5)] )
                        
                        fig.text(0.8, 0.89,r"$T_{LN} ($"+'-'+r"$)$",color='k')
                        
                        fig.savefig(dir_out_img_diagram+'temp_LN.png')
                        plt.close(fig)
                        sys.exit()

                print(' ref_{:04d}  prev_{:04d} prev_ssim {:04d} loaded'.format(iref,iprev, iprev_ssim))

                iprev = igeo 
                iprev_ssim = max([  int(os.path.basename(lwir_geo_names[max([ii_geo-ii_lag_ssim,0])]).split('_')[-2]), 0])
                ii_geo += 1
                
                continue


            #-------------------  
            # compute mask and arrivalTime
            #-------------------  
            '''
            temp, maskfull, gtemp, mask, arrivalTime_here = get_radgradmask(params_lwir_camera, temp, maskfull, True, tree, burnplot, plot_mask, plot_mask_enlarged_small, 
                                                                             arrivalTime, frame_time_igni,ssim2dF_mean, dist_behind=dist_behind,
                                                                             temp_fire=temp_threshFire,time_behind_input=time_behind, 
                                                                             ssim2D_thres=ssim2D_thres, ssim2dF_filter=ssim2dF_filter, diskSize=diskSize)
            '''

            #-------------------  
            if igeo == igeostart : 
            #-------------------  

                rad1,    temp1,    maskfull1,    mask1,    gtemp1    = rad, temp, maskfull, mask, gtemp
                rad_ref, temp_ref, maskfull_ref, mask_ref, gtemp_ref = rad, temp, maskfull, mask, gtemp 
                iref = igeo
                corr_status = 'applied'
                
                EP_id.append(igeo)
                EP_id_ref.append(iref)
                EP_plotUncovered.append(ratio_uncovered_plot_mask)  
                EP08_0_prev.append(1)
                EP08_1_prev.append(1)
                status_arr.append(corr_status)
                correc_ecc.append(0)
                EP08_F_arr.append(1)
                EP08_F_00 = 1
                pixErr_F_arr.append(0)
                np.save(dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),\
                        np.array([[iref,iprev,frame_info,
                         corr_status,EP08_0_prev[-1],EP08_1_prev[-1],0,EP08_F_arr[-1],pixErr_F_arr[-1]],
                         scaling_mat.dot(homogra_mat_ori),rad, mask, maskfull, temp, arrivalTime_here, ssim2dF,
                        ],dtype=object))
                #save native resolution
                shape_     = plot_mask00.shape
                rad11      = cv2.warpPerspective( radiance00, scaling_mat.dot(homogra_mat_ori), shape_, flags=cv2.INTER_LINEAR)
                rad11      = np.where(rad11<param_set_temperature.radiance.min(),0,rad11)
                
                maskfull11 = cv2.warpPerspective( maskfull00, scaling_mat.dot(homogra_mat_ori), shape_, flags=cv2.INTER_NEAREST)
                
                temp11 = spectralTools.conv_Rad2Temp( rad11, param_set_temperature )
                
                ssim2dF_mean00 = ndimage.zoom(ssim2dF_mean, reso, order=0, cval=0)[:temp11.shape[0],:temp11.shape[1]]
                arrivalTime00  = ndimage.zoom(arrivalTime, reso, order=0, cval=0)[:temp11.shape[0],:temp11.shape[1]]
                
                #temp11, maskfull11, gtemp11, mask11, arrivalTime11_here = get_radgradmask(params_lwir_camera, temp11, maskfull11, True, tree00, burnplot00, plot_mask00, plot_mask_enlarged_small00,
                #                                                                           arrivalTime00, frame_time_igni, ssim2dF_mean00,
                #                                                                           dist_behind=dist_behind, 
                #                                                                           temp_fire=temp_threshFire, time_behind_input=time_behind, 
                #                                                                           ssim2D_thres=ssim2D_thres, ssim2dF_filter=ssim2dF_filter, diskSize=diskSize) 
                #
                #np.save(dir_out_refine11_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),  \
                #                                       np.array([frame_info, scaling_mat.dot(homogra_mat_ori),
                #                                        maskfull11, temp11, rad11, mask11],dtype=object) )
                np.save(dir_out_refine11_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),  \
                                                       np.array([frame_info, scaling_mat.dot(homogra_mat_ori),
                                                        maskfull11, temp11, rad11,],dtype=object) )
                #plot png native reso
                mpl.rcdefaults()
                mpl.rcParams['text.usetex'] = True
                mpl.rcParams['font.size'] = 16.
                mpl.rcParams['axes.linewidth'] = 1
                mpl.rcParams['axes.labelsize'] = 14.
                mpl.rcParams['xtick.labelsize'] = 14.
                mpl.rcParams['ytick.labelsize'] = 14.
                mpl.rcParams['figure.subplot.left'] = .0
                mpl.rcParams['figure.subplot.right'] = 1.
                mpl.rcParams['figure.subplot.top'] = 1.
                mpl.rcParams['figure.subplot.bottom'] = 0.
                mpl.rcParams['figure.subplot.hspace'] = 0.
                mpl.rcParams['figure.subplot.wspace'] = 0.0
                fig = plt.figure(figsize=(8.,8))
                ax = plt.subplot(111)
                idx_plot =np.where(burnplot00.mask==2)
                buff=20
                xmin,xmax,ymin,ymax= max([0,idx_plot[0].min()-buff]), min([temp11.shape[0]-1,idx_plot[0].max()+buff]), max([0,idx_plot[1].min()-buff]), min([temp11.shape[1]-1,idx_plot[1].max()+buff])
                ax.imshow(np.ma.masked_where(maskfull11==0,temp11)[xmin:xmax,ymin:ymax].T,origin='lower',vmin=290,vmax=500)
                fig.savefig(dir_out_refine11_png + '{:s}_georef2nd_{:06d}_{:s}.png'.format(plotname,igeo,georefMode) )
                plt.close(fig)  

                temp_prev = temp_ref
                rad_prev  = rad_ref
                maskfull_prev = maskfull_ref
                mask_prev     = mask_ref
                time_igni_prev = time_igni
                frame_prev = frame.copy()
                arrivalTime = np.copy(arrivalTime_here)
                id_prev_arr.append(iprev)
                print('')
                iprev_ssim = max([  int(os.path.basename(lwir_geo_names[max([ii_geo-ii_lag_ssim,0])]).split('_')[-2]), 0])
                iprev    = igeo
                ii_geo += 1
                #test_ok_metric_arr.append(-999)
                
                continue
            

            #update ref variable
            #------------------
            tailRef_here = tailRef
            
            if not(flag_change_iref ):
                update_ref_status = ''
            
            elif (flag_change_iref) & (ii_geo>tailRef_here):
                iref_new = EP_id[-2*tailRef_here:old_div(-tailRef_here,2)][ np.array(EP08_F_arr[-2*tailRef_here:old_div(-tailRef_here,2)]).argmax() ] # changeTail
                
                if iref_new != iref:
                    if EP_plotUncovered[iref_new]  < 0.02:
                        data_ref = np.load(dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,iref,georefMode), allow_pickle=True)
                        _, _, rad_ref, mask_ref, maskfull_ref, temp_ref, _, _ =  data_ref
                        
                        iref = iref_new
                        rad_ref = np.array(rad_ref,dtype=np.float32)
                        gtemp_ref = np.array(tools.get_gradient(temp_ref)[0],dtype=np.float32) 
                        update_ref_status = ' ## update ref: done'
                        flag_change_iref = False
                    else: 
                        update_ref_status = ' ## update ref: plot_mask is not covered, ratio = {:.4f}'.format(ratio_uncovered_plot_mask)
                else: 
                    update_ref_status = ' ## update ref: last ref is still the best'
            else: 
                update_ref_status = ' ## update ref: wait for ii_geo > {:d}'.format(tailRef_here)


            #print('ref_{:04d} prev_{:04d} |  {:}'.format(iref, iprev, update_ref_status))
            print('ref_{:04d} prev_{:04d} prev_ssim {:04d} |  {:}'.format(iref,iprev, iprev_ssim, update_ref_status))


            EP08_0 = tools.get_EP08_from_img(temp, temp_ref, inputMask=mask, inputMask_ref=mask_ref)
            EP08_0_prev.append(EP08_0)
            EP_id.append(igeo)
            EP_id_ref.append(iref)
            EP_plotUncovered.append(ratio_uncovered_plot_mask)  
            

            #-------------------  
            if igeo < igeostart+tailStart:
            #-------------------  
                rad1, temp1, mask1, maskfull1, gtemp1 = rad, temp, mask, maskfull, gtemp
                
                EP08_1_prev.append(EP08_0)
                EP08_F_arr.append(EP08_0)
                temp1, mask1, maskfull1, gtemp1 = temp, mask, maskfull, gtemp
                corr_status = 'applied'
                status_arr.append(corr_status)
                np.save(dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),
                        np.array([[iref,iprev,frame_info,corr_status,EP08_0_prev[-1],EP08_1_prev[-1],0,EP08_F_arr[-1],pixErr_F_arr[-1]],\
                         scaling_mat.dot(homogra_mat_ori),rad,mask1,maskfull1,temp1,arrivalTime_here,ssim2dF,
                        ],dtype=object))
                #save native resolution
                shape_     = plot_mask00.shape
                rad11      = cv2.warpPerspective( radiance00, scaling_mat.dot(homogra_mat_ori), shape_, flags=cv2.INTER_LINEAR)
                rad11      = np.where(rad11<param_set_temperature.radiance.min(),0,rad11)
                
                maskfull11 = cv2.warpPerspective( maskfull00, scaling_mat.dot(homogra_mat_ori), shape_, flags=cv2.INTER_NEAREST)
                
                temp11 = spectralTools.conv_Rad2Temp( rad11, param_set_temperature )
                
                ssim2dF_mean00 = ndimage.zoom(ssim2dF_mean, reso, order=0, cval=0)[:temp11.shape[0],:temp11.shape[1]]
                arrivalTime00  = ndimage.zoom(arrivalTime, reso, order=0, cval=0)[:temp11.shape[0],:temp11.shape[1]]
                
                #temp11, maskfull11, gtemp11, mask11, arrivalTime11_here = get_radgradmask(params_lwir_camera, temp11, maskfull11, True, tree00, burnplot00, plot_mask00, plot_mask_enlarged_small00,
                #                                                                           arrivalTime00, frame_time_igni, ssim2dF_mean00,
                #                                                                           dist_behind=dist_behind, 
                #                                                                           temp_fire=temp_threshFire, time_behind_input=time_behind, 
                #                                                                           ssim2D_thres=ssim2D_thres, ssim2dF_filter=ssim2dF_filter, diskSize=diskSize) 

                #np.save(dir_out_refine11_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),  \
                #                                       np.array([frame_info, scaling_mat.dot(homogra_mat_ori),
                #                                         maskfull11, temp11, rad11, mask11],dtype=object) )
                np.save(dir_out_refine11_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),  \
                                                       np.array([frame_info, scaling_mat.dot(homogra_mat_ori),
                                                         maskfull11, temp11, rad11,],dtype=object) )
                
                #plot png native reso
                mpl.rcdefaults()
                mpl.rcParams['text.usetex'] = True
                mpl.rcParams['font.size'] = 16.
                mpl.rcParams['axes.linewidth'] = 1
                mpl.rcParams['axes.labelsize'] = 14.
                mpl.rcParams['xtick.labelsize'] = 14.
                mpl.rcParams['ytick.labelsize'] = 14.
                mpl.rcParams['figure.subplot.left'] = .0
                mpl.rcParams['figure.subplot.right'] = 1.
                mpl.rcParams['figure.subplot.top'] = 1.
                mpl.rcParams['figure.subplot.bottom'] = 0.
                mpl.rcParams['figure.subplot.hspace'] = 0.
                mpl.rcParams['figure.subplot.wspace'] = 0.0
                fig = plt.figure(figsize=(8.,8))
                ax = plt.subplot(111)
                idx_plot =np.where(burnplot00.mask==2)
                buff=20
                xmin,xmax,ymin,ymax= max([0,idx_plot[0].min()-buff]), min([temp11.shape[0]-1,idx_plot[0].max()+buff]), max([0,idx_plot[1].min()-buff]), min([temp11.shape[1]-1,idx_plot[1].max()+buff])
                ax.imshow(np.ma.masked_where(maskfull11==0,temp11)[xmin:xmax,ymin:ymax].T,origin='lower',vmin=290,vmax=500)
                fig.savefig(dir_out_refine11_png + '{:s}_georef2nd_{:06d}_{:s}.png'.format(plotname,igeo,georefMode) )
                plt.close(fig)       
                        
                print('')
                temp_prev = temp1
                rad_prev  = rad1
                maskfull_prev = maskfull1
                mask_prev     = mask1
                time_igni_prev = time_igni
                frame_prev = frame.copy()
                arrivalTime = np.copy(arrivalTime_here)
                id_prev_arr.append(iprev)
                iprev = igeo
                iprev_ssim = max([  int(os.path.basename(lwir_geo_names[max([ii_geo-ii_lag_ssim,0])]).split('_')[-2]), 0])
                ii_geo += 1
                #test_ok_metric_arr.append(-999)
                continue
         
            ####################### 
            # start core loop here 
            #######################

            #optical flow on prev
            #--------------------

            #Using background temperature
            blockSize_ = int(15./reso)
            feature_on_frame0, feature_on_prev0, nbrept_badLoc0, nbrept_badTemp0, _ = get_matching_feature_opticalFlow(temp, mask, temp_prev, temp_range=[280,330],blockSize=blockSize_, qualityLevel=0.3)
            feature_on_frame1, feature_on_prev1, nbrept_badLoc1, nbrept_badTemp1, _ = get_matching_feature_opticalFlow(temp, mask, temp_prev, temp_range=[320,400],blockSize=blockSize_, qualityLevel=0.3)
            feature_on_frame2, feature_on_prev2, nbrept_badLoc2, nbrept_badTemp2, _ = get_matching_feature_opticalFlow(temp, mask, temp_prev, temp_range=[390,470],blockSize=blockSize_, qualityLevel=0.3)
            feature_on_frame3, feature_on_prev3, nbrept_badLoc3, nbrept_badTemp3, _ = get_matching_feature_opticalFlow(temp, mask, temp_prev, temp_range=[460,550],blockSize=blockSize_, qualityLevel=0.3)
            feature_on_frame4, feature_on_prev4, nbrept_badLoc4, nbrept_badTemp4, _ = get_matching_feature_opticalFlow(temp, mask, temp_prev, temp_range=[540,620],blockSize=blockSize_, qualityLevel=0.3)
            
            feature_on_frame = np.concatenate((feature_on_frame0, feature_on_frame1,feature_on_frame2,feature_on_frame3,feature_on_frame4))
            feature_on_prev  = np.concatenate((feature_on_prev0, feature_on_prev1, feature_on_prev2, feature_on_prev3, feature_on_prev4))

            '''
            NOTE:
            this was not coded properly in the version that produce final result. For Sku6, correcting temp1 to temp in get_matching_feature_opticalFlow does not seem to have improve final georef.
            In future, we ll ha to concider howt to deal with that. remove the get_matching_feature_opticalFlow step? 
            '''

            #remove pair
            df_pair = pandas.DataFrame({'xnew':feature_on_frame[:,0], 'ynew':feature_on_frame[:,1]})
            idx_  = np.array(df_pair.drop_duplicates().index)
            feature_on_frame = feature_on_frame[idx_,:]
            feature_on_prev  = feature_on_prev[idx_,:]
          
            if feature_on_frame.shape[0]>0:
                H_corr0, _ = cv2.findHomography(feature_on_frame, feature_on_prev, cv2.RANSAC,5.0)
            else: 
                H_corr0 = None
            

            if H_corr0 is not None:
                rad0      = cv2.warpPerspective( radiance00, H_corr0.dot(homogra_mat_ori), rad.shape[::-1], flags=cv2.INTER_LINEAR)
                rad0      = np.where(rad0<param_set_temperature.radiance.min(),0,rad0)
                maskfull0 = cv2.warpPerspective( maskfull00, H_corr0.dot(homogra_mat_ori), rad.shape[::-1], flags=cv2.INTER_NEAREST)
                temp0     = spectralTools.conv_Rad2Temp( rad0, param_set_temperature )

                temp0, maskfull0, gtemp0, mask0, arrivalTime_here = get_radgradmask(params_lwir_camera, temp0, maskfull0, True, tree, burnplot, plot_mask, plot_mask_enlarged_small,
                                                                                     arrivalTime, frame_time_igni,ssim2dF_mean, dist_behind=dist_behind,
                                                                                     temp_fire=temp_threshFire,time_behind_input=time_behind, 
                                                                                     ssim2D_thres=ssim2D_thres, ssim2dF_filter=ssim2dF_filter, diskSize=diskSize)  
                try: 
                    EP08_i = tools.get_EP08_from_img(temp0, temp_ref, inputMask=mask0, inputMask_ref=mask_ref)
                except: 
                    #if we are here, it means H_corr0 was pretty bad, eg maskfull0 is not useable to get mask0
                    EP08_i = EP08_0-1

            else: 
                EP08_i = EP08_0-1

            if EP08_i > EP08_0:
                optical_flow_res = 'good'
                rad1, temp1, maskfull1, gtemp1, mask1 = rad0, temp0, maskfull0, gtemp0, mask0
                homogra_mat = H_corr0.dot(homogra_mat_ori)
                EP08_0 = EP08_i
            else:
                rad1, temp1, maskfull1, gtemp1, mask1 = rad, temp, maskfull, gtemp, mask
                homogra_mat = homogra_mat_ori
                optical_flow_res = 'bad'
            #current temp and rad in temp1 and rad1
            


            # ECC on prev
            #--------------------
            #mask
            mask1_    = mask1    
            mask_ref_ = mask_prev
            #temp
            temp1_    = temp1
            temp_ref_ = temp_prev
          
            input_temp     = temp1_     #input
            input_temp_ref = temp_ref_  #template
            id_ecc, warp_matrix_, _ = warp_lwir_mir.findTransformECC_stepbystep(input_temp, 
                                                                                          input_temp_ref,
                                                                                          mask1_, mask_ref_, 
                                                                                          trans_len_limit=[0,10], ep08_limit=[0.9,0.6], 
                                                                                          flag_plot_=False,)
            H_corr1 = np.linalg.inv(warp_matrix_)
            #0 var are the variables with ECC applied
            rad0      = cv2.warpPerspective( radiance00, H_corr1.dot(homogra_mat), rad.shape[::-1], flags=cv2.INTER_LINEAR)
            rad0      = np.where(rad0<param_set_temperature.radiance.min(),0,rad0)
            maskfull0 = cv2.warpPerspective( maskfull00, H_corr1.dot(homogra_mat), rad.shape[::-1], flags=cv2.INTER_NEAREST)
            temp0     = spectralTools.conv_Rad2Temp( rad0, param_set_temperature )
            
            temp0, maskfull0, gtemp0, mask0, arrivalTime_here = get_radgradmask(params_lwir_camera, temp0, maskfull0, True, tree, burnplot, plot_mask, plot_mask_enlarged_small,
                                                                                 arrivalTime, frame_time_igni, ssim2dF_mean, dist_behind=dist_behind, 
                                                                                 temp_fire=temp_threshFire,time_behind_input=time_behind, 
                                                                                 ssim2D_thres=ssim2D_thres, ssim2dF_filter=ssim2dF_filter, diskSize=diskSize) 
            

            EP08_1 = tools.get_EP08_from_img(temp0, temp_ref, inputMask=mask0, inputMask_ref=mask_ref)
            print('    |  on prev  init:{:.5f} opt:{:.5f} ecc:{:.5f} ({:.5f}) '.format(EP08_0, EP08_i, EP08_1, EP08_1-EP08_0,), end=' ') #correc_ecc_ref_),

            corr_status = 'applied'
            if  EP08_1 > EP08_0: 
                rad1, temp1, maskfull1, gtemp1, mask1 = rad0, temp0, maskfull0, gtemp0, mask0
                homogra_mat = H_corr1.dot(homogra_mat)
            else:
                EP08_1 = EP08_0
                corr_status = 'skipped'
            EP08_1_prev_ = EP08_1

            print('of:{:4s} ecc:{:7s}'.format(optical_flow_res, corr_status), end=' ')
            print('')
            print('    || {:.5f}'.format(EP08_1))  
            #current temp and rad in temp1 and rad1
            
            
            
            # optical flow and ECC on length_tail_ringCorr last image
            # two steps refine, if ((EP08_R >= limit_EP08_i*EP08_i0) & (EP08_RF > limit_EP08_F*EP08_iF0) )   -> temp1  var
            #                   if                                      EP08_RF > EP08_FM                    -> temp1M var  main comparison var

            #--------------------

            nbre_corr_applied_on_ring = 0
            length_tail_ringCorr_now = 0 
            
            temp1M, rad1M, maskfull1M, gtemp1M, mask1M = temp1, rad1, maskfull1, gtemp1, mask1
            homogra_matM = homogra_mat
            EP08_FM = EP08_1
            #current temp and rad in tempM1 and rad1M
            
            EP_id_here =  EP_id[:-1][::-1][:length_tail_ringCorr]
            if iref not in EP_id_here: EP_id_here = EP_id_here + [iref]
            
            for iref_ in EP_id_here:
                print('    ', end=' ')
                _, _, rad_iref, mask_iref, maskfull_iref, temp_iref, _ , _= np.load(dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,iref_,georefMode), allow_pickle=True)

                ## with refined mask
                EP08_i0 = tools.get_EP08_from_img(temp1,  temp_iref, inputMask=mask1, inputMask_ref=mask_iref)
                EP08_iF0 = tools.get_EP08_from_img(temp1, temp_ref,  inputMask=mask1, inputMask_ref=mask_ref)

                temp1_     = local_normalization(temp1    , mask1,    diskSize=diskSize)
                temp_iref_ = local_normalization(temp_iref, mask_iref, diskSize=diskSize)
      
                mask1_     = mask1     
                mask_iref_ = mask_iref 
               
                #apply optical flow on localy normalized temp before ECC (focus on smoldering)
                blockSize_ = 11 #int(205./reso)
                feature_on_frameR, feature_on_refR, nbrept_badLocR, nbrept_badTempR, _ = get_matching_feature_opticalFlow(temp1_, mask1_, temp_iref_,
                                                                                                                       temp_range=[0.2,0.8], blockSize=blockSize_, qualityLevel=0.1, 
                                                                                                                       relative_err=None, maxLevel=3)
                
                # ... same on background temperature
                blockSize_ = 11 #int(205./reso)
                mask_here  = np.array(np.where( (temp1   <= temp_threshFire) & (temp_iref <= temp_threshFire) & (plot_mask_enlarged_small == 2) ,1,0), dtype=np.uint8)
                feature_on_frameT1, feature_on_refT1, nbrept_badLocT1, nbrept_badTempT1, _ = get_matching_feature_opticalFlow(temp1, mask_here, temp_iref,
                                                                                                                       temp_range=[290,320], blockSize=blockSize_, qualityLevel=0.3,
                                                                                                                       relative_err=None, maxLevel=3)
                feature_on_frameT2, feature_on_refT2, nbrept_badLocT2, nbrept_badTempT2, _ = get_matching_feature_opticalFlow(temp1, mask_here, temp_iref,
                                                                                                                       temp_range=[320,340], blockSize=blockSize_, qualityLevel=0.3,
                                                                                                                       relative_err=None, maxLevel=3)
                
                #remove pair
                feature_on_frame_ = np.concatenate((feature_on_frameR, feature_on_frameT1,feature_on_frameT2 ))
                feature_on_ref_  = np.concatenate((feature_on_refR,   feature_on_refT1, feature_on_refT2    ))
                df_pair = pandas.DataFrame({'xnew':feature_on_frame_[:,0], 'ynew':feature_on_frame_[:,1]})
                idx_  = np.array(df_pair.drop_duplicates().index)
                feature_on_frameR = feature_on_frame_[idx_,:]
                feature_on_refR   = feature_on_ref_[idx_,:]
                #ax=plt.subplot(121)
                #ax.imshow(np.ma.masked_where(mask1_==0,temp1_).T,origin='lower')
                #ax.scatter(feature_on_frameR[:,1],feature_on_frameR[:,0],c='k')
                #ax=plt.subplot(122)
                #ax.imshow(np.ma.masked_where(mask_iref_==0,temp_iref_).T,origin='lower')
                #ax.scatter(feature_on_refR[:,1],feature_on_refR[:,0],c='k')
                #plt.show()
                #pdb.set_trace()
                
                print(' {:04d} | '.format(iref_), end=' ')  
                print('of: #pt{:4d}'.format(feature_on_frameR.shape[0]), end=' ') 
                if feature_on_frameR.shape[0]>0:
                    H_corrR, _ = cv2.findHomography(feature_on_frameR, feature_on_refR, cv2.RANSAC,5.0)
                else: 
                    H_corrR = None

                if H_corrR is not None:
                    radR      = cv2.warpPerspective( radiance00, H_corrR.dot(homogra_mat), rad.shape[::-1], flags=cv2.INTER_LINEAR)
                    radR      = np.where(radR<param_set_temperature.radiance.min(),0,radR)
                    maskfullR = cv2.warpPerspective( maskfull00, H_corrR.dot(homogra_mat), rad.shape[::-1], flags=cv2.INTER_NEAREST)
                    tempR     = spectralTools.conv_Rad2Temp( radR, param_set_temperature )

                    tempR, maskfullR, gtempR, maskR, arrivalTime_here = get_radgradmask(params_lwir_camera, tempR, maskfullR, True, tree, burnplot, plot_mask, plot_mask_enlarged_small,
                                                                                         arrivalTime, frame_time_igni,ssim2dF_mean, dist_behind=dist_behind,
                                                                                         temp_fire=temp_threshFire,time_behind_input=time_behind, 
                                                                                         ssim2D_thres=ssim2D_thres, ssim2dF_filter=ssim2dF_filter, diskSize=diskSize)  
                    
                    EP08_R  = tools.get_EP08_from_img(tempR, temp_iref, inputMask=maskR, inputMask_ref=mask_iref_)
                    EP08_RF = tools.get_EP08_from_img(tempR, temp_ref,  inputMask=maskR, inputMask_ref=mask_ref)
                  
                    if ((EP08_R >= limit_EP08_i*EP08_i0) & (EP08_RF > limit_EP08_F*EP08_iF0) ): 
                        temp1, rad1, maskfull1, gtemp1, mask1 = tempR, radR, maskfullR, gtempR, maskR
                        homogra_mat = H_corrR.dot(homogra_mat)
                    
                        EP08_i0 = tools.get_EP08_from_img(temp1,  temp_iref, inputMask=mask1, inputMask_ref=mask_iref)
                        EP08_iF0 = tools.get_EP08_from_img(temp1, temp_ref,  inputMask=mask1, inputMask_ref=mask_ref)
                        print('^', end=' ')
                        
                        if EP08_RF > EP08_FM:
                            temp1M, rad1M, maskfull1M, gtemp1M, mask1M = temp1, rad1, maskfull1, gtemp1, mask1
                            homogra_matM = homogra_mat
                            EP08_FM = EP08_RF
                            print('^', end=' ')
                        else:
                            print(' ', end=' ')

                    else:
                        print('   ', end=' ')
                
                else: 
                    print('--')



                #ECC
                #-----------
                input_temp     = temp1
                input_temp_ref = temp_iref

                mask1_     = mask1     
                mask_iref_ = mask_iref 
                id_ecc, warp_matrix_, _ = warp_lwir_mir.findTransformECC_stepbystep(input_temp, 
                                                                                              input_temp_ref, 
                                                                                              mask1_, mask_iref_, 
                                                                                              trans_len_limit=[0,10], ep08_limit=[0.9,0.6], 
                                                                                              flag_plot_=False)
                H_corri = np.linalg.inv(warp_matrix_)

                radi  = cv2.warpPerspective( radiance00, H_corri.dot(homogra_mat), rad.shape[::-1], flags=cv2.INTER_LINEAR)
                radi  = np.where(radi<param_set_temperature.radiance.min(), 0, radi)
                maskfulli = cv2.warpPerspective( maskfull00,     H_corri.dot(homogra_mat), rad.shape[::-1], flags=cv2.INTER_NEAREST)
                tempi = spectralTools.conv_Rad2Temp( radi, param_set_temperature )

                tempi, maskfulli, gtempi, maski, arrivalTime_here = get_radgradmask(params_lwir_camera, tempi, maskfulli, True, tree, burnplot, plot_mask,plot_mask_enlarged_small, 
                                                                                   arrivalTime, frame_time_igni, ssim2dF_mean, 
                                                                                   dist_behind=dist_behind, 
                                                                                   temp_fire=temp_threshFire,time_behind_input=time_behind, 
                                                                                   ssim2D_thres=ssim2D_thres, ssim2dF_filter=ssim2dF_filter, diskSize=diskSize)
                
                #EP08 tempi vs iref and ref
                EP08_i1  = tools.get_EP08_from_img(tempi, temp_iref, inputMask=maski, inputMask_ref=mask_iref_)
                EP08_iF1 = tools.get_EP08_from_img(tempi, temp_ref,  inputMask=maski, inputMask_ref=mask_ref)
               
                
                # EP08 temp1 vs ref with erosion
                mask12    = cv2.erode(mask1,np.ones((7,7),dtype=np.uint8),iterations = 1)
                EP08_iF02 = tools.get_EP08_from_img(temp1, temp_ref,  inputMask=mask12, inputMask_ref=mask_ref)
               
                # EP08 tempi vs ref with erosion
                maski2    = cv2.erode(maski,np.ones((7,7),dtype=np.uint8),iterations = 1)
                EP08_iF12 = tools.get_EP08_from_img(tempi, temp_ref,  inputMask=maski2, inputMask_ref=mask_ref)

                print(' | ecc: iref({:.4f};{:.4f}), ref({:.4f};{:.4f}), erod ({:.4f};{:.4f})'.format(EP08_i0,EP08_i1,EP08_iF0,EP08_iF1,EP08_iF02,EP08_iF12), end=' ')

                length_tail_ringCorr_now += 1
                if  (EP08_i1 >= limit_EP08_i*EP08_i0) & (EP08_iF1 > limit_EP08_F*EP08_iF0) & (EP08_iF12 > EP08_iF02): 
                    nbre_corr_applied_on_ring += 1
                    temp1, rad1, maskfull1, gtemp1, mask1 = tempi, radi, maskfulli, gtempi, maski
                    homogra_mat = H_corri.dot(homogra_mat)
                    print(' *', end=' ')
              
                    if EP08_iF1 > EP08_FM:
                        temp1M, rad1M, maskfull1M, gtemp1M, mask1M = temp1, rad1, maskfull1, gtemp1, mask1
                        homogra_matM = homogra_mat
                        EP08_FM = EP08_iF1
                        print(' *', end=' ')
                    else: 
                        print('  ', end=' ')
                
                else: 
                    print('  ', end=' ')
                
                print('')

            print('    {:2d}/{:2d}'.format(nbre_corr_applied_on_ring,length_tail_ringCorr_now), end=' ') 

            EP08_F = tools.get_EP08_from_img(temp1, temp_ref,  inputMask=mask1, inputMask_ref=mask_ref)
            if EP08_FM > EP08_F:
                temp1, rad1, maskfull1, gtemp1, mask1 = temp1M, rad1M, maskfull1M, gtemp1M, mask1M
                homogra_mat = homogra_matM
                EP08_F = EP08_FM 
                print(' back to max EP08 {:.4f}'.format(EP08_F), end=' ')
            print('')

            #end loop on ref
            #--------------
            
            
            #back to native resolution
            if temp1.size != plot_mask00.size:
                shape_     =  plot_mask00.shape
                rad11      = cv2.warpPerspective( radiance00, scaling_mat.dot(homogra_mat), shape_, flags=cv2.INTER_LINEAR)
                rad11      = np.where(rad11<param_set_temperature.radiance.min(),0,rad11)
                maskfull11 = cv2.warpPerspective( maskfull00, scaling_mat.dot(homogra_mat), shape_, flags=cv2.INTER_NEAREST)
                temp11 = spectralTools.conv_Rad2Temp( rad11, param_set_temperature )
                
                ssim2dF_mean00 = ndimage.zoom(ssim2dF_mean, reso, order=0, cval=0)[:temp11.shape[0],:temp11.shape[1]]
                arrivalTime00  = ndimage.zoom(arrivalTime, reso, order=0, cval=0)[:temp11.shape[0],:temp11.shape[1]]
           
                #get temp for prev frame at native resolution
                #[frame_info_prev, homogra_mat_prev, maskfull11_prev, temp11_prev, rad11_prev, mask11_prev] = np.load(dir_out_refine11_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,iprev,georefMode), allow_pickle=True)
                [frame_info_prev, homogra_mat_prev, maskfull11_prev, temp11_prev, rad11_prev,] = np.load(dir_out_refine11_npy+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,iprev,georefMode), allow_pickle=True)
            
            else:
                rad11         = rad1
                maskfull11    = maskfull1
                temp11        = temp1
                ssim2dF_mean00 = ssim2dF_mean
                arrivalTime00  = arrivalTime

                temp11_prev    = temp_prev

            #temp11, maskfull11, gtemp11, mask11, arrivalTime11_here = get_radgradmask(params_lwir_camera, temp11, maskfull11, True, tree00, burnplot00, plot_mask00, plot_mask_enlarged_small00,
            #                                                                       arrivalTime00, frame_time_igni, ssim2dF_mean00,
            #                                                                       dist_behind=dist_behind, 
            #                                                                       temp_fire=temp_threshFire, time_behind_input=time_behind, 
            #                                                                       ssim2D_thres=ssim2D_thres, ssim2dF_filter=ssim2dF_filter, diskSize=diskSize) 
        
            EP08_F = tools.get_EP08_from_img( temp1, temp_ref, inputMask=mask1, inputMask_ref=mask_ref)

            
            #ssim to prev
            #######
            
            #temp1_     = local_normalization(temp1    , mask1,     diskSize=diskSize)
            #temp_prev_ssim_ = local_normalization(temp_prev_ssim, mask_prev_ssim, diskSize=diskSize)
            temp1_     = local_normalization(temp1    , np.where(temp1<540,1,0),     diskSize=diskSize)
            temp_prev_ssim_ = local_normalization(temp_prev_ssim, np.where(temp_prev_ssim<540,1,0), diskSize=diskSize)
            #_, ssim2dF = skimage.measure.compare_ssim(temp1_, temp_prev_, full=True, win_size=params_lwir_camera['diskSize_ssim_refined'])
            _, ssim2dF = skimage.metrics.structural_similarity(temp1_, temp_prev_ssim_, full=True, win_size=params_lwir_camera['diskSize_ssim_refined'])
            
            #ssim2dF_mean_arr.append(ssim2dF)
            ssim2dF_mean_arr[0].append(ssim2dF); ssim2dF_mean_arr[1].append(maskfull1) 
            if len(ssim2dF_mean_arr[0]) > 2*length_tail_ringCorr:
                ssim2dF_mean_arr[0].pop(0); ssim2dF_mean_arr[1].pop(0)
            
           
                
            ssim2dF_nan = np.where(np.array(ssim2dF_mean_arr[1])==1, np.array(ssim2dF_mean_arr[0]), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ssim2dF_mean = np.nanmean(ssim2dF_nan, axis=0)
            #ssim2dF_mean = np.nanmean(ssim2dF_nan, axis=0)
            ssim2dF_mean = np.ma.array(np.where(np.isnan(ssim2dF_mean),1.e-6,ssim2dF_mean), mask=np.where(np.isnan(ssim2dF_mean),True, False))
        
            ''' 
            if len(ssim2dF_mean_arr[0]) > 1: 
                ssim2dF_masked = np.ma.array(np.array(ssim2dF_mean_arr[0]),mask=np.where(np.array(ssim2dF_mean_arr[1])==0,True,False))
                ssim2dF_masked_clipped = np.where(ssim2dF_masked<1e-6, 1.e-6, ssim2dF_masked)
                ssim2dF_mean = ssim2dF_masked_clipped.mean(axis=0) 
            else: 
                ssim2dF_mean = np.ma.array(np.array(ssim2dF_mean_arr[0][0]),mask=np.where(np.array(ssim2dF_mean_arr[1][0])==0,True,False))
            '''
            
            #ssim2dF_mean = np.ma.array( np.array(ssim2dF_mean_arr[0]).mean(axis=0), mask=np.where( np.prod( np.array(ssim2dF_mean_arr[1]), axis=0 ) == 0 ,True, False) )
            
            #pixxErr
            #######
            temp1_     = local_normalization(temp1    , mask1,     diskSize=diskSize)
            temp_prev_ = local_normalization(temp_prev, mask_prev, diskSize=diskSize)    
            mask1_     = mask1   
            blockSize_ = 11 #int(205./reso)
            feature_on_frameF, feature_on_prevF, nbrept_badLocF, nbrept_badTempF, _ = get_matching_feature_opticalFlow(temp1_, mask1_, temp_prev_,
                                                                                                                    temp_range=[0.2,0.8], blockSize=blockSize_, qualityLevel=0.1, 
                                                                                                                    relative_err=None, maxLevel=3)
            
            # ... same on background temperature
            blockSize_ = 11 #int(205./reso)
            mask_here  = np.array(np.where( (temp1   > temp_threshFire) & (temp_prev > temp_threshFire),0,1),dtype=np.uint8)
            feature_on_frameT1, feature_on_prevT1, nbrept_badLocT1, nbrept_badTempT1, _ = get_matching_feature_opticalFlow(temp1, mask_here, temp_prev,
                                                                                                                   temp_range=[290,320], blockSize=blockSize_, qualityLevel=0.3,
                                                                                                                   relative_err=None, maxLevel=3)
            feature_on_frameT2, feature_on_prevT2, nbrept_badLocT2, nbrept_badTempT2, _ = get_matching_feature_opticalFlow(temp1, mask_here, temp_prev,
                                                                                                                   temp_range=[320,340], blockSize=blockSize_, qualityLevel=0.3,
                                                                                                                   relative_err=None, maxLevel=3)
            
            #remove pair
            feature_on_frame_ = np.concatenate((feature_on_frameF, feature_on_frameT1,feature_on_frameT2 ))
            feature_on_prev_  = np.concatenate((feature_on_prevF,   feature_on_prevT1, feature_on_prevT2    ))
            df_pair = pandas.DataFrame({'xnew':feature_on_frame_[:,0], 'ynew':feature_on_frame_[:,1]})
            idx_  = np.array(df_pair.drop_duplicates().index)
            feature_on_frameF = feature_on_frame_[idx_,:]
            feature_on_prevF   = feature_on_prev_[idx_,:]
            
           
            error_gridded, outlier_flag, ng = get_gridded_error(feature_on_frameF, feature_on_prevF, temp1, plot_mask_enlarged_small00, 1)
            idx_selected_feature_on_frameF    = np.where(outlier_flag==0)
            idx_nonselected_feature_on_frameF = np.where(outlier_flag==1)
            if np.where(error_gridded!=-999)[0].shape[0] <= 2: 
                pixErr_F = 999
            else:
                pixErr_F = np.percentile(error_gridded[np.where(error_gridded!=-999)],80)
            
            print('    --> ep08_F={:.4f} err={:.4f} (n={:d}/ng={:d}/nout={:d})'.format(EP08_F,pixErr_F,feature_on_frameF.shape[0],ng,np.where(outlier_flag==1)[0].shape[0]), end=' ')
            
            pixErr_F_limit = 1.5 if ((igeo-iprev)>3) else 3.0
            

            flag_again = False

            EP08_1_prev.append(EP08_1_prev_)
            EP08_F_arr.append( EP08_F )    
            correc_ecc.append(0) 
            pixErr_F_arr.append(pixErr_F)
            
            ##test_ok # deprecated. keep loging it. 
            ########
            '''
            temp1_ = temp1
            temp_prev_ = temp_prev 

            temp1_     = np.array(temp1_,dtype=float)
            temp_prev_ = np.array(temp_prev_,dtype=float)
           
            #mask_test_ok = np.where( ((mask1==1)&(mask_prev==1)&(temp_prev_>0.)) & ((arrivalTime>0)&(arrivalTime<time_igni-20)), 1, 0 )
            mask_test_ok = np.where( ((mask1==1)&(mask_prev==1)&(temp_prev_>0.)), 1, 0 )
            idx_ = np.where( (mask_test_ok == 1) )
            test_ok_2d = np.zeros(temp1_.shape)
            test_ok_2d[idx_] =old_div(np.abs( temp1_[idx_] - temp_prev_[idx_] ), temp_prev_[idx_]) #/ (time_igni-time_igni_prev)
            #test_ok_metric = old_div(test_ok_2d[np.where(mask_test_ok==1)].sum(), np.where(mask_test_ok==1)[0].shape[0])

            #print 'test_ok={:.4f} '.format(test_ok_metric), 
            test_ok_metric_arr.append(test_ok_metric)
            '''
            id_prev_arr.append(iprev)

            
            #--------------------------
            # control plot
            #--------------------------
            flag_plot = True
            if flag_plot:
                mpl.rcdefaults()
                fig = plt.figure(figsize=(10,10))
                ax = plt.subplot(221)
                '''
                mask_     = mask #np.where(plot_mask_enlarged_small,0,mask)
                ax.imshow(np.ma.masked_where(mask_==0,temp).T,origin='lower',vmin=300,vmax=340)#,vmin=300,vmax=340)#,vmin=10,vmax=160)
                #ax.imshow(np.ma.masked_where(mask==0,gtemp).T,origin='lower',vmin=1,vmax=20)
                ax.set_title('init EP08={:.5f} \n iref={:03d}, iprev={:03d}'.format(EP08_0, iref, iprev))
                ax.scatter(feature_on_frame0[:,1],feature_on_frame0[:,0],c='k',alpha=.5) 
                ax.scatter(feature_on_frame1[:,1],feature_on_frame1[:,0],c='b',alpha=.5) 
                ax.scatter(feature_on_frame2[:,1],feature_on_frame2[:,0],c='y',alpha=.5) 
                ax.scatter(feature_on_frame3[:,1],feature_on_frame3[:,0],c='m',alpha=.5) 
                ax.scatter(feature_on_frame4[:,1],feature_on_frame4[:,0],c='r',alpha=.5) 
                '''
                #ax.imshow(np.ma.masked_where(mask_of==0,ssim2dF_mean).T,origin='lower',vmin=0.2,vmax=.8)
                ax.imshow(np.ma.masked_where(error_gridded<0,error_gridded).T,origin='lower',extent=(0,error_gridded.shape[0],0,error_gridded.shape[1]),vmin=0.001,vmax=3)
                if error_gridded.max()>0:
                    ax.set_title('max errPix={:.3f}'.format(error_gridded[np.where(error_gridded>0)].max()))
                else: 
                    ax.set_title('max errPix= NA' )
                if pixErr_F > 0:
                    idx_pts_ = np.where(outlier_flag==1)
                    ax.scatter(feature_on_frameF[idx_pts_,1],feature_on_frameF[idx_pts_,0],alpha=.5,s=20,c='r') 
                    idx_pts_ = np.where(outlier_flag==0)
                    ax.scatter(feature_on_frameF[idx_pts_,1],feature_on_frameF[idx_pts_,0],alpha=.5,s=20,c='k') 

        
                ax = plt.subplot(222)
               
                temp1_    = local_normalization(temp1    , mask1,   diskSize=diskSize)
                temp_ref_ = local_normalization(temp_ref, mask_ref, diskSize=diskSize)

                ax.imshow(np.ma.masked_where(mask1==0,temp1_).T,origin='lower')#,vmin=300,vmax=340)#,vmin=300,vmax=340)#,vmin=10,vmax=160)
                status=r' good' if(pixErr_F_arr[-1] < pixErr_F_limit) else ''
                ax.set_title('corr EP08f={:.3f} - errPx={:.2f}'.format(EP08_F_arr[-1],pixErr_F_arr[-1]) + status )
                
                ax = plt.subplot(223)
                ax.imshow( ssim2dF_mean.T, origin='lower', vmin=0.4, vmax=.95, alpha=.4)
                ax.imshow( np.ma.masked_where( (mask1==0), ssim2dF_mean).T, origin='lower', vmin=0.4, vmax=.95)
                ax.set_title('min={:.2f} max={:.2f}'.format(ssim2dF_mean[np.where(mask1==1)].min(), ssim2dF_mean[np.where(mask1==1)].max())) 
                ax = plt.subplot(224)
                ax.imshow(np.ma.masked_where(maskfull1==0,temp1_).T,origin='lower',)#vmin=330,vmax=600)
                ax.set_title('t = {:.2f} s'.format(time_igni)) 
                
                fig.savefig(dir_out_refine_png+'{:s}_georef2nd_{:06d}_{:s}.png'.format(plotname,igeo,georefMode))
                plt.close(fig)
           
            #print 'debug stop here'
            #sys.exit()
            
            status_arr.append(corr_status)
            #save  resolution
            np.save(dir_out_refine_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),np.array( [[iref,iprev,frame_info,corr_status,EP08_0_prev[-1],EP08_1_prev[-1],correc_ecc[-1],
                                                                                                       EP08_F_arr[-1], pixErr_F_arr[-1] ],
                                                                                                      scaling_mat.dot(homogra_mat),
                                                                                                      rad1,
                                                                                                      mask1, maskfull1,
                                                                                                      temp1, arrivalTime_here, ssim2dF,], dtype=object))

            #save native resolution
            #np.save(dir_out_refine11_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),  \
            #                                       np.array([frame_info, scaling_mat.dot(homogra_mat), 
            #                                         maskfull11, temp11, rad11, mask11], dtype=object) )
            np.save(dir_out_refine11_npy+'{:s}_georef2nd_{:06d}_{:s}'.format(plotname,igeo,georefMode),  \
                                                   np.array([frame_info, scaling_mat.dot(homogra_mat), 
                                                     maskfull11, temp11, rad11,], dtype=object) )
                
            
           
            #plot png native reso
            mpl.rcdefaults()
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['font.size'] = 16.
            mpl.rcParams['axes.linewidth'] = 1
            mpl.rcParams['axes.labelsize'] = 14.
            mpl.rcParams['xtick.labelsize'] = 14.
            mpl.rcParams['ytick.labelsize'] = 14.
            mpl.rcParams['figure.subplot.left'] = .0
            mpl.rcParams['figure.subplot.right'] = 1.
            mpl.rcParams['figure.subplot.top'] = 1.
            mpl.rcParams['figure.subplot.bottom'] = 0.
            mpl.rcParams['figure.subplot.hspace'] = 0.
            mpl.rcParams['figure.subplot.wspace'] = 0.0
            fig = plt.figure(figsize=(8.,8))
            ax = plt.subplot(111)
            idx_plot =np.where(burnplot00.mask==2)
            buff=20
            xmin,xmax,ymin,ymax= max([0,idx_plot[0].min()-buff]), min([temp11.shape[0]-1,idx_plot[0].max()+buff]), max([0,idx_plot[1].min()-buff]), min([temp11.shape[1]-1,idx_plot[1].max()+buff])
            ax.imshow(np.ma.masked_where(maskfull11==0,temp11)[xmin:xmax,ymin:ymax].T,origin='lower',vmin=290,vmax=500)
            fig.savefig(dir_out_refine11_png + '{:s}_georef2nd_{:06d}_{:s}.png'.format(plotname,igeo,georefMode) )
            plt.close(fig)


            #--------------------------
            # update prev for next loop
            #--------------------------
            temp_prev = temp1
            maskfull_prev = maskfull1
            frame_prev = frame.copy()
            time_igni_prev = time_igni
            iprev = igeo
            iprev_ssim = max([  int(os.path.basename(lwir_geo_names[max([ii_geo-ii_lag_ssim,0])]).split('_')[-2]), 0])
        
            #monitor EP08_F and activate change iref
            print('')
            print('    ref_thr = {:.3f}, plot_unCov = {:.3f}'.format(0.99* EP08_F_00, EP_plotUncovered[-1]), end=' ') 
            if EP_id_ref[-1] != EP_id_ref[-2]: 
                EP08_F_00 = EP08_F
            if np.array(EP08_F_arr[-1*length_tail_ringCorr:]).mean() < 0.99* EP08_F_00: 
                flag_change_iref = True
                print('    ## update ref ##')
            else:
                print('')

            if  (old_div(np.abs(EP08_F_arr[-1] - EP08_F_arr[-2]), EP08_F_arr[-2]) ) < .1 : arrivalTime = arrivalTime_here
            ii_geo += 1
            print('')
            
            #if igeo == 775: 
            #    pdb.set_trace()

        '''
        ax=plt.subplot(311)
        ax.plot(corr_ref)
        ax=plt.subplot(312)
        ax.plot(corr_ref00)
        for xc in id_ref00: ax.axvline(x=xc,c='r')
        ax=plt.subplot(313)
        ax.plot(corr_ref00_init)
        '''
        np.save(dir_out_refine11+'EP08_F.npy',[EP_id, EP08_F_arr, EP_id_ref, EP_plotUncovered])
        np.save(dir_out_refine11+'arrivalTime_lowRes.npy', ndimage.zoom(arrivalTime, reso, order=0, cval=0))

    else:
        arrivalTime = np.load(dir_out_refine11+'arrivalTime_lowRes.npy') 
        #arrivalTime = np.load('/media/paugam/goulven/data/2014_SouthAfrica/Postproc/Shabeni1/LWIR301e/Georef_refined_new_SH/front_LN/shabeni1_arrivalTime_LN.npy')
   
    
    time_end_run = datetime.datetime.now()
    print('')
    print('---------')
    print('cpu time elapse (h) = ', old_div((time_end_run - time_start_run).total_seconds(),3600))     

    '''
    #####################
    #remove outlier frame
    #####################
    import refine_mir
    reload(refine_mir)
    flag_parallel = False
    lwir_info = lwir_info[370:410]
    [badLwirId,badLwirTime],[goodLwirId,goodLwirTime] = refine_mir.filter_badMirFrame(inputConfig, flag_parallel, lwir_info, dir_out_frame, dir_out_refine11_npy, plotname, burnplot00, mask_burnNoburn, arrivalTime, window_ssimNeighbors=2, flag_save_ssim_npy=True, flag_camera='LWIR')
    '''
