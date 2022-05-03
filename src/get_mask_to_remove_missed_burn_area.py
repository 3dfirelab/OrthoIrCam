from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import cv2 
from scipy import ndimage, spatial 
import sys
import pdb 
from PIL import Image, ImageDraw
#import keras
#import tensorflow
#from keras.models import load_model
import os 
import argparse
#import imp
import glob
import sys 
import socket



#####################################################################
def gradient_here(z):

    normal_x = np.zeros_like(z)
    normal_y = np.zeros_like(z)

    #along x
    normal_x[1:-1,:   ] = .5*(z[2:,: ]-z[:-2,:  ])

    idx = np.where((z[2:,: ]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_x[i+1,j] = z[i+1,j]-z[i,j]

    idx = np.where((z[:-2,:  ]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_x[i+1,j] = z[i+2,j]-z[i+1,j]

    idx = np.where((z[2:,: ]==-999)&(z[:-2,:  ]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_x[i+1,j] = 0

    #along y
    normal_y[:   ,1:-1] = .5*(z[: ,2:]-z[:  ,:-2])
    
    idx = np.where((z[:,2:]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_y[i,j+1] = z[i,j+1]-z[i,j]

    idx = np.where((z[:,:-2 ]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_y[i,j+1] = z[i,j+2]-z[i,j+1]

    idx = np.where((z[:,2:]==-999)&(z[:,:-2]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_y[i,j+1] = 0

    return normal_x, normal_y


############################################
def get_mask(inputConfig, maps,  georefMode, npyDirectory, modelType='thr', modelType_val=11, flag_plot=False, flag_save=True):
    root_postproc     = inputConfig.params_rawData['root_postproc']
    plot_name         = inputConfig.params_rawData['plotname']
   
    #root_postproc+inputConfig.params_lwir_camera['dir_input']+'{:s}_{:s}'.format(radDirectory,georefMode)+'/npy/'
    try:
        filenames_georef = sorted(glob.glob(npyDirectory+plot_name+'_georef*.npy'))
        first_georef_file = filenames_georef[0]; itemp = 3
        tmp = np.load(first_georef_file, allow_pickle=True)
        tile_time    = tmp[0][1] # ignition time
        georef_temp = tmp[itemp]    # temp
        georef_rad = tmp[4]    # radiance
    except: 
        pdb.set_trace()
        filenames_georef = sorted(glob.glob(npyDirectory+plot_name+'_0*.npy'))
        first_georef_file = filenames_georef[0]; itemp = 1
        tmp = np.load(first_georef_file)
        tile_time    = tmp[0][1] # ignition time
        georef_temp = tmp[itemp]    # temp
        georef_rad = None    # radiance


    
    if modelType == 'cnn':
        import tools_cnn
        cnnV = modelType_val 
        if cnnV == 'v5': 
            print('bad cnnV selected for maskburnNoburn, cnnV = ', cnnV)
            sys.exit()
        
        model_front_delimitation, firemap_mask = tools_cnn.loadCnn(cnnV, maps)
        mask = tools_cnn.runCnn(cnnV, georef_rad, georef_temp, firemap_mask, model_front_delimitation,inputConfig.params_georef['cnn_meanTemp'])  # mask is simplified to full plot mask here as we only run it on first image

    elif modelType == 'thr':
        mask = np.where((georef_rad>float(modelType_val))&(maps.mask==2),np.ones_like(georef_rad),np.zeros_like(georef_rad))
    

    mask = np.where( (mask==1) & (maps.mask==2) , 0, old_div(maps.mask,2))

    if mask.max() != 0:
        kernel = np.ones((3,3),np.uint8)
        img_ = np.array(mask,dtype=np.uint8)*255
        img_eroded = cv2.erode(img_,kernel,iterations = 2)
        mask2 = old_div(np.array(img_eroded,dtype=np.int64),255)

        s = [[0,1,0], \
             [1,1,1], \
             [0,1,0]]
        labeled, num_cluster = ndimage.label(mask2, structure=s )
        cluster_size = np.zeros(num_cluster)
        for icluster in range(num_cluster):
            cluster_size[icluster] = np.where(labeled==icluster+1)[0].shape[0]
        

        mask3 = np.where(labeled==cluster_size.argmax()+1, np.ones_like(georef_rad),np.zeros_like(georef_rad) )

        img_ = np.array(mask3,dtype=np.uint8)*255
        img_dilated = cv2.dilate(img_,kernel,iterations = 2)
        mask4 = old_div(np.array(img_dilated,dtype=np.int64),255)
    
    else: 
        mask4 = np.ones_like(mask)

    maskfinal = np.where( (mask4==0) & (maps.mask==2), old_div(maps.mask,2), 0)
    maskfinal= old_div(cv2.morphologyEx(255*maskfinal, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8)),255)

    if flag_save:
        np.save(os.path.dirname(first_georef_file)+'/mask_nodata_'+plot_name+'_georef',maskfinal)


    if flag_plot:
        ax = plt.subplot(121)
        ax.imshow(mask4.T,origin='lower')
        ax = plt.subplot(122)
        ax.imshow((maskfinal+2*mask4).T,origin='lower')

        plt.figure()
        plt.imshow(georef_rad.T,origin='lower')

        plt.show()
        pdb.set_trace()

    return maskfinal



################################
if __name__ == '__main__':
#################################
    
    parser = argparse.ArgumentParser(description='this is the driver of the GeorefCam Algo.')
    parser.add_argument('-i','--input', help='Input run name',required=True)
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
    
    
    
    inputConfig = imp.load_source('config_'+runName,os.getcwd()+'/../../georefircam/input_config/config_'+runName+'.py')
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


    params_lwir_camera     = inputConfig.params_lwir_camera
    
    # control flag
    flag_georef_mode = inputConfig.params_flag['flag_georef_mode']
    if flag_georef_mode   == 'WithTerrain'     : 
        georefMode = 'WT'
        params_lwir_camera['dir_input'] = params_lwir_camera['dir_input'][:-1] + '_WT/'
    elif flag_georef_mode == 'SimpleHomography': 
        georefMode = 'SH'


    root_postproc     = inputConfig.params_rawData['root_postproc']
    plot_name          = inputConfig.params_rawData['plotname']
   
    npyDirectory = root_postproc + params_lwir_camera['dir_input'] + 'Georef_{:s}/npy/'.format(georefMode)

    mask_type     = inputConfig.params_lwir_camera['mask_burnNobun_type_refined']
    mask_type_val = inputConfig.params_lwir_camera['mask_burnNobun_type_val_refined']



    maps = np.load(root_postproc+'/grid_'+plot_name+'.npy')
    maps = maps.view(np.recarray)
   
    mask_final = get_mask(inputConfig, maps,  georefMode, npyDirectory, mask_type, mask_type_val, flag_plot=True, flag_save=False)

    sys.exit()







'''
s = [[0,1,0], \
     [1,1,1], \
     [0,1,0]]
labeled, num_cluster = ndimage.label(mask, structure=s )
cluster_size = np.zeros(num_cluster)
for icluster in range(num_cluster):
    cluster_size[icluster] = np.where(labeled==icluster+1)[0].shape[0]

mask2 = np.where(labeled==cluster_size.argmax()+1, np.ones_like(georef_rad),np.zeros_like(georef_rad) )

kernel = np.ones((3,3),np.uint8)
img_ = np.array(mask2,dtype=np.uint8)*255
img_dilated = cv2.dilate(img_,kernel,iterations = 2)
mask2 = np.array(img_dilated,dtype=np.int64)/255

gradient_x, gradient_y = gradient_here(georef_rad)
gradient_norm = np.sqrt(gradient_x**2+gradient_y**2) 

idx_fire = np.where((gradient_norm >= 2 )      &\
                     (maps.mask==2)             )

mask3 = np.zeros_like(georef_rad)
mask3[idx_fire] = 1
tresh = np.array(mask3,dtype=np.uint8)
image, contours, hierarchy = cv2.findContours(tresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

len_contours = np.zeros(len(contours))
for ic, contour in enumerate(contours):
    len_contours[ic]=len(contour)

idx_ic = len_contours.argmax()

mask4 = np.zeros_like(georef_rad)
polygon =[ tuple( [pt[0][1],pt[0][0]] ) for pt in contours[idx_ic] ]
img = Image.new('L', georef_rad.shape , 0)
ImageDraw.Draw(img).polygon(polygon, 1, 1)
mask_gap = np.copy(img).T
idx_mask = np.where(mask_gap==1)
mask4[idx_mask]=1
plt.imshow(mask4.T,origin='lower')
plt.show()



mask4_out = np.where((maps.mask==2) & (mask4!=1), np.ones_like(georef_rad), np.zeros_like(georef_rad))
s = [[1,1,1], \
     [1,1,1], \
     [1,1,1]]
labeled, num_cluster = ndimage.label(mask4_out, structure=s )

pts_fire = zip(contours[idx_ic][:,0,1],contours[idx_ic][:,0,0])
tree_neighbour = spatial.cKDTree(pts_fire) # all point tree

mask5 = np.zeros_like(georef_rad)
pts_to_check = np.where((mask2==1)&(maps.mask==2))
mm = []
for label in range(num_cluster):
  
    idx = np.where(labeled == label + 1)
    z=[]
    for pt in zip(*idx):
        dists, inds = tree_neighbour.query(pt,k=100)
        z.append([ pt[0]-pts_fire[ind][0] + (pt[1]-pts_fire[ind][1])*1j for ind in inds])
    
    mask5[idx] = (np.angle(np.array(z))).mean() 
    #if (np.angle(z) >= np.pi/2) & (np.angle(z) <= np.pi):
    #if ((pt[0]-close_pt_on_front[0] <= 0) & (pt[1]-close_pt_on_front[1] >= 0)) | ((pt[1]-close_pt_on_front[1] >= 0)) :
    #    mask4[pt[0],pt[1]] = 1
        
    #plt.imshow(np.ma.masked_where((mask2!=1)|(maps.mask!=2),georef_rad).T,origin='lower')
    #plt.scatter(pt[0],pt[1],c='g')
    #plt.scatter(close_pt_on_front[0],close_pt_on_front[1],c='r')
    #[plt.scatter(pt_[0][1],pt_[0][0],c='k',s=10) for pt_ in contours[idx_ic]]
    #plt.show()
    #sys.exit()

maskfinal = np.where((mask4==1)|(mask5>0),np.ones_like(georef_rad), np.zeros_like(georef_rad))

mask6 = np.where( (maskfinal!=1), maps.mask, np.zeros_like(maskfinal) )
mask7 = np.where(mask6==2, np.ones_like(maskfinal), np.zeros_like(maskfinal))
s = [[1,1,1], \
     [1,1,1], \
     [1,1,1]]
labeled, num_cluster = ndimage.label(mask7, structure=s )
cluster_size = np.zeros(num_cluster)
for icluster in range(num_cluster):
    idx = np.where(labeled==icluster+1)
    if idx[0].shape[0]<10:
        maskfinal[idx] = 1


#np.save(rootdir+'2014_SouthAfrica/Postproc/Shabeni1/LWIR301e/Georef_refined_SH/npy_final/mask_nodata_shabeni1_georef',maskfinal)
np.save(os.path.dirname(first_georef_file)+'mask_nodata_'+os.path.basename(first_georef_file).split('_')[0]+'_georef',maskfinal)
ax = plt.subplot(121)
ax.imshow(maskfinal.T,origin='lower')
[plt.scatter(pt_[0][1],pt_[0][0],c='k',s=10) for pt_ in contours[idx_ic]]
ax = plt.subplot(122)
ax.imshow(np.where(maskfinal==1, 0, maps.mask).T,origin='lower')

plt.show()
'''
