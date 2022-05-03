from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import cv2 
import numpy as np
import matplotlib as mpl
import socket
if 'ibo' in socket.gethostname(): mpl.use('Agg')
if 'kraken' in socket.gethostname(): mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import io, ndimage, interpolate, stats
import asciitable
import datetime 
import os
import sys
import glob 
import argparse
import imp
import shutil 
from osgeo import gdal,osr,ogr
import pdb
import itertools
import socket 
import sklearn.datasets, sklearn.decomposition
import skimage 
import math
import importlib
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
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
import matplotlib.patches as mpatches

from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors

#homebrewed
import agema
import optris
import tools
import camera_tools
import spectralTools
import warp_lwir_mir
import refine_lwir
import hist_matching
sys.path.append('../../Ros/src/')
#import map_georefImage
import get_mask_to_remove_missed_burn_area

#reload(map_georefImage)

#############################################
def load_georefnpy(filename, flag_out_masks = False):
    tmp_     = np.load(filename)
    if len(tmp_) == 7: 
        frame_info,                          \
             homogra_mat_ori,                \
             georef_radiance,                \
             georef_mask, georef_maskfull,   \
             georef_temp, arrivalTime_here, = tmp_
        frame_time_igni = frame_info[2][1]
    elif len(tmp_) == 6: 
        frame_info,                           \
             georef_img,                      \
             georef_maskfull,                 \
             georef_temp,                     \
             georef_radiance, _              = tmp_
        frame_time_igni = frame_info[2][1]
    elif len(tmp_) == 5: 
        frame_info,                           \
             georef_img,                      \
             georef_maskfull,                 \
             georef_temp,                     \
             georef_radiance,                = tmp_
        frame_time_igni = frame_info[2][1]


    if flag_out_masks: 
        return frame_time_igni,  georef_radiance, georef_temp, georef_maskfull, georef_mask
        

    return frame_time_igni,  georef_radiance, georef_temp, georef_maskfull


#######################################
def get_mean_var_frame(georef_temp, georef_maskfull, plotmask, trange,):
   
    #dilate plotmask
    kernel = np.ones((31,31),np.uint8)
    img_ = np.array(np.where(plotmask==2,1,0),dtype=np.uint8)*255
    mask_ = cv2.dilate(img_, kernel, iterations = 1)    
    plot_mask = np.where(mask_==255,2,0)

    mask_bckgrd = np.where((georef_temp>=trange[0])&(georef_temp<=trange[1])&(georef_maskfull==1)&(plotmask!=2), 1, 0)

    #temp_scale = np.linspace(trange[0], trange[1], (trange[1]-trange[0])/1)
    #kernel = stats.gaussian_kde(georef_temp[np.where(mask_bckgrd==1)].flatten())
    #pdf = kernel(temp_scale)
    #mean_bckgrd = temp_scale[pdf.argmax()]
    mean_bckgrd = np.median(georef_temp[np.where(mask_bckgrd==1)])

    #print mean_bckgrd, np.median(georef_temp[np.where(mask_bckgrd==1)]), np.where(mask_bckgrd==1)[0].shape[0]
    #sys.stdout.flush()

    var_fulldata = georef_temp[np.where(georef_maskfull==1)].std()

    return mean_bckgrd, var_fulldata
    

#########################################
def grid_ep08(temp, temp_ref, box_size, flag_returnGrid = False, gridded_idx = None):
    
    nn = int( np.round(old_div(1.*temp.shape[0], box_size), 0))
    bb = int( old_div(1.*temp.shape[0],nn) )

    if flag_returnGrid:
        #grid data on grid with resolution x nn
        grid_x, grid_y = np.mgrid[0:temp.shape[0]:(nn+1)*1j, 0:temp.shape[1]:(nn+1)*1j]
        gridded_idx = np.zeros(temp.shape)-999

        grid_x, grid_y = np.array(np.round(grid_x,0),dtype=np.int), np.array(np.round(grid_y,0),dtype=np.int)
        grid_x_l, grid_y_b = grid_x[:-1,:-1], grid_y[:-1,:-1]
        grid_x_r, grid_y_u = grid_x[1:,1:], grid_y[1:,1:]

        for ii, idx_ in enumerate(zip(grid_x_l.flatten(),grid_x_r.flatten(),grid_y_b.flatten(),grid_y_u.flatten())):
            gridded_idx[idx_[0]:idx_[1], idx_[2]:idx_[3]] = ii
  
        return gridded_idx
    
    gridded_ep08 = np.zeros_like(gridded_idx)

    for ii in range(int(gridded_idx.max())+1):
    
        idx_mask_all = np.where( (gridded_idx == ii) )
        idx_mask_ = np.where( (gridded_idx == ii) & (temp > 0) & (temp_ref > 0) )
        
        if idx_mask_[0].shape[0] < .5*(nn*2): 
            gridded_ep08[idx_mask_all] = -888
            continue

        temp_mean     = temp[idx_mask_].mean()
        temp_ref_mean = temp_ref[idx_mask_].mean()

        iw = np.array(     temp[idx_mask_].flatten() - temp_mean,  dtype=np.float64)
        ir = np.array( temp_ref[idx_mask_].flatten() - temp_ref_mean,  dtype=np.float64)

        if (np.linalg.norm(ir) == 0) | (np.linalg.norm(iw) == 0) :
            gridded_ep08[idx_mask_all] = -777
            continue

        ep08 =  old_div(np.dot(ir,iw),(np.linalg.norm(ir)*np.linalg.norm(iw)))
        ep08 = ep08 if (not(np.isnan(ep08))) else -999.

        gridded_ep08[idx_mask_all] = ep08


    return gridded_ep08


#########################################
if __name__ == '__main__':
#########################################
    importlib.reload(agema)
    importlib.reload(optris)
    importlib.reload(camera_tools)
    importlib.reload(tools)
    importlib.reload(spectralTools)
    importlib.reload(warp_lwir_mir) 
    importlib.reload(refine_lwir)
    importlib.reload(hist_matching)

    time_start_run = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='this is the driver of the GeorefCam Algo.')
    parser.add_argument('-i','--input', help='Input run name',required=True)
    parser.add_argument('-s','--newStart',  help='True then it uses existing data',required=False)
    parser.add_argument('-it','--iteration',  help='iteration loop',required=False)
    #parser.add_argument('-ir','--irun',  help='(nbre of time lwir_refine was run) - 1',required=True)
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

    if args.iteration is None:
        itr = 0
    else: 
        itr = int(args.iteration)

    runMode = 'mir'
    inputConfig = imp.load_source('config_'+runName,os.getcwd()+'/../input_config/config_'+runName+'.py')
   
    #irunS = args.irun
    #if irunS == '0': irunS =''

    if socket.gethostname() == 'coreff':
        path_ = 'goulven/data/' # 'Kerlouan/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/paugam/data/','/media/paugam/'+path_)
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/paugam/data/','/media/paugam/'+path_)
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/paugam/data/','/media/paugam/'+path_)     
    
    if socket.gethostname() == 'ibo':
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/','/space/')
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/','/space/')
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/','/space/')     

    # input parameters
    params_grid       = inputConfig.params_grid
    params_gps        = inputConfig.params_gps 
    params_lwir_camera     = inputConfig.params_lwir_camera
    params_rawData    = inputConfig.params_rawData
    params_georef     = inputConfig.params_georef

    # control flag
    flag_georef_mode = inputConfig.params_flag['flag_georef_mode']
    flag_parallel = inputConfig.params_flag['flag_parallel']
    flag_restart  = inputConfig.params_flag['flag_restart']
    if flag_georef_mode   == 'WithTerrain'     : 
        georefMode = 'WT'
        params_lwir_camera['dir_input'] = params_lwir_camera['dir_input'][:-1] + '_WT/'
    elif flag_georef_mode == 'SimpleHomography': 
        georefMode = 'SH'
    if (args.newStart is None) : 
        flag_restart = True
    elif args.newStart is not None:
        flag_restart = tools.string_2_bool(args.newStart)

    plotname          = params_rawData['plotname']
    root_postproc     = params_rawData['root_postproc']

    dir_out           = root_postproc + params_lwir_camera['dir_input']
    dir_dem           = root_postproc + params_georef['dir_dem_input']
  
    dir_out_frame      = dir_out + 'Frames/'
    #
    #
    dir_out_refine     = dir_out + 'Georef_refined_{:s}/'.format(georefMode)
    
    #dir_out_refine_npy = '/scratch/globc/paugam/MERDE_new_gpu_distbehind8/npy{:s}/'.format(irunS)   #dir_out_refine + 'npy{:s}/'.format(irunS)
    dir_out_refine_npy = dir_out_refine + 'npy/'
   

    dir_out_lwir_image = dir_out #+ 'ImageDiagram/'

    #load grid 
    ###########################################
    burnplot = np.load(root_postproc+'grid_'+plotname+'.npy')
    burnplot = burnplot.view(np.recarray)
    grid_e, grid_n = burnplot.grid_e, burnplot.grid_n

    kernel_largePlotMask = 121
    kernel = np.ones((kernel_largePlotMask,kernel_largePlotMask),np.uint8)
    img_ = np.array(np.where(burnplot.mask==2,1,0),dtype=np.uint8)*255
    mask_out = cv2.dilate(img_, kernel, iterations = 1)    
    
    kernel_largePlotMask = 21
    kernel = np.ones((kernel_largePlotMask,kernel_largePlotMask),np.uint8)
    img_ = np.array(np.where(burnplot.mask==2,1,0),dtype=np.uint8)*255
    mask_in = cv2.dilate(img_, kernel, iterations = 1)    
    
    plot_mask_enlarged00 = np.where((mask_out==255),2,0)#&(mask_in!=255),2,0)

    ''' 
    filenames = sorted(glob.glob(dir_out_refine_npy + plotname + '_georef*.npy') )
    trange = [280,350]
    meanvar = np.zeros([len(filenames),2])
    for ifile, filename in enumerate(filenames):
        out_tmp = np.zeros_like(grid_e)-999
        time_igni,  georef_radiance, georef_temp, georef_maskfull  = load_georefnpy(filename)
        meanvar[ifile,:] = get_mean_var_frame(georef_temp, georef_maskfull, plot_mask_enlarged, trange,)

    ax = plt.subplot(111)
    ax.plot(meanvar[:,0])
    bx = ax.twinx()
    bx.plot(meanvar[:,1],c='r')
    plt.show()
    sys.exit()
    '''
    
    img_ = np.array(np.where(burnplot.mask==2,1,0),dtype=np.uint8)*255
    tmp_= cv2.erode(img_, np.ones((2,2),np.uint8), iterations = 1)    
    burnplot_mask_ = np.where(tmp_==255,1,0)  
    
    filenames = sorted(glob.glob(dir_out_refine_npy + plotname + '_georef*.npy') )
    '''
    if os.path.isfile(os.path.dirname(filenames[0])+'/mask_nodata_'+plotname+'_georef.npy'): 
        mask_burnNoburn = np.load(os.path.dirname(filenames[0])+'/mask_nodata_'+plotname+'_georef.npy')
    else:
        print '#### mask_burnNoburn not found'
        mask_burnNoburn =  get_mask_to_remove_missed_burn_area.get_mask(inputConfig, burnplot, georefMode, 
                                                                        dir_out_refine_npy,
                                                                        'cnn', 'v4', flag_plot=False)
    '''
    mask_type_ = inputConfig.params_lwir_camera['mask_burnNobun_type_refined']
    mask_type_val = inputConfig.params_lwir_camera['mask_burnNobun_type_val_refined']
    if os.path.isfile(os.path.dirname(filenames[0])+'/mask_nodata_{:s}_georef.npy'.format(plotname)): 
        mask_burnNoburn00 = np.load(os.path.dirname(filenames[0])+'/mask_nodata_{:s}_georef.npy'.format(plotname))
    else:
        print('#### mask_burnNoburn not found. now generated with mask type:', mask_type_)
        mask_burnNoburn00 =  get_mask_to_remove_missed_burn_area.get_mask(inputConfig, burnplot, georefMode, 
                                                                        dir_out_refine_npy,
                                                                        mask_type_, mask_type_val, flag_plot=False)


    idx_bad_prev = []
    filenames2 = np.copy(filenames)
    if itr > 0:
        for it_ in range(itr):
            idx_bad_ = np.load(dir_out_refine_npy+'badFrameID_it{:02d}.npy'.format(it_))
            [idx_bad_prev.append(idx_bad__) for idx_bad__ in idx_bad_]

        filenames  = []
        for ifile, filename in enumerate(filenames2):
            igeo = int(os.path.basename(filename).split('.')[0].split('_')[-2])

            if igeo in idx_bad_prev: continue
            filenames.append(filename)


    print('files are in: ', dir_out_refine_npy + plotname + '_georef*.npy')
    print('number of file removed : {:04d}'.format(len(filenames2)-len(filenames)))
    if True: 
        
        #nssim = 1
        #res_arr = np.arange(1,2) #,6) onlyt consider native resolution
        #out = np.zeros([2,len(filenames),burnplot_mask_.shape[0],burnplot_mask_.shape[1],nssim,res_arr.shape[0]])-999
        #out_dt = np.zeros([len(filenames),nssim])
        #out_id = np.zeros([len(filenames)], dtype=np.int)
        #errPix =  np.zeros([len(filenames),nssim])
        igeo_prev = []; time_igni_prev=[]; georef_radiance_prev=[]; georef_temp_prev=[]; georef_maskfull_prev=[]#; georef_maskring_prev=[]
        
        arrivalTime = np.zeros_like(grid_e) - 999
         
        #ssim_20Percentile_arr_av = np.load('mm.npy')

        ep08_bb = 21
        winav=10

        ssim_20Percentile_arr = []
        ssim_20Percentile_arr2 = []
        ssim_20Percentile_arr_all = []
        ssim_20Percentile_arr2_all = []
        ssim_20Percentile_id  = []
        ssim_20Percentile_id_all = []


        if (not(os.path.isfile(dir_out_refine_npy+plotname+'_ssim_averaged_it{:02d}.npy'.format(itr)))) | (flag_restart == False) :
            print('compute mean ssim ...')
            for ifile, filename in enumerate(filenames):
                
                print('\r{:d} {:s}'.format(ifile,os.path.basename(filename)), end=' ')
                igeo = int(os.path.basename(filename).split('.')[0].split('_')[-2])
                out_tmp = np.zeros_like(grid_e)-999

                if igeo in idx_bad_prev: continue

                frame_info, _ , georef_maskfull, georef_temp, georef_radiance, = np.load(filename, allow_pickle=True)
                time_igni = frame_info[1]

                if ifile == 0: 
                    time_igni_prev.append(time_igni)
                    georef_radiance_prev.append(georef_radiance)
                    georef_temp_prev.append(georef_temp)
                    georef_maskfull_prev.append(georef_maskfull)
                    #georef_maskring_prev.append(georef_maskring)

                    ssim_20Percentile_arr.append( 0  ) 
                    ssim_20Percentile_id.append(igeo)
                    
                    continue

                ssim,      ssim2d      = skimage.metrics.structural_similarity(georef_temp,     georef_temp_prev[-1], full=True, win_size=ep08_bb) #params_lwir_camera['diskSize_ssim_refined'])
                    
                kernel = np.ones((21,21),np.uint8)
                
                img_ = np.array(np.where(georef_maskfull==1,1,0),dtype=np.uint8)*255
                mask_ = cv2.erode(img_, kernel, iterations = 1)    
                georef_maskfull_ = np.where(mask_==255,1,0)

                img_ = np.array(np.where(georef_maskfull_prev[-1]==1,1,0),dtype=np.uint8)*255
                mask_ = cv2.erode(img_, kernel, iterations = 1)    
                georef_maskfull_prev_ = np.where(mask_==255,1,0)
                
                mask_data = np.where( (georef_maskfull_==1) & (georef_maskfull_prev_==1) & (georef_temp<390) & (georef_temp_prev[-1]<390) & (plot_mask_enlarged00==2), 1, 0)

                ssim_20Percentile_arr.append( np.nanpercentile( np.ma.filled(np.ma.masked_where( mask_data==0, ssim2d),np.nan), 80, )  ) 
                ssim_20Percentile_arr_all.append(ssim_20Percentile_arr[-1])
                

                ssim_20Percentile_id.append(igeo)
                ssim_20Percentile_id_all.append(igeo)
                
                igeo_prev.append(igeo)
                time_igni_prev.append(time_igni)
                georef_radiance_prev.append(georef_radiance)
                georef_temp_prev.append(georef_temp)
                georef_maskfull_prev.append(georef_maskfull)
                #georef_maskring_prev.append(georef_maskring)
                
                print('* \r', end=' ')
                sys.stdout.flush()

            ssim_20Percentile_arr_av = np.zeros(len(ssim_20Percentile_arr))
            for i in range(old_div(winav,2),len(ssim_20Percentile_arr)-old_div(winav,2)):
                ssim_20Percentile_arr_av[i] = old_div(np.array(ssim_20Percentile_arr[i-old_div(winav,2):i+old_div(winav,2)]).sum(),winav)
            ssim_20Percentile_id_av = np.array(ssim_20Percentile_id)
            np.save ( dir_out_refine_npy+plotname+'_ssim_averaged_it{:02d}'.format(itr), [ssim_20Percentile_id_av,ssim_20Percentile_arr_av,ssim_20Percentile_id_all,ssim_20Percentile_arr_all])
            print('done                                 ')
           
        else:
            print('load mean ssim')
            ssim_20Percentile_id_av,ssim_20Percentile_arr_av,ssim_20Percentile_id_all,ssim_20Percentile_arr_all = np.load(dir_out_refine_npy+plotname+'_ssim_averaged_it{:02d}.npy'.format(itr), allow_pickle=True)
            

        ssim_20Percentile_arr = []
        ssim_20Percentile_id = []
        nbre_missed_consecutive = 0
        
        for ifile, filename in enumerate(filenames):
            
            print('\r{:d} {:s}'.format(ifile,os.path.basename(filename)), end=' ')
            igeo = int(os.path.basename(filename).split('.')[0].split('_')[-2])
            out_tmp = np.zeros_like(grid_e)-999

            if igeo in idx_bad_prev: continue

            frame_info, _ , georef_maskfull, georef_temp, georef_radiance, = np.load(filename, allow_pickle=True)
            time_igni = frame_info[1]

            if ifile == 0: 
                igeo_prev.append(igeo)
                time_igni_prev.append(time_igni)
                georef_radiance_prev.append(georef_radiance)
                georef_temp_prev.append(georef_temp)
                georef_maskfull_prev.append(georef_maskfull)
                #georef_maskring_prev.append(georef_maskring)

                gridded_idx = grid_ep08(georef_temp, georef_temp_prev[0], ep08_bb, flag_returnGrid = True)
                continue

            ssim,      ssim2d      = skimage.metrics.structural_similarity(georef_temp,     georef_temp_prev[-1], full=True, win_size=ep08_bb) #params_lwir_camera['diskSize_ssim_refined'])
            
            kernel = np.ones((21,21),np.uint8)
            
            img_ = np.array(np.where(georef_maskfull==1,1,0),dtype=np.uint8)*255
            mask_ = cv2.erode(img_, kernel, iterations = 1)    
            georef_maskfull_ = np.where(mask_==255,1,0)

            img_ = np.array(np.where(georef_maskfull_prev[-1]==1,1,0),dtype=np.uint8)*255
            mask_ = cv2.erode(img_, kernel, iterations = 1)    
            georef_maskfull_prev_ = np.where(mask_==255,1,0)
            
            mask_data = np.where( (georef_maskfull_==1) & (georef_maskfull_prev_==1) & (georef_temp<390) & (georef_temp_prev[-1]<390)& (plot_mask_enlarged00==2), 1, 0)

            ssim_20Percentile_arr.append( np.nanpercentile( np.ma.filled(np.ma.masked_where( mask_data==0, ssim2d),np.nan), 80, )  ) 
            ssim_20Percentile_id.append(igeo)
            
            lastitem = min([len(ssim_20Percentile_arr),winav+1])
            if lastitem == 1: 
                idx_ = np.where(np.array(ssim_20Percentile_arr) > -999)
            elif lastitem < winav:
                idx_ = np.where(np.array(ssim_20Percentile_arr[-lastitem:-1]) > -999)
            else:
                idx_ = np.where(np.array(ssim_20Percentile_arr[-lastitem:-1]) < np.array(ssim_20Percentile_arr[-lastitem:-1]).mean()+2*np.array(ssim_20Percentile_arr[-lastitem:-1]).std())
            
            if idx_[0].shape[0]>1:
                ssim_20Percentile_arr2.append( [ssim_20Percentile_arr_av[ifile], max([0.01,np.array(ssim_20Percentile_arr[-lastitem:-1])[idx_].std()]) ] )
                ssim_20Percentile_arr2_all.append( [ssim_20Percentile_arr_av[ifile], max([0.01,np.array(ssim_20Percentile_arr[-lastitem:-1])[idx_].std()]) ] )
            else: 
                
                ssim_20Percentile_arr2.append( [ssim_20Percentile_arr_av[ifile], 0 ] )
                ssim_20Percentile_arr2_all.append( [ssim_20Percentile_arr_av[ifile], 0 ] )
            
            print('{:03d} {:02d} | {:.3f} | {:.3f} {:.4f}'.format(igeo_prev[-1], idx_[0].shape[0], ssim_20Percentile_arr[-1], *ssim_20Percentile_arr2[-1]), end=' ')
          
           
            #if igeo == 161: 
            #    plt.imshow(np.ma.masked_where( mask_data==0, ssim2d).T, origin='lower'); plt.show()
            #    sys.exit()
            
            if ((lastitem > old_div(winav,2)) & (ssim_20Percentile_arr[-1] < ssim_20Percentile_arr2[-1][0]-2*ssim_20Percentile_arr2[-1][1]) \
                                     & (igeo not in params_lwir_camera['final_selection_force2keep']) \
               ) \
             | (igeo in params_lwir_camera['final_selection_4bin'] ) :
  
            #if ((lastitem > winav/2) & (ssim_20Percentile_arr[-1] < ssim_20Percentile_arr2[-1][0]-2*ssim_20Percentile_arr2[-1][1]) & (igeo not in [376,623,838,888,889,970,1038,1071,1099,1142,1160,1176,1182,1186,1191,1198,1225,1244,1298,1307,1310,1333,1336,])) \
            # | (igeo in [894,895,898,966,967,968,969,1006,1007,1097,1100,1114,1139,1140,1141,1155,1156,1157,1158,1159,1164,1165,1169,1174,1175,1180,1181,1183,1184,1186,1190,1239,1240,1241,1242,1243,1296,1297,1305,1306,1308,1309,1317,1324,1327,1331,1332,1334,1335,1338,1345,1348,1358,1360,1384,1387,1469,1648,1649]) : #sha3
            
            #if ((lastitem > winav/2) & (ssim_20Percentile_arr[-1] < ssim_20Percentile_arr2[-1][0]-2*ssim_20Percentile_arr2[-1][1]) & (igeo not in [332,489,532,564,602,619,640,653,728])) \
            # | (igeo in [324,342,331,345,380,414,443,444,455,461,481,483,488,506,559,562,563,566,584,597,614,623,631,632,637,638,639,650,651,652,663,733,753,758,798,799,800,801,805,836,852,855,872,910,921,928,945,972,974,975,976,977,978]) : #sku6
            
            #if ((lastitem > winav/2) & (ssim_20Percentile_arr[-1] < ssim_20Percentile_arr2[-1][0]-2*ssim_20Percentile_arr2[-1][1]) & (igeo not in [85,111,140,171,221,262])) \
            # | (igeo in [114,139,146,148,164,230,259,261,330,342,345,349,413,414,415,416,]) : #sha1
            
            #if ((lastitem > winav/2) & (ssim_20Percentile_arr[-1] < ssim_20Percentile_arr2[-1][0]-2*ssim_20Percentile_arr2[-1][1]) & (igeo not in [183,248,278,306])) \
            # | (igeo in [220,228,238,364,366,391,394,396,397,453,480,481,481,482,483,]) : #sku4
                ssim_20Percentile_arr.pop(-1)
                ssim_20Percentile_id.pop(-1)
                ssim_20Percentile_arr2.pop(-1)
                print('')
                nbre_missed_consecutive += 1
                if nbre_missed_consecutive > 10: break
                continue
            nbre_missed_consecutive = 0
            
            igeo_prev.append(igeo)
            time_igni_prev.append(time_igni)
            georef_radiance_prev.append(georef_radiance)
            georef_temp_prev.append(georef_temp)
            georef_maskfull_prev.append(georef_maskfull)
            #georef_maskring_prev.append(georef_maskring)

            print('*')
            sys.stdout.flush()

        idx_bad = np.array ( np.sort(np.array(list(set(ssim_20Percentile_id_all)- set(ssim_20Percentile_id)))) )
        np.save(dir_out_refine_npy+'badFrameID_it{:02d}.npy'.format(itr),idx_bad)
   
        #ssim_20Percentile_arr_av = np.zeros(len(ssim_20Percentile_arr))
        #for i in range(winav/2,len(ssim_20Percentile_arr)-winav/2):
        #    ssim_20Percentile_arr_av[i] = np.array(ssim_20Percentile_arr[i-winav/2:i+winav/2]).sum()/winav

        #ssim_bad = [ ssim_20Percentile_arr_all[ np.where(ssim_20Percentile_id_all==xx) ]  for xx in idx_bad ] 
        ssim_bad = [ np.array(ssim_20Percentile_arr_all)[ np.where(ssim_20Percentile_id_all==xx) ][0]  for xx in idx_bad ]
        
        ssim_bad_auto   = [[],[]]
        ssim_bad_manual = [[],[]]
        for idx_,ssim_ in zip(idx_bad,ssim_bad): 
            iii = np.where( ssim_20Percentile_id_all == idx_ )[0][0]
            if (idx_ in  params_lwir_camera['final_selection_4bin']) \
               & ((ssim_20Percentile_arr_all[iii] >= ssim_20Percentile_arr2_all[iii][0]-2*ssim_20Percentile_arr2_all[iii][1])): 
                ssim_bad_manual[0].append(idx_)
                ssim_bad_manual[1].append(ssim_)
            else:
                ssim_bad_auto[0].append(idx_)
                ssim_bad_auto[1].append(ssim_)

        ssim_good_manual = [[],[]]
        for idx_,ssim_ in zip(ssim_20Percentile_id,ssim_20Percentile_arr): 
            if idx_ in  params_lwir_camera['final_selection_force2keep']: 
                ssim_good_manual[0].append(idx_)
                ssim_good_manual[1].append(ssim_)

       
        #########
        #plot final time series of SSIM
        #########
        
        mpl.rcdefaults()
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'STIXGeneral'
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}\usepackage{amssymb}'] #for \text command   
        mpl.rcParams['font.size'] = 14.
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['axes.labelsize'] = 14.
        mpl.rcParams['xtick.labelsize'] = 12.
        mpl.rcParams['ytick.labelsize'] = 12.
        mpl.rcParams['figure.subplot.left'] = .1
        mpl.rcParams['figure.subplot.right'] = .86
        mpl.rcParams['figure.subplot.top'] = 0.87
        mpl.rcParams['figure.subplot.bottom'] = .2
        mpl.rcParams['figure.subplot.hspace'] = 0.01
        mpl.rcParams['figure.subplot.wspace'] = 0.01
        
        fig = plt.figure(figsize=(9,6))
        ax = plt.subplot(111)
        
        ax.plot(ssim_20Percentile_id_all,                 ssim_20Percentile_arr_all,                 c='k',alpha=.3, label=r'$\text{SSIM}^{\text{prev}}$')
        ax.plot(ssim_20Percentile_id,                     ssim_20Percentile_arr,                     '-', c='k', label=r'$\text{SSIM}^{\text{prev}}_f$')
        axhandles, axlabels = ax.get_legend_handles_labels()

        axhandles.append(Line2D([0], [0], color='k', linewidth=1, linestyle=':'))
        axlabels.append('moving average\n'+r'$\text{SSIM}^{\text{prev}}[\text{id-}10:\text{id+}10]$')
        
        axhandles.append(mpatches.Patch(color='k',alpha=.3))
        axlabels.append('filtered moving std (x2)\n'+r'$\text{SSIM}^{\text{prev}}_f[\text{id-}10:\text{id}]$')

        x = np.array(ssim_20Percentile_id_all)
        ax.xaxis.tick_top() # x axis on top
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks(np.linspace(x.min(),x.max(),5))
        ax.xaxis.set_ticklabels([ r'${:3.0f}$'.format(xx) for xx in np.linspace(x.min(),x.max(),5)])
        ax.set_ylim(.5, 1.) 
        ax.set_xlim(x.min(),x.max()) 
        ax.set_xlabel('image id',labelpad=15)
        ax.set_ylabel(r'$\text{SSIM}^{\text{prev}}$',labelpad=15)
        
        axins = zoomed_inset_axes(ax, 6, loc='lower right', bbox_to_anchor=[.98,-0.15,], bbox_transform=fig.transFigure, axes_kwargs={'aspect':7.e2, },)

        axins.plot(ssim_20Percentile_id,                     ssim_20Percentile_arr,                     '-', c='k')
        axins.plot(ssim_20Percentile_id_all,                 ssim_20Percentile_arr_all,                 c='k',alpha=.3)
        
        axins.scatter(ssim_bad_manual[0],ssim_bad_manual[1], marker=r'$\bigotimes$', facecolors='none', edgecolors='black',s=69, label='False Positive')
        #axins.scatter(ssim_bad_auto[0],ssim_bad_auto[1],     marker=r'$\times$', facecolors='none', edgecolors='black',s=60, label='True Positive')
        axins.scatter(ssim_good_manual[0],ssim_good_manual[1], marker='o',c='k', s=60, label='False Negative')

        #axins.plot(ssim_20Percentile_id_av[winav/2:-winav/2],ssim_20Percentile_arr_av[winav/2:-winav/2],c='k',linestyle=':')
        axins.plot(ssim_20Percentile_id_av, ssim_20Percentile_arr_av,c='k',linestyle=':')

        ssim_20Percentile_arr2 = np.array(ssim_20Percentile_arr2)
        #axins.fill_between(ssim_20Percentile_id[winav/2:-winav/2],ssim_20Percentile_arr2[winav/2:-winav/2,0]-2*ssim_20Percentile_arr2[winav/2:-winav/2,1],
        #                                                        ssim_20Percentile_arr2[winav/2:-winav/2,0]+2*ssim_20Percentile_arr2[winav/2:-winav/2,1],interpolate=True,color='k',alpha=.2)
        axins.fill_between(ssim_20Percentile_id,ssim_20Percentile_arr2[:,0]-2*ssim_20Percentile_arr2[:,1],
                                                                ssim_20Percentile_arr2[:,0]+2*ssim_20Percentile_arr2[:,1],interpolate=True,color='k',alpha=.2)

        x1,x2 = 400, 500
        y1,y2 = .84, .96
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)


        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    
        legins = axins.legend(title=r'outliers')
        legins._legend_box.align = "left" 
        
        leg = ax.legend(handles=axhandles, labels=axlabels, loc='lower left', facecolor='white', framealpha=.5)
        leg._legend_box.align = "left" 

        fig.savefig(dir_out_lwir_image+'{:s}_filter_lwir_ssimprev4.png'.format(runName.split('_')[0]),dpi=400)
        plt.close(fig)


