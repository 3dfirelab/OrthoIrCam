from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import input
from builtins import range
from past.utils import old_div
import sys, os, glob, pdb
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
#mpl.use('TkAgg')
#if mpl.get_backend() != 'TkAgg':
#    reload(mpl)
#    mpl.use('TkAgg', warn=False, force=True)
import matplotlib.pyplot as plt 
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2 
import scipy
import asciitable
import argparse
#import imp
import shutil
import datetime
from osgeo import gdal,osr,ogr
import pickle 
from PIL import Image, ImageDraw
from netCDF4 import Dataset
import subprocess
from scipy import ndimage
import multiprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import importlib
from pyproj import CRS


import tools 
import visible as camera_visible
#########################################################
if __name__ == '__main__':
#########################################################

    factor_finalReds = 10

    time_start_run = datetime.datetime.now()

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

    inputConfig = importlib.machinery.SourceFileLoader('config_'+runName,os.getcwd()+'/../input_config/config_'+runName+'.py').load_module()
    if socket.gethostname() == 'moritz':
        path_ = '/media/paugam/toolcoz/data/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/paugam/data/',path_)
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/paugam/data/',path_)
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/paugam/data/',path_)
    if socket.gethostname() == 'coreff':
        path_ = '/media/paugam/goulven/data/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/paugam/data/',path_)
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/paugam/data/',path_)
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/paugam/data/',path_)
    elif socket.gethostname() == 'ibo':
        path_ = '/space/paugam/data/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/','/space/')
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/','/space/')
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/','/space/')
    # control flag
    flag_georef_mode = inputConfig.params_flag['flag_georef_mode']
    if flag_georef_mode   == 'WithTerrain'     : georefMode = 'WT'
    elif flag_georef_mode == 'SimpleHomography': georefMode = 'SH'
    flag_parallel = inputConfig.params_flag['flag_parallel']
   
    # input parameters
    params_grid           = inputConfig.params_grid
    params_gps            = inputConfig.params_gps 
    params_camera_lwir    = inputConfig.params_lwir_camera
    params_camera_mir     = inputConfig.params_mir_camera
    params_camera_visible = inputConfig.params_vis_camera
    params_rawData        = inputConfig.params_rawData
    params_georef         = inputConfig.params_georef

    plotname          = params_rawData['plotname']
    root_postproc     = params_rawData['root_postproc']
    camera_lwir_name        = params_camera_lwir['camera_name'] 
    camera_visible_name     = params_camera_visible['camera_name'] 
    
    dir_dem           = root_postproc + params_georef['dir_dem_input']

    dir_out_mir           = root_postproc + params_camera_mir['dir_input']
    dir_out_lwir           = root_postproc + params_camera_lwir['dir_input']
    dir_out_visible         = root_postproc + params_camera_visible['dir_input']
    dir_out_Rawvisible         = params_rawData['root_data'] + params_rawData['root_data_DirVis']
    dir_dem           = root_postproc + params_georef['dir_dem_input']
    
    dir_out_lwir_georef_npy = dir_out_lwir + 'Georef_{:s}/npy/'.format(georefMode)
    dir_out_lwir_frame      = dir_out_lwir + 'Frames/'

    dir_out_visible_georef_npy = dir_out_visible + 'Georef_{:s}/npy/'.format(georefMode)
    dir_out_visible_frame = dir_out_visible + 'Frames/'
   
    dir_out_full_RGB = dir_out_visible + 'Georef_{:s}/full_rgb/'.format(georefMode)
    tools.ensure_dir(dir_out_full_RGB)


    #load grid
    grid = np.load(root_postproc+'grid_'+plotname+'.npy')
    grid = grid.view(np.recarray)
    # Load the .prj file
    with open(root_postproc+'grid_'+plotname+'.prj', 'r') as file:
        prj_data = file.read()
    # Create a CRS object
    crs = CRS.from_wkt(prj_data)
    
    #define homography from lwir to vis
    gcps_vis = np.array([  [ 2807, 240],[1387, 189],[2824,2054],[1423,2048]])
    gcps_lwir = np.array([ [  496, 91 ],[221,  85],[490,  451],[228,  448]])

    H_lwir2vis, _ = cv2.findHomography(gcps_lwir, gcps_vis)


    #loop over image in raw 3bands img in data
    #filenames = np.load(dir_out+ params_camera_visible['dir_img_input'] + 'filename.npy' ) 
    filenames_lwir_time = np.load(dir_out_lwir+ params_camera_lwir['dir_img_input'] + 'filename_time.npy', allow_pickle=True) 
    filenames_lwir_time = filenames_lwir_time.view(np.recarray)
    
    filenames_vis_time = np.load(dir_out_visible+ params_camera_visible['dir_img_input'] + 'filename_time.npy', allow_pickle=True) 
    filenames_vis_time = filenames_vis_time.view(np.recarray) 
    #to correct time
    filenamesRaw_vis_time = np.load(dir_out_Rawvisible + 'filename_time_raw.npy', allow_pickle=True) 
    filenamesRaw_vis_time = filenamesRaw_vis_time.view(np.recarray)

    for ii in range(len(filenames_vis_time)):
        if '.png' not in filenames_vis_time.name[ii]: continue
        idx = np.where(filenames_vis_time.name[ii] == filenamesRaw_vis_time.filename.astype(str))
        filenames_vis_time.time[ii] = filenamesRaw_vis_time.time_igni[idx]

    dir_in_vis = dir_out_visible+ params_camera_visible['dir_img_input']

    #set the grid to 5cm
    dxy = 5.e-2
    nx5, ny5 = grid.grid_e.shape[0]*10, grid.grid_e.shape[1]*10
    grid5_e = np.arange(nx5)*dxy + grid.grid_e.min()
    grid5_n = np.arange(ny5)*dxy + grid.grid_n.min()
    xv, yv = np.meshgrid(grid5_e, grid5_n)
    maskburnt = np.zeros((nx5,ny5)) 

    grid5 = np.zeros_like(maskburnt, dtype=([('grid_e',float),('grid_n',float),('mask',float)]))
    grid5 = grid5.view(np.recarray)
    grid5.grid_e, grid5.grid_n, grid5.mask = xv.T, yv.T, maskburnt
    dx5 = grid5.grid_e[1,1]-grid.grid_e[0,0]
    dy5 = grid5.grid_n[1,1]-grid.grid_n[0,0]
    nx5, ny5 = grid5.grid_e.shape

    #save all frames in netcdf

    ncfile = Dataset(root_postproc+params_camera_visible['dir_input']+'Georef_SH/netcdf/{:s}.nc'.format(params_rawData['plotname']),'w')

    ncfile.description = 'orthorectified data for plot ' + plotname 

    # Global attributes
    setattr(ncfile, 'created', 'R. Paugam') 
    setattr(ncfile, 'title', ' ortho data ' + plotname)
    setattr(ncfile, 'Conventions', 'CF')

    # dimensions
    ncfile.createDimension('easting',grid5.grid_e.shape[0])
    ncfile.createDimension('northing',grid5.grid_n.shape[1])
    ncfile.createDimension('time',None)

    # set dimension
    ncx = ncfile.createVariable('easting', 'f8', ('easting',))
    setattr(ncx, 'long_name', 'easting UTM WGS84 UTM Zone {:s}'.format(crs.utm_zone))
    setattr(ncx, 'standard_name', 'easting')
    setattr(ncx, 'units','m')

    ncy = ncfile.createVariable('northing', 'f8', ('northing',))
    setattr(ncy, 'long_name', 'northing UTM WGS84 UTM Zone {:s}'.format(crs.utm_zone))
    setattr(ncy, 'standard_name', 'northing')
    setattr(ncy, 'units','m')
    
    ncTime = ncfile.createVariable('time', 'f8', ('time',))
    setattr(ncTime, 'long_name', 'time')
    setattr(ncTime, 'standard_name', 'time')
    setattr(ncTime, 'units','seconds since 1970-1-1')
    time_ref_nc=datetime.datetime(1970,1,1)

    # set Variables
    ncVarlwir    = ncfile.createVariable('btLwir','float32', (u'time',u'northing',u'easting',), fill_value=-999.)
    setattr(ncVarlwir, 'long_name', 'lwir brightness temperature') 
    setattr(ncVarlwir, 'standard_name', 'btLwir') 
    setattr(ncVarlwir, 'units', 'K') 
    
    ncVarvisr    = ncfile.createVariable('visr',np.uint8, (u'time',u'northing',u'easting',), fill_value=-999.)
    setattr(ncVarvisr, 'long_name', 'visr') 
    setattr(ncVarvisr, 'standard_name', 'visr') 
    setattr(ncVarvisr, 'units', '-') 
    setattr(ncVarvisr, 'grid_mapping', "wkt")
    ncVarvisg    = ncfile.createVariable('visg',np.uint8, (u'time',u'northing',u'easting',), fill_value=-999.)
    setattr(ncVarvisg, 'long_name', 'visg') 
    setattr(ncVarvisg, 'standard_name', 'visg') 
    setattr(ncVarvisg, 'units', '-') 
    setattr(ncVarvisg, 'grid_mapping', "wkt")
    ncVarvisb    = ncfile.createVariable('visb',np.uint8, (u'time',u'northing',u'easting',), fill_value=-999.)
    setattr(ncVarvisb, 'long_name', 'visb') 
    setattr(ncVarvisb, 'standard_name', 'visb') 
    setattr(ncVarvisb, 'units', '-') 
    setattr(ncVarvisb, 'grid_mapping', "wkt")
    
    # set projection
    ncprj    = ncfile.createVariable('wkt',    'c', ())
    setattr(ncprj, 'crs', crs.to_wkt()) 

    

    #write grid
    ncx[:] = grid5.grid_e[:,0]
    ncy[:] = grid5.grid_n[0,:]
    
    i_time_nc = 0


    out4bands = []
    outtime = []
    filenametimelwir = []
    for idframe, (filename, time_igni, time) in enumerate(filenames_vis_time): 
        filename = dir_in_vis + filename

        #img_raw = np.transpose(np.array( Image.open( filename)), [1,0,2] ) 
        img_raw = np.array( Image.open( filename) ) 
        #info    = np.load('{:s}{:s}_georef_{:06}_SH.npy'.format(dir_out_visible_georef_npy,plotname,idframe), allow_pickle=True)
        frame    = camera_visible.load_existing_file(params_camera_visible, dir_out_visible_frame+'frame{:06d}.nc'.format(idframe))

        
        ni, nj = grid.shape
        georef_img = np.zeros([ni*factor_finalReds,nj*factor_finalReds,3],dtype=np.uint8)    
        for ii in range(img_raw.shape[2]):
            img_raw_ = img_raw[:,:,ii]
            bufferZ =  int(frame.bufferZone * frame.shrink_factor)
            img_band2 = np.zeros( ((frame.img.shape[0]-frame.bufferZone)*frame.shrink_factor+2*bufferZ,  
                                   (frame.img.shape[1]-frame.bufferZone)*frame.shrink_factor+2*bufferZ) , dtype=img_raw_.dtype)
            img_band2[bufferZ:-bufferZ,bufferZ:-bufferZ] = img_raw_
           
            srhk = np.zeros([3,3])
            srhk[0,0] = 1./frame.shrink_factor
            srhk[1,1] = 1./frame.shrink_factor
            srhk[2,2] = 1

            usrhk = np.zeros([3,3])
            usrhk[0,0] = factor_finalReds
            usrhk[1,1] = factor_finalReds
            usrhk[2,2] = 1

            georef_img[:,:,ii] = cv2.warpPerspective(img_band2,                 \
                                                 np.array(usrhk.dot(frame.H2Grid).dot(srhk)),      \
                                                 georef_img.shape[:2],                 \
                                                 borderValue=0,flags=cv2.INTER_LINEAR  ) 

        

        idx_lwir = np.abs(filenames_lwir_time.time - time_igni).argmin()
        info_lwir = np.load(dir_out_lwir+params_camera_lwir['dir_img_input']+filenames_lwir_time.name[idx_lwir],allow_pickle=True)

        lwir_ = info_lwir[2]* 0.04 # conversion to K as spec in https://github.com/LJMUAstroecology/flirpy/blob/main/README.md 
        
        print(os.path.basename(filename), time_igni, idx_lwir,info_lwir[1], end='\r')
        
        flag_lwir_available = True
        if abs(info_lwir[1] - time_igni) > 0.5: flag_lwir_available = False
        
        '''
        fig = plt.figure()
        ax = plt.subplot(131)
        ax.imshow(img_raw_.T, origin='lower')
        ax.scatter(gcps_vis[:,1],gcps_vis[:,0],c='k')
        
        ax = plt.subplot(132)
        ax.imshow(lwir_.T, origin='lower')
        ax.scatter(gcps_lwir[:,1],gcps_lwir[:,0],c='k')
        '''
        onvis_lwir = cv2.warpPerspective(lwir_[:,::-1].T,                 \
                                                 np.array(H_lwir2vis),      \
                                                 img_raw_.shape[:2][::-1],                 \
                                                 borderValue=0,flags=cv2.INTER_LINEAR  ) 
        onvis_lwir_band = np.zeros(img_band2.shape[:2])
        onvis_lwir_band[bufferZ:-bufferZ,bufferZ:-bufferZ] = onvis_lwir

        georef_lwir = cv2.warpPerspective(onvis_lwir_band,                 \
                                                 np.array(usrhk.dot(frame.H2Grid).dot(srhk)), \
                                                 georef_img.shape[:2][::-1],                 \
                                                 borderValue=0,flags=cv2.INTER_LINEAR  ) 
   
        if flag_lwir_available:
            out4bands.append([georef_img, georef_lwir])
            outtime.append([time_igni,info_lwir[1]])
            filenametimelwir.append([filenames_lwir_time.name[idx_lwir], info_lwir[1], info_lwir[0] ])
            #save lwir npy
            np.save(dir_out_lwir_georef_npy + filenames_lwir_time.name[idx_lwir], np.array([ [info_lwir[0],info_lwir[1],[None],[None],None], \
                                                                                          None, None,None,None,  \
                                                                                          georef_lwir,    \
                                                                                          None ],dtype=object) )
        #plotting    
        mpl.rcdefaults()
        mpl.rcParams['text.usetex'] = True
        #mpl.rcParams['font.family'] = 'Comic Sans MS'
        mpl.rcParams['font.size'] = 16.
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['axes.labelsize'] = 14.
        mpl.rcParams['xtick.labelsize'] = 14.
        mpl.rcParams['ytick.labelsize'] = 14.
        mpl.rcParams['figure.subplot.left'] = .00
        mpl.rcParams['figure.subplot.right'] = 1.
        mpl.rcParams['figure.subplot.top'] = 1.
        mpl.rcParams['figure.subplot.bottom'] = 0.09
        mpl.rcParams['figure.subplot.hspace'] = 0.02
        mpl.rcParams['figure.subplot.wspace'] = 0.02     
        fig = plt.figure(figsize=(8.2,8))
            
        lwirmin = 29
        lwirmax = 900
        values = np.linspace(100,700,10)
        ax = plt.subplot(111)
        ax.imshow(np.transpose(georef_img,[1,0,2]), origin='lower')
        divider = make_axes_locatable(ax)
        if flag_lwir_available: 
            im = ax.contour(np.ma.masked_where(georef_lwir<lwirmin,georef_lwir).T, origin='lower',levels=values)
            cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
            cbar = fig.colorbar(im ,cax = cbaxes,orientation='horizontal')
            cbar.set_label('T (K)')
        ax.set_axis_off()
       
        ax.text(.01, .99, 't={:.3f}s | diff vis-lwir {:04.1f}ms'.format(info_lwir[1],1000*(info_lwir[1] - time_igni) ), ha='left', va='top', transform=ax.transAxes,c='w',size=18)

        fig.savefig(dir_out_full_RGB+'frame{:06d}.png'.format(idframe),dpi=200)
        plt.close(fig)

        
        #save netcdf
        #time
        ncTime[i_time_nc] =  info_lwir[1]

        #layer
        ncVarlwir[i_time_nc,:,:] = np.swapaxes(georef_lwir,0,1)
        ncVarvisr[i_time_nc,:,:] =  np.swapaxes(georef_img[:,:,0],0,1).astype(np.uint8)
        ncVarvisg[i_time_nc,:,:] =  np.swapaxes(georef_img[:,:,1],0,1).astype(np.uint8)
        ncVarvisb[i_time_nc,:,:] =  np.swapaxes(georef_img[:,:,2],0,1).astype(np.uint8)

        i_time_nc += 1
    
    #close file
    ncfile.close()

