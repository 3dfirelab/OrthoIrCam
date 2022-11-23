from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys,os,glob
from netCDF4 import Dataset
import datetime
import argparse
import imp 
from osgeo import gdal,osr,ogr
import subprocess
from PIL import Image
import cv2 
import pdb 
from scipy import ndimage
import socket 
import shutil
from PIL import Image, ImageDraw
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import importlib 
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#homebrwed
import spectralTools
import tools
import camera_tools as cameraTools

def rotateImage(image, angle):
  image_center = tuple(old_div(np.array(image.shape[1::-1]), 2))
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


#########################################################
if __name__ == '__main__':
#########################################################

    time_start_run = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='this combines georeferenced lwir, mir and visible in the same time stamped nc file.')
    parser.add_argument('-i','--input', help='Input run name',required=True)
    parser.add_argument('-a','--angle', help='rotate angle',required=True)
    parser.add_argument('-t','--timeperiod', help='rgb image time period',required=False)
    parser.add_argument('-s','--newStart',  help='True then it uses existing data',required=False)

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
        rotationAngle = float(args.angle)
        if args.timeperiod is not None: 
            rgbTimePeriod = float(args.timeperiod)
        else:
            rgbTimePeriod = 0
    if (args.newStart is None) : 
        flag_restart = True    
    else: 
        flag_restart = tools.string_2_bool(args.newStart)
    
    inputConfig = imp.load_source('config_'+runName,os.getcwd()+'/../input_config/config_'+runName+'.py')
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
    params_rawData        = inputConfig.params_rawData
    params_georef         = inputConfig.params_georef

    plotname          = params_rawData['plotname']
    root_postproc     = params_rawData['root_postproc']
    camera_lwir_name        = params_camera_lwir['camera_name'] 

    dir_out_lwir           = root_postproc + params_camera_lwir['dir_input']
    dir_dem           = root_postproc + params_georef['dir_dem_input']
    
    dir_out_lwir_georef_npy = dir_out_lwir + 'Georef_refined_{:s}/npy/'.format(georefMode)
    dir_out_lwir_frame      = dir_out_lwir + 'Frames/'
    
    print('input directories for LWIR is: '); print('#####')
    print(dir_out_lwir)

    if 'agema' in params_camera_lwir['camera_name']: 
        import flir as camera_lwir

    elif 'optris' in params_camera_lwir['camera_name']:
        import optris as camera_lwir
    
    importlib.reload(camera_lwir)
    importlib.reload(spectralTools)
    importlib.reload(tools)

    dir_out_combined = root_postproc+'OrthoData/'
    if flag_restart is False: 
        if os.path.isdir(dir_out_combined): shutil.rmtree(dir_out_combined)

    tools.ensure_dir(dir_out_combined)
    tools.ensure_dir(dir_out_combined+'Raw/')
    tools.ensure_dir(dir_out_combined+'Raw/LWIR/')
    tools.ensure_dir(dir_out_combined+'LWIR/')

    my_cmap_rgb = plt.get_cmap('bwr')(np.arange(256))
    alpha = 0.3
    for i in range(3): # Do not include the last column!
        my_cmap_rgb[:,i] = (1 - alpha) + alpha*my_cmap_rgb[:,i]
    my_cmap = mpl.colors.ListedColormap(my_cmap_rgb, name='my_cmap')


    #read ignition time
    file_ignition_time = params_rawData['root_data'] + 'ignition_time.dat'
    f = open(file_ignition_time,'r')
    lines = f.readlines()
    ignitionTime = datetime.datetime.strptime(params_rawData['fire_date']+'_'+lines[0].rstrip(), "%Y-%m-%d_%H:%M:%S")
    endTime = datetime.datetime.strptime(params_rawData['fire_date']+'_'+lines[1].rstrip(), "%Y-%m-%d_%H:%M:%S")
    fire_durationTime = (endTime-ignitionTime).total_seconds()


    #load grid
    grid = np.load(root_postproc+'grid_'+plotname+'.npy')
    grid = grid.view(np.recarray)
    grid_e, grid_n, plotMask  = grid.grid_e, grid.grid_n, grid.mask
    dx = grid_e[1,1]-grid_e[0,0]
    dy = grid_n[1,1]-grid_n[0,0]
    nx, ny = grid_e.shape
    prj_info = open(root_postproc+'grid_'+plotname+'.prj').readline()
    prj = osr.SpatialReference()
    prj.ImportFromWkt(prj_info)
    idx_plot = np.where(rotateImage(plotMask,rotationAngle)==2)


    #load file name
    lwir_georef00    = sorted(glob.glob(dir_out_lwir_georef_npy+'*'+georefMode+'*.npy'))
 
    frameLwir_ref00 = camera_lwir.load_existing_file(params_camera_lwir, dir_out_lwir_frame + '/frame{:06d}.nc'.format(0))
    
    print('set Distortion to 0 for optris')
    D = np.zeros(5) 
    
    #load time
    print('load time ', end=' ') 
    sys.stdout.flush()
    if (not(os.path.isfile(dir_out_combined+'time_cameras_info.npy'))) | (flag_restart == False):
        print('from lwir ... ', end=' ')
        sys.stdout.flush()
        lwir_time = []
        lwir_id = []  
   
        lwir_id, lwir_time, lwir_georef = tools.load_good_lwir_frame_selection(flag_restart, dir_out_lwir_georef_npy,
                                                                               georefMode=georefMode)
       
        lwir_time = np.array(lwir_time)
        lwir_id = np.array(lwir_id)    


        np.save(dir_out_combined+'time_cameras_info.npy',[lwir_time,lwir_id])
        np.save(dir_out_combined+'lwir_selection.npy',lwir_georef)
    
    else:
        print('... ', end=' ')
        lwir_time,lwir_id = np.load(dir_out_combined+'time_cameras_info.npy', allow_pickle=True, encoding='latin1')
        lwir_georef = np.load(dir_out_combined+'lwir_selection.npy')

    print('done')


    #Create Nectdf
    #loop over the visible and overlay the closest lwir frame in time
    ncfile = Dataset(dir_out_combined+plotname+'_georef_lwir_mir_visible.nc','w')
    ncfile.description = 'lwir frames generated by OrthoIrCam'

    # Global attributes
    setattr(ncfile, 'created', 'R. Paugam') 
    setattr(ncfile, 'title', 'lwir frames')
    setattr(ncfile, 'Conventions', 'CF')

    # create dimensions
    ncfile.createDimension('easting' ,nx)
    ncfile.createDimension('northing',ny)
    ncfile.createDimension('time',None)

    # set dimension
    ncx = ncfile.createVariable('easting', 'f8', ('easting',))
    setattr(ncx, 'long_name', 'easting UTM WGS84 UTM Zone {:d}'.format(prj.GetUTMZone()))
    setattr(ncx, 'standard_name', 'easting')
    setattr(ncx, 'units','m')
    ncy = ncfile.createVariable('northing', 'f8', ('northing',))
    setattr(ncy, 'long_name', 'northing UTM WGS84 UTM Zone {:d}'.format(prj.GetUTMZone()))
    setattr(ncy, 'standard_name', 'northing')
    setattr(ncy, 'units','m')
    ncTime = ncfile.createVariable('time', 'f8', ('time',))
    setattr(ncTime, 'long_name', 'time')
    setattr(ncTime, 'standard_name', 'time')
    setattr(ncTime, 'units','seconds since 1970-1-1')
    time_ref_nc=datetime.datetime(1970,1,1)

    # create Variables
    nc_lwirRad    = ncfile.createVariable('lwir_rad','f8', (u'time',u'northing',u'easting',), fill_value=-999.)
    setattr(nc_lwirRad, 'long_name', 'lwir radiance') 
    setattr(nc_lwirRad, 'standard_name', 'lwir_rad') 
    setattr(nc_lwirRad, 'units', 'kW/m2') 
    nc_lwirTemp    = ncfile.createVariable('lwir_temp','f8', (u'time',u'northing',u'easting',), fill_value=-999.)
    setattr(nc_lwirTemp, 'long_name', 'lwir brightness temperature') 
    setattr(nc_lwirTemp, 'standard_name', 'lwir_temp') 
    setattr(nc_lwirTemp, 'units', 'K') 

    # set projection
    ncprj    = ncfile.createVariable('transver_mercator',    'c', ())
    setattr(ncprj, 'spatial_ref', prj.ExportToWkt() ) 

    #write grid
    ncx[:] = grid_e[:,0] + .5*dx
    ncy[:] = grid_n[0,:] + .5*dy
        
    ignitionTime_since1970 = (ignitionTime - datetime.datetime(1970,1,1)).total_seconds()
    levels_lwirb=np.linspace(290,800,10)
        
    #camera pose
    #cameraPosition = []
    #cameraAngle = []

    #loop over the lwir and overlay the closest visible and mir frame in time
    for i_time, [lwir_filename,lwir_id_,lwir_time_] in enumerate(zip(lwir_georef, lwir_id, lwir_time)):
        
        #time
        ncTime[i_time] = np.round( ((ignitionTime + datetime.timedelta(seconds=lwir_time[i_time])) - time_ref_nc).total_seconds() ,3 )
        
        #closest lwir
        frameLwir = camera_lwir.load_existing_file(params_camera_lwir, dir_out_lwir_frame + '/frame{:06d}.nc'.format(int(lwir_id_)))
        lwir_infos = np.load(lwir_filename,allow_pickle=True,encoding='latin1')
        
        nc_lwirTemp[i_time,:,:] = np.array(np.swapaxes(lwir_infos[3],0,1),dtype=float)
        nc_lwirRad[i_time,:,:]  = np.array(np.swapaxes(lwir_infos[4],0,1),dtype=float)
        temp_lwir = np.array(lwir_infos[3],dtype=float)
        mask_lwir = np.array(lwir_infos[2],dtype=float)
        Href_lwir = lwir_infos[1]
        
        print(frameLwir.id, end=' ') 

           
        print(' || {:5.2f} | {:5.2f} | {:5.2f}'.format(ncTime[i_time]-ignitionTime_since1970, ncTime[i_time]-ncTime[i_time-1],
                                                                              lwir_time_))
       
        temp_lwir = rotateImage(temp_lwir,rotationAngle)

        # plot raw 
        ########
       

        #lwir
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
        mpl.rcParams['figure.subplot.hspace'] = 0.0
        mpl.rcParams['figure.subplot.wspace'] = 0.0
        ratio = old_div(1.* frameLwir.temp.shape[0],frameLwir.temp.shape[1])
        fig = plt.figure(figsize=(8.*ratio,8))
        ax = plt.subplot(111)
        ax.imshow(frameLwir.temp[50:-50,50:-50].T,origin='lower',vmin=levels_lwirb.min(),vmax=levels_lwirb.max(),cmap=mpl.cm.jet)
        ax.set_axis_off()
        time_sec      = int(lwir_time_)
        time_millisec = int(1000*(lwir_time_-int(lwir_time_)))

        fig.savefig('{:s}{:s}_t_{:06d}_{:03d}s.png'.format(dir_out_combined+'Raw/LWIR/',plotname,time_sec,time_millisec))
        plt.close(fig)


        # plot georef lwir
        ########
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
        mpl.rcParams['figure.subplot.hspace'] = 0.0
        mpl.rcParams['figure.subplot.wspace'] = 0.0
        buffer = 40
        ratio = old_div(1.*( min([idx_plot[0].max()+buffer,grid.shape[0]]) - max([idx_plot[0].min()-buffer,0]) ),\
                   ( min([idx_plot[1].max()+buffer,grid.shape[1]]) - max([idx_plot[1].min()-buffer,0]) ))
        fig = plt.figure(figsize=(8.*ratio,8))
        ax = plt.subplot(111)

        im = ax.imshow(np.ma.masked_where(mask_lwir==0,temp_lwir)[ max([idx_plot[0].min()-buffer,0]):min([idx_plot[0].max()+buffer,temp_lwir.shape[0]]),
                                   max([idx_plot[1].min()-buffer,0]):min([idx_plot[1].max()+buffer,temp_lwir.shape[1]])].T,
                                   origin='lower', cmap=mpl.cm.jet, 
                                   vmin=levels_lwirb.min(),vmax=levels_lwirb.max())
        ax.annotate('t={:04.1f}s'.format(lwir_time[i_time]), (0.1,0.9), ha='left', va='top', transform=ax.transAxes)
        ax.set_axis_off()

        ax.text(0.9,0.94, 't={: >8.2f}s'.format(lwir_time[i_time]), fontdict=dict(fontsize=14, fontweight='bold'), bbox=dict(facecolor='white', alpha=0.4, edgecolor='none'),transform=ax.transAxes)
        
        cbbox = inset_axes(ax, '15%', '90%', loc = 'center left')
        [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
        cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        cbbox.set_facecolor([1,1,1,0.5])

        cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
        #cbaxes = fig.add_axes([0.05, .08, .4, .05])
        ticks_levels_lwirb = [ r'${:.0f}$'.format(t) for t in levels_lwirb]
        ticks_levels_lwirb[-1] = r'$>$'+ticks_levels_lwirb[-1]
        cbar = fig.colorbar(im ,cax = cbaxes,orientation='vertical',ticks=levels_lwirb,)
        cbar.ax.set_yticklabels(ticks_levels_lwirb)
        cbar.set_label('LWIR BT (K)') 


        fig.savefig('{:s}{:s}_t_{:06d}_{:03d}s.png'.format(dir_out_combined+'LWIR/',plotname,time_sec,time_millisec))
        plt.close(fig)

    '''
    #plot colorbar in seprated fig for lwir imshow
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
    mpl.rcParams['figure.subplot.hspace'] = 0.0
    mpl.rcParams['figure.subplot.wspace'] = 0.0
    fig=plt.figure(figsize=(6,.9))
    cbaxes = fig.add_axes([0.1, .4, .8, .4])
    norm= mpl.colors.Normalize(vmin=levels_lwirb.min(), vmax=levels_lwirb.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
    sm.set_array([])
    ticks_levels_lwirb = [ r'${:.0f}$'.format(t) for t in levels_lwirb]
    ticks_levels_lwirb[-1] = r'$>$'+ticks_levels_lwirb[-1]
    cbar = fig.colorbar(sm ,cax = cbaxes,orientation='horizontal',ticks=levels_lwirb,)
    cbar.ax.set_xticklabels(ticks_levels_lwirb)
    fig.savefig('{:s}{:s}_colorbarLWIR.png'.format(dir_out_combined+'LWIR/',plotname))
    plt.close(fig)
    '''

    #close file
    ncfile.close()


