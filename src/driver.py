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
import socket
import importlib

#import warnings
#warnings.filterwarnings('error')
#np.seterr(all='raise')

#homebrewed
import tools
import camera_tools as cameraTools 
import spectralTools
import cornerFirePicker as cFP
import tools_georefWithTerrain as georefWT

def new_increment(i_file, incr_last, incr_default, status, dir_out_frame):
    if status == 'ok':
        if os.path.isfile(dir_out_frame+'frame{:06d}.nc'.format(i_file+1)):
            return 1
        else:
            return incr_default
    if status == 'failed':
        if incr_last<=1:
            return 1
        else:
            if i_file + (-1*incr_last + 1) > 0: 
                return -1*incr_last + 1
            else: 
                return 1


#########################################################
if __name__ == '__main__':
#########################################################

    time_start_run = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='this is the driver of the GeorefCam Algo.')
    parser.add_argument('-i','--input', help='Input run name',required=True)
    parser.add_argument('-m','--mode',  help='Camera type: "lwir" or "visible"',required=True)
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
    runMode = args.mode

    inputConfig = importlib.machinery.SourceFileLoader('config_'+runName,os.getcwd()+'/../input_config/config_'+runName+'.py').load_module()
    if socket.gethostname() == 'coreff':
        data_path_ = 'goulven/data/' # 'Kerlouan/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/paugam/data/','/media/paugam/'+data_path_)
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/paugam/data/','/media/paugam/'+data_path_)
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/paugam/data/','/media/paugam/'+data_path_)
    elif socket.gethostname() == 'ibo':
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/','/space/')
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/','/space/')
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/','/space/')
    
    #print '##'
    #print '##'

    # input parameters
    params_grid       = inputConfig.params_grid
    params_gps        = inputConfig.params_gps 
    if runMode == 'lwir'    : params_camera     = inputConfig.params_lwir_camera
    if runMode == 'visible' : params_camera     = inputConfig.params_vis_camera
    params_rawData    = inputConfig.params_rawData
    params_georef     = inputConfig.params_georef

    # control flag
    flag_georef_mode = inputConfig.params_flag['flag_georef_mode']
    if flag_georef_mode   == 'WithTerrain'     : 
        georefMode = 'WT'
        params_camera['dir_input'] = params_camera['dir_input'][:-1] + '_WT/'
    elif flag_georef_mode == 'SimpleHomography': 
        georefMode = 'SH'
    flag_parallel = inputConfig.params_flag['flag_parallel']
    flag_restart  = inputConfig.params_flag['flag_restart']
    if (args.newStart is None) : 
        if not(flag_restart): 
            res = 'na'
            while res not in {'y','n'}:
                res = input('are you sure you want to delete the existing data in {:s}? (y/n)'.format(params_camera['dir_input']))
            if res == 'n':
                print('stopped here')
                sys.exit()
    elif args.newStart is not None:
        flag_restart = tools.string_2_bool(args.newStart)

    flag_time2Changeref00 = False
    if runMode == 'visible':
        flag_processRawData = inputConfig.params_flag['flag_vis_processRawData']
        flag_plot_warp  = inputConfig.params_flag['flag_vis_plot_warp']
        flag_georef      = inputConfig.params_flag['flag_vis_georef']
        flag_plot_georef = inputConfig.params_flag['flag_vis_plot_georef']
    elif runMode == 'lwir':
        flag_processRawData = inputConfig.params_flag['flag_lwir_processRawData']
        flag_plot_warp  = inputConfig.params_flag['flag_lwir_plot_warp']
        flag_georef      = inputConfig.params_flag['flag_lwir_georef']
        flag_plot_georef = inputConfig.params_flag['flag_lwir_plot_georef']

    plotname          = params_rawData['plotname']
    root_postproc     = params_rawData['root_postproc']
    camera_name       = params_camera['camera_name'] 

    flag_track_feature = params_camera['track_mode'] #'track_cf' # 'track_background'

    dir_out           = root_postproc + params_camera['dir_input']
    dir_dem           = root_postproc + params_georef['dir_dem_input']
    
    dir_out_wkdir      = dir_out + 'Wkdir/'
    dir_out_georef_npy = dir_out + 'Georef_{:s}/npy/'.format(georefMode)
    dir_out_georef_png = dir_out + 'Georef_{:s}/png/'.format(georefMode)
    dir_out_georef_tif = dir_out + 'Georef_{:s}/tif/'.format(georefMode)
    dir_out_georef_kml = dir_out + 'Georef_{:s}/kml/'.format(georefMode)
    dir_out_georef_nc  = dir_out + 'Georef_{:s}/netcdf/'.format(georefMode)
    dir_out_frame      = dir_out + 'Frames/'
    dir_out_warping    = dir_out + 'Warping/'

    if (runMode == 'lwir') & ('optris' in params_camera['camera_name'])  :   import optris as camera
    if (runMode == 'lwir') & ('agema570' in params_camera['camera_name'])    :   import flir as camera
    if (runMode == 'lwir') & ('xt' in params_camera['camera_name'])    :   import flir as camera
    if runMode == 'visible' :   import visible as camera
    
    importlib.reload(camera)
    importlib.reload(cameraTools)
    importlib.reload(tools)
    importlib.reload(cFP)
    importlib.reload(georefWT)
    if runMode == 'lwir'    : importlib.reload(spectralTools)
    
    if not(flag_restart):
        if os.path.isdir(dir_out_wkdir): shutil.rmtree(dir_out_wkdir)
        if os.path.isdir(dir_out_georef_npy): shutil.rmtree(dir_out_georef_npy)
        if os.path.isdir(dir_out_georef_png): shutil.rmtree(dir_out_georef_png)
        if os.path.isdir(dir_out_georef_nc): shutil.rmtree(dir_out_georef_nc)
        if os.path.isdir(dir_out_georef_tif): shutil.rmtree(dir_out_georef_tif)
        if os.path.isdir(dir_out_frame): shutil.rmtree(dir_out_frame)
        if os.path.isdir(dir_out_warping): shutil.rmtree(dir_out_warping)
        if os.path.isdir(dir_out_georef_kml): shutil.rmtree(dir_out_georef_kml)

    tools.ensure_dir(dir_out_georef_npy)
    tools.ensure_dir(dir_out_georef_png)
    tools.ensure_dir(dir_out_georef_nc)
    tools.ensure_dir(dir_out_georef_tif)
    tools.ensure_dir(dir_out_frame)
    tools.ensure_dir(dir_out_warping)
    tools.ensure_dir(dir_out_georef_kml)
    
    if flag_georef_mode  == 'WithTerrain':
        tools.ensure_dir(dir_out_wkdir)
        wkdir = dir_out_wkdir + 'grid_img_lookuptable/'
        tools.ensure_dir(wkdir)
  

    #copy config file in output dir
    ############
    shutil.copy(os.getcwd()+'/../input_config/config_{:s}.py'.format(runName), dir_out+'config_{:s}_{:s}.py'.format(runName,georefMode))

    #read ignition time
    ############
    file_ignition_time = params_rawData['root_data'] + 'ignition_time.dat'
    f = open(file_ignition_time,'r')
    lines = f.readlines()
    ignitionTime = datetime.datetime.strptime(params_rawData['fire_date']+'_'+lines[0].rstrip(), "%Y-%m-%d_%H:%M:%S")
    endTime = datetime.datetime.strptime(params_rawData['fire_date']+'_'+lines[1].rstrip(), "%Y-%m-%d_%H:%M:%S")
    fire_durationTime = (endTime-ignitionTime).total_seconds()


    #tools for conversion to UTM
    ############################
    print('buiild grid tools')
    wgs84 = osr.SpatialReference( ) # Define a SpatialReference object
    wgs84.ImportFromEPSG( 4326 ) # And set it to WGS84 using the EPSG code
    utm = osr.SpatialReference()
    #utm.SetWellKnownGeogCS( 'WGS84' )
    try: 
        centerpt = tools.get_center_Coord_ll(params_gps['ctr_format'], root_postproc+params_gps['dir_gps']+params_gps['contour_file'], params_gps)
    except: 
        centerpt = tools.get_center_Coord_ll(params_gps['cf_format'],  root_postproc+params_gps['dir_gps']+params_gps['loc_cf_file'],  params_gps)
    flag_hemisphere = 1 if centerpt[1] > 0 else  0
    utm.SetUTM( tools.UTMZone( *centerpt) , flag_hemisphere)
    conv_ll2utm = osr.CoordinateTransformation(wgs84, utm)
    conv_utm2ll = osr.CoordinateTransformation(utm,wgs84)

    
    #process raw data
    ##################
    if flag_processRawData:
        plt.ioff()
        camera.processRawData(ignitionTime, params_rawData, params_camera, flag_restart)

    #if (mpl.get_backend() != 'Agg') & (mpl.get_backend() != 'agg'): 
    #    plt.ion()


    #Select the file name that are going to be read
    ###############################################
    filenames = np.array(sorted(glob.glob(dir_out + params_camera['dir_img_input'] +  params_camera['dir_img_input_*'])))
    filenames_timelookupTable = np.load(dir_out+ params_camera['dir_img_input'] + 'filename_time.npy')
    filenames_timelookupTable = filenames_timelookupTable.view(np.recarray)
    if os.path.isfile(dir_out+ params_camera['dir_img_input'] + 'filename.npy'):
        filenames  = np.load(dir_out+ params_camera['dir_img_input'] + 'filename.npy')
        filenames  = tools.bytesStringConv(filenames)
        
        if "/".join(filenames[0].split('/')[:-1])+'/' !=  dir_out+ params_camera['dir_img_input']:
            for ifilename, filename in enumerate(filenames):
                filenames[ifilename] =  dir_out+ params_camera['dir_img_input'] + os.path.basename(filename)
    else:
        filenames_basename = np.array(len(filenames)*[filenames[0]])
        for i, filename in enumerate(filenames):
            filenames_basename[i] = os.path.basename(filename)  
        
        #keep only file happening after  params_camera['time_start_processing'] and files which are not been manually removed
        intersect = np.intersect1d(filenames_timelookupTable.name,filenames_basename)
        filenames_removed = np.zeros(filenames_timelookupTable.shape)
        for i, filename_ in enumerate(filenames_timelookupTable.name):
            if filename_ not in intersect: filenames_removed[i]=1 
        idx = np.where( (filenames_removed == 0) & (filenames_timelookupTable.time>=params_camera['time_start_processing']) )
        filenames = np.core.defchararray.add( np.array(len(idx[0])* [os.path.dirname(filenames[0])+'/']) ,  filenames_timelookupTable.name[idx[0]] )

        np.save(dir_out+ params_camera['dir_img_input'] + 'filename', filenames ) 



    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 5000,
                           qualityLevel = params_camera['of_qualityLevel'], #lwir for sku4
                           #qualityLevel = 0.3, #lwir for sku4 and sha1
                           #qualityLevel = 0.2, #vis
                           minDistance = 11,
                           blockSize = params_camera['of_blockSize']) #11)  #21 lwir for sku4
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (21,21),
                      maxLevel = 7,  
                      criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1000, 0.01))
    
    '''
    # for Mulch 
    feature_params = dict( maxCorners = 1000,
                           qualityLevel = 0.3, 
                           minDistance = 7,
                           blockSize = 21 )
    lk_params = dict( winSize  = (11,11),
                      maxLevel = 11,  
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.05))
    '''


    #load grid, mask and location for gcp
    ###########################################
    print('load gcps')
    if ('cornerFireName' in list(params_gps.keys())):
        gcps_world    = tools.load_loc_cornerFire(root_postproc, params_gps, conv_ll2utm, cornerFireName=params_gps['cornerFireName'] )
    else:  
        gcps_world    = tools.load_loc_cornerFire(root_postproc, params_gps, conv_ll2utm )
    try: 
        camera_world  = tools.load_loc_camera(root_postproc+params_gps['dir_gps']+params_gps['loc_camera_file'], ignitionTime, conv_ll2utm) 
    except: 
        camera_world = None

    if (not(flag_restart)) | (not os.path.isfile(root_postproc+'grid_'+plotname+'.npy')):
        print('generate grid')
        try: 
            dem_filename = dir_dem+params_georef['dem_file']
        except: 
            dem_filename = None
        burnplot = tools.create_grid(root_postproc, plotname, 
                                     gcps_world, camera_world,
                                     params_gps, params_grid, params_camera,
                                     utm, conv_ll2utm, conv_utm2ll, flag_georef_mode, 
                                     dem_filename=dem_filename, 
                                     flag_plot=flag_plot_warp, dir_out=dir_dem)
        ''' 
        #apply correction on terrain if SimpleHomography to get it as a plan 
        #-------------------
        if flag_georef_mode  == 'SimpleHomography':
            burnplot.terrain = tools.get_best_plane(burnplot.grid_e.flatten(), burnplot.grid_n.flatten(), burnplot.terrain.flatten(), 
                                                    flag_plot=flag_plot_warp, dir_out=dir_dem, dimension=burnplot.shape, 
                                                    maskPlot=burnplot.mask ).reshape(burnplot.shape)
        '''
    else:
        burnplot = np.load(root_postproc+'grid_'+plotname+'.npy')
        burnplot = burnplot.view(np.recarray)

    grid_e, grid_n = burnplot.grid_e, burnplot.grid_n


    #and correct gcp to reflet terrain 
    #---------------------------------
    gcps_world_px = np.zeros([2,4])
    print('ensure gcps gps point with DEM are the same')
    print('terrain  gcps_world')  
    for icf in range(gcps_world.shape[1]):
        idx = np.where( (gcps_world[0,icf]>=grid_e[:-1,:-1]) & (gcps_world[0,icf]<grid_e[1:,1:]) &\
                        (gcps_world[1,icf]>=grid_n[:-1,:-1]) & (gcps_world[1,icf]<grid_n[1:,1:])  )
        if len(idx[0])!=1:
            pdb.set_trace()
        idx = (idx[1][0], idx[0][0]) 
        print(burnplot.terrain[idx], gcps_world[2,icf])
        gcps_world[2,icf] = burnplot.terrain[idx]
        gcps_world_px[:2,icf] = idx

    '''# activate for testing
    nx, ny = grid_e.shape
    grid_e   = fS2D.downgrade_resolution_4nadir(grid_e, np.zeros([nx/8,ny/8]) , flag_interpolation='min' )
    grid_n   = fS2D.downgrade_resolution_4nadir(grid_n, np.zeros([nx/8,ny/8]) , flag_interpolation='min' ) 
    plotMask = fS2D.downgrade_resolution_4nadir(plotMask, np.zeros([nx/8,ny/8]) , flag_interpolation='min' )
    terrain  = fS2D.downgrade_resolution_4nadir(terrain, np.zeros([nx/8,ny/8]) , flag_interpolation='average' )
    '''
    dx, dy = grid_e[1,1]- grid_e[0,0], grid_n[1,1]- grid_n[0,0]
    nx, ny = grid_e.shape

    if flag_georef_mode  == 'WithTerrain':
        #triangulate terrain and map triangle
        #------------------------------------
        if (not(flag_restart)) | (not(os.path.isfile(dir_dem + 'tri_angles.npy'))):
            tri, points, triangles, triangles_grid, triangles_area = georefWT.set_triangle(burnplot, dir_dem)
        else:
            print('')
            print('load triangulation of the terrain')
            tri, points    = np.load(dir_dem + 'tri_angles.npy'    )
            triangles      = np.load(dir_dem + 'triangle_coord.npy').tolist()
            triangles_grid = np.load(dir_dem + 'triangle_grid.npy' ).tolist()
            triangles_area = np.load(dir_dem + 'triangle_aera.npy' ).tolist()


    #load mask with bare ground
    try:
        kml_file = root_postproc+params_gps['dir_gps']+params_gps['contour_file_bareGround']
        pts_extra = tools.load_polygon_from_kml(kml_file,params_gps['contour_file_bareGround_polygonName'])
        
        polygon = []
        for pt_ll in zip(pts_extra[0],pts_extra[1]):
            pt_utm = list(conv_ll2utm.TransformPoint(*pt_ll)[:2])
            idx = np.where( (pt_utm[0]>=burnplot.grid_e[:-1,:-1]) & (pt_utm[0]<burnplot.grid_e[1:,1:]) &\
                            (pt_utm[1]>=burnplot.grid_n[:-1,:-1]) & (pt_utm[1]<burnplot.grid_n[1:,1:])  )
            polygon.append( tuple([idx[0][0], idx[1][0]]) )

        img = Image.new('L', (burnplot.grid_e.shape), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask_bare_ground_georef = np.where( np.array(img).T ==1, np.ones_like(burnplot.grid_e), np.zeros_like(burnplot.grid_e))
    except: 
        mask_bare_ground_georef = np.ones_like(burnplot.grid_e)



    # load camera info
    #####################################
    K, D = np.load('../data_static/Camera/'+camera_name.split('_')[0]+'/Geometry/'+camera_name+'_cameraMatrix_DistortionCoeff.npy', allow_pickle=True, encoding='latin1')
    #K = np.array([[423.63788267,   0.       ,  120.        ],
    #              [  0.         ,753.13401364, 160.        ],
    #              [  0.         ,  0.        ,   1.        ]])
    #D = 0
    #if 'optris' in params_camera['camera_name']:
    #    print('set Distortion to 0 for optris')
    #    D = np.zeros(5) 
    if 'fix_mask' in list(params_camera.keys()):
        params_camera['fix_mask'] = params_rawData['root_data']+params_camera['fix_mask']

    # blur image was only consider for mulch, for now it is disregarded 
    if 'mulch' in plotname:
        flag_blur = np.load('mulch_blur.npy', allow_pickle=True)
        flag_blur = flag_blur.view(np.recarray)
    else: 
        flag_blur = None



    # define input argument for loading frame
    #####################################
    if runMode == 'visible':
        dir_ = params_rawData['root_postproc'] + params_camera['dir_input']
        #print 'image_pdf_reference_tools REMOVED'
        
        if (not(flag_restart)) | (not(os.path.isfile(dir_+'/'+ params_camera['dir_img_input']+'image_pdf_reference_tools.npy'))):
            if params_camera['filenames_no_helico_mask'] is not None:
                image_pdf_reference_tools = camera.build_reference_pdf(filenames, params_camera['filenames_no_helico_mask'], K, D, params_camera['shrink_factor'])
                np.save(dir_+'/'+ params_camera['dir_img_input']+'image_pdf_reference_tools',image_pdf_reference_tools)    
            else:
                image_pdf_reference_tools = None
        else:
            if params_camera['filenames_no_helico_mask'] is not None:
                image_pdf_reference_tools = np.load(dir_+'/'+ params_camera['dir_img_input']+'image_pdf_reference_tools.npy')
            else:
                image_pdf_reference_tools = None

        #image_pdf_reference_tools = None
        args_frame_ref00 = (filenames, filenames_timelookupTable, K, D, inputConfig, burnplot.shape, feature_params, flag_blur,       image_pdf_reference_tools, )
        args_frame       = (filenames, filenames_timelookupTable, K, D, inputConfig, burnplot.shape, feature_params, flag_blur,       image_pdf_reference_tools, ) 
    
    elif runMode == 'lwir':
        args_frame_ref00 = (filenames, filenames_timelookupTable, K, D, inputConfig, burnplot.shape, feature_params, flag_blur,                            )
        args_frame       = (filenames, filenames_timelookupTable, K, D, inputConfig, burnplot.shape, feature_params, flag_blur,                            )



    # load gcp location on ref image
    #####################################
    if not( os.path.isfile(dir_out+'gcp_ref00_{:s}.pts'.format(os.path.basename(filenames[0]).split('.')[0])) ):
         
        print('')
        print('need to define gcp location in ref image (location without buffer at the resolution of img)')
        print('saved in ', dir_out+'gcp_ref00_{:s}.pts'.format(os.path.basename(filenames[0]).split('.')[0]))
        frame_ref00 = camera.loadFrame(params_camera)
        #if runMode == 'lwir': frame_ref00.set_trange(params_georef['trange'])
        if runMode == 'visible': frame_ref00.set_trange([-999,-999])
        frame_ref00.init(0, None, *args_frame_ref00) 
        fig = plt.figure(figsize=(14,7))
        ax = plt.subplot(121)
        ax.imshow(burnplot.mask.T, origin='lower')
        ax.scatter(gcps_world_px[1,:],gcps_world_px[0,:],c='r',s=20)
        for i_gcp in range(4):
            ax.annotate('{:d}'.format(i_gcp), (gcps_world_px[1,i_gcp],gcps_world_px[0,i_gcp]))

        ax = plt.subplot(122)
        if  runMode == 'lwir':
            print('follow instruction in plot')
            img2plot = frame_ref00.temp[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),  \
                                        old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)]
            img2plot, _, _ = tools.get_gradient(img2plot)
            #ax.imshow(img2plot.T, origin='lower',interpolation='nearest', vmax=460)
            ax.imshow(img2plot.T, origin='lower',interpolation='nearest', vmax=300) #sycan19sg1
            
            line, = ax.plot([0],[0],linestyle='None', marker='+', color='r', markersize=20)
            #ax.set_xlim(0,nx-1)
            #ax.set_ylim(0,ny-1)
            filename_suffic = 'gcp_ref00_'
            filename_ = '{:s}.pts'.format(os.path.basename(filenames[0]).split('.')[0])
            linebuilder = cFP.cornerFirePicker(line,img2plot,filename_, 0, suffix=filename_suffic, outdir=dir_out, temp_threshold=params_camera['temperature_threshold_cornerFirePicker'], flag='4ref00')
        
        
        elif runMode == 'visible':
            print ('this needs to be done manually for now. program will stop here')
            img_fullReso = camera.get_img_fullReso(filenames[frame_ref00.id], K, D,)
            img2plot = img_fullReso[0]
            ax.imshow(np.transpose(img_fullReso,[1,0,2]), origin='lower',interpolation='nearest', cmap=mpl.cm.Greys)

            #ax.imshow(frame_ref00.img[frame_ref00.bufferZone/2:-frame_ref00.bufferZone/2,  \
            #                          frame_ref00.bufferZone/2:-frame_ref00.bufferZone/2].T, origin='lower',interpolation='nearest',
            #                          cmap=mpl.cm.Greys)
        
        
        else:
            print('bad run mode, runMode = ',  runMode)
            pdb.set_trace()
        

        plt.show()
        #print('exit now, need to restart')
        if runMode == 'visible':
            sys.exit()

    
    reader = asciitable.NoHeader()
    reader.data.splitter.delimiter = ' '
    reader.data.splitter.process_line = None
    data = reader.read(dir_out+'gcp_ref00_{:s}.pts'.format(os.path.basename(filenames[0]).split('.')[0]))
    gcps_cam = np.zeros([4,2])
    gcps_cam[:,1] = np.array(data['col1'][:]) 
    gcps_cam[:,0] = data['col2']              
    gcps_cam = old_div(gcps_cam,params_camera['shrink_factor'])


    if runMode == 'lwir':
        #set parameter for radiance/temperature conversion 
        #####################################
        srf_file = '../data_static/Camera/'+camera_name.split('_')[0]+'/SpectralResponseFunction/'+camera_name.split('_')[0]+'.txt'
        wavelength_resolution = 0.01
        param_set_radiance = [srf_file, wavelength_resolution]
        param_set_temperature = spectralTools.get_tabulated_TT_Rad(srf_file, wavelength_resolution)


   
    # Take first frame and find corners in it
    #####################################
    i_ref = 0
    print('')
    print('frame {:6d}'.format(i_ref), end=' ')
    if False: #(flag_restart) & (os.path.isfile(dir_out_frame+'frame{:06d}.nc'.format(i_ref))) :  
        frame_ref00 = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(i_ref))
        print('loaded')
        ni, nj = frame_ref00.ni, frame_ref00.nj
        plotMask_onref00_withbuffer = frame_ref00.plotMask_withBuffer
        
        if flag_georef_mode  == 'WithTerrain':
            img_grid_list, grid_img_list = pickle.load(open( wkdir+'frame{:06d}_grid_img_lookuptable.p'.format(i_ref), 'rb'))
            pts_img_xyz_map_ref00_init   = np.load(          wkdir+'frame{:06d}_2d23d.npy'.format(i_ref)                    ) 

    else: 
        print('')
        frame_ref00 = camera.loadFrame(params_camera)
        #if runMode == 'lwir': frame_ref00.set_trange(params_georef['trange'])
        if runMode == 'visible': frame_ref00.set_trange([-999,-999])
        frame_ref00.init(0, None, *args_frame_ref00)
        ni, nj = frame_ref00.ni, frame_ref00.nj
        
        #get camera pose, here we only use 4 points that were selected manually so we fully trust them and get the pose out of them.
        flag, rvec, tvec = cv2.solvePnP(gcps_world.T, gcps_cam, frame_ref00.K_undistorted_imgRes, D)

        #img to grid
        if flag_georef_mode  == 'SimpleHomography':
            H_ref2Grid, _ = cv2.findHomography(gcps_cam+old_div(frame_ref00.bufferZone,2), gcps_world_px.T, cv2.RANSAC,5.0)
          
            plotMask_onref00_withbuffer = cv2.warpPerspective(burnplot.mask,np.linalg.inv(H_ref2Grid),                           \
                                                   (frame_ref00.nj+frame_ref00.bufferZone,frame_ref00.ni+frame_ref00.bufferZone),\
                                                   borderValue=0,flags=cv2.INTER_NEAREST                                         )
            kernel_ring = params_camera['kernel_ring'] #201 #91 sku4? #51 
            kernel = np.ones((kernel_ring,kernel_ring),np.uint8)
            img_ = np.array(np.where(burnplot.mask==2,1,0),dtype=np.uint8)*255
            mask_ = cv2.dilate(img_, kernel, iterations = 1)    
            burnplot_mask_ring = np.where(mask_==255,2,0)

            plotMask_onref00_withbuffer_ring = cv2.warpPerspective(burnplot_mask_ring,np.linalg.inv(H_ref2Grid),                           \
                                                   (frame_ref00.nj+frame_ref00.bufferZone,frame_ref00.ni+frame_ref00.bufferZone),\
                                                   borderValue=0,flags=cv2.INTER_NEAREST                                         )

            if mask_bare_ground_georef.min() == 1: 
                bareGroundMask_onref00_withbuffer = np.ones_like(plotMask_onref00_withbuffer) 
            else:
                bareGroundMask_onref00_withbuffer = cv2.warpPerspective(mask_bare_ground_georef,np.linalg.inv(H_ref2Grid),           \
                                                       (frame_ref00.nj+frame_ref00.bufferZone,frame_ref00.ni+frame_ref00.bufferZone),\
                                                       borderValue=0,flags=cv2.INTER_NEAREST                                         ) 
        elif flag_georef_mode  == 'WithTerrain':
            #find match between triangles and image pixels
            img_grid_list, grid_img_list, pts_img_xyz_map_ref00_init = tools.triangle_img_pixel_match(i_ref, triangles, (ni, nj),   \
                                                                         rvec, tvec, frame_ref00.K_undistorted_imgRes, D, wkdir,    \
                                                                         flag_parallel=flag_parallel,\
                                                                         flag_restart=flag_restart, 
                                                                         flag_outptmap=True)

            if (img_grid_list is None) | (grid_img_list is None): 
                print('georef of first image failed')
                print('stop here')
                sys.exit()

            #mask plot countour on ref Image 
            plotMask_onref00 = np.zeros((ni,nj))
            bareGroundMask_onref00 = np.zeros((ni,nj))
            for ii in range(ni*nj):
                idx_img = np.unravel_index(ii,(ni,nj))
                total_img_pixel_area = 0.
                
                for tri_idx, area_intersect in img_grid_list[ii]:
                    total_img_pixel_area += area_intersect 
                    plotMask_onref00[idx_img]       += area_intersect * burnplot.mask[ triangles_grid[tri_idx] ]
                    bareGroundMask_onref00[idx_img] += area_intersect * mask_bare_ground_georef[ triangles_grid[tri_idx] ]
                if total_img_pixel_area!=0: 
                    plotMask_onref00[idx_img] /= total_img_pixel_area
                    bareGroundMask_onref00[idx_img] /= total_img_pixel_area
            
            plotMask_onref00 = np.round(plotMask_onref00,0).reshape(ni,nj)
            plotMask_onref00_withbuffer = np.zeros([ni+frame_ref00.bufferZone,nj+frame_ref00.bufferZone])
            plotMask_onref00_withbuffer[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),\
                                        old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)] = plotMask_onref00
            bareGroundMask_onref00 = np.round(bareGroundMask_onref00,0).reshape(ni,nj)
            bareGroundMask_onref00_withbuffer = np.zeros([ni+frame_ref00.bufferZone,nj+frame_ref00.bufferZone])
            bareGroundMask_onref00_withbuffer[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),\
                                        old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)] = bareGroundMask_onref00
     

        #update frame
        #------------
        frame_ref00.set_correlation(1., 1., 1., 1.)
        frame_ref00.set_warp(frame_ref00.img)
        frame_ref00.set_homography_to_ref(np.identity(3))
        if flag_georef_mode  == 'SimpleHomography': frame_ref00.set_homography_to_grid(H_ref2Grid)
        frame_ref00.set_pause(rvec,tvec)
        frame_ref00.set_plotMask(plotMask_onref00_withbuffer, plotMask_onref00_withbuffer_ring)
        frame_ref00.set_bareGroundMask(bareGroundMask_onref00_withbuffer)
        frame_ref00.set_maskWarp(frame_ref00.mask_img)

        if (frame_ref00.type == 'lwir') & (params_georef['look4cf']):
            gcps_img = cv2.perspectiveTransform( (gcps_cam+old_div(frame_ref00.bufferZone,2)).reshape(-1,1,2), np.linalg.inv(frame_ref00.H2Ref) )[:,0,:]
            cf_status, cf_on_img, cf_T, cf_hist =  tools.get_stat_info_cluster(gcps_img, frame_ref00, None, params_georef)
            frame_ref00.set_cf_on_img(np.dstack((cf_on_img))[0].T, np.dstack((cf_hist))[0].T)
        frame_ref00.set_flag_cfMode('anchored')

        frame_ref00.set_flag_inRefList('yes')
        
        p0 =  np.array( (gcps_cam+old_div(frame_ref00.bufferZone,2)).reshape(-1,1,2), dtype = np.float32)
        frame_ref00.save_feature_old_new(p0[:,0,:],p0[:,0,:],None)
       
        if frame_ref00.type == 'lwir':
            cf_status, cf_on_img, cf_T, cf_hist =  tools.get_stat_info_cluster(gcps_cam+old_div(frame_ref00.bufferZone,2), frame_ref00, None, params_georef)
            if len(cf_on_img) != gcps_cam.shape[0]:
                print('cf not all found on ref00')
            else:
                frame_ref00.set_cf_on_img(np.dstack((cf_on_img))[0].T, np.dstack((cf_hist))[0].T)
        
        #save frame 
        frame_ref00.dump(dir_out_frame+'frame{:06d}.nc'.format(frame_ref00.id))


    #georef ref image 
    #-----------------
    if ( not(flag_restart) ) | (not os.path.isfile(dir_out_georef_npy+plotname+'_georef_{:06d}_{:s}.npy'.format(frame_ref00.id,georefMode)))  :
   
        georef_img_ref00 = np.zeros([nx,ny])
        if frame_ref00.type == 'lwir': 
            georef_radiance_ref00 = np.zeros([nx,ny])
            radiance_ref00 = camera.return_radiance(frame_ref00, srf_file, wavelength_resolution=0.01)
        
        if flag_georef_mode  == 'SimpleHomography':
            georef_img_ref00 = cv2.warpPerspective(frame_ref00.img,frame_ref00.H2Grid,   \
                                                   (ny,nx),                              \
                                                   borderValue=0,flags=cv2.INTER_LINEAR )
            georef_mask_ref00 = cv2.warpPerspective(frame_ref00.mask_img,frame_ref00.H2Grid,   \
                                                   (ny,nx),                              \
                                                   borderValue=0,flags=cv2.INTER_NEAREST )
            if frame_ref00.type == 'lwir': 
                georef_radiance_ref00 =  cv2.warpPerspective(radiance_ref00,frame_ref00.H2Grid,  \
                                                            (ny,nx),                             \
                                                            borderValue=0,flags=cv2.INTER_LINEAR)
                georef_temp_ref00 = spectralTools.conv_Rad2Temp(georef_radiance_ref00, param_set_temperature)

        elif flag_georef_mode  == 'WithTerrain':
            input_array_1 = np.copy(frame_ref00.img)[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)]
            if frame_ref00.type == 'lwir': 
                input_array_2 = np.copy(radiance_ref00)[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)]
            
            total_grid_pixel_area = np.zeros((nx,ny))
            for i_tri in range(len(triangles)):
                for img_idx, area_intersect in grid_img_list[i_tri]:
                    total_grid_pixel_area[triangles_grid[i_tri]] += area_intersect 
                    georef_img_ref00[triangles_grid[i_tri]] += area_intersect * input_array_1[ img_idx ]
                    if frame_ref00.type == 'lwir': 
                        georef_radiance_ref00[triangles_grid[i_tri]] += area_intersect * input_array_2[ img_idx ]
            
            idx = np.where(total_grid_pixel_area!=0)
            georef_img_ref00[idx] /= total_grid_pixel_area[idx]
            if frame_ref00.type == 'lwir': 
                georef_radiance_ref00[idx] /= total_grid_pixel_area[idx]
                georef_temp_ref00 = spectralTools.conv_Rad2Temp(georef_radiance_ref00, param_set_temperature)
        
        #plot georef
        #-----------------
        if flag_plot_georef:
            mpl.rcdefaults()
            mpl.rcParams['text.usetex'] = True
            #mpl.rcParams['font.family'] = 'Comic Sans MS'
            mpl.rcParams['font.size'] = 16.
            mpl.rcParams['axes.linewidth'] = 1
            mpl.rcParams['axes.labelsize'] = 14.
            mpl.rcParams['xtick.labelsize'] = 14.
            mpl.rcParams['ytick.labelsize'] = 14.
            mpl.rcParams['figure.subplot.left'] = .002
            mpl.rcParams['figure.subplot.right'] = .93
            mpl.rcParams['figure.subplot.top'] = .9
            mpl.rcParams['figure.subplot.bottom'] = .1
            mpl.rcParams['figure.subplot.hspace'] = 0.02
            mpl.rcParams['figure.subplot.wspace'] = 0.02
            fig_georef = plt.figure(2, figsize=(13,5)) 
            
            ax = plt.subplot(131)
            if runMode == 'lwir':
                vmin_ = np.nanpercentile(np.ma.filled(np.ma.masked_where(georef_temp_ref00<=0,georef_temp_ref00),np.nan), 20)
                vmax_ = max([np.nanpercentile(np.ma.filled(np.ma.masked_where(georef_temp_ref00<=0,georef_temp_ref00),np.nan), 90), 320])
                im = ax.imshow(np.ma.masked_where(georef_temp_ref00<=0,georef_temp_ref00).T,origin='lower',cmap=mpl.cm.inferno,vmin=vmin_,vmax=vmax_)
                divider = make_axes_locatable(ax)
                cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
                cbar = fig_georef.colorbar(im ,cax = cbaxes,orientation='horizontal')
                cbar.set_label('T (K)')
            elif runMode == 'visible':
                ax.imshow(np.ma.masked_where(georef_img_ref00<=0,georef_img_ref00).T,origin='lower',cmap=mpl.cm.Greys_r)

            ax.set_axis_off()
            ax.set_title('georeferenced')

            ax = plt.subplot(132)
            if runMode == 'lwir':
                temp_warp = frame_ref00.temp[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),
                                             old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)] + 273.14 
                plot_warp = frame_ref00.plotMask_withBuffer[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),
                                                            old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)] 
                vmin_ = np.percentile(temp_warp[np.where(temp_warp>280)],20)
                vmax_ = np.max( [ np.percentile(temp_warp, 90), 320] )
                ax.imshow(temp_warp.T,origin='lower',interpolation='nearest',cmap=mpl.cm.jet,vmin=vmin_,vmax=vmax_)
                ax.imshow(np.ma.masked_where(plot_warp!=2,plot_warp).T,origin='lower',interpolation='nearest',alpha=0.7,cmap=mpl.cm.Greys_r)
            elif runMode == 'visible':
                ax.imshow(frame_ref00.warp.T,origin='lower',interpolation='nearest',cmap=mpl.cm.Greys_r)
                ax.imshow(np.ma.masked_where(frame_ref00.plotMask_withBuffer!=2,frame_ref00.plotMask_withBuffer).T,origin='lower',interpolation='nearest',alpha=0.5)
            
            ax.set_axis_off()
            ax.set_title('warp on ref image')
            
            ax = plt.subplot(133)
            if runMode == 'lwir':
                temp_raw = frame_ref00.temp[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)] + 273.14
                im = ax.imshow(temp_raw.T,origin='lower',interpolation='nearest',cmap=mpl.cm.jet,vmin=vmin_,vmax=vmax_)
                ax.imshow(np.ma.masked_where(frame_ref00.plotMask_withBuffer[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)]==0,
                                             frame_ref00.plotMask_withBuffer[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)]).T,\
                          origin='lower',interpolation='nearest',alpha=.2)
            elif runMode == 'visible':
                ax.imshow(frame_ref00.img[old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2),old_div(frame_ref00.bufferZone,2):old_div(-frame_ref00.bufferZone,2)].T,origin='lower',interpolation='nearest',cmap=mpl.cm.Greys_r)

            ax.set_axis_off()
            ax.set_title('raw image')
            
            if runMode == 'lwir':
                cbaxes = fig_georef.add_axes([0.935, 0.2, 0.015, 0.6])
                cbar = fig_georef.colorbar(im ,cax = cbaxes,orientation='vertical')
                cbar.set_label('T (K)')
           
            #add time
            fig_georef.text(.45,.9, frame_ref00.time_date.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + '    since igni t = {:.2f} s'.format(frame_ref00.time_igni))


            fig_georef.savefig(dir_out_georef_png+'{:06d}.png'.format(frame_ref00.id))
            plt.close(fig_georef) 

        #save georef npy 
        #-----------------
        if frame_ref00.type == 'visible': 
            np.save(dir_out_georef_npy+plotname+'_georef_{:06d}_{:s}'.format(frame_ref00.id,georefMode), np.array([ [frame_ref00.time_igni,\
                                                                                                            frame_ref00.rvec,     \
                                                                                                            frame_ref00.tvec,     \
                                                                                                            frame_ref00.corr_ref, \
                                                                                                            frame_ref00.corr_ref00,\
                                                                                                            frame_ref00.corr_ref00_init],  \
                                                                                                            georef_img_ref00,     \
                                                                                                            georef_mask_ref00,    ],dtype=object))    
        if frame_ref00.type == 'lwir': 
            np.save(dir_out_georef_npy+plotname+'_georef_{:06d}_{:s}'.format(frame_ref00.id,georefMode), np.array([ [frame_ref00.time_date, \
                                                                                                            frame_ref00.time_igni, \
                                                                                                            frame_ref00.rvec,      \
                                                                                                            frame_ref00.tvec,      \
                                                                                                            frame_ref00.corr_ref, \
                                                                                                            frame_ref00.corr_ref00,\
                                                                                                            frame_ref00.corr_ref00_init],  \
                                                                                                            georef_img_ref00,      \
                                                                                                            georef_mask_ref00,     \
                                                                                                            georef_temp_ref00,    \
                                                                                                            georef_radiance_ref00 ],dtype=object))    
    
        if not os.path.isfile(dir_out_georef_tif+plotname+'_img_{:06d}.tif'.format(frame_ref00.id)):
            tools.dump_geotiff(dir_out_georef_tif,grid_e,grid_n,utm,np.dstack([georef_img_ref00,burnplot.mask]), plotname+'_img_{:06d}'.format(frame_ref00.id),nodata_value=0)

        if not os.path.isfile(dir_out_georef_kml+'{:s}_{:s}_{:s}_id{:06d}.kmz'.format(plotname, camera_name,frame_ref00.time_date.strftime('%Y-%m-%dT%H%M%S%fZ'),frame_ref00.id)):
            tools.dump_kml(dir_out_georef_kml, 
                           camera_name.split('_')[0], plotname, 
                           grid_e, grid_n, utm, 
                           np.dstack([georef_img_ref00,]), 
                           [frame_ref00.time_date,], 
                           [frame_ref00.time_igni,], 
                           frameid=frame_ref00.id
                           )


    #Read correlation level form conf file
    #################################
    #energy_good_1       = params_camera['energy_good_1'] # * frame, successful georef
    #energy_good_2       = params_camera['energy_good_2'] # limit to break gcp scan (hard limit on gcp is good_3), lower limit to get * when 10 frames stay above consecutively 
    #energy_good_3       = params_camera['energy_good_3'] # limit to accpet first warp on gcp 
    corr_good_limit   = .2                             # limit below which frame is considered not georef 
    #ssim_2d_lower_limit = params_camera['ssim_2d_lower_limit']


    #load already processed frame
    #################################
    frames_done = sorted(glob.glob(dir_out_frame+'frame*.nc'))
    id_frames_done = np.array( [int(os.path.basename(frames_done_).split('.')[0].split('rame')[1]) \
                                for frames_done_ in frames_done] )


    #Open figure 
    #################################
    mpl.rcdefaults()
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.size'] = 10.
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelsize'] = 10.
    mpl.rcParams['xtick.labelsize'] = 10.
    mpl.rcParams['ytick.labelsize'] = 10.
    mpl.rcParams['figure.subplot.left'] = .02
    mpl.rcParams['figure.subplot.right'] = .98
    mpl.rcParams['figure.subplot.top'] = .98
    mpl.rcParams['figure.subplot.bottom'] = .05
    mpl.rcParams['figure.subplot.hspace'] = 0.01
    mpl.rcParams['figure.subplot.wspace'] = 0.01

    if (plt.fignum_exists(1)):
        fig = plt.figure(1)
    else:
        print('open figure')
        if params_camera['flag_costFunction']=='ssim': 
            fig = plt.figure(1,figsize=(12,9))
        elif params_camera['flag_costFunction']=='EP08':
            if params_georef['run_opti'] is False:
                fig = plt.figure(1,figsize=(14,6))
            else: 
                fig = plt.figure(1,figsize=(10,3.8))
        else: 
            print('bad flag_costFunction. stop here')
            sys.exit()



    # Loop over Frames
    #####################################
    
    #init
    #----
    frame_ref00_init = frame_ref00.copy()
    last_time2Changeref00 = frame_ref00.time_igni 
    
    i_file = i_ref+1
    
    item=(0,0,-999,0,0,0,0,0)
    info_loop = np.array( [item]*len(filenames), dtype=np.dtype([ ('time_igni',float),('inRef',int),('id',int),
                                                                  ('corr_ref',float),('corr_ref00',float),('corr_ref00_init',float),
                                                                  ('ep08_mask_size_around_plot',float), ('nbreCfTracked',float) ]) )
    info_loop = info_loop.view(np.recarray)
    info_loop.id[i_ref]                         = frame_ref00.id 
    info_loop.inRef[i_ref]                      = 1
    info_loop.time_igni[i_ref]                  = frame_ref00.time_igni 
    info_loop.corr_ref[i_ref]                   = frame_ref00.corr_ref
    info_loop.corr_ref00[i_ref]                 = frame_ref00.corr_ref00
    info_loop.corr_ref00_init[i_ref]            = frame_ref00.corr_ref00_init
    info_loop.ep08_mask_size_around_plot[i_ref] = tools.get_covergaeOfExtendedMaskPlot(frame_ref00,inputConfig)
    if params_georef['look4cf']: 
        info_loop.nbreCfTracked[i_ref]              = frame_ref00.cf_on_img.shape[0] if ('cf_on_img' in frame_ref00.__dict__) else 0  
    else: 
        info_loop.nbreCfTracked[i_ref]              = 999

    #framesID_ref = [frame_ref00.id]
    #energy_frames_ref = [1.]
    #ep08MaskSize_frames_ref = [tools.get_covergaeOfExtendedMaskPlot(frame_ref00) ]
    #framesID_all = [[frame_ref00.id],[frame_ref00.energy]]
    #time_frames_ref = [frame_ref00.time_igni]
    
    nbre_consecutive_frame_above_35  = 0
    id_last_anchored = i_ref
            
    try: 
        if runMode == 'lwir'   : win_size_ssim = params_georef['ssim_win_lwir'] 
        if runMode == 'visible': win_size_ssim = params_georef['ssim_win_visible'] 
    except: 
        if runMode == 'visible': win_size_ssim = 21
        if runMode == 'lwir'   : win_size_ssim = 21

    if frame_ref00.type == 'lwir'   : add_more_frame = params_georef['#frames_history_tail_lwir']
    if frame_ref00.type == 'visible': add_more_frame = params_georef['#frames_history_tail_visible']
    
    flag_frame_already_init = False

    if frame_ref00.type == 'visible':
        if 'backgrdimg' not in frame_ref00.__dict__ :
            flag_update_bckgrdImg = frame_ref00_init.create_backgrdimg()
        elif frame_ref00.backgrdimg.max() == 0: 
            flag_update_bckgrdImg = frame_ref00_init.create_backgrdimg()
        else:
            flag_update_bckgrdImg = 'done'
        
        frame_ref00.backgrdimg = frame_ref00_init.backgrdimg
        frame_ref00.mask_backgrdimg = frame_ref00_init.mask_backgrdimg
  
    incr_ifile_default = params_camera['incr_ifile_default'] #15
    incr_ifile = incr_ifile_default
    #flag_need_operator_input = False
    nbre_frame_below_corr_ref00_limit = 0
    nbre_frame_below_corref00_threshold = 0
    if len(params_camera['forceRef00Updateat'])==0:
        forceRef00Updateat_lowerlimit = -9e9
    else: 
        forceRef00Updateat_lowerlimit = params_camera['forceRef00Updateat'][0]

    #MERDEMERDE
    #frame_ref00 = camera.load_existing_file(params_camera,dir_out_frame+'frame{:06d}.nc'.format(8))
    #last_time2Changeref00 = 400

    while(1):  
        flag_frame_created = False 
         
        #MERDEMERDE
        #limit_to_start_debug = 1200
        #if i_file < limit_to_start_debug:
        #    if i_file == limit_to_start_debug-1:
        #        frame       = camera.load_existing_file(params_camera,dir_out_frame+'frame{:06d}.nc'.format(i_file))
        #        frame_ref00 = camera.load_existing_file(params_camera,dir_out_frame+'frame{:06d}.nc'.format(frame.id_ref00))
        #    i_file += incr_ifile
        #    continue
        
        if 'frame2skip' in list(params_camera.keys()):
            if i_file in  params_camera['frame2skip']:
                i_file += incr_ifile
                continue

        #----------------------------------------------------------------------------------
        # warp frame on ref frame
        #      or only get atching feature if not using simpleHomography for georef method
        #----------------------------------------------------------------------------------
        

        #if alredy done 
        #~~~~
        if (flag_restart) & (os.path.isfile(dir_out_frame+'frame{:06d}.nc'.format(i_file))) & (flag_frame_already_init is False):
            frame = camera.load_existing_file(params_camera,dir_out_frame+'frame{:06d}.nc'.format(i_file))
            print('frame {:6d} loaded'.format(frame.id), end=' ') 
            sys.stdout.flush()
            #flag_need_operator_input = False


        #else find gcp (+ warp if simpleHomography) and try to get cf if present 
        #~~~~
        else:
            if i_file == len(filenames):
                break
            flag_frame_created = True
            frame = camera.loadFrame(params_camera)
            #if runMode == 'lwir': frame.set_trange(params_georef['trange'])
            if runMode == 'visible': frame.set_trange([-999,-999])
            frame.init(i_file, frame_ref00, *args_frame)
            if runMode == 'visible': frame.save_backgrdimg(frame_ref00_init.backgrdimg, frame_ref00_init.mask_backgrdimg )

            frame.set_plotMask(frame_ref00.plotMask_withBuffer, frame_ref00.plotMask_withBuffer_ring)
            print('frame {:6d}'.format(frame.id), end=' ') 
            sys.stdout.flush()
        
            #check for lwir that we have feature
            if frame.type == 'lwir' : 

                if (frame.feature.shape[0] == 0) | (frame.blurred):
                    incr_ifile = new_increment(i_file, incr_ifile, incr_ifile_default, 'failed', dir_out_frame)
                    i_file += incr_ifile
                    print('blur')
                    if i_file == len(filenames):
                        break
                    continue

            # find gcp (+ warp if simpleHomography) and try to get cf if present
            if flag_georef_mode  == 'SimpleHomography':
                nbre_frame_availale_since_last_anchor = np.where((info_loop.inRef==1))[0].shape[0] #&(info_loop.id>=id_last_anchored))[0].shape[0]
                frame = tools.set_frame_using_homography(frame, frame_ref00, frame_ref00_init, info_loop.id[np.where(info_loop.inRef==1)], info_loop,
                                                         camera,
                                                         params_camera, params_georef, flag_parallel,
                                                         gcps_cam,
                                                         win_size_ssim, lk_params,
                                                         dir_out_frame,
                                                         nbre_frame_availale_since_last_anchor,add_more_frame)
            
            elif flag_georef_mode  == 'WithTerrain':
                frame = tools.set_frame_gcps(frame, info_loop.id[np.where(info.loop.inRef==1)], 
                                             camera,
                                             params_camera, params_georef, flag_parallel,
                                             dir_out_frame, wkdir, 
                                             win_size_ssim, lk_params)
                frame.set_flag_cfMode('nowarp')
                frame.set_correlation(999., 999.,999.,999.) # set to access the georef

        if (frame.cfMode == 'anchored'): id_last_anchored = frame.id



        #----------------------------------------------------------------------------------
        # get camera pose 
        #----------------------------------------------------------------------------------
        if flag_georef_mode  == 'SimpleHomography':
            warpSeemsOK = False 
            if 'H2Ref'   in frame.__dict__:
                if (frame.H2Ref-np.zeros([3,3])).sum()!=0:
                    #set homography to Grid
                    frame.set_homography_to_grid(frame_ref00_init.H2Grid.dot(np.linalg.inv(frame_ref00_init.H2Ref)).dot(frame.H2Ref) )
               
                    idx_plot = np.where( (frame.plotMask_withBuffer==2) & (frame.mask_warp==1)) 
                    gcps_cam_plot   = np.dstack((idx_plot[1],idx_plot[0]))[0].reshape(-1,1,2); gcps_cam_plot = np.array(gcps_cam_plot,dtype=np.float32)
                    if  idx_plot[0].shape[0] > 4: 
                        gcps_frame_plot = cv2.perspectiveTransform( gcps_cam_plot, np.linalg.inv(frame.H2Ref) ) - old_div(frame.bufferZone,2)
                        
                        gcps_frame_world_ = np.array( np.round( cv2.perspectiveTransform( gcps_cam_plot, frame_ref00_init.H2Grid), 0), dtype=int)
                        idx_ = (gcps_frame_world_[:,0,1], gcps_frame_world_[:,0,0])
                        gcps_frame_world  = np.vstack( (burnplot.grid_e[idx_], burnplot.grid_n[idx_], burnplot.terrain[idx_]) ).T
                
                        warpSeemsOK = True 

        elif flag_georef_mode  == 'WithTerrain':
            gcps_frame_plot  = frame.gcps_cam[  np.where(np.prod(frame.gcps_cam,axis=1)!=0),:].reshape(-1,1,2)
            gcps_frame_world = frame.gcps_world[np.where(np.prod(frame.gcps_cam,axis=1)!=0),:] 
            warpSeemsOK = False 
            print('define warpSeemsOK - stop here'); sys.exit() 
        
        if warpSeemsOK:
            flag, rvec, tvec = cv2.solvePnP(gcps_frame_world, gcps_frame_plot[:,0,:], frame.K_undistorted_imgRes, D)
        else: 
            rvec, tvec = np.array([-999.]), np.array([-999.])

        frame.set_pause(rvec,tvec)

            

        #----------------------------------------------------------------------------------
        # georef frame 
        #----------------------------------------------------------------------------------
        flag_georef_now = False
        if ( (incr_ifile>0) & (frame.corr_ref00>0) & (flag_georef) & ( not(flag_restart) & 
             (os.path.isfile(dir_out_georef_npy+plotname+'_georef_{:06d}_{:s}.npy'.format(frame.id,georefMode))) ) ):
           
            flag_georef_now = True
            georef_img = np.zeros([nx,ny])
            if frame.type == 'lwir': 
                georef_radiance = np.zeros((nx,ny))
                radiance = camera.return_radiance(frame, srf_file, wavelength_resolution=0.01)

            
            if flag_georef_mode  == 'SimpleHomography':
                georef_img = cv2.warpPerspective(frame.img,          \
                                                 frame.H2Grid,      \
                                                 (ny,nx),                               \
                                                 borderValue=0,flags=cv2.INTER_LINEAR  )
                georef_mask = cv2.warpPerspective(frame.mask_img,          \
                                                 frame.H2Grid,      \
                                                 (ny,nx),                               \
                                                 borderValue=0,flags=cv2.INTER_NEAREST  )
                if frame.type == 'lwir': 
                    georef_radiance =  cv2.warpPerspective(radiance,                            \
                                                           frame.H2Grid,                        \
                                                           (ny,nx),                             \
                                                           borderValue=0,flags=cv2.INTER_LINEAR )

            elif flag_georef_mode  == 'WithTerrain':
                #find match between triangles and image pixels
                img_grid_list, grid_img_list, pts_img_xyz_map= tools.triangle_img_pixel_match(i_file, triangles, (ni, nj), \
                                                                              frame.rvec, frame.tvec, frame.K_undistorted_imgRes, D, \
                                                                              wkdir, flag_parallel=flag_parallel, flag_restart=flag_restart, \
                                                                              flag_outptmap=True)
            
                if  (img_grid_list is not None) & (grid_img_list is not None) :
               
                    #georef ref image 
                    input_array_1 = np.copy(frame.img)[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)]
                    if frame.type == 'lwir': 
                        input_array_2 = np.copy(radiance)[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)]
                    
                    total_grid_pixel_area = np.zeros((nx,ny))
                    for i_tri in range(len(triangles)):
                        for img_idx, area_intersect in grid_img_list[i_tri]:
                            total_grid_pixel_area[triangles_grid[i_tri]] += area_intersect 
                            georef_img[triangles_grid[i_tri]] += area_intersect * input_array_1[ img_idx ]
                            if frame.type == 'lwir': 
                                georef_radiance[triangles_grid[i_tri]] += area_intersect * input_array_2[ img_idx ]
                    
                    idx = np.where(total_grid_pixel_area!=0)
                    georef_img[idx] /= total_grid_pixel_area[idx]
                    if frame.type == 'lwir': 
                        georef_radiance[idx] /= total_grid_pixel_area[idx]

            if frame.type == 'lwir': 
                georef_temp = spectralTools.conv_Rad2Temp(georef_radiance, param_set_temperature)
            
            #save npy 
            if frame.type == 'visible': 
                np.save(dir_out_georef_npy+plotname+'_georef_{:06d}_{:s}'.format(frame.id,georefMode), np.array([[frame.time_igni,frame.rvec,frame.tvec,         \
                                                                                                         frame.corr_ref,frame.corr_ref00,frame.corr_ref00_init], \
                                                                                                         georef_img,georef_mask    ]            , dtype=object))    
            if frame.type == 'lwir': 
                np.save(dir_out_georef_npy+plotname+'_georef_{:06d}_{:s}'.format(frame.id,georefMode), np.array([[frame.time_date,                               \
                                                                                                        frame.time_igni,frame.rvec,frame.tvec,                   \
                                                                                                        frame.corr_ref,frame.corr_ref00,frame.corr_ref00_init],  \
                                                                                                        georef_img,georef_mask,                                  \
                                                                                                        georef_temp,                                             \
                                                                                                        georef_radiance ]                        , dtype=object))    


        #------------------------
        #plot georef 
        #------------------------
        if (incr_ifile>0)  & (frame.corr_ref00>0) & (flag_plot_georef) & (not(os.path.isfile(dir_out_georef_png+'{:06d}.png'.format(frame.id)))):
            mpl.rcdefaults()
            mpl.rcParams['text.usetex'] = True
            #mpl.rcParams['font.family'] = 'Comic Sans MS'
            mpl.rcParams['font.size'] = 16.
            mpl.rcParams['axes.linewidth'] = 1
            mpl.rcParams['axes.labelsize'] = 14.
            mpl.rcParams['xtick.labelsize'] = 14.
            mpl.rcParams['ytick.labelsize'] = 14.
            mpl.rcParams['figure.subplot.left'] = .002
            mpl.rcParams['figure.subplot.right'] = .93
            mpl.rcParams['figure.subplot.top'] = .9
            mpl.rcParams['figure.subplot.bottom'] = .1
            mpl.rcParams['figure.subplot.hspace'] = 0.02
            mpl.rcParams['figure.subplot.wspace'] = 0.02
            fig_georef = plt.figure(2, figsize=(13,5))
            
            ax = plt.subplot(131)
            if frame.type == 'lwir':
                if not(flag_georef_now):  _ , georef_img, georef_mask, georef_temp, georef_radiance = \
                                                 np.load(dir_out_georef_npy+plotname+'_georef_{:06d}_{:s}.npy'.format(frame.id,georefMode), allow_pickle=True) 
                
                vmin_ =        np.nanpercentile( np.ma.filled(np.ma.masked_where(georef_temp<=0,georef_temp),np.nan), 20) 
                vmax_ =  max( [np.nanpercentile( np.ma.filled(np.ma.masked_where(georef_temp<=0,georef_temp),np.nan), 90), 320]) 

                im = ax.imshow(np.ma.masked_where(georef_temp<=0,georef_temp).T,origin='lower',cmap=mpl.cm.inferno,vmin=vmin_,vmax=vmax_)
                divider = make_axes_locatable(ax)
                cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
                cbar = fig_georef.colorbar(im ,cax = cbaxes,orientation='horizontal')
                cbar.set_label('T (K)')
            elif frame.type == 'visible':    
                if not(flag_georef_now):    _ ,georef_img, georef_mask = \
                                                 np.load(dir_out_georef_npy+plotname+'_georef_{:06d}_{:s}.npy'.format(frame.id,georefMode)) 
                ax.imshow(np.ma.masked_where(georef_img<=0,georef_img).T,origin='lower',cmap=mpl.cm.Greys_r)

            ax.set_axis_off()
            ax.set_title('georeferenced')

            ax = plt.subplot(132)
            if (frame.type == 'lwir') & (flag_georef_mode  == 'SimpleHomography'): 
                temp_warp = cv2.warpPerspective(frame.temp, frame.H2Ref, (nj+100,ni+100), borderValue=0,flags=cv2.INTER_LINEAR) + 273.14
                
                temp_warp = temp_warp[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)]
                vmin_ = np.percentile(temp_warp[np.where(temp_warp>280)],20)
                vmax_ = np.max( [ np.percentile(temp_warp, 90), 320] )

                ax.imshow(temp_warp.T,
                          origin='lower',interpolation='nearest',cmap=mpl.cm.jet,vmin=vmin_,vmax=vmax_)
            elif (frame.type == 'visible') & (flag_georef_mode  == 'SimpleHomography'):    
                ax.imshow(frame.warp.T,origin='lower',interpolation='nearest',cmap=mpl.cm.Greys_r)
            
            ax.set_axis_off()
            ax.set_title('warped on ref image')
            
            ax = plt.subplot(133)
            if frame.type == 'lwir': 
                img = frame.temp[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)] + 273.14
                im = ax.imshow(img.T,origin='lower',interpolation='nearest',cmap=mpl.cm.jet,vmin=vmin_,vmax=vmax_)
            elif frame.type == 'visible':    
                ax.imshow(frame.img[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)].T,origin='lower',interpolation='nearest',cmap=mpl.cm.Greys_r)
            
            ax.set_axis_off()
            ax.set_title('raw image')
            
            if frame.type == 'lwir': 
                cbaxes = fig_georef.add_axes([0.935, 0.2, 0.015, 0.6])
                cbar = fig_georef.colorbar(im ,cax = cbaxes,orientation='vertical')
                cbar.set_label('T (K)')
            
            #add time
            fig_georef.text(.45,.9, frame.time_date.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + '    since igni t = {:.2f} s'.format(frame.time_igni))

            fig_georef.savefig(dir_out_georef_png+'{:06d}.png'.format(frame.id))
            plt.close(fig_georef) 
            
        
       
        #------------------------
        # plot Warping info 
        #------------------------
        if (incr_ifile>0)  & (frame.corr_ref00>0) & (flag_plot_warp) & (not(os.path.isfile(dir_out_warping+'{:06d}.png'.format(frame.id))))  :
            mpl.rcdefaults()
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['font.size'] = 10.
            mpl.rcParams['axes.linewidth'] = 1
            mpl.rcParams['axes.labelsize'] = 10.
            mpl.rcParams['xtick.labelsize'] = 10.
            mpl.rcParams['ytick.labelsize'] = 10.
            mpl.rcParams['figure.subplot.left'] = .02
            mpl.rcParams['figure.subplot.right'] = .98
            mpl.rcParams['figure.subplot.top'] = .98
            mpl.rcParams['figure.subplot.bottom'] = .05
            mpl.rcParams['figure.subplot.hspace'] = 0.01
            mpl.rcParams['figure.subplot.wspace'] = 0.01 
            fig = plt.figure(1)
            plt.clf()
            flag_ = 'yes' if frame.blurred else 'no'
            fig.text(.02,.96,r'filename: {:s}  -  blurred: {:s}'.format(os.path.basename(filenames[i_file]).split('.')[0].replace('_','\_'),flag_))
         


            if False:#frame.id > 203:
                frame_ = frame_ref00 #camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(frame.id-1))
                
                temp_warp_ = cv2.warpPerspective(frame_.temp, frame_.H2Ref, frame_.img.shape[::-1], flags=cv2.INTER_LINEAR)
                plt.imshow(np.ma.masked_where(frame_.mask_warp==0,temp_warp_).T,origin='lower', vmax=70)
               
                plt.figure()
                temp_warp_ = cv2.warpPerspective(frame.temp,       frame.H2Ref,       frame.img.shape[::-1],       flags=cv2.INTER_LINEAR)
                plt.imshow(np.ma.masked_where(frame.mask_warp==0,      temp_warp_).T,origin='lower', vmax=70)
                plt.show()
                pdb.set_trace()



            #warp
            #---        
            if frame.type == 'lwir':
                plot_warp = cv2.warpPerspective(frame.temp, frame.H2Ref,                       \
                                                (frame.nj+frame.bufferZone,frame.ni+frame.bufferZone),\
                                                borderValue=0,flags=cv2.INTER_LINEAR                 )
                colorbar = mpl.cm.inferno
            elif frame.type == 'visible': 
                plot_warp = frame.warp
                colorbar = mpl.cm.Greys_r
           
            tempmin_ = plot_warp[np.where(plot_warp>0)].min()
            if frame.temp[np.where(frame.temp>0)].mean() < 300: 
                tempmin_ = 10
                tempmax_ = 50
            else:
                tempmin_ = 290
                tempmax_ = 440

            ax =  tools.get_plot_axis('warp',params_camera, params_georef)
            if frame.type == 'lwir':   im=ax.imshow(plot_warp.T,origin='lower',cmap=colorbar,vmin=tempmin_,vmax=tempmax_)
            if frame.type == 'visible':im=ax.imshow(plot_warp.T,origin='lower',cmap=colorbar)
            ax.set_title('warped img{:06d} - t={:5.1f}s'.format(frame.id,frame.time_igni))
            ax.set_axis_off()
            if frame.type == 'lwir':
                #colorbar
                if params_camera['flag_costFunction']=='ssim':
                    cbaxes = fig.add_axes([0.03, 0.55, 0.02, 0.3])
                    cbar = fig.colorbar(im,orientation='vertical',cax = cbaxes)
                    fig.text(.03,.88, '$T(K)$')
                elif params_camera['flag_costFunction']=='EP08':
                    cbaxes = fig.add_axes([0.03, 0.12, 0.3, 0.05])
                    cbar = fig.colorbar(im,orientation='horizontal',cax = cbaxes)
                    cbar.ax.set_xlabel('$T(K)$')  

            if params_camera['flag_costFunction']=='ssim': 
                #ssim
                ax =  tools.get_plot_axis('ssim',params_camera, params_georef)
                im = ax.imshow(np.ma.masked_where( frame.mask_ssim!= 1,                                                  \
                                                   frame.ssim_2d).T,origin='lower',vmin=-.1,vmax=.9)
                ax.set_title('$\overline{ssim}'+'= {:4.2f}$ with img{:06d}'.format(frame.ssim,frame.id_best_ref))
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                #colorbar
                cbaxes = fig.add_axes([0.03, 0.18, 0.02, 0.3])
                cbar = fig.colorbar(im,orientation='vertical',cax = cbaxes)
                fig.text(.025,.44, '$ssim$')

            #img
            #---        
            ax =  tools.get_plot_axis('img',params_camera, params_georef)
            ax.imshow(np.ma.masked_where(frame.mask_img!=1,frame.img).T,origin='lower',cmap=mpl.cm.Greys_r)
            if 'good_new_4plot' in frame.__dict__ : ax.scatter(frame.good_new_4plot[:,1],frame.good_new_4plot[:,0],marker='o',s=20,facecolors='none',edgecolors='r')
            ax.set_title('raw img{:06d}'.format(frame.id))
            ax.set_xlim(.5*frame.bufferZone,frame.img.shape[0]-.5*frame.bufferZone)
            ax.set_ylim(.5*frame.bufferZone,frame.img.shape[1]-.5*frame.bufferZone)
            ax.set_axis_off()
            if 'cf_on_img' in frame.__dict__: 
                if frame.cfMode == 'anchored':
                    ax.scatter(frame.cf_on_img[:,1], 
                               frame.cf_on_img[:,0],color='g', s=100, facecolors='none', edgecolors='g')
                else: 
                    ax.scatter(frame.cf_on_img[:,1], 
                               frame.cf_on_img[:,0],color='g',marker='x',s=100)
           
            #best ref img
            #---      
            if frame.id_best_ref >= 0: 
                ax =  tools.get_plot_axis('bestrefimg',params_camera, params_georef)
                
                frame_best_ref = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(frame.id_best_ref))
                if frame.type == 'lwir':
                    mask_, mask_fix = tools.mask_lowT(frame, frame_best_ref, kernel_warp=frame.kernel_warp, kernel_plot=frame.kernel_plot, \
                                                      lowT=params_camera['lowT_param'][0], kernel_lowT=params_camera['lowT_param'][1] ) 
                else: 
                    mask_, mask_fix = tools.mask_onlyImageMask(frame, frame_best_ref, kernel_warp=frame.kernel_warp, kernel_plot=frame.kernel_plot)
                ax.imshow(np.ma.masked_where( (mask_==0) | (mask_fix==0) ,frame_best_ref.warp).T,origin='lower')#,cmap=mpl.cm.Greys_r)
                #ax.imshow(np.ma.masked_where( frame_best_ref.mask_warp==0 ,frame_best_ref.warp).T,origin='lower')#,cmap=mpl.cm.Greys_r)
                if 'good_old' in frame.__dict__ : 
                    ax.scatter(frame.good_old[:,1],frame.good_old[:,0],marker='o',s=15,facecolors='none',edgecolors='r',alpha=1./min([frame.number_ref_frames_used,20]))
                ax.set_title('img{:06d} with gcp from {:6d} frames '.format(frame.id_best_ref, frame.number_ref_frames_used))
                #ax.set_axis_off()
                ax.set_xlim(0,frame.img.shape[0])
                ax.set_ylim(0,frame.img.shape[1])
                ax.tick_params(labelbottom=False, bottom=False)
                ax.tick_params(labelleft=False, left=False)

            #ssim
            #---  
            if win_size_ssim!= 0: 
                if (params_camera['flag_costFunction']=='EP08') & params_georef['run_opti']:
                    ax =  tools.get_plot_axis('ssim',params_camera, params_georef)
                    im = ax.imshow(np.ma.masked_where( frame.mask_ssim!= 1,                                                  \
                                                       #np.logical_not(tools.idx_ssim_ok(frame.mask_ssim,                    \
                                                       #                                 win_size_ssim,                      \
                                                       #                                 frame_ref00.plotMask_withBuffer,    \
                                                       #                                 tools.plume_mask(frame,frame_ref00),\
                                                       #                                 frame.ssim_2d                      )),\
                                                       frame.ssim_2d).T,origin='lower',vmin=-.1,vmax=.9)
                    ax.set_title('$\overline{ssim}'+'= {:4.2f}$ with ref00 img{:06d}'.format(frame.ssim,frame.id_ref00))
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    #colorbar
                    cbaxes = fig.add_axes([0.955, 0.22, 0.01, 0.5])
                    cbar = fig.colorbar(im,orientation='vertical',cax = cbaxes)
                    fig.text(.985,.5, '$ssim$', rotation=90)


            #if (mpl.get_backend() != 'Agg') & (mpl.get_backend() != 'agg'): 
            #    plt.draw()
            #    plt.pause(0.1)
            #    #tools.local_pause_4_figure(.01)

            try: 
                fig.savefig(dir_out_warping+'{:06d}.png'.format(frame.id))
            except: 
                pdb.set_trace()


        #if frame.id > 440: 
        #    tools.get_matching_feature_opticalFlow_prev_frames(frame, params_camera, dir_out_frame, camera, lk_params)


        #------------------------
        # set next loop
        #------------------------
        info_loop.id[i_file]                         = frame.id 
        info_loop.time_igni[i_file]                  = frame.time_igni 
        info_loop.corr_ref[i_file]                   = frame.corr_ref
        info_loop.corr_ref00[i_file]                 = frame.corr_ref00
        info_loop.corr_ref00_init[i_file]            = frame.corr_ref00_init
        info_loop.ep08_mask_size_around_plot[i_file] = tools.get_covergaeOfExtendedMaskPlot(frame,inputConfig)
        if params_georef['look4cf']: 
            info_loop.nbreCfTracked[i_file]              = frame.cf_on_img.shape[0] if ('cf_on_img' in frame.__dict__) else 0  
        else: 
            info_loop.nbreCfTracked[i_file]              = 999

        if frame.corr_ref00_init >= 0: print(' ep08={:4.3f},{:4.3f} '.format(frame.corr_ref00,frame.corr_ref00_init), end=' ') 
        if frame.ssim >= 0:            print(' ssim={:4.3f} '.format(frame.ssim), end=' ') 
        
        print('{:d}  s={:.2f}'.format(frame_ref00.id, info_loop.ep08_mask_size_around_plot[i_file]), end=' ') 
        
        if frame.corr_ref00 >= params_camera['energy_good_2']:
            nbre_consecutive_frame_above_35 += 1
        else:
            nbre_consecutive_frame_above_35  = 0

        idx_ = np.where( (info_loop.id>=frame.id-60) & (info_loop.id<frame.id) & (info_loop.corr_ref00<1) & (info_loop.inRef==1) & (info_loop.time_igni>last_time2Changeref00) )
        if idx_[0].shape[0] > 3: 
            test_on_corr_ref00 = (frame.corr_ref00 >= (info_loop.corr_ref00[idx_].mean()-3*info_loop.corr_ref00[idx_].std())) | (frame.corr_ref00 >= params_camera['energy_good_1'] ) 
        else: 
            test_on_corr_ref00 = (frame.corr_ref00 >= params_camera['energy_good_1'] )

        if ( test_on_corr_ref00                                                                         )    |\
           ( (nbre_consecutive_frame_above_35>4) & (frame.corr_ref00 >= params_camera['energy_good_2']) )    :
          
            if nbre_consecutive_frame_above_35 > 4:
                idx_last_10_candidate = np.where( (info_loop.id>frame.id-4 ) & (info_loop.corr_ref00 > params_camera['energy_good_2']) )[0]
                idx_frame = idx_last_10_candidate[ info_loop.corr_ref00[idx_last_10_candidate].argmax() ]
                if info_loop.id[idx_frame] == frame.id:
                    frame_here = frame
                else: 
                    frame_here = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(info_loop.id[idx_frame]))
            
            else: 
                frame_here = frame
            
            if info_loop.inRef[frame_here.id] == 0: #else frame is already saved and in ref so we pass 
                info_loop.inRef[frame_here.id] = 1 
                
                if frame_here.type == 'visible':
                    if flag_update_bckgrdImg != 'done': 
                        print(' + bckgrdimg ', end=' ')
                        flag_update_bckgrdImg = frame_ref00_init.update_backgrdimg(frame_here)
                        if flag_update_bckgrdImg == 'done':
                            #update reference frame
                            frame_ref00_init.dump(dir_out_frame+'frame{:06d}.nc'.format(frame_ref00_init.id))
               
                    frame_ref00.backgrdimg = frame_ref00_init.backgrdimg
                    frame_ref00.mask_backgrdimg = frame_ref00_init.mask_backgrdimg

                print('*({:d})'.format(frame_here.id), end=' ') 
                #frame_here.set_flag_inRefList('yes')
                nbre_consecutive_frame_above_35 = 0
            
            incr_ifile = new_increment(i_file, incr_ifile, incr_ifile_default, 'ok', dir_out_frame)
            i_file += incr_ifile
       
        else:
            incr_ifile = new_increment(i_file, incr_ifile, incr_ifile_default, 'failed', dir_out_frame)
            i_file += incr_ifile
            
  

        #------------------------
        #save frame
        #------------------------
        if (flag_frame_created) & (incr_ifile>0):
            frame.dump(dir_out_frame+'frame{:06d}.nc'.format(frame.id))
       
            if (frame.corr_ref00 < corr_good_limit                                                                ):
                nbre_frame_below_corr_ref00_limit += 1
            else: 
                nbre_frame_below_corr_ref00_limit = 0
        

        #----------------------------------------------------------------------------------
        # control ref00 
        #----------------------------------------------------------------------------------
        #if (frame.time_igni-last_time2Changeref00 > params_camera['time_change_ref00']) | (frame.id in params_camera['forceRef00Updateat'] ):
        accepted_ref00_decrease = params_camera['ref00Update_threshold'] 
        idx_ = np.where( (info_loop.time_igni>=last_time2Changeref00) & (info_loop.corr_ref00<1) & (info_loop.inRef==1) )
        if len(idx_[0]) > 0:
            if ( frame.corr_ref00 < accepted_ref00_decrease * info_loop.corr_ref00[ idx_ ].max() ):
                nbre_frame_below_corref00_threshold += 1        
            print(' {:.2f} {:1d}'.format(old_div(frame.corr_ref00,info_loop.corr_ref00[ idx_ ].max()), nbre_frame_below_corref00_threshold), end=' ')
        
        if (frame.id in params_camera['forceRef00Updateat'] ) | (nbre_frame_below_corref00_threshold>3) :
                flag_time2Changeref00 = True
        
        if nbre_consecutive_frame_above_35 > params_camera['nbre_consecutive_frame_above_35']: #4 
            flag_time2Changeref00 = True 


        #------------------------
        #update frame_ref00
        #------------------------
        if (flag_time2Changeref00): #& ((params_georef['look4cf'])&('cf_on_img' in frame.__dict__)): 
          
            if (frame.id not in params_camera['forceRef00Updateat']) & (frame.id > forceRef00Updateat_lowerlimit):   
                
                lowerlim_ = info_loop.id[info_loop.inRef==1]\
                                        [np.abs(frame.time_igni - info_loop.time_igni[np.where(info_loop.inRef==1)] - params_camera['time_tail_ref00']).argmin()]
                lowerlim_ = max([-1*(frame.id-forceRef00Updateat_lowerlimit), lowerlim_ ])
                
                if frame.type == 'lwir': 
                    idx_candidate = np.where( (info_loop.inRef==1)                          &
                                              (info_loop.id>lowerlim_)                      & 
                                              (info_loop.ep08_mask_size_around_plot < 0.15 )   &
                                              (info_loop.nbreCfTracked >= gcps_cam.shape[0]) 
                                            )[0] 
                elif frame.type == 'visible': 
                    idx_candidate = np.where( (info_loop.inRef==1)                          &
                                              (info_loop.id>lowerlim_)                      & 
                                              (info_loop.ep08_mask_size_around_plot < 0.15 )   
                                            )[0]
                else: 
                    print(' frame type not set in ref00 update.')
                    print(' frame type is {:s}'.format(frame.type))
                    pdb.set_trace()

                if len(idx_candidate)>0:
                    idx = info_loop.corr_ref00[idx_candidate].argmax()
                    new_iframe_ref00 = info_loop.id[idx_candidate[idx]]
                else:
                    new_iframe_ref00 = -1

                '''
                energy_arr_       = np.array(energy_frames_ref[lowerlim_:])
                ep08MaskSize_arr  = np.array(ep08MaskSize_frames_ref[lowerlim_:])
                framesID_ref_arr_ = np.array(framesID_ref[lowerlim_:])
                if len(framesID_ref_arr_)>0:
                    idx_ = np .where( ep08MaskSize_arr > .8 * ep08MaskSize_arr.max() )[0]
                    new_iframe_ref00 = framesID_ref_arr_[ idx_[np.array(energy_arr_[idx_]).argmax()] ]
                else:
                    new_iframe_ref00 = -1
                new_iframe_ref00 = 
                '''

            elif (frame.id not in params_camera['forceRef00Updateat']) & (frame.id <= forceRef00Updateat_lowerlimit):
                new_iframe_ref00 = -1
            
            else: 
                new_iframe_ref00 = frame.id

            if (new_iframe_ref00>0) & (new_iframe_ref00 != frame_ref00.id):
                frame_ref00 = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(new_iframe_ref00))
                if frame_ref00.type == 'visible':
                    frame_ref00.backgrdimg = frame_ref00_init.backgrdimg
                    frame_ref00.mask_backgrdimg = frame_ref00_init.mask_backgrdimg
                flag_time2Changeref00 = False
                print(' ref00 update {:6d}'.format(frame_ref00.id), end=' ')
                last_time2Changeref00 = frame.time_igni
                nbre_frame_below_corref00_threshold = 0

            else:
                print(' ref00 update no candidate', end=' ')
            
            if len(params_camera['forceRef00Updateat']) > 0: 
                if frame.id == params_camera['forceRef00Updateat'][0]: # then we change ref00_init
                    if frame_ref00.type == 'visible':
                        frame.backgrdimg = frame_ref00_init.backgrdimg
                        frame.mask_backgrdimg = frame_ref00_init.mask_backgrdimg
                    frame_ref00_init = frame.copy()

        if (nbre_frame_below_corr_ref00_limit>20):          
            print('')
            print('**')
            print('too many files (20) with low correlation (< {:.3f})'.format(corr_good_limit))
            print('exit main loop here')
            break
            #flag_need_operator_input = True

            

        print('')
        if i_file == len(filenames):
            break

        #------------------------
        # end loop
        #------------------------
    np.save(dir_out+runName+'_info_loop.npy', info_loop)

    #save netcdf file with all georef image on same grid for visu in qgis
    ##############################
    if (flag_georef == True) & (runMode == 'lwir'): 
        
        if (os.path.isfile(dir_out_georef_nc+plotname+'_allFrame.nc')) : os.remove(dir_out_georef_nc+plotname+'_allFrame.nc')
   
        print('save all frames in netcdf') 

        ncfile = Dataset(dir_out_georef_nc+plotname+'_allFrame.nc','w')
    
        ncfile.description = 'frp/m2 for plot ' + plotname 
   
        # Global attributes
        setattr(ncfile, 'created', 'R. Paugam') 
        setattr(ncfile, 'title', 'frp/m2 ' + plotname)
        setattr(ncfile, 'Conventions', 'CF')

        # dimensions
        ncfile.createDimension('easting',grid_e.shape[0])
        ncfile.createDimension('northing',grid_n.shape[1])
        ncfile.createDimension('time',None)

        # set dimension
        ncx = ncfile.createVariable('easting', 'f8', ('easting',))
        setattr(ncx, 'long_name', 'easting UTM WGS84 UTM Zone {:d}'.format(utm.GetUTMZone()))
        setattr(ncx, 'standard_name', 'easting')
        setattr(ncx, 'units','m')

        ncy = ncfile.createVariable('northing', 'f8', ('northing',))
        setattr(ncy, 'long_name', 'northing UTM WGS84 UTM Zone {:d}'.format(utm.GetUTMZone()))
        setattr(ncy, 'standard_name', 'northing')
        setattr(ncy, 'units','m')
        
        ncTime = ncfile.createVariable('time', 'f8', ('time',))
        setattr(ncTime, 'long_name', 'time')
        setattr(ncTime, 'standard_name', 'time')
        setattr(ncTime, 'units','seconds since 1970-1-1')
        time_ref_nc=datetime.datetime(1970,1,1)
    
        # set Variables
        ncVar    = ncfile.createVariable('btLwir','float32', (u'time',u'northing',u'easting',), fill_value=-999.)
        #ncVar    = ncfile.createVariable('frpm2',    'f', ('northing', 'easting',), fill_value=-999.)
        setattr(ncVar, 'long_name', 'lwir brightness temperature') 
        setattr(ncVar, 'standard_name', 'btLwir') 
        setattr(ncVar, 'units', 'K') 
        setattr(ncVar, 'grid_mapping', "transver_mercator")
        
        # set projection
        ncprj    = ncfile.createVariable('transver_mercator',    'c', ())
        setattr(ncprj, 'spatial_ref', utm.ExportToWkt() ) 

        #write grid
        ncx[:] = grid_e[:,0]
        ncy[:] = grid_n[0,:]
        
        #loop over the npy georef
        i_time = 0
        for i_file, filename in enumerate(sorted(glob.glob(dir_out_georef_npy+'/'+plotname+'*.npy'))):
          
            #load georef lwir 
            [time_date_, time_igni_, rvec_, tvec_,           \
                     corr_ref, corr_ref00, corr_ref00_init], \
                     georef_img, georef_mask,                \
                     georef_temp, georef_radiance            = np.load(filename,allow_pickle=True) 
          

            if corr_ref00 < params_camera['energy_good_2'] : continue 

            #set outside of the plot with do nodata flag, ie = -999
            #idx = np.where(burnplot.mask!=2)
            #mosaic_geo_frp[idx] = -999.

            #time
            ncTime[i_time] = (time_date_ - time_ref_nc).total_seconds()

            #layer
            ncVar[i_time,:,:] = np.swapaxes(georef_temp,0,1)
            
            i_time += 1
            
        #close file
        ncfile.close()

        #and copy a version without hdf 
        filename_out_2 = dir_out_georef_nc+plotname+'_allFrame_nohdf.nc'
        subprocess.call(["nccopy", "-k", "1",dir_out_georef_nc+plotname+'_allFrame.nc' , filename_out_2 ])


    time_end_run = datetime.datetime.now()
    print('')
    print('---------')
    print('cpu time elapse (h) = ', old_div((time_end_run - time_start_run).total_seconds(),3600))     

