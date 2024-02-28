from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import sys, os, glob, pdb
import numpy as np
import shapefile 
from shapely.geometry import Polygon
from osgeo import gdal,osr,ogr
import itertools
import cv2 
import transformation 
import multiprocessing
import pickle 
import pandas
from skimage import feature, measure, exposure, filters
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import asciitable
import datetime
from PIL import Image, ImageDraw
import simplekml
from mpl_toolkits.basemap import Basemap, cm, shiftgrid
from fastkml import kml
from scipy import ndimage, interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage 
import psutil
import gc 
import warnings

from matplotlib import _pylab_helpers
from matplotlib.rcsetup import interactive_bk as _interactive_bk

#homebrewed
#path_SrcPython = os.environ['PATH_SRC_PYTHON_LOCAL']
#sys.path.append(path_SrcPython+'/georefircam/src/')
import hist_matching
import tools_georefWithTerrain as georefWT
sys.path.append('./Factor_number/')
import factor

#####################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)  


########################################################
def bytesStringConv(input_):

    data = input_[0]
    try:
        data = data.decode()
    except (UnicodeDecodeError, AttributeError): # string is not bytes
        return input_
    
    len_ = input_.shape[0]
    output_ = np.array(['mm']*len_, dtype=np.dtype('U200'))
    for i, data in enumerate(input_):
        output_[i] = data.decode()

    return output_
    

########################################################
def load_npy(filename):
   
    try: 
        tmp_ = np.load(filename, allow_pickle=True)
    except:
        tmp_ = np.load(filename, allow_pickle=True,encoding='latin1')

    if len(tmp_)==6: # georef
        [[time_date, time_igni, rvec, tvec, corr_ref, corr_ref00, corr_ref00_init], \
                georef_img, georef_mask, georef_temp, georef_radiance, georef_pixelError ] = tmp_
    if len(tmp_)==5: # final
        [[time_date, time_igni, rvec, tvec, corr_ref, corr_ref00, corr_ref00_init], \
                georef_img, georef_mask, georef_temp, georef_radiance,                   ] = tmp_
    if len(tmp_)==7: # refined 
        [[_,_,frame_info,_,_,_,_,_,_],
                  homogra_mat,
                  _,
                  _, georef_mask,
                  georef_temp, _,]  = tmp_
        [time_date, time_igni, rvec, tvec, corr_ref, corr_ref00, corr_ref00_init] = frame_info 
        georef_img = None
        georef_radiance = None
    
    if len(tmp_)==4: # firefront
        [[time_date, time_igni, rvec, tvec, corr_ref, corr_ref00, corr_ref00_init],
                  georef_temp, _, georef_mask]  = tmp_
        georef_img = None
        georef_radiance = None

    return [[time_date, time_igni, rvec, tvec, corr_ref, corr_ref00, corr_ref00_init], \
                georef_img, georef_mask, georef_temp, georef_radiance,]


######################################################
def create_grid(root_postproc, fireName, 
                cf_gps_utm, cameraLocation,
                params_gps, params_grid, params_camera,
                utm, conv_ll2utm, conv_utm2ll, flag_georef_mode, 
                contour_extra_file=None, 
                dem_filename=None, flag_plot=False, dir_out=None):
      
    #input paramerter
    contour_file  = root_postproc+params_gps['dir_gps']+params_gps['contour_file']
    flag_ctr_format = params_gps['ctr_format']
    resolution = params_grid['grid_resolution']
    domain_size = params_grid['grid_size']
    cameraLens = params_camera['cameraLens']
    cameraDimension = params_camera['cameraDimension']

    if (domain_size == 'na'):
        print('need input of domain size')
        print('terminate here')
        print('--------------')
        # use center location
        sys.exit()
        
    if (resolution == 'na' ) :
        resolution = get_resolution_from_camera_location(cameraLocation,cf_gps_utm,cameraLens,cameraDimension) 

    #define grid on flat ground
    center = cf_gps_utm.mean(axis=1)[0:2]
    resolution, grid_e, grid_n = defineGrid(center[0],center[1],box_size=domain_size,res = resolution)

    if flag_ctr_format.split('_')[0] == 'shapeFile':
        #load shape file
        ctr = shapefile.Reader(contour_file)
        fields_name = np.array([ctr.fields[i][0] for i in range(len(ctr.fields))])
        idx_name = np.where(fields_name == 'Comment')[0][0] - 1

        records = ctr.shapeRecords() #will store the geometry separately

        plotmask = np.zeros(grid_e.shape) 
        for i, record in enumerate(records):
            pts        = record.shape.points #will show you the points of the polygon
            attributes = record.record       #will show you the attributes
        
            if 'in' in  attributes[idx_name]:
                fill_val = 2
            else:
                fill_val =1
            
            pts_utm = []
            for ii, pt in enumerate(pts):
                pts_utm.append(list(conv_ll2utm.TransformPoint(*pt[::-1])[:2]))

            # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
            width, height = plotmask.shape
            polygon =[ tuple( np.array(np.round(old_div((np.array(pt)-np.array(center)),resolution),0) + .5*np.array([width, height]),dtype=int) ) for pt in pts_utm ]

            img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=i+1, fill=i+1)
            mask = np.array(img)
        
            idx=np.where(mask != 0)
            plotmask[idx] = fill_val

        if contour_extra_file is not None:
            polygon = []
            #load shape file
            ctr = shapefile.Reader(contour_extra_file)
            fields_name = np.array([ctr.fields[i][0] for i in range(len(ctr.fields))])
            idx_name = np.where(fields_name == 'Comment')[0] - 1
            records = ctr.shapeRecords() #will store the geometry separately
            for i, record in enumerate(records):
                pts        = record.shape.points #will show you the points of the polygon
                attributes = record.record       #will show you the attributes
                pts_utm = []
                for ii, pt in enumerate(pts):
                    pts_utm.append(list(conv_ll2utm.TransformPoint(*pt[::-1])[:2]))
                # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
                width, height = plotmask.shape
                polygon_ =[ tuple( np.array(np.round(old_div((np.array(pt)-np.array(center)),resolution),0) + .5*np.array([width, height]),dtype=int) ) for pt in pts_utm ]
                [polygon.append(polygon_[i]) for i in range(len(polygon_))]
            img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=i+1, fill=i+1)
            mask = np.array(img)
            idx=np.where((mask != 0) & (plotmask==0))
            plotmask[idx] = 1
        plotmask = plotmask.T

    #deprecated
    elif False :#flag_ctr_format.split('_')[0] == 'textFile':
        #the contour of the plot
        flag_textFormat  = flag_ctr_format.split('_')[1]
        f = open(contour_file,'r')
        lines = f.readlines()
        pts_utm3D = np.zeros([3,len(lines)-1])
        f.close()
        for i_line, line in enumerate(lines[1:]):
            pts_utm3D[:,i_line] = line.rstrip().split(' ')[0:3] 

            if flag_textFormat == 'latlong':
                pts_utm3D[:,i_line] = (conv_ll2utm.TransformPoint(*pts_utm3D[:,i_line]) )

        center   = [pts_utm3D[0,:].mean(), pts_utm3D[1,:].mean()]

        plotmask = np.zeros(grid_e.shape) 
        # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
        width, height = plotmask.shape
        pts_utm_list = []
        for ii in range(pts_utm3D.shape[1]):
            pts_utm_list.append((pts_utm3D[:,ii]).tolist())
        #polygon =[ tuple( np.array(np.round((np.array(pt[:2])-np.array(center))/resolution,0) + .5*np.array([width, height]),dtype=int) ) for pt in pts_utm_list ]
    
        polygon = []
        for pt in pts_utm_list:
            idx = np.where( (np.abs(grid_e - pt[0])<.5*resolution) & (np.abs(grid_n - pt[1])<.5*resolution) )
            if len(idx[0]) != 1:
                pdb.set_trace()
            polygon.append((idx[0][0],idx[1][0]))

        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=2)
        mask = np.array(img)
        idx=np.where(mask != 0)
        plotmask[idx] = mask[idx]
        plotmask = plotmask.T
    
    elif flag_ctr_format.split('_')[0] == 'kml':
        pts = load_polygon_from_kml(contour_file,params_gps['contour_feature_name'])
        pts_utm = []
        for ii, pt in enumerate(zip(pts[0],pts[1])):
            pts_utm.append(list(conv_ll2utm.TransformPoint(*pt[::-1])[:2]))

        # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
        width, height = grid_e.shape
        polygon =[ tuple( np.array(np.round(old_div((np.array(pt)-np.array(center)),resolution),0) + .5*np.array([width, height]),dtype=int) ) for pt in pts_utm ]

        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=2, fill=2)
        plotmask = np.array(img).T
   
    else: 
        pdb.set_trace()
   

    try: 
        #shrink plot mask if wanted
        kernel = np.ones((2,2),np.uint8)
        img_ = np.array(np.where(plotmask==2,np.ones(mask.shape,dtype=np.uint8),np.zeros(mask.shape,dtype=np.uint8)),dtype=np.uint8)
        mask = cv2.erode(img_,kernel,iterations = params_gps['contour_file_shrinkPlotMask'])

        idx=np.where(mask == 0)
        plotmask[idx] = 0
    except: 
        pass

    '''ax = plt.subplot(121)
    ax.imshow(img_.T,origin='lower')
    ax = plt.subplot(122)
    ax.imshow(plotmask.T,origin='lower')
    plt.show()
    pdb.set_trace()'''

    # load DEM if exists
    if dem_filename is not None: 
        try: 
            terrain = np.load(dem_filename)
        except: 
            print('*****')
            print('DEM file path in config file was not found ')
            print('gris is saved in ', root_postproc)
            print('for you to run DEM_from_googleEarthDEM ')
            print('stop here')
            np.save(root_postproc + 'grid_'+fireName,[grid_e, grid_n, plotmask])
            open(   root_postproc + 'grid_'+fireName+'.prj','w').writelines(utm.ExportToWkt())    
            pdb.set_trace()
            sys.exit()
            
    else:
        terrain = np.zeros_like(grid_e)

    if flag_georef_mode  == 'SimpleHomography':
        burnplot = np.zeros(grid_e.shape, dtype=np.dtype([('grid_e',float),('grid_n',float),('terrain',float),('mask',float),('terrain_original',float),('area',float)]) )
    else:
        burnplot = np.zeros(grid_e.shape, dtype=np.dtype([('grid_e',float),('grid_n',float),('terrain',float),('mask',float)]) )
    burnplot = burnplot.view(np.recarray)
    burnplot.grid_e  = grid_e
    burnplot.grid_n  = grid_n
    burnplot.mask    = plotmask
    
    try: 
        burnplot.terrain = terrain
        
    except: 
        print('terrain was not set to the right resolution or right extension.')
        print('rerun DEM_from_googleEarthDEM.py with the file now saved in ', root_postproc)
        print('and restart. grid is: ')
        print('resolution  = ',   burnplot.grid_e[1,1]-burnplot.grid_e[0,0], burnplot.grid_n[1,1]-burnplot.grid_n[0,0])
        print('domain size = ',   burnplot.grid_e[-1,-1]-burnplot.grid_e[0,0], burnplot.grid_n[-1,-1]-burnplot.grid_n[0,0])
        print('domain dim. = ',   burnplot.grid_e.shape)
        print('--')
        print('terrain dim = ', terrain.shape)

        np.save(root_postproc + 'grid_'+fireName,[grid_e, grid_n, plotmask])
        open(   root_postproc + 'grid_'+fireName+'.prj','w').writelines(utm.ExportToWkt())    
        sys.exit()


    #apply correction on terrain if SimpleHomography to get it as a plan 
    #-------------------
    if flag_georef_mode  == 'SimpleHomography':
        burnplot.terrain_original = burnplot.terrain
        burnplot.terrain = get_best_plane(burnplot.grid_e.flatten(), burnplot.grid_n.flatten(), burnplot.terrain.flatten(), 
                                                flag_plot=flag_plot, dir_out=dir_out, dimension=burnplot.shape, 
                                                maskPlot=burnplot.mask ).reshape(burnplot.shape)


    #compute pixel area in m2
    #--------- 
    idxi = np.arange(0,burnplot.mask.shape[0]) 
    idxj = np.arange(0,burnplot.mask.shape[1]) 
    for iii,jjj in itertools.product(idxi, idxj):
        
        pts = [ ( iii  , jjj   ), \
                ( iii+1, jjj   ), \
                ( iii+1, jjj+1 ), \
                ( iii  , jjj+1 ), \
              ]
        if not ( (iii+1 >= burnplot.mask.shape[0] ) | (jjj+1 >= burnplot.mask.shape[1]) ): 
           
            pts_world = np.array([ (burnplot.grid_e[pts[i]], burnplot.grid_n[pts[i]], burnplot.terrain[pts[i]]) for i in range(4)])
        
            burnplot.area[iii,jjj]= .5*np.linalg.norm(np.cross( (pts_world[1] -pts_world[0]), (pts_world[0] -pts_world[2]) )) + \
                                    .5*np.linalg.norm(np.cross( (pts_world[3] -pts_world[2]), (pts_world[2] -pts_world[0]) ))

        elif (iii+1 >= burnplot.mask.shape[0] ) & (jjj+1 < burnplot.mask.shape[1]):
            burnplot.area[iii,jjj] = burnplot.area[iii-1,jjj]

        elif (iii+1 < burnplot.mask.shape[0] ) & (jjj+1 >= burnplot.mask.shape[0] ):
            burnplot.area[iii,jjj] = burnplot.area[iii,jjj-1]

        elif (iii+1 >= burnplot.mask.shape[0] ) & (jjj+1 >= burnplot.mask.shape[0] ):
            burnplot.area[iii,jjj] = burnplot.area[iii-1,jjj-1]

    np.save(root_postproc + 'grid_'+fireName,burnplot)
    open(   root_postproc + 'grid_'+fireName+'.prj','w').writelines(utm.ExportToWkt())
    
    return burnplot

######################################################
def get_resolution_from_camera_location(cameraLocation,cf_gps_utm,cameraLens,cameraDimension):
    
    center3D = np.average(cf_gps_utm,axis=1)
    nx, ny = cameraDimension

    if len(cameraLocation) == 1:
        loc = np.array([cameraLocation.x,cameraLocation.y,cameraLocation.z])
        distance    = np.sqrt( ((loc-center3D)**2).sum())
        pixel_ground_size = 1./(.5*nx) * distance * np.tan(.5* cameraLens * 3.14/180.) 
        return round(pixel_ground_size,2)
    
    else: # get the closest point
        distance = np.zeros(len(cameraLocation))
        for ii_loc in range(len(cameraLocation)):
            loc = np.array([cameraLocation.x[ii_loc],cameraLocation.y[ii_loc],cameraLocation.z[ii_loc]])
            distance[ii_loc] = np.sqrt( ((loc-center3D)**2).sum())
        
        distance_min = distance.min()
        pixel_ground_size = 1./nx * distance_min * np.tan(.5* cameraLens * 3.14/180.) 
        return round(pixel_ground_size,2)



#######################################################################
def defineGrid(e,n,box_size=50.,res = 1.):
    #get the resolution 
    dxy = res           # in meters
    boxSize = box_size  # in meters
    #border
    max_e = e + boxSize*.5
    min_e = e - boxSize*.5
    max_n = n + boxSize*.5
    min_n = n - boxSize*.5

    acceptable_number = np.arange(1000) * 16.

    nx =  int( round( old_div((max_e - min_e),dxy)) //2 * 2 ) 
    ny =  int( round( old_div((max_n - min_n),dxy)) //2 * 2 ) 
   
    nx = acceptable_number[np.abs(acceptable_number-nx).argmin()]
    ny = acceptable_number[np.abs(acceptable_number-ny).argmin()]

    #reajust resolution to conserve the box size as define
    dxy = old_div(boxSize,(nx))

    grid_e = np.arange(nx)*dxy + min_e 
    grid_n = np.arange(ny)*dxy + min_n
    
    xv, yv = np.meshgrid(grid_e, grid_n)

    return dxy, xv.T, yv.T



#######################################################################
def UTMZone(lon, lat):
    return int(old_div((lon + 180), 6)) + 1 


######################################################
def downgrade_resolution_4nadir(arr, diag_res_cte_shape, flag_interpolation='conservative'):

    '''
    flag_interpolation is conservative, or use max value in the new grid box
    '''
    factor = old_div(1.*arr.shape[0],diag_res_cte_shape[0])
    if factor == int(np.floor(factor)): factor = int(factor)
    else: factor = int(factor) + 1
    #factor = int(np.round(old_div(1.*arr.shape[0],diag_res_cte_shape[0]),0))
    
    if np.mod( arr.shape[0], factor )!=0:
        extra_pixel0 = factor-np.mod( arr.shape[0], factor )
        extra_pixel1 = factor-np.mod( arr.shape[1], factor )
    else: 
        extra_pixel0 = 0
        extra_pixel1 = 0
  
    if (extra_pixel0>0) |  (extra_pixel1>0):
        x = np.arange(0,arr.shape[0],1)
        y = np.arange(0,arr.shape[1],1)
        z = arr.flatten()
        f = interpolate.interp2d(x, y, z, kind='linear')
        
        grid_x = np.arange(0-int(0.5*extra_pixel0),extra_pixel0-int(0.5*extra_pixel0)+arr.shape[0],1)
        grid_y = np.arange(0-int(0.5*extra_pixel1),extra_pixel1-int(0.5*extra_pixel1)+arr.shape[1],1) 
        arr = f(grid_x, grid_y)
        arr = arr.T
    
    if flag_interpolation == 'max':
        return shrink_max(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
   
    elif flag_interpolation == 'min':
        return shrink_min(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
    
    elif flag_interpolation == 'conservative':
        
        mask = np.where(arr!=-999, 1, 0)
        sum_pixel = shrink_sum(mask, diag_res_cte_shape[0], diag_res_cte_shape[1]) 
        
        sum = shrink_sum(arr, diag_res_cte_shape[0], diag_res_cte_shape[1]) 
        return np.where(sum != -999, old_div(sum,sum_pixel), sum)

    elif flag_interpolation == 'average':
        return shrink_average(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
    
    elif flag_interpolation == 'sum':
        return shrink_sum(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])

    else:
        print('bad flag')
        pdb.set_trace()


######################################################
def shrink_sum(data, nx, ny, nodata=-999):
    
    data_masked = np.ma.array(data, mask = np.where(data==nodata,1,0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    return np.where(rshp.sum(3).sum(1).mask==False, rshp.sum(3).sum(1).data, nodata)

######################################################
def shrink_max(data, nx, ny, nodata=-999):
    data_masked = np.ma.array(data, mask = np.where(data==nodata,1,0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    return np.where(rshp.max(3).max(1).mask==False, rshp.max(3).max(1).data, nodata)
    
    #return min3    return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).max(axis=1).max(axis=2)


######################################################
def shrink_min(data, nx, ny, nodata=-999):
    
    data_masked = np.ma.array(data, mask = np.where(data==nodata,1,0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    return np.where(rshp.min(3).min(1).mask == False, rshp.min(3).min(1).data, nodata)

    #return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).min(axis=1).min(axis=2)


######################################################
def shrink_average(data, nx, ny, nodata=-999.):
   
    data_masked = np.ma.array(data, mask = np.where(data==nodata, 1, 0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    mean3 = rshp.mean(3)
    count3 = rshp.count(3)

    # Compute weighted mean along axis 1
    mean1 = old_div((count3*mean3).sum(1),count3.sum(1))
    
    return np.where( mean1.mask, nodata, mean1.data)
    
    '''
    if flag_avergae_nodataMask:
        out = np.zeros([rows,cols]) + nodata
        id_cell = ndimage.zoom(np.arange(rows*cols).reshape([rows,cols]), (data.shape[0]/rows, data.shape[1]/cols), order=0)
        for id_cell_val in np.unique(id_cell):
            idx=np.where( (id_cell==id_cell_val) & (data!=nodata) )
            if len(idx[0])==0: 
                continue
            out[np.unravel_index(id_cell_val, (rows,cols))] = data[idx].mean()
        return out
    
    else: 
        return ((data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).sum(axis=1)/(data.shape[0]/rows)).sum(axis=2)/(data.shape[1]/cols))
    '''

'''
######################################################
def downgrade_resolution_4nadir(arr, diag_res_cte, flag_interpolation='conservative'):

    if flag_interpolation == 'max':
        return np.array( shrink_max(arr, diag_res_cte.shape[0], diag_res_cte.shape[1]), dtype=arr.dtype)
    
    if flag_interpolation == 'min':
        return np.array( shrink_min(arr, diag_res_cte.shape[0], diag_res_cte.shape[1]), dtype=arr.dtype)

    elif flag_interpolation == 'conservative':
        scaling_ratio = arr.shape[0] / diag_res_cte.shape[0]
        return np.array( shrink_sum(arr, diag_res_cte.shape[0], diag_res_cte.shape[1]) / scaling_ratio**2 , dtype=arr.dtype)

    elif flag_interpolation == 'average':
        return np.array( shrink_average(arr, diag_res_cte.shape[0], diag_res_cte.shape[1]), dtype=arr.dtype)
    
    elif flag_interpolation == 'sum':
        return np.array( shrink_sum(arr, diag_res_cte.shape[0], diag_res_cte.shape[1]), dtype=arr.dtype)

    else:
        print 'bad flag'
        pdb.set_trace()

#----------------------------------------------------
def shrink_sum(data, rows, cols):
    return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).sum(axis=1).sum(axis=2)

#----------------------------------------------------
def shrink_max(data, rows, cols):
    return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).max(axis=1).max(axis=2)

#----------------------------------------------------
def shrink_min(data, rows, cols):
    return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).min(axis=1).min(axis=2)

#----------------------------------------------------
def shrink_average(data, rows, cols):
    return ((data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).sum(axis=1)/(data.shape[0]/rows)).sum(axis=2)/(data.shape[1]/cols))

'''

######################################################
def load_loc_camera(cameraLocation_file, ignitionTime, conv_ll2utm):
    
    f = open(cameraLocation_file, 'r')
    lines = f.readlines()
    f.close()
    lines = lines[1:] #remove header
    
    if len(lines) > 1:  # file is coming from helicoptercamera_gpsGPS/mergeGpsKestrel.py
        reader = asciitable.NoHeader()
        reader.data.splitter.delimiter = ' '
        reader.data.start_line = 1
        reader.data.splitter.process_line = None
        data = reader.read(cameraLocation_file)
        
        helico_lon   = data['col3']
        helico_lat   = data['col4']
        helico_alt   = data['col7']
        helico_time_ = data['col1']

        helico_time = []
        for time in helico_time_:
            helico_time.append( (datetime.datetime.strptime(time, "%Y-%m-%d-%H:%M:%S") - ignitionTime).total_seconds() )
        
        item = (0.,0.,0.,0.,0.,0.,0.)
        camera_gps = np.array([item]*len(helico_lon),dtype=np.dtype([('time',float),('lon',float),('lat',float),('alt',float),('x',float),('y',float),('z',float)]))
        camera_gps = camera_gps.view(np.recarray)
        camera_gps.time = helico_time
        camera_gps.lon = helico_lon
        camera_gps.lat = helico_lat
        camera_gps.alt = helico_alt

        #conversion to utm
        for i in range(len(helico_lon)):
            camera_gps.x[i], camera_gps.y[i], camera_gps.z[i] = conv_ll2utm.TransformPoint(camera_gps.lon[i],camera_gps.lat[i],camera_gps.alt[i]) 

    else: # only one location
        item = (0.,0.,0.,0.,0.,0.,0.)
        camera_gps = np.array([item]*1,dtype=np.dtype([('time',float),('lon',float),('lat',float),('alt',float),('x',float),('y',float),('z',float)]))
        camera_gps = camera_gps.view(np.recarray)
        camera_gps.x[0] = float(lines[-1].rstrip().split(' ')[0])
        camera_gps.y[0] = float(lines[-1].rstrip().split(' ')[1])
        camera_gps.z[0] = float(lines[-1].rstrip().split(' ')[2])

    return camera_gps



######################################################
def load_loc_cornerFire(root_postproc, params_gps, conv_ll2utm, cornerFireName='cornerFire_Name.txt'):
    
    dir_gps, cf_file, flag_cf_format = params_gps['dir_gps'],params_gps['loc_cf_file'], params_gps['cf_format']
     
    if flag_cf_format.split('_')[0] == 'shapeFile':
        #get first the name of the cf. it must match the order of the cf in the cornerFireLocator
        cf_gps_ll = np.zeros([3,4])
        cf_gps_utm = np.zeros([3,4])
        f = open(root_postproc+ dir_gps + cornerFireName,'r')
        cf_name = np.array([line.rstrip() for line in f.readlines()])
        f.close()
        
        ctr = shapefile.Reader(root_postproc+dir_gps+cf_file)
        fields_name = np.array([ctr.fields[i][0] for i in range(len(ctr.fields))])
        idx_name   = np.where(fields_name == 'Comment')[0][0]    - 1  # remove the tuple at the start
        idx_heihgt = np.where(fields_name == 'GNSS_Heigh')[0][0] - 1
        
        records = ctr.shapeRecords() #will store the geometry separately
        found_cf_name = 0
        for record in records:
            pts        = record.shape.points #will show you the points of the polygon
            attributes = record.record       #will show you the attributes
         
            idx = np.where( cf_name == attributes[idx_name])
            if len(idx[0]) == 1:
                cf_gps_ll[:2,idx[0][0]] =  pts[0][::-1]
                cf_gps_ll[2,idx[0][0]]  = attributes[idx_heihgt]
                found_cf_name += 1
        if found_cf_name < 4:
            print('shape file for cf location is not well set up.')
            pdb.set_trace()
   
    #deprecated
    elif flag_cf_format.split('_')[0] == 'textFile':
        cf_gps_ll = np.zeros([3,4])
        cf_gps_utm = np.zeros([3,4])
        f = open(root_postproc+dir_gps+cf_file,'r')
        lines = f.readlines()
        f.close()
        for i_line, line in enumerate(lines[1:]):
           cf_gps_ll[:,i_line] = [float(xx) for xx in line.rstrip().split(' ')[0:3]] # conversion to float

        cf_gps_ll = cf_gps_ll[[1,0,2],:]

    elif flag_cf_format.split('_')[0] == 'kml':
        #get first the name of the cf. it must match the order of the cf in the cornerFireLocator
        f = open(root_postproc+ dir_gps + cornerFireName,'r')
        cf_names = np.array([line.rstrip() for line in f.readlines()])
        f.close()
        pts = load_polygon_from_kml(root_postproc+dir_gps+cf_file, params_gps['cf_feature_name'])
        cf_gps_ll  = np.zeros([3,len(cf_names)])
        cf_gps_utm = np.zeros([3,len(cf_names)])
        
        pts_name = np.array([ pt[0] for pt in pts])
        for icf, cf_name in enumerate(cf_names):
            idx_ = np.where(pts_name==cf_name)[0][0]
            cf_gps_ll[:,icf] = [pts[idx_][1][1], pts[idx_][1][0], pts[idx_][2] ]

    else: 
        print('merde stop here')
        pdb.set_trace()

    #wgs84 = osr.SpatialReference( ) # Define a SpatialReference object
    #wgs84.ImportFromEPSG( 4326 ) # And set it to WGS84 using the EPSG code
    #utm = osr.SpatialReference()
    #utm.SetWellKnownGeogCS( 'WGS84' )
    #utm.SetUTM(UTMZone(  *cf_gps_ll[:2,0] ), )
    #conv_ll2utm = osr.CoordinateTransformation(wgs84, utm)
    #conv_utm2ll = osr.CoordinateTransformation(utm,wgs84)
  
    #conversion to utm
    for i in range(cf_gps_ll.shape[1]):
        cf_gps_utm[:,i] = conv_ll2utm.TransformPoint(*cf_gps_ll[:,i])
  
    return cf_gps_utm



###########################################################
def get_center_Coord_ll(flag_format, contour_file, params_gps=None):
    
    if flag_format.split('_')[0] == 'shapeFile':
        #load shape file
        ctr = shapefile.Reader(contour_file)
        records = ctr.shapeRecords() #will store the geometry separately
        #read a firest time just to get 1 point 
        #to use in the conversion to CTM for the zone
        ###########################
        for record in records:
            pts        = record.shape.points #will show you the points of the polygon
            attributes = record.record       #will show you the attributes
            #print attributes[0]
        center = np.array(pts).mean(axis=0)
      
    #deprecated
    elif flag_format.split('_')[0] == 'textFile':
        f = open(contour_file,'r')
        lines = f.readlines()
        f.close()
        pts = np.zeros([3,len(lines)-1])
        for i_line, line in enumerate(lines[1:]):
            #pts[:,i_line] = line.rstrip().split(' ')[1:] # conversion to float
            pts[:,i_line] = [float(xx) for xx in line.rstrip().split(' ')]
        center   = [pts[0,:].mean(), pts[1,:].mean()]
        

    elif flag_format.split('_')[0] == 'kml':
        pts = load_polygon_from_kml(contour_file,params_gps['contour_feature_name'])
        center   = [pts[0,:].mean(), pts[1,:].mean()]
    return center



########################################
def star_triangle_img_pixel_intersection(param):
    return triangle_img_pixel_intersection(*param)


#---------------------------------------
def triangle_img_pixel_intersection( i_tri,pts_xy,rvec, tvec, K, D, ni, nj, cam_loc, img_polygons):
    
    out = []

    pts_xy3D, pts_xy2D, tri_coord = to_planar_coord(np.array(pts_xy))

    tri_center = pts_xy3D[:3,:].mean(axis=0)

    view_direction = cam_loc.T-tri_center

    pts_ij_, _ = cv2.projectPoints(np.array(pts_xy), rvec, tvec, K, D)
    pts_ij = np.fliplr(pts_ij_[:,0,:])

    ii_img_polygon_l = max([0, int(np.round(pts_ij[:,0].min())) -1])
    ii_img_polygon_u = min([ni,int(np.round(pts_ij[:,0].max())) +1])
    jj_img_polygon_l = max([0, int(np.round(pts_ij[:,1].min())) -1])
    jj_img_polygon_u = min([nj,int(np.round(pts_ij[:,1].max())) +1])

    if ((ii_img_polygon_l>=ni) & (ii_img_polygon_u>=ni)) |\
       ((ii_img_polygon_l<=0   ) & (ii_img_polygon_u<=0   )) |\
       ((jj_img_polygon_l>=nj) & (jj_img_polygon_u>=nj)) |\
       ((jj_img_polygon_l<=0   ) & (jj_img_polygon_u<=0   )) : 
        return out

    grid_H, mask = cv2.findHomography(pts_ij, pts_xy2D, cv2.RANSAC,5.0)
    
    grid_polygon = Polygon(pts_ij[:3])#.convex_hull
    grid_area       = grid_polygon.area
    if grid_area < 5.e-3 : #for point on the edge where distortion is big due to the bad distortion model
                           #for now we neglect those points. To be removed when the distortion is better estimated ##MERDE
        return out

    grid_area_m2    = .5*np.linalg.norm(np.cross( (pts_xy2D[1]-pts_xy2D[0]), (pts_xy2D[0]-pts_xy2D[2]) ))

    idxi = np.arange(ii_img_polygon_l,ii_img_polygon_u) 
    idxj = np.arange(jj_img_polygon_l,jj_img_polygon_u) 
    for iii,jjj in itertools.product(idxi, idxj):
        
        pts = [ [ iii  , jjj   ], \
                [ iii+1, jjj   ], \
                [ iii+1, jjj+1 ],
                [ iii  , jjj+1 ], \
              ]
        img_polygons_ = Polygon(pts)

        intersection = grid_polygon.intersection(img_polygons_)
        #intersection = grid_polygon.intersection(img_polygons[iii,jjj])
        if intersection.area!=0:
            intersect_ij = np.array(intersection.exterior.coords.xy,dtype=np.float32).T.reshape(-1,1,2)
            intersect_xy = cv2.perspectiveTransform(intersect_ij,grid_H).reshape(-1,2)
            intersect_xyz = to_3d_coord(intersect_xy,*tri_coord)
            grid_img_intersection_area_m2 = Polygon( intersect_xy).area
           
            if old_div((grid_img_intersection_area_m2- grid_area_m2),grid_area_m2) > 1.e-3 :
                
                if mpl.get_backend() == 'Agg':
                    plt.clf()
                    ax = plt.subplot(121)
                    ax.scatter(intersect_xy[:,0],intersect_xy[:,1],c='m',s=100)
                    ax.scatter(pts_xy2D[:3,0],pts_xy2D[:3,1],s=10)
                    ax = plt.subplot(122)
                    #imgi,imgj = img_polygons[iii,jjj].exterior.coords.xy
                    imgi,imgj = img_polygons_.exterior.coords.xy
                    ax.scatter(imgi,imgj,c='r',s=100)
                    ax.scatter(intersect_ij[:,0,0],intersect_ij[:,0,1],c='m',s=100)
                    ax.scatter(pts_ij[:3,0],pts_ij[:3,1],s=10)
                    pdb.set_trace()
                out.append([None])
            else:
                out.append([iii,jjj,i_tri,grid_img_intersection_area_m2])

    return out



########################################
def to_planar_coord(pts_xy):
    xx = pts_xy[1] - pts_xy[0]
    xx = old_div(xx,np.linalg.norm(xx))
    zz = np.cross(xx,pts_xy[2] - pts_xy[0])
    zz = old_div(zz,np.linalg.norm(zz))
    yy = np.cross(zz,xx)
    return pts_xy, np.array( [ [np.dot(xx,pts_xy[i]-pts_xy[0]), np.dot(yy,pts_xy[i]-pts_xy[0])] for i in range(pts_xy.shape[0])]), (xx,yy,zz,pts_xy[0])



########################################
def to_3d_coord(pts2d,xx,yy,zz,x0):
    return np.array([x0 + pts2d[i,0]*xx + pts2d[i,1]*yy   for i in range(pts2d.shape[0])])



########################################
def triangle_img_pixel_match(id_img, triangles, img_dimension, rvec, tvec, K, D, wkdir, flag_parallel=False, flag_restart=False, flag_outptmap=False):

    ni,nj = img_dimension

    #project triangle in image plan
    #-----------
    #print 'project triangles on images plan'
    #triangles_img = []
    #for triangle in triangles:
    #    triangle_img, _ = cv2.projectPoints(np.array(triangle), rvec, tvec, K, D)
    #    triangles_img.append(np.fliplr(triangle_img[:,0,:]))
    '''
    img_cam_j, img_cam_i = np.meshgrid(np.arange(nj+1),np.arange(ni+1))
    img_polygons = []
    for ii,jj in itertools.product(range(ni), range(nj)):
        pts = [ [ img_cam_i[ii,  jj  ],img_cam_j[ii,  jj  ] ], \
                [ img_cam_i[ii+1,jj  ],img_cam_j[ii+1,jj  ] ], \
                [ img_cam_i[ii+1,jj+1],img_cam_j[ii+1,jj+1] ], \
                [ img_cam_i[ii  ,jj+1],img_cam_j[ii  ,jj+1] ], \
              ]
        img_polygons.append(Polygon(pts))
    img_polygons = np.array(img_polygons).reshape(ni,nj)
    '''
    img_polygons = None
    cam_loc, cam_angle = get_cam_loc_angle(rvec,tvec)
    #print '----'
    #print cam_loc, cam_angle
    #print '----'
    
    img_grid_list = np.empty((ni*nj, 0)).tolist()
    grid_img_list = np.empty((len(triangles), 0)).tolist()

    #match triangles and image pixels
    #-----------
    if (not(flag_restart)) | (not os.path.isfile(wkdir+'frame{:06d}_grid_img_lookuptable.p'.format(id_img))):
        #print '   match triangles and image pixels'
       
        params = []
        for i_tri, pts_xy in enumerate(triangles):
            params.append([i_tri,pts_xy, rvec, tvec, K, D, ni, nj]) 

        flag_parallel_ = True
        if flag_parallel_:
            # set up a pool to run the parallel processing
            cpus = cpu_count()
            pool = multiprocessing.Pool(processes=cpus)

            # then the map method of pool actually does the parallelisation  
            results = pool.map(georefWT.star_triangle_img_pixel_intersection, params)
            pool.close()
            pool.join()
           
        else:
            results = []
            for param in params:
                print('{:5.2f}\r'.format(old_div(100.*param[0],len(triangles))), end=' ')
                sys.stdout.flush()
                results.append(georefWT.triangle_img_pixel_intersection(*param))

        if flag_outptmap: pts_img_xyz_map = np.zeros([ni,nj,3])
        for (out, out_pt_xyz) in results:
            if None in list(itertools.chain.from_iterable(out)):
                 print('triangle_img_pixel_match failed. do not do geroef')
                 return None, None
            for (iii,jjj,i_tri,grid_img_intersection_area_m2) in out:
                idx_img_pixel = np.ravel_multi_index([[iii],[jjj]],(ni,nj))[0]
                img_grid_list[idx_img_pixel].append([i_tri,grid_img_intersection_area_m2])

                grid_img_list[i_tri].append([(iii,jjj),grid_img_intersection_area_m2])

            if (flag_outptmap) & (len(out_pt_xyz)!=0):
                iii,jjj,pt_xyz = out_pt_xyz[0]
                pts_img_xyz_map[iii,jjj,:] = pt_xyz[0]
        
        pickle.dump([img_grid_list,grid_img_list], open( wkdir+'frame{:06d}_grid_img_lookuptable.p'.format(id_img), "wb" ) )
        if (flag_outptmap): np.save( wkdir+'frame{:06d}_2d23d.npy'.format(id_img), pts_img_xyz_map)
   
    else:
        #print '   load triangle img pixel lookup table'
        img_grid_list,grid_img_list = pickle.load(open( wkdir+'frame{:06d}_grid_img_lookuptable.p'.format(id_img), 'rb'))
        pts_img_xyz_map = np.load( wkdir+'frame{:06d}_2d23d.npy'.format(id_img) )

    if flag_outptmap: 
        return img_grid_list,grid_img_list,pts_img_xyz_map
    else:
        return img_grid_list,grid_img_list



#################################################
def get_cam_loc_angle(rvec,tvec):
    
    rotM_cam = cv2.Rodrigues(rvec)[0]
    cameraPosition = np.array(-(np.matrix(rotM_cam).T * np.matrix(tvec)))
   
    tmp = transformation.euler_from_matrix(rotM_cam,'rzyz')
    cameraAngle = [180 + 180/3.14*tmp[1], -tmp[2]*180/3.14, 180+(tmp[0]*180/3.14)]
    
    return cameraPosition, cameraAngle


##################################################
def warp_frame_on_prev_frame(camera, params_camera, params_georef, frame_in, frame_ref, 
                             frame_ref00, frame_ref00_init,
                             flag_parallel, win_size_ssim, lk_params, dir_out_frame, flag_update=True):

    if flag_update:
        frame = frame_in
    else:
        frame = frame_in.copy()

    frames_ref = [frame_ref]

    param = [frame.type,frame,frame_ref,params_camera,lk_params]
    results_matching_feature = [star_get_matching_feature(param) ]
    results_frame_ref_id = [param[2].id]
    
    good_new = []
    good_old = []
    nbrept_badLoc= 0; nbrept_badTemp2 = 0
    for res in results_matching_feature:
        p0_good, p1_good_on_ref00, nbrept_badLoc_, nbrept_badTemp2_ = res
        if p0_good is None : continue
        nbrept_badLoc   += nbrept_badLoc_
        nbrept_badTemp2 += nbrept_badTemp2_
        for ii in range(p0_good.shape[0]):
            good_new.append(p0_good[ii,:]) 
            good_old.append(p1_good_on_ref00[ii,:]) 

    good_old = np.copy(good_old).reshape(-1,2)
    good_new = np.copy(good_new).reshape(-1,2)
    good_new_4plot = good_new

    try:
        H_new, _ = cv2.findHomography(good_new, good_old, cv2.RANSAC,5.0)
    except: 
        H_new = None

    if H_new is None:
        frame.set_correlation(0., 0., 0., 0.)
        #frame.set_trange(params_georef['trange']) 
        #frame.set_img()
        return frame, results_matching_feature


    nx,ny = frame.img.shape
    frame.set_warp(cv2.warpPerspective(frame.img, H_new,               \
                                 (ny,nx),                              \
                                 borderValue=0))


    frame.set_maskWarp( cv2.warpPerspective(frame.mask_img, H_new,         \
                                     (ny,nx),                              \
                                     borderValue=0,flags=cv2.INTER_NEAREST))
    
    frame.set_homography_to_ref(H_new)
    frame.save_feature_old_new(good_new_4plot,good_new,good_old)

    mask_func_param = frame.lowT_param  if frame.type=='lwir' else [1.e6, 0]   #[150, 3] # lowT, kernel_lowT
    mask_func = mask_lowT if frame.type=='lwir' else mask_onlyImageMask # add plumemask

    #compute EP08 and ssim agains each ref to get best match
    #-------------
    params = []
    for i_frame_ref, frame_ref in  enumerate(frames_ref):
        params.append([params_camera['flag_costFunction'], frame, frame_ref, mask_func, mask_func_param]) 

    flag_parallel_ = False

    if flag_parallel_: 
        # set up a pool to run the parallel processing
        cpus = cpu_count()
        pool = multiprocessing.Pool(processes=cpus)

        # then the map method of pool actually does the parallelisation  
        result_costFunction = pool.map(star_get_costFunction, params)
        pool.close()
        pool.join()
       
    else:
        result_costFunction = []
        for param in params:
            result_costFunction.append(star_get_costFunction(param))
            
    correlation_arr = [] 
    for i_frame_ref, res in enumerate(result_costFunction):
        correlation_arr.append(res)

    idx_energgy = np.array(correlation_arr).argmax()
    corr_ref = correlation_arr[idx_energgy]
    
    
    #compute EP08 and ssim agains ref00
    #-------------
    if win_size_ssim != 0: 
        mask_ssim, ssim_2d, ssim = star_get_costFunction([ 'ssim', frame, frame_ref00, win_size_ssim ] )
        frame.set_similarity_info(ssim_2d, mask_ssim)
    else: 
        ssim = -999.
    frame.set_correlation(corr_ref,
                          star_get_costFunction(['EP08', frame, frame_ref00,      mask_EP08, mask_func_param]),
                          star_get_costFunction(['EP08', frame, frame_ref00_init, mask_EP08, mask_func_param]),
                          ssim) 
  
    if flag_update:
        #print status
        #------------------
        if frame.type == 'lwir':
            print('{:5d} {:5d} {:5d} | {:6d} {:6d} {:6d} | {:5.3f} {:04d} | {:5.3f} {:5.3f} |'.format( 
                                                                           nbrept_badLoc, 
                                                                           nbrept_badTemp2,
                                                                           0, 
                                                                           good_old.shape[0], good_new_4plot.shape[0], len(frames_ref),
                                                                           frame.corr_ref,frames_ref[idx_energgy].id, 
                                                                           frame.corr_ref00,frame.corr_ref00_init ), end=' ') 
        elif frame.type == 'visible': 
            print('{:5d} {:5d} {:5d} | {:6d} {:6d} {:6d} | {:5.3f} {:04d} | {:5.3f} {:5.3f} |'.format(nbrept_badLoc,nbrept_badTemp2, 
                                                                            0, 
                                                                            good_old.shape[0], good_new_4plot.shape[0], len(frames_ref),
                                                                            frame.corr_ref,frames_ref[idx_energgy].id, 
                                                                            frame.corr_ref00,frame.corr_ref00_init ), end=' ')

    
    frame.set_id_best_ref(frames_ref[idx_energgy].id)
    
    return frame, results_matching_feature


#############################################
def warp_frame(camera, params_camera, params_georef, frame_in, framesID_ref, 
               frame_ref00, frame_ref00_init,
               flag_parallel, win_size_ssim, lk_params, dir_out_frame, feature_last_call=[None, None]):

    frame = frame_in.copy()

    frames_ref = []
    for id in framesID_ref:
        frames_ref.append(camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(id)))

    lastCall_refId            = np.array(feature_last_call[0])
    lastCall_matching_feature = feature_last_call[1]
    params = []
    results_matching_feature = []
    results_frame_ref_id = []
    for i_frame_ref, frame_ref in  enumerate(frames_ref):
        if frame_ref.id in lastCall_refId:
            results_matching_feature.append( lastCall_matching_feature[np.where(lastCall_refId == frame_ref.id)[0][0] ] )
            results_frame_ref_id.append(frame_ref.id)
        else:
            params.append([frame.type,frame,frame_ref,params_camera,lk_params])

    if len(frames_ref) == 1: 
        flag_parallel_ = False
    else:
        flag_parallel_ = flag_parallel
   
    if flag_parallel_: 
        # set up a pool to run the parallel processing
        cpus = cpu_count()
        pool = multiprocessing.Pool(processes=cpus)

        # then the map method of pool actually does the parallelisation  
        [results_matching_feature.append( res ) for res in pool.map(star_get_matching_feature, params)]
        [results_frame_ref_id.append( params[ii][2].id ) for ii in range(len(params))]

        pool.close()
        pool.join()
       
    else:
        for param in params:
            results_matching_feature.append(star_get_matching_feature(param))
            results_frame_ref_id.append(param[2].id)
            
    good_new = []
    good_old = []
    nbrept_badLoc= 0; nbrept_badTemp2 = 0
    for res in results_matching_feature:
        p0_good, p1_good_on_ref00, nbrept_badLoc_, nbrept_badTemp2_ = res
        if p0_good is None : continue
        nbrept_badLoc   += nbrept_badLoc_
        nbrept_badTemp2 += nbrept_badTemp2_
        for ii in range(p0_good.shape[0]):
            good_new.append(p0_good[ii,:]) 
            good_old.append(p1_good_on_ref00[ii,:]) 

    good_old_ = np.copy(good_old)
    good_new_ = np.copy(good_new)
   
    #remove duplicate
    if len(good_old)> 0:
        df_pair = pandas.DataFrame({'xnew':good_new_[:,0],'ynew':good_new_[:,1], 'xold':good_old_[:,0],'yold':good_old_[:,1]})
        df_pair = df_pair.drop_duplicates()
    else: 
        frame.set_correlation(0., 0., 0., 0.)
        return frame, [None, None]

    #
    df_new = df_pair[['xnew','ynew']].reset_index(drop=True); df_new.columns = ['x','y'] #pandas.DataFrame({'x':good_new_[:,0],'y':good_new_[:,1]})
    df_old = df_pair[['xold','yold']].reset_index(drop=True); df_old.columns = ['x','y'] #pandas.DataFrame({'x':good_old_[:,0],'y':good_old_[:,1]})
    gg = df_new.groupby(['x','y'])
    group_indices = gg.indices
    
    nbre_pt_too_sparse = 0
    idx_df_2_rm = []
    replacement_4df_new = []
    replacement_4df_old = []
    for pts in group_indices:
        if df_new.iloc[group_indices[pts]].shape[0]>3:
            std_pts = np.std(df_old.iloc[group_indices[pts]],axis=0)
            
            if (std_pts['x']<.5) & (std_pts['y']<.5):
                continue

            pt_new = np.array(df_new.iloc[group_indices[pts]])
            pt_old = np.array(df_old.iloc[group_indices[pts]])
            
            #kde = scipy.stats.kde.gaussian_kde(pt_old.T)
            
            m1, m2 = pt_old[:,0], pt_old[:,1]
            xmin = m1.min()
            xmax = m1.max()+0.0001
            ymin = m2.min()
            ymax = m2.max()+0.0001

            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            dX, dY = X[1,1]-X[0,0], Y[1,1]-Y[0,0] 
            positions = np.vstack([X.ravel(), Y.ravel()])
            values = np.vstack([m1, m2])
            try:
                kernel = scipy.stats.gaussian_kde(values)
                Z = np.reshape(kernel(positions).T, X.shape)
                zone_ok = np.where( Z>.66*Z.max(), np.ones_like(X), np.zeros_like(X) )  
            except: 
                zone_ok = np.zeros_like(X) 

            #if (frame.id == 4) & (pts[0] ==97.0) & (pts[1] == 340.0): 
            #    pdb.set_trace()

            pt_new2 = []
            pt_old2 = []
            pt_2_indices_df = []
            for i_pt, [pt,pt_index_df] in enumerate(zip(pt_old,group_indices[pts])):
                idx_pt = np.where( (pt[0]>=X[:-1,:-1]) & (pt[0]<X[1:,1:]) & (pt[1]>=Y[:-1,:-1]) & (pt[1]<Y[1:,1:]) )
                if len(idx_pt[0])!=1: pdb.set_trace()
                if zone_ok[idx_pt] == 0:
                    idx_df_2_rm.append(pt_index_df) 
                    nbre_pt_too_sparse += len(idx_pt[0])
                else:
                    pt_new2.append(pt_new[i_pt])
                    pt_old2.append(pt)
                    pt_2_indices_df.append(pt_index_df)

            #plt.clf()
            #plt.imshow(Z.T,extent=[xmin, xmax, ymin, ymax],origin='lower')
            #[plt.plot(mm1, mm2, 'k.', markersize=20) for mm1,mm2 in pt_2_plot ]
            #pdb.set_trace()
            
            '''
            idx = np.where(Z==Z.max())
   
            df_new2add = pandas.DataFrame({'x':pt_new[0,0],'y':pt_new[0,1], 'index': [df_old.iloc[group_indices[pt]].index[0]]  } )  
            df_new2add = df_new2add.set_index('index')
            df_old2add = pandas.DataFrame({'x':X[idx]     ,'y':Y[idx]     , 'index': [df_new.iloc[group_indices[pt]].index[0]]  } ) 
            df_old2add = df_old2add.set_index('index')
        
            replacement_4df_new.append(df_new2add)
            replacement_4df_old.append(df_old2add)
            '''
            
            pt_new2 = np.array(pt_new2)
            pt_old2 = np.array(pt_old2)
          
            if pt_old2.shape[0] <= 2: 
                [idx_df_2_rm.append(idx_) for idx_ in pt_2_indices_df ]
                nbre_pt_too_sparse += len(pt_2_indices_df)
                continue

            dist_between_pts = scipy.spatial.distance.pdist(pt_old2)
            median_dist = np.median(dist_between_pts)

            if median_dist>2:
                [idx_df_2_rm.append(idx_) for idx_ in pt_2_indices_df ]
                nbre_pt_too_sparse += len(pt_2_indices_df)

    try:
        if len(idx_df_2_rm) > 0:
            df_new = df_new.drop(idx_df_2_rm)
            df_old = df_old.drop(idx_df_2_rm)

            #for df in replacement_4df_new:
            #    df_new = df_new.append(df) 
            #for df in replacement_4df_old:
            #    df_old = df_old.append(df) 
    except: 
        pdb.set_trace()

    good_new_4plot = df_new.drop_duplicates().values

    good_old = np.array(df_old)
    good_new = np.array(df_new)

    if (len(good_new)>=4)  & (len(good_old)>=4):
        try:
            H_new, _ = cv2.findHomography(good_new, good_old, cv2.RANSAC,5.0)
        except: 
            pdb.set_trace()
    else:
        frame.set_correlation(0., 0., 0., 0.)
        return frame, [None, None]

    if H_new is None:
        frame.set_correlation(0., 0., 0., 0.)
        return frame, [None, None]


    nx,ny = frame.img.shape
    frame.set_warp(cv2.warpPerspective(frame.img, H_new,               \
                                 (ny,nx),                              \
                                 borderValue=0))


    frame.set_maskWarp( cv2.warpPerspective(frame.mask_img, H_new,         \
                                     (ny,nx),                              \
                                     borderValue=0,flags=cv2.INTER_NEAREST))
    
    frame.set_homography_to_ref(H_new)
    frame.save_feature_old_new(good_new_4plot,good_new,good_old)

    mask_func_param = frame.lowT_param  if frame.type=='lwir' else [1.e6, 0]   #[150, 3] # lowT, kernel_lowT
    mask_func = mask_lowT if frame.type=='lwir' else mask_onlyImageMask # add plumemask

    #compute EP08 and ssim agains each ref to get best match
    #-------------
    params = []
    for i_frame_ref, frame_ref in  enumerate(frames_ref):
        params.append([params_camera['flag_costFunction'], frame, frame_ref, mask_func, mask_func_param]) 

    flag_parallel_ = flag_parallel

    if flag_parallel_: 
        # set up a pool to run the parallel processing
        cpus = cpu_count()
        pool = multiprocessing.Pool(processes=cpus)

        # then the map method of pool actually does the parallelisation  
        result_costFunction = pool.map(star_get_costFunction, params)
        pool.close()
        pool.join()
       
    else:
        result_costFunction = []
        for param in params:
            result_costFunction.append(star_get_costFunction(param))
            
    correlation_arr = [] 
    for i_frame_ref, res in enumerate(result_costFunction):
        correlation_arr.append(res)

    idx_energgy = np.array(correlation_arr).argmax()
    corr_ref = correlation_arr[idx_energgy]
    
    
    #compute EP08 and ssim agains ref00
    #-------------
    if win_size_ssim != 0: 
        mask_ssim, ssim_2d, ssim = star_get_costFunction([ 'ssim', frame, frame_ref00, win_size_ssim ] )
        frame.set_similarity_info(ssim_2d, mask_ssim)
    else: 
        ssim = -999.
    frame.set_correlation(corr_ref,
                          star_get_costFunction(['EP08', frame, frame_ref00,      mask_EP08, mask_func_param]),
                          star_get_costFunction(['EP08', frame, frame_ref00_init, mask_EP08, mask_func_param]),
                          ssim) 
   
    

    #print status
    #------------------
    if frame.type == 'lwir':
        print('{:5d} {:5d} {:5d} {:5d} {:5d} | {:6d} {:6d} {:6d} | {:5.3f} {:04d} | {:5.3f} {:5.3f} |'.format(frame.nbrept_badTemp, 
                                                                       frame.nbrept_helico, 
                                                                       nbrept_badLoc, 
                                                                       nbrept_badTemp2,
                                                                       nbre_pt_too_sparse, 
                                                                       good_old.shape[0], good_new_4plot.shape[0], len(frames_ref),
                                                                       frame.corr_ref,frames_ref[idx_energgy].id, 
                                                                       frame.corr_ref00,frame.corr_ref00_init ), end=' ') 
    elif frame.type == 'visible': 
        print('{:5d} {:5d} {:5d} | {:6d} {:6d} {:6d} | {:5.3f} {:04d} | {:5.3f} {:5.3f} |'.format(nbrept_badLoc,nbrept_badTemp2, 
                                                                        nbre_pt_too_sparse, 
                                                                        good_old.shape[0], good_new_4plot.shape[0], len(frames_ref),
                                                                        frame.corr_ref,frames_ref[idx_energgy].id, 
                                                                        frame.corr_ref00,frame.corr_ref00_init ), end=' ')

    
    frame.set_id_best_ref(frames_ref[idx_energgy].id)

    return frame, [results_frame_ref_id, results_matching_feature]




###################################################
def star_get_matching_feature(param):
   
    if param[0] == 'lwir':  
        try: 
            p0_good, p1_good_on_ref00, nbrept_badLoc, nbrept_badTemp2 = get_matching_feature_opticalFlow(*param[1:],input_img=['img','feature','trange'])
        except: 
            pdb.set_trace()
        if 'trange2' in param[1].__dict__:
            p0_good2, p1_good_on_ref002, nbrept_badLoc2, nbrept_badTemp22 = get_matching_feature_opticalFlow(*param[1:],input_img=['img2','feature2','trange2'])
        else:
            p0_good2 = None 

        
        if (p0_good2 is not None) & (p0_good is not None):
            return np.concatenate((p0_good,p0_good2),axis=0), np.concatenate((p1_good_on_ref00,p1_good_on_ref002),axis=0), nbrept_badLoc+nbrept_badLoc2, nbrept_badTemp2+nbrept_badTemp22
        elif (p0_good2 is None): 
            return p0_good, p1_good_on_ref00, nbrept_badLoc, nbrept_badTemp2
        elif (p0_good is None):
            #frame_here = param[1]
            #frame_here.trange = frame_here.trange2
            #frame_here.img    = frame_here.img2
            #frame_here.trange2 = None 
            #frame_here.img2    = None 
            return p0_good2, p1_good_on_ref002, nbrept_badLoc2, nbrept_badTemp22

    if param[0] == 'visible': return get_matching_feature_SIFT(*param[1:])



#---------------------------------------------
def get_matching_feature_SIFT(frame,frame_ref, params_camera, lk_params, flag='use img',newtrange=None, mask_in=None):
    MIN_MATCH_COUNT = 10

    nbre_feature_to_select = 2000
    nbre_match_pt_1 = 0
    nbre_call_sift = 1

    if flag == 'use img':
        input      = getattr(frame,'img')
        input_ref  = getattr(frame_ref,'img')
        input_mask = 'mask_img'
    elif flag == 'use warp': 
        input      = getattr(frame,'warp')
        input_ref  = getattr(frame_ref,'warp')
        input_mask = 'mask_warp'
    elif flag == 'use new trange':
        input      = frame.return_warp(newtrange)
        input_ref  = frame_ref.return_warp(newtrange)
        input_mask = 'mask_warp'
    if flag == 'use for Wt':
        input      = getattr(frame,'img')
        input_ref  = frame_ref.return_img(frame.trange)
        input_mask = 'mask_img'

    #ax = plt.subplot(121)
    #ax.imshow(input.T,origin='lower',cmap=mpl.cm.Greys_r)
    #ax = plt.subplot(122)
    #ax.imshow(input_ref.T,origin='lower',cmap=mpl.cm.Greys_r)
    #plt.show()

    while nbre_match_pt_1 < 200: 

        sift = cv2.xfeatures2d.SIFT_create(nbre_call_sift*nbre_feature_to_select) # limit number of feature
       
        if mask_in is None:
            mask_ = mask_in
        else:
            mask_ = mask_in[  old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)]
        # find the keypoints and descriptors with SIFT
        kp1_vis, des1_vis = sift.detectAndCompute(input[    old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)],mask_)
        kp2_vis, des2_vis = sift.detectAndCompute(input_ref[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)],mask_)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)  # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(des1_vis,des2_vis,k=2)

        good_vis = []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good_vis.append(m)

        '''
        ##brutefoce
        bf_vis = cv2.BFMatcher()
        matches_vis  = bf_vis.knnMatch(des1_vis,des2_vis, k=2)
        matches_vis2 = bf_vis.knnMatch(des2_vis,des1_vis, k=2)

        # store all the good matches as per Lowe's ratio test.
        good_vis = []
        for m,n in matches_vis:
            if m.distance < 0.5*n.distance:
                good_vis.append(m)
        
        good_vis2 = []
        for m,n in matches_vis2:
            if m.distance < 0.5*n.distance:
                good_vis2.append(m)
        '''
        
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
    src_pts = np.float32([ kp1[m.queryIdx].pt for m,kp1 in zip(good_vis,kp1_all) ]).reshape(-1,1,2) + .5*frame.bufferZone
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m,kp2 in zip(good_vis,kp2_all) ]).reshape(-1,1,2) + .5*frame.bufferZone
   

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



#---------------------------------------------
def get_matching_feature_opticalFlow(frame, frame_ref, params_camera, lk_params, input_img=['img','feature','trange']):

    #print '##MM', frame.id, frame_ref.id, getattr(frame,input_img[2])

    p0 = getattr(frame,input_img[1]).reshape(-1,1,2)
    nx,ny = frame.img.shape

    # calculate optical flow
    if frame.type == 'lwir':
        try:
            p1,  st, err = cv2.calcOpticalFlowPyrLK(getattr(frame,input_img[0])                     , frame_ref.return_img(getattr(frame,input_img[2])), p0, None, **lk_params)
        except: 
            pdb.set_trace()
        p0r, st, err = cv2.calcOpticalFlowPyrLK(frame_ref.return_img(getattr(frame,input_img[2])), getattr(frame,input_img[0])                         , p1, None, **lk_params)
    
    elif frame.type == 'visible' :
        img_ref = hist_matching.hist_matching(getattr(frame,input_img[0]),frame_ref.img) # set histogram from img to ref img
        p1,  st, err = cv2.calcOpticalFlowPyrLK(getattr(frame,input_img[0]), img_ref  , p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img_ref  , getattr(frame,input_img[0]), p1, None, **lk_params)

   
    try:
        d = np.sqrt( ((p0[:,0,:]-p0r[:,0,:])**2).sum(-1) )
    except: 
        pdb.set_trace()

    marge_extra = 0 if (frame.img.shape[0]< 500) else 20
    marge = int(frame.bufferZone/2) + marge_extra
    good = np.where( (d < 1)                              &\
                     (p1[:,0,0]>=marge) & (p1[:,0,0]<=ny-marge) &\
                     (p1[:,0,1]>=marge) & (p1[:,0,1]<=nx-marge), \
                     np.ones(d.shape,dtype=bool), np.zeros(d.shape,dtype=bool) )
    
    nbrept_badLoc = len(np.where( good == False)[0])

    if frame.type == 'lwir':
        #average temperature at each good point on both side
        sbox=5
        idx_t1 = (np.array(np.round(p1[good,0,1],0),dtype=int),np.array(np.round(p1[good,0,0],0,),dtype=int))
        t1 = []
        for i,j in zip(idx_t1[0],idx_t1[1]):
            i_l = max([0,i-sbox]); i_r = min([i+sbox,nx])
            j_l = max([0,j-sbox]); j_r = min([j+sbox,nx])
            t1.append(frame_ref.temp[i_l:i_r,j_l:j_r].mean())

        t0 = []
        idx_t0 = (np.array(np.round(p0[good,0,1],0),dtype=int),np.array(np.round(p0[good,0,0],0,),dtype=int))
        for i,j in zip(idx_t0[0],idx_t0[1]):
            i_l = max([0,i-sbox]); i_r = min([i+sbox,nx])
            j_l = max([0,j-sbox]); j_r = min([j+sbox,nx])
            t0.append(frame.temp[i_l:i_r,j_l:j_r].mean())

        idx_temp_diff = np.where( np.abs( old_div((np.array(t0)-np.array(t1)),np.array(t0)) ) > .1 )
        nbrept_badTemp2 = len(idx_temp_diff[0])
        good[idx_temp_diff] = False

        # Select good points
        p1_good = p1[good,:,:] 
        p0_good = p0[good,:,:] 
    
    else:
        nbrept_badTemp2 = 0
        p1_good = p1[:,:,:] 
        p0_good = p0[:,:,:] 
    
    if p0_good.size == 0 :
        return None, None, None, None 
    
    # convert point from ref to ref00
    p1_good_on_ref00 = cv2.perspectiveTransform(p1_good,frame_ref.H2Ref)

    #set point on that are out on the edge so that they are remove when checking if in the ring or not
    try:
        idx_ = np.array(np.round(p1_good_on_ref00,0),dtype=int)
        idx_[:,0,1] = np.where(idx_[:,0,1]<0,0,idx_[:,0,1])
        idx_[:,0,1] = np.where(idx_[:,0,1]>nx-1,nx-1,idx_[:,0,1])
        idx_[:,0,0] = np.where(idx_[:,0,0]<0,0,idx_[:,0,0])
        idx_[:,0,0] = np.where(idx_[:,0,0]>ny-1,ny-1,idx_[:,0,0])
    except: 
        pdb.set_trace()
    
    if params_camera['of_feature_kernel_ring']:
        flag_ring = frame_ref.plotMask_withBuffer_ring[(idx_[:,0,1],idx_[:,0,0])]
    else:
        flag_ring = np.zeros_like(idx_[:,0,1])+2

    idx_good_good = np.where(flag_ring==2)[0]
    p0_good = p0_good[idx_good_good,:,:]
    p1_good_on_ref00 = p1_good_on_ref00[idx_good_good,:,:]

    
    ''' 
    plt.clf()
    ax = plt.subplot(121)
    ax.imshow(frame.img.T,origin='lower',cmap=mpl.cm.Greys_r,interpolation='nearest')
    ax.scatter(p0[:,0,1],p0[:,0,0],c='g',s=80)
    ax.scatter(p0_good[:,0,1],p0_good[:,0,0],c='r')
    ax = plt.subplot(122)
    ax.imshow(frame_ref.img.T,origin='lower',cmap=mpl.cm.Greys_r,interpolation='nearest')
    ax.scatter(p1_good[:,0,1],p1_good[:,0,0],c='r')
    plt.show()
    pdb.set_trace()
    '''
    
    if p0_good.size == 0 :
        return None, None, None, None 
    else: 
        return p0_good[:,0,:], p1_good_on_ref00[:,0,:], nbrept_badLoc, nbrept_badTemp2

'''
################################################################
def get_matching_feature_opticalFlow_prev_frames(frame, params_camera, dir_out_frame, camera, lk_params): 

    input_img=['warp','feature','trange']

    feature_params = dict( maxCorners = 5000,
                           qualityLevel = 0.2, #lwir KNP14 0.3
                           #qualityLevel = 0.2, #vis
                           minDistance = 2,
                           blockSize = 31 )

    img      = getattr(frame,     input_img[0])
    nx,ny = frame.img.shape
    p0 = cv2.goodFeaturesToTrack(img, mask = np.where(img>200,np.ones_like(frame.mask_img),np.zeros_like(frame.mask_img)), **feature_params)

    iprev = 1
    frame_prev = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(frame.id-iprev))
    img_prev_ = getattr(frame_prev,input_img[0])
   
    plumeMask = np.zeros_like(frame.mask_img)
    while( (iprev<10) & ((frame.id-iprev)>=0) ):

        img_prev = hist_matching.hist_matching(img,img_prev_) 
        
        diff = (np.array(img,dtype=np.float)-np.array(img_prev,dtype=np.float))
       
        idx = np.where((img>200)&(img_prev<200)&(np.abs(diff)>50))
        plumeMask[idx] += 1

        print(iprev)
        
        iprev += 1
        frame_prev = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(frame.id-iprev))
        img_prev_ = getattr(frame_prev,input_img[0])

    img_ = np.array(np.where(plumeMask>=1,1,0),dtype=np.uint8)*255
    tmp_= cv2.morphologyEx(img_, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))

    tmp_= cv2.dilate(tmp_, np.ones((9,9),np.uint8) , iterations = 1)



    #ax = plt.subplot(121)
    #ax.imshow(diff.T,origin='lower')
    #ax = plt.subplot(122)
    #ax.imshow(np.ma.masked_where(tmp_==255,img).T,origin='lower')
    #plt.show()

    pdb.set_trace()


    while( (iprev<20) & ((frame.id-iprev)>=0) ):
   
        # calculate optical flow
        img_prev = hist_matching.hist_matching(img,img_prev_) 
        p1,  st, err = cv2.calcOpticalFlowPyrLK(img, img_prev  , p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img_prev  , img, p1, None, **lk_params)

       
        d = np.sqrt( ((p0[:,0,:]-p0r[:,0,:])**2).sum(-1) )
        good = np.where( (d < 1)                              &\
                         (p1[:,0,0]>=old_div(frame.bufferZone,2)) & (p1[:,0,0]<=ny-old_div(frame.bufferZone,2)) &\
                         (p1[:,0,1]>=old_div(frame.bufferZone,2)) & (p1[:,0,1]<=nx-old_div(frame.bufferZone,2)), \
                         np.ones(d.shape,dtype=bool), np.zeros(d.shape,dtype=bool) )
        
        nbrept_badLoc = len(np.where( good == False)[0])

        p1_good = p1[np.where(good)[0],:,:] 
        p0_good = p0[np.where(good)[0],:,:] 

        plt.imshow(img.T,origin='lower')
        plt.scatter(p0_good[:,0,1],p0_good[:,0,0],c=d[np.where(good)])
        plt.show()
        pdb.set_trace()
       
        img = img_prev.copy()
        p0 = p1_good.reshape(-1,1,2)
    
        iprev += 1
        frame_prev = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(frame.id-iprev))
        img_prev_ = getattr(frame_prev,input_img[0])

        pdb.set_trace()


    if p0_good.size == 0 :
        return None, None, None, None 
    

    
    if p0_good.size == 0 :
        return None, None, None, None 
    else: 
        return p0_good[:,0,:], p1_good_on_ref00[:,0,:], nbrept_badLoc, nbrept_badTemp2
'''


###############################################################
def mask_EP08(frame, frame_ref, kernel_warp=11, kernel_plot=11, lowT=1.e6, kernel_lowT=0): 


    mask_img_    = np.where( (frame.mask_img==1),      255*np.ones_like(frame_ref.warp), np.zeros_like(frame_ref.warp) )
    mask_fix = np.where( (frame_ref.mask_warp==1), 255*np.ones_like(frame.warp), np.zeros_like(frame.warp) )
    
    if  kernel_warp != 0: 
        kernel = np.ones((kernel_warp,kernel_warp),np.uint8)
        img_  = np.array(mask_img_,dtype=np.uint8)
        mask_img_ = cv2.erode(img_, kernel, iterations = 1)
        
        img_fix  = np.array(mask_fix,dtype=np.uint8)
        mask_fix = cv2.erode(img_fix, kernel, iterations = 1)

    mask_ = cv2.warpPerspective(mask_img_, frame.H2Ref, frame.warp.shape[::-1], flags=cv2.INTER_NEAREST)



    if frame.type == 'lwir':

        mask_plot_grid = cv2.warpPerspective(frame_ref.plotMask_withBuffer, frame_ref.H2Grid.dot(np.linalg.inv(frame_ref.H2Ref)), frame_ref.grid_shape[::-1], flags=cv2.INTER_NEAREST)
        mask_plot = np.where( (mask_plot_grid ==2), 255*np.ones(mask_plot_grid.shape,dtype=np.uint8), np.zeros(mask_plot_grid.shape,dtype=np.uint8) )
        if  kernel_plot != 0: 
            kernel = np.ones((kernel_plot,kernel_plot),np.uint8)
            img_      = np.array(mask_plot,dtype=np.uint8)
            mask_plot = cv2.erode(img_, kernel, iterations = 1)
        mask_plot_warp = cv2.warpPerspective(mask_plot,      frame_ref.H2Grid.dot(np.linalg.inv(frame_ref.H2Ref)), frame_ref.warp.shape[::-1], flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP)
        idx_1 = np.where(mask_plot_warp==255)
        idx_2 = np.where(mask_plot_warp==0)
        mask_plot_warp[idx_1] += 1
        mask_plot_warp[idx_2] -= 1

        
        mask_lowT_     = np.where(  (frame.temp  < lowT),         255*np.ones_like(frame.warp), np.zeros_like(frame.warp) )
        frame_ref_temp_warp = cv2.warpPerspective(frame_ref.temp, frame_ref.H2Ref, frame_ref.warp.shape[::-1], flags=cv2.INTER_LINEAR)
        mask_lowT_ref_ = np.where(  (frame_ref_temp_warp  < lowT), 255*np.ones_like(frame.warp), np.zeros_like(frame.warp) )
        if  kernel_lowT != 0: 
            kernel = np.ones((kernel_lowT,kernel_lowT),np.uint8)
            img_ = np.array(mask_lowT_,dtype=np.uint8)
            mask_lowT_ = cv2.erode(img_, kernel, iterations = 1)
            
            img_ = np.array(mask_lowT_ref_,dtype=np.uint8)
            mask_lowT_ref_ = cv2.erode(img_, kernel, iterations = 1)
        mask_lowT_ = cv2.warpPerspective(mask_lowT_, frame.H2Ref, frame.warp.shape[::-1], flags=cv2.INTER_NEAREST)
    else: 
        mask_lowT_     = 255*np.ones_like(mask_)
        mask_lowT_ref_ = 255*np.ones_like(mask_)
        mask_plot_warp = 255*np.ones_like(mask_)



    if frame.type == 'visible':
        mask_plume_ = 255*np.array( plume_mask(frame, force=False) )
    else:
        mask_plume_ = np.zeros_like(mask_)


    return np.array( np.where( (mask_==255)    & (mask_lowT_==255)                                                                                            , 1, 0 ), dtype=np.uint8) ,\
           np.array( np.where( (mask_fix==255) & (mask_lowT_ref_ ==255) & (mask_plot_warp==255) & (frame_ref.bareGroundMask_withBuffer ==1)&(mask_plume_!=255), 1, 0 ), dtype=np.uint8)


#########################################################
def mask_lowT(frame, frame_ref, kernel_warp=11, kernel_plot=None, lowT=100, kernel_lowT=31, flag_output_onWarp=True): 

    '''
    output 1: img mask + kernel warped on
    output 2: warp mask of fix ref + kernel
    '''

    mask_img_ = np.where( (frame.mask_img==1),      255*np.ones_like(frame.img)    , np.zeros_like(frame.img)     )
    mask_fix  = np.where( (frame_ref.mask_warp==1), 255*np.ones_like(frame_ref.warp), np.zeros_like(frame_ref.warp) )
    
    if  kernel_warp != 0: 
        kernel = np.ones((kernel_warp,kernel_warp),np.uint8)
        img_ = np.array(mask_img_,dtype=np.uint8)
        mask_img_ = cv2.erode(img_, kernel, iterations = 1)
        
        img_fix = np.array(mask_fix,dtype=np.uint8)
        mask_fix = cv2.erode(img_fix, kernel, iterations = 1)

    if flag_output_onWarp:
        mask_ = cv2.warpPerspective(mask_img_, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)
    else: 
        mask_ = mask_img_

    frame_ref_temp_warp = cv2.warpPerspective(frame_ref.temp, frame_ref.H2Ref, frame_ref.warp.shape[::-1], flags=cv2.INTER_LINEAR)

    mask_lowT_ = np.where(  (frame.temp  < lowT),         255*np.ones_like(frame.img), np.zeros_like(frame.img) )
    mask_lowT_ref_ = np.where(  (frame_ref_temp_warp  < lowT), 255*np.ones_like(frame_ref.warp), np.zeros_like(frame_ref.warp) )
    if  kernel_lowT != 0: 
        kernel = np.ones((kernel_lowT,kernel_lowT),np.uint8)
        img_ = np.array(mask_lowT_,dtype=np.uint8)
        mask_lowT_ = cv2.erode(img_, kernel, iterations = 1)
  
        kernel = np.ones((kernel_lowT,kernel_lowT),np.uint8)
        img_ = np.array(mask_lowT_ref_,dtype=np.uint8)
        mask_lowT_ref_ = cv2.erode(img_, kernel, iterations = 1)

    if flag_output_onWarp:
        mask_lowT_ = cv2.warpPerspective(mask_lowT_, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)

    return np.array( np.where( (mask_==255)    & (mask_lowT_ ==255)                                           , 1, 0 ), dtype=np.uint8),\
           np.array( np.where( (mask_fix==255) & (mask_lowT_ref_ ==255) & (frame.bareGroundMask_withBuffer==1), 1, 0 ), dtype=np.uint8)


#########################################################
def mask_onlyImageMask(frame, frame_ref, kernel_warp=11, kernel_plot=None, lowT=None, kernel_lowT=None): 

    '''
    if flag == 'warp':
        mask_img_    = np.where( (frame.mask_warp==1),     255*np.ones_like(frame.warp), np.zeros_like(frame.warp) )
        mask_fix = np.where( (frame_ref.mask_warp==1), 255*np.ones_like(frame.warp), np.zeros_like(frame.warp) )
        if  kernel_warp != 0: 
            kernel = np.ones((kernel_warp,kernel_warp),np.uint8)
            img_ = np.array(mask_img_,dtype=np.uint8)
            mask_img_ = cv2.erode(img_, kernel, iterations = 1)
            
            img_fix  = np.array(mask_fix,dtype=np.uint8)
            mask_fix = cv2.erode(img_fix, kernel, iterations = 1)

        mask_ = cv2.warpPerspective(mask_img_, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)

        return np.array( np.where( (mask_==255), 1, 0 ), dtype=np.uint8),\
               np.array( np.where( (mask_fix==255) & (frame_ref.bareGroundMask_withBuffer==1), 1, 0 ), dtype=np.uint8)
    '''

    mask_img_ = np.where( (frame.mask_img==1), 255*np.ones_like(frame.warp), np.zeros_like(frame.warp) )
    mask_fix = np.where( (frame_ref.mask_warp==1), 255*np.ones_like(frame.warp), np.zeros_like(frame.warp) )
    
    if  kernel_warp != 0: 
        kernel = np.ones((kernel_warp,kernel_warp),np.uint8)
        img_ = np.array(mask_img_,dtype=np.uint8)
        mask_img_ = cv2.erode(img_, kernel, iterations = 1)
        
        img_fix = np.array(mask_fix,dtype=np.uint8)
        mask_fix = cv2.erode(img_fix, kernel, iterations = 1)
    
    mask_ = cv2.warpPerspective(mask_img_, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)
    
    
    if frame.type == 'visible':
        mask_plume_ = 255*np.array( plume_mask(frame, force=False) )
    else:
        mask_plume_ = np.zeros_like(mask_)
    
    
    return np.array( np.where( (mask_==255), 1, 0 ), dtype=np.uint8),\
           np.array( np.where( (mask_fix==255) & (frame_ref.bareGroundMask_withBuffer==1) &(mask_plume_!=255), 1, 0 ), dtype=np.uint8)


#############################################
def star_get_costFunction(param):
    if param[0] == 'ssim':
        return get_ssim(*param[1:])
    elif  param[0] == 'EP08':
        return get_EP08(*param[1:])
    else: 
        print('bad flag: ', param[0])

#---------------------------------------------
def get_EP08(frame, frame_ref, inputMask_function=mask_EP08, inputMask_param=[1.e6,0]):
 
    if frame_ref is None: return None

    inputMask, inputMask_fix = inputMask_function( frame, frame_ref, kernel_warp=frame.kernel_warp, kernel_plot=frame.kernel_plot, 
                                                   lowT       =inputMask_param[0],
                                                   kernel_lowT=inputMask_param[1] )


    if frame.type == 'lwir':
        img     = cv2.warpPerspective(frame.templn, frame.H2Ref,                                \
                                      (frame.nj+frame.bufferZone,frame.ni+frame.bufferZone),\
                                      flags=cv2.INTER_LINEAR                      )
        img_ref = cv2.warpPerspective(frame_ref.templn, frame_ref.H2Ref,                                          \
                                      (frame_ref.nj+frame_ref.bufferZone,frame_ref.ni+frame_ref.bufferZone),\
                                      flags=cv2.INTER_LINEAR                                          ) 
    else: 
        img     = frame.return_warp(frame_ref.trange)
        img_ref = frame_ref.warp

    #plt.imshow(np.ma.masked_where(inputMask==0,img).T,origin='lower')
    #plt.imshow(np.ma.masked_where(inputMask==0,img_ref).T,origin='lower',alpha=.7,cmap=mpl.cm.Greys_r)
    #plt.show()
    
    return get_EP08_from_img(img, img_ref, inputMask=np.where(frame.plotMask_withBuffer_ring!=2,0,inputMask), inputMask_ref=inputMask_fix )


#---------------------------------------------
def get_EP08_from_img(img, img_ref, inputMask=None, inputMask_ref=None):
    
    if inputMask     is None: inputMask     = np.zeros_like(img)
    if inputMask_ref is None: inputMask_ref = np.zeros_like(img)

    idx_mask_ = np.where( (inputMask==1) & (inputMask_ref==1) )
    
    if idx_mask_[0].shape[0] == 0 :
        return -999

    if 1.*idx_mask_[0].shape[0] < .3*min([np.where(inputMask    ==1)[0].shape[0],
                                       np.where(inputMask_ref==1)[0].shape[0]]): 
        #print 'get_EP08_from_img: no mask intersection'
        return -999.
    
    img_mean     = img[idx_mask_].mean()
    img_ref_mean = img_ref[idx_mask_].mean()

    iw = np.array(     img[idx_mask_].flatten() - img_mean,  dtype=np.float64)
    ir = np.array( img_ref[idx_mask_].flatten() - img_ref_mean,  dtype=np.float64)

    #print '**',  img_mean,img_ref_mean, np.linalg.norm(iw), np.linalg.norm(ir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ep08 =  old_div(np.dot(ir,iw),(np.linalg.norm(ir)*np.linalg.norm(iw)))
    ep08 = ep08 if (not(np.isnan(ep08))) else -999.
   
    return ep08

#---------------------------------------------
def get_maskCommunAera(img, img_ref, inputMask=None, inputMask_ref=None):
    
    if inputMask     is None: inputMask     = np.zeros_like(img)
    if inputMask_ref is None: inputMask_ref = np.zeros_like(img)

    idx_mask_ = np.where( (inputMask==1) & (inputMask_ref==1) )

    return old_div(1.*idx_mask_[0].shape[0], min([np.where(inputMask    ==1)[0].shape[0],
                                        np.where(inputMask_ref==1)[0].shape[0]])) 


#---------------------------------------------
def get_ssim(frame, frame_ref, win_size_ssim=21 ):
   
    ni_, nj_ = frame.img.shape
    flag_shrink_img = False

    
    if frame_ref.img.size > 5.e5:
        flag_shrink_img = True
        shrink_factor=2
        ni_, nj_ = old_div(ni_,shrink_factor), old_div(nj_,shrink_factor) 
        
        plotMask_      = downgrade_resolution_4nadir(frame.plotMask_withBuffer, [ni_,nj_] , flag_interpolation='min' )
        if frame.type == 'visible':
            plumeMask_     = downgrade_resolution_4nadir(plume_mask(frame),       [ni_,nj_] , flag_interpolation='max' )
        else:
            plumeMask_     = np.zeros_like(plotMask_)
        warp_          = downgrade_resolution_4nadir(frame.warp,          [ni_,nj_] , flag_interpolation='min' )
        
        if frame.type == 'lwir':
            warp_ref_  = downgrade_resolution_4nadir(frame_ref.return_warp(frame.trange),      [ni_,nj_] , flag_interpolation='min' )
        elif frame.type == 'visible':
            warp_ref_  = downgrade_resolution_4nadir(frame_ref.warp,      [ni_,nj_] , flag_interpolation='min' )
        
        ssim_mask_, ssim_mask_fix = mask_EP08(frame, frame_ref, kernel_plot=0, kernel_warp=0., lowT=1.e6, kernel_lowT=0) # kernel is instead applied in idx_ssim_ok
        ssim_mask_    = downgrade_resolution_4nadir(ssim_mask_   ,     [ni_,nj_] , flag_interpolation='min' ) 
        ssim_mask_fix = downgrade_resolution_4nadir(ssim_mask_fix,     [ni_,nj_] , flag_interpolation='min' ) 


    else:
        plotMask_      = frame.plotMask_withBuffer
        if frame.type == 'visible':
            plumeMask_     = plume_mask(frame)
        else:
            plumeMask_     = np.zeros_like(plotMask_)
        warp_          = frame.warp
        
        if frame.type == 'lwir':
            warp_ref_  = frame_ref.return_warp(frame.trange)
        elif frame.type == 'visible':
            warp_ref_  = frame_ref.warp

        ssim_mask_, ssim_mask_fix = mask_EP08(frame, frame_ref, kernel_plot=0, kernel_warp=0., lowT=1.e6, kernel_lowT=0) # kernel is instead applied in idx_ssim_ok


    #enlarge plot mask to stop effect from hot grounda around the plot that make mean ssim goes down when further away ssim is ok
    #kernel = np.ones((15,15),np.uint8)
    #img_ = np.array(np.where(plotMask_==2,1,0),dtype=np.uint8)*255
    #plotMask_ = cv2.dilate(img_, kernel, iterations = 1)
    #plotMask_ = np.where(plotMask_==255,2,0) 
    plotMask_ = np.zeros_like(plotMask_)
    
    
    idx_ssim = np.where( (ssim_mask_==1) & (ssim_mask_fix==1) )

    img_ssim_ref = np.zeros([ni_,nj_]); img_ssim_ref[idx_ssim] = warp_ref_[idx_ssim] 
    img_ssim     = np.zeros([ni_,nj_]); img_ssim[idx_ssim] = warp_[idx_ssim]    
    
    _, ssim_2d = measure.compare_ssim(img_ssim_ref, img_ssim, win_size=win_size_ssim, full=True)

    mask_ssim = np.zeros_like(ssim_2d); mask_ssim[idx_ssim] = 1
    idx = idx_ssim_ok(mask_ssim, win_size_ssim, plotMask_, plumeMask_, ssim_2d)
    mask_ssim = np.zeros_like(ssim_2d); mask_ssim[idx] = 1

    if np.where(idx)[0].shape[0] > 1: 
        mean_ssim = ssim_2d[idx].mean()
    else: 
        mean_ssim = 0.

    if flag_shrink_img: 
        ssim_2d   = scipy.ndimage.zoom(ssim_2d,   shrink_factor, order=0)
        mask_ssim = scipy.ndimage.zoom(mask_ssim, shrink_factor, order=0)

    return mask_ssim, ssim_2d, mean_ssim



#####################################################
def plume_mask(frame, force=False):

    if ('plumeMask' in frame.__dict__ ) & (not(force)) : return frame.plumeMask

    '''
    good_new_warped = cv2.perspectiveTransform(frame.good_new.reshape([-1,1,2]),frame.H2Ref)

    m1, m2 = good_new_warped[:,0,1], good_new_warped[:,0,0]
    xmin = m1.min()
    xmax = m1.max()+0.0001
    ymin = m2.min()
    ymax = m2.max()+0.0001

    X, Y = np.mgrid[0:frame.img.shape[0]:frame.img.shape[0]/4j, 0:frame.img.shape[1]:frame.img.shape[1]/4j]
    dX, dY = X[1,1]-X[0,0], Y[1,1]-Y[0,0] 
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = scipy.stats.gaussian_kde(values)

    Z = np.reshape(kernel(positions).T, X.shape)
    ZZ = scipy.ndimage.zoom(Z,   4, order=0)
    mask = np.where( (frame.mask_warp==1) & (ZZ>=.5*ZZ.max()) & (frame.plotMask_withBuffer!=2 ), np.ones_like(ZZ), np.zeros_like(ZZ) )

    ImgRef  = filters.gaussian( np.where( frame.mask_warp==1, exposure.equalize_hist(frame_ref00.backgrdimg,mask=mask), np.zeros_like(frame.warp)), sigma=15.0)
    ImgWarp = filters.gaussian( np.where( frame.mask_warp==1, exposure.equalize_hist(frame.warp,            mask=mask), np.zeros_like(frame.warp)), sigma=15.0)
    mask = np.where( (np.abs( 1.*ImgRef - 1.*ImgWarp) > .2) & (frame.plotMask_withBuffer!=2), np.ones_like(ImgRef), np.zeros_like(ImgRef) )

    return mask
    '''
    
    #####################

    '''
    if 'good_new' in frame.__dict__: 
        good_new_warped = cv2.perspectiveTransform(frame.good_new.reshape([-1,1,2]),frame.H2Ref)

        m1, m2 = good_new_warped[:,0,1], good_new_warped[:,0,0]
        xmin = m1.min()
        xmax = m1.max()+0.0001
        ymin = m2.min()
        ymax = m2.max()+0.0001

        X, Y = np.mgrid[0:frame.img.shape[0]:frame.img.shape[0]/2j, 0:frame.img.shape[1]:frame.img.shape[1]/2j]
        dX, dY = X[1,1]-X[0,0], Y[1,1]-Y[0,0] 
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([m1, m2])
        kernel = scipy.stats.gaussian_kde(values)

        Z = np.reshape(kernel(positions).T, X.shape)
        ZZ = scipy.ndimage.zoom(Z,   2, order=0) # measure of where we have control points
        
        limit_ssim_2d = .1

    else: 
        ZZ = np.zeros_like(frame.mask_warp)
        limit_ssim_2d = 0.05

    mask = np.where( (frame.mask_warp==1) & (frame.mask_backgrdimg==1) & (ZZ>=.2*ZZ.max()) & (frame.plotMask_withBuffer!=2 ), np.ones_like(ZZ), np.zeros_like(ZZ) )

    if mask.max() > 0: 
        kernel_ = 3
        ImgRef  = filters.gaussian( np.where( frame.mask_warp==1, exposure.equalize_hist(frame.backgrdimg,mask=mask), np.zeros_like(frame.warp)), sigma=kernel_)
        ImgWarp = filters.gaussian( np.where( frame.mask_warp==1, exposure.equalize_hist(frame.warp,            mask=mask), np.zeros_like(frame.warp)), sigma=kernel_)
    else: 
        ImgRef  = frame.backgrdimg
        ImgWarp = frame.warp

    mask_ = np.where( (frame.mask_warp==1) & (frame.mask_backgrdimg==1), 1, 0)
    kernel = np.ones( (frame.kernel_warp, frame.kernel_warp),np.uint8)
    img_  = np.array(mask_,dtype=np.uint8)
    mask_ = cv2.erode(img_, kernel, iterations = 1)
    
    ni_, nj_ = frame.img.shape
    flag_shrink_img = False
    shrink_factor = 1
    if frame.img.size > 5.e5:
        flag_shrink_img = True
        listi = factor.get_factor(ni_)
        listj = factor.get_factor(nj_)
        intersec = np.array(list(set(listi).intersection(listj)))
        shrink_factor= intersec[np.abs(intersec-10).argmin()]
        ni_, nj_ = ni_/shrink_factor, nj_/shrink_factor 
        ImgRef      = downgrade_resolution_4nadir(ImgRef,  np.zeros([ni_,nj_]) , flag_interpolation='min' )
        ImgWarp     = downgrade_resolution_4nadir(ImgWarp, np.zeros([ni_,nj_]) , flag_interpolation='min' )
        mask_       = downgrade_resolution_4nadir(mask_,   np.zeros([ni_,nj_]) , flag_interpolation='min' )

    ImgRef[np.where(mask_==0)]  = 0
    ImgWarp[np.where(mask_==0)] = 0

    _, ssim_2d = measure.compare_ssim(ImgRef, ImgWarp, win_size=11, full=True)

    if flag_shrink_img: 
        ssim_2d   = scipy.ndimage.zoom(ssim_2d,   shrink_factor, order=0)
        mask_     = scipy.ndimage.zoom(mask_  ,   shrink_factor, order=0)
    
    #plumeMask = np.where( (mask_==1) & (frame.plotMask_withBuffer!=2) & (ssim_2d<=.7), np.ones_like(ZZ), np.zeros_like(ZZ))
    plumeMask = np.where( (mask_==1) & (ssim_2d<=limit_ssim_2d), np.ones_like(ZZ), np.zeros_like(ZZ))
    '''

    plumeMask =  np.where(frame.plotMask_withBuffer==2, 1, 0)

    img_ = np.array(plumeMask*255,dtype=np.uint8)
    plumeMask = cv2.erode(img_,np.ones((3,3),dtype=np.uint8),iterations = 1)
    plumeMask =  np.where( plumeMask==255, 1, 0)

    if 'plumeMask' not in frame.__dict__: frame.set_plumeMask(plumeMask)
    return plumeMask

    #####################

    if frame.type == 'lwir': 
        import optris as camera
        params_camera = 'params_lwir_camera'
    elif frame.type == 'visible': 
        import visible as camera
        params_camera = 'params_vis_camera'

    input_img=['warp','feature','trange']
    dir_out_frame = frame.inputConfig.params_rawData['root_postproc'] + frame.inputConfig.__dict__[params_camera]['dir_input'] + 'Frames/'

    frame_ = camera.load_existing_file(frame.inputConfig.__dict__[params_camera], dir_out_frame+'frame{:06d}.nc'.format(frame.id-1))
    img      = getattr(frame_,     input_img[0])
    img_bckgrd = frame.backgrdimg 
    nx,ny = frame_.img.shape

    iprev = 1
   
    plumeMask = np.zeros_like(frame.mask_img)
    while( (iprev<20) & ((frame_.id-iprev)>=0) ):

        frame_prev = camera.load_existing_file(frame.inputConfig.__dict__[params_camera], dir_out_frame+'frame{:06d}.nc'.format(frame_.id-iprev))
        img_prev_ = getattr(frame_prev,input_img[0])
        img_prev = hist_matching.hist_matching(img,img_prev_) 
       
        diff        = np.where( (frame_.mask_warp==1)&(frame_prev.mask_warp==1), np.abs(np.array(img,dtype=np.float)-np.array(img_prev,dtype=np.float)),0)
        diff_bckgrd = np.where( (frame_.mask_warp==1)&(frame_.mask_backgrdimg==1), np.abs(np.array(img,dtype=np.float)-np.array(img_bckgrd,dtype=np.float)), 0)
  
        #diff_ln = local_normalization( np.array(255*diff/diff.max(),dtype=np.uint8), 
        #                               np.where( (frame_.mask_warp==1)&(frame_prev.mask_warp==1), np.ones_like(frame_.mask_img), np.zeros_like(frame.mask_img)), diskSize=40)

        #img_ = np.array(np.where(diff>=20,1,0),dtype=np.uint8)*255
        #diff_close= cv2.morphologyEx(img_, cv2.MORPH_CLOSE,  np.ones((5,5),np.uint8))

        idx = np.where((diff_bckgrd>75)&(np.abs(diff)>20))
        plumeMask[idx] += 1

        iprev += 1

    img_ = np.array(np.where(plumeMask>=1,1,0),dtype=np.uint8)*255
    tmp_= cv2.morphologyEx(img_, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))

    tmp_= cv2.dilate(tmp_, np.ones((9,9),np.uint8) , iterations = 1)
    plumeMask = np.where(tmp_==255,np.ones_like(frame.mask_img),np.zeros_like(frame.mask_img))

    '''
    ax = plt.subplot(141)
    ax.imshow(diff.T,origin='lower')
    ax = plt.subplot(142)
    ax.imshow(diff_bckgrd.T,origin='lower')
    ax = plt.subplot(143)
    ax.imshow(img_.T,origin='lower')
    ax = plt.subplot(144)
    ax.imshow(plumeMask.T,origin='lower')
    plt.show()
    pdb.set_trace()
    ''' 
    #if plumeMask is not in the frame we set it, so that next call can load it 
    if 'plumeMask' not in frame.__dict__: frame.set_plumeMask(plumeMask)

    #plt.imshow(np.ma.masked_where(frame.mask_warp==0,ssim_2d).T,origin='lower'); plt.show()
    #plt.imshow(np.ma.masked_where(plumeMask==1,frame.warp).T,origin='lower'); plt.show()
    #pdb.set_trace()

    return plumeMask


#########################################
def local_normalization(img, mask, diskSize=30):
    
    if img.dtype == float: 
        tmp_ = (img/img.max() * 2) - 1
    else:
        tmp_ = img
    
    tmp_ = skimage.util.img_as_ubyte(tmp_)

    selem = skimage.morphology.disk(diskSize)
    img_eq = skimage.filters.rank.equalize(tmp_, footprint=selem, mask=mask)

    return np.float32(old_div(img_eq,255) )


#####################################################
def idx_ssim_ok(mask_data_combined, winsize, plotMask, plumeMask, ssim_2d):

    def test_func(values):
        if 0 in values:
            return 0
        else:
            return 1 

    def padwithzeros(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector

    footprint = np.ones([winsize,winsize])

    mask_data_combined_plume = np.where( (mask_data_combined== 1) & (np.logical_not( (plumeMask==1) & (ssim_2d<0.3) )), \
                                         np.ones_like(mask_data_combined),                                              \
                                         np.zeros_like(mask_data_combined)                                              )
    x_padded = np.lib.pad(mask_data_combined_plume, winsize, padwithzeros)

    mask_without_winsize =  np.array( scipy.ndimage.generic_filter(x_padded, test_func, footprint=footprint)[winsize:-winsize,winsize:-winsize], dtype=bool )

    #if np.where((mask_without_winsize) & (plotMask!=2))[0].shape[0] == 0:
    #    plt.clf(); plt.imshow(mask_without_winsize.T,origin='lower'); plt.show(); pdb.set_trace()

    return np.where( (mask_without_winsize) & (plotMask!=2),                                      \
                     np.ones(plotMask.shape,dtype=np.bool), np.zeros(plotMask.shape,dtype=np.bool)) 


#####################################################
def runningAverageNeighborPixels_withMask(data,mask,winsize):

    '''
    slow
    '''

    def func(values,data,mask):
        idx = np.unravel_index(np.array(values[np.where(values>=0)],dtype=int),data.shape)
        if len(idx[0])==0: return -999
        data_ = data[idx]
        mask_ = mask[idx]
        return data_[np.where( (data_>=0) & (mask_==1) )].mean()

    def padwithNegativeNumbre(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = -999
        vector[-pad_width[1]:] = -999
        return vector

    footprint = np.ones([winsize,winsize])

    x_padded = np.lib.pad(np.arange(data.size,dtype=np.float).reshape(data.shape), winsize, padwithNegativeNumbre)

    return scipy.ndimage.generic_filter(x_padded, func, footprint=footprint, extra_arguments=(data,mask) )[winsize:-winsize,winsize:-winsize]



######################################################
def dump_geotiff(out_dir_tiff,grid_e,grid_n,utm,field,filename,nodata_value=-999):

 
    ensure_dir(out_dir_tiff)
    #  Initialize the Image Size
    image_size = field.shape[:2]

    dx = grid_e[1,1]-grid_e[0,0]
    dy = grid_n[1,1]-grid_n[0,0]

    # set geotransform
    nx = image_size[0]
    ny = image_size[1]
    xmin, ymin, xmax, ymax = [grid_e.min(), grid_n.min(),  grid_e.max()+dx,  grid_n.max()+dy]
    xres = dx
    yres = dy
    geotransform = (xmin, xres, 0, ymax, 0, -yres)

    nband = 1
    if len(field.shape) > 2: 
        nband = field.shape[2]

    # create the 1-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(out_dir_tiff+filename+'.tif', nx, ny, nband, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    dst_ds.SetProjection(utm.ExportToWkt()) # export coords to file
    for iband in range(nband): 
        band = dst_ds.GetRasterBand(iband+1)
        if nband == 1: 
            band.WriteArray(np.array(field[:,:],dtype=np.float32).T[::-1])   # write r-band to the raster
        else:
            band.WriteArray(np.array(field[:,:,iband],dtype=np.float32).T[::-1])   # write r-band to the raster
        band.SetNoDataValue(nodata_value)
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None

    return 0


###############################################
def dump_png(out_dir_png,camera_name, plot_name, grid_e, grid_n, utm, layer_name, data, time, conv_utm2ll):
    
    mpl.rcdefaults()
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'Comic Sans MS'
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelsize'] = 28.
    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['font.size'] = 28.
    mpl.rcParams['xtick.labelsize'] = 28.
    mpl.rcParams['ytick.labelsize'] = 28.
    mpl.rcParams['figure.subplot.left'] = .0
    mpl.rcParams['figure.subplot.right'] = 1.
    mpl.rcParams['figure.subplot.top'] = 1.
    mpl.rcParams['figure.subplot.bottom'] = .0
    mpl.rcParams['figure.subplot.hspace'] = 0.1
    mpl.rcParams['figure.subplot.wspace'] = 0.18
    
    dx = grid_e[1,1]-grid_e[0,0]
    dy = grid_n[1,1]-grid_n[0,0]
    
    nx, ny = grid_e.shape
    lat_0,     lon_0,     tmp         = conv_utm2ll.TransformPoint(grid_e[old_div(nx,2),old_div(ny,2)],grid_n[old_div(nx,2),old_div(ny,2)])
    lat_scale, lon_scale, tmp         = conv_utm2ll.TransformPoint(grid_e[int(old_div(nx,10)),int(old_div(ny,14))],grid_n[int(old_div(nx,10)),int(old_div(ny,14))])

    fig_ = plt.figure(2,figsize=(12,12))
    ax = plt.subplot(111)
    # setup of basemap ('lcc' = lambert conformal conic).
    # use major and minor sphere radii from WGS84 ellipsoid.
    llcrnrlon, llcrnrlat, tmp = conv_utm2ll.TransformPoint(grid_e[0,0]      -old_div(dx,2), grid_n[0,0]     -old_div(dy,2))
    urcrnrlon, urcrnrlat, tmp = conv_utm2ll.TransformPoint(grid_e[-1,-1]    +old_div(dx,2), grid_n[-1,-1]   +old_div(dy,2))
    width_ =   (grid_e[-1,-1]    +old_div(dx,2)) - (grid_e[0,0]      -old_div(dx,2))
    height_ =  (grid_n[-1,-1]    +old_div(dy,2)) - (grid_n[0,0]      -old_div(dy,2))

    m = Basemap(width=width_, height=height_, projection='tmerc', lon_0=lon_0, lat_0=lat_0, resolution='i')

    extent=(llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat)
    # plot image over map with imshow.
    im = m.imshow(data.T,origin='lower',interpolation='nearest',extent=extent,cmap=mpl.cm.Greys_r)

    # draw coastlines and political boundaries.
    #m.drawcoastlines()
    #m.drawcountries()
    #m.drawstates() 

    # Draw a map scale
    #dev = m.scatter(fire_product_roi.lon[idx],fire_product_roi.lat[idx],latlon=True,
    #    marker='+', lw=2., s=100,
    #    facecolor='r', edgecolor='r',
    #    alpha=1., antialiased=True,
    #    label='MOD14 hot spot', zorder=3)

    #m.drawmapscale(
    #    lon_scale, lat_scale,
    #    lon_0,        lat_0 ,
    #    #llcrnrlon+0.19, llcrnrlat+.05 + 0.015,
    #    ##urcrnrlon-0.25 + 0.01, llcrnrlat+.05 + 0.015,
    #    #urcrnrlon-0.25,        llcrnrlat+.05 ,
    #    10.,
    #    barstyle='fancy', labelstyle='simple',
    #    fillcolor1='w', fillcolor2='k',
    #    fontcolor='k',fontsize=24,
    #    zorder=5)
    
    #add compass
    #north_indicator =  np.array(Image.open('../data_static/compass_invertedColor.png'))
    #im = OffsetImage(north_indicator, zoom=.3)
    #ab = AnnotationBbox(im, [.08*m.urcrnrx,.9*m.urcrnry], xycoords='data', frameon=False)
    #ax.add_artist(ab)

    #add time
    #ax.set_title(r'{:04d}-{:02d}-{:02d} - {:02d}:{:02d} - {:s} vz={:3.1f} va={:3.1f}'.\
    #             format(date_modis.year,date_modis.month,date_modis.day,date_modis.hour,date_modis.minute,sat_,out.viewAngle[0],out.azimuthAngle[0]), y=.96)

    #add cluster info
    # these are matplotlib.patch.Patch properties
    #props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    filename = out_dir_png+'geoRef_{:s}_{:s}_{:s}.png'.format(plot_name,camera_name, time.strftime('%Y-%m-%dT%H%M%S%fZ') )
    fig_.savefig(filename,dpi=300)
    plt.close(fig_)

    return filename


######################################################
def dump_kml(out_dir_kml, camera_name, plot_name, grid_e, grid_n, utm, data, datetime, time_igni, frameid=None ):

    out_dir_png = out_dir_kml + 'png/'
    ensure_dir(out_dir_png)

    #create kml file
    ################################# 
    kml = simplekml.Kml()
    #image corner location of the kml
    # tool to convert lon lat to UTM
    wgs84 = osr.SpatialReference( ) # Define a SpatialReference object
    wgs84.ImportFromEPSG( 4326 ) # And set it to WGS84 using the EPSG code
    conv_ll2utm = osr.CoordinateTransformation(wgs84, utm)
    conv_utm2ll = osr.CoordinateTransformation(utm,wgs84) 
   
    dx = grid_e[1,1]-grid_e[0,0]
    dy = grid_n[1,1]-grid_n[0,0]

    pt1 = conv_utm2ll.TransformPoint(grid_e[0,0]-old_div(dx,2),grid_n[0,0]-old_div(dy,2))[:2][::-1]
    pt3 = conv_utm2ll.TransformPoint(grid_e[0,-1]-old_div(dx,2),grid_n[0,-1]+old_div(dy,2))[:2][::-1]
    pt2 = conv_utm2ll.TransformPoint(grid_e[-1,0]+old_div(dx,2),grid_n[-1,0]-old_div(dy,2))[:2][::-1]
    pt4 = conv_utm2ll.TransformPoint(grid_e[-1,-1]+old_div(dx,2),grid_n[-1,-1]+old_div(dy,2))[:2][::-1]

    
    layer_name =  camera_name
    for i_time in  range(len(datetime)):
        time_ = datetime[i_time]
        layer = kml.newgroundoverlay(name=layer_name+time_.strftime('%Y-%m-%dT%H:%M:%S.%fZ') )
        #layer = kml.newgroundoverlay(name=name_layer[i_layer]+r'{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}Z'.format(time_.year,
        #                                                                    time_.month,
        #                                                                    time_.day,
        #                                                                    time_.hour,
        #                                                                    time_.minute,
        #                                                                    time_.second))
        layer.icon.href = dump_png(out_dir_png, camera_name, plot_name, grid_e, grid_n, utm, layer_name, data[:,:,i_time], time_, conv_utm2ll ) 
        layer.gxlatlonquad.coords = [pt1, pt2, \
                                     pt4, pt3]
        layer.timespan.begin   =  time_.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        #layer.timespan.begin   = r'{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:00Z'.format(time_.year,
        #                                                                    time_.month,
        #                                                                    time_.day,
        #                                                                    time_.hour,
        #                                                                    time_.minute)
        if i_time!=len(datetime)-1:
            layer.timespan.end   = datetime[i_time+1].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            #time_ = out.fireDatetime[i_time+1]
            #layer.timespan.end   = r'{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:00Z'.format(time_.year,
            #                                                                    time_.month,
            #                                                                    time_.day,
            #                                                                    time_.hour,
            #                                                                    time_.minute)
        else:
            layer.timespan.end   = datetime[i_time].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            #time_end = out.fireDatetime[i_time] + datetime.timedelta(minutes = 5.)
            #layer.timespan.end   = r'{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:00Z'.format(time_end.year,
            #                                                                    time_end.month,
            #                                                                    time_end.day,
            #                                                                    time_end.hour,
            #                                                                    time_end.minute)
    
    if len(datetime) == 1: 
        kml.savekmz(out_dir_kml + '{:s}_{:s}_{:s}_id{:06d}.kmz'.format(plot_name,layer_name,datetime[0].strftime('%Y-%m-%dT%H%M%S%fZ'), frameid) )
    else:
        kml.savekmz(out_dir_kml + '{:s}_{:s}.kmz'.format(plot_name,layer_name))


###############################################
def load_polygon_from_kml(kml_file,polygon_name):
    
    with open(kml_file, 'rt') as myfile:
        doc=myfile.read().encode('utf-8')
   
    k = kml.KML()
    k.from_string(doc)

    # feature in document 
    features = list( list(k.features())[0].features())

    i_feature = 0
    name_feature= 'mm'
    while features[i_feature].name != polygon_name:
        i_feature =+1 
        if i_feature == len(features): 
            print('feature {:s} not found in kml file {:s}'.format(polygon_name,os.path.basename(kml_file)))
            pdb.set_trace()
    
    if '_geometry' not in features[i_feature].__dict__:
        pts = []
        for ff in features[i_feature].features(): 
            pts.append( [ff.name, np.dstack(ff.geometry.coords.xy)[0][0], ff.geometry.z] )
        return pts

    else:
        polygon = features[i_feature].geometry
        try:
            pts = np.dstack(polygon.exterior.coords)[0][:2,:]
        except:
            pdb.set_trace()
    
        return np.dstack(pts)[0].T


###################################################
def get_plot_axis(field, params_camera, params_georef):
    if params_camera['flag_costFunction']=='ssim':
        if field == 'warp': 
            return plt.subplot2grid((2,2), (0,0)) 
        elif field == 'img': 
            return plt.subplot2grid((2,2), (0,1)) 
        elif field == 'bestrefimg': 
            return plt.subplot2grid((2,2), (1,1)) 
        elif field == 'ssim': 
            return plt.subplot2grid((2,2), (1,0)) 

    if params_camera['flag_costFunction']=='EP08':
        
        if (params_georef['run_opti'] is False) | (params_georef['ssim_win_lwir'] == 0): 
            if field == 'warp': 
                return plt.subplot2grid((1,3), (0,0)) 
            elif field == 'img': 
                return plt.subplot2grid((1,3), (0,1)) 
            elif field == 'bestrefimg': 
                return plt.subplot2grid((1,3), (0,2)) 
        else:
            if field == 'warp': 
                return plt.subplot2grid((1,4), (0,0)) 
            elif field == 'img': 
                return plt.subplot2grid((1,4), (0,1)) 
            elif field == 'bestrefimg': 
                return plt.subplot2grid((1,4), (0,2)) 
            elif field == 'ssim': 
                return plt.subplot2grid((1,4), (0,3)) 




###################################################
def get_stat_info_cluster(gcps_img, frame, frame_ref00, params_georef): 
    
    cf_on_img = []
    cf_T      = []
    cf_hist   = []
    cf_status = []
   
    #if visible exit now with no stat info
    #if 'temp' not in frame.__dict__:
    #    return ['vis','vis','vis','vis'], [ gcp for gcp in gcps_img], [-999]*4, [np.zeros(53)]*4

    acceptable_dist_to_cluster = 10 # px
    acceptable_hist_change = 50 
    
    #mark all cluster in the whole image
    idx = np.where(frame.temp > params_georef['cornerFire_Temp_threshold'])
    mask = np.zeros(frame.temp.shape)
    mask[idx] = 1
    s = [[1,1,1], \
         [1,1,1], \
         [1,1,1]] # for diagonal
    labeled, num_cluster = ndimage.label(mask, structure=s )
    if num_cluster == 0:
        return ['out','out','out','out'], cf_on_img, cf_T, cf_hist

    #get info for each cluster
    cluster_temp   = []
    cluster_nbrepx = []
    cluster_idx    = []
    cluster_hist   = []
    cluster_loc    = []
    for i_cluster in range(num_cluster):
        idx = np.where(labeled==i_cluster+1)
        cluster_temp.append(frame.temp[idx].max())
        cluster_nbrepx.append(idx[0].shape[0])
        cluster_idx.append(idx)
        cluster_loc.append( np.array([np.average(idx[1],weights=frame.temp[idx]), np.average(idx[0],weights=frame.temp[idx])]) )
        cluster_hist.append(np.histogram(frame.temp[idx].flatten(),
                                         bins=np.arange(30,300,5))[0])
    cluster_temp   = np.array(cluster_temp)
    cluster_nbrepx = np.array(cluster_nbrepx)

    #plt.clf()
    #plt.imshow(frame.temp.T,origin='lower')
    #plt.scatter(gcps_img[:,1],gcps_img[:,0])
    #plt.show()

    for icf in range(gcps_img.shape[0]):
        gcp =  np.array(np.round(gcps_img[icf,:],0),dtype=int)
        if (gcp[0] > old_div(frame.bufferZone,2)) & (gcp[0] < frame.img.shape[1]-old_div(frame.bufferZone,2)) & \
           (gcp[1] > old_div(frame.bufferZone,2)) & (gcp[1] < frame.img.shape[0]-old_div(frame.bufferZone,2))   : 
               
                dist_to_cluster = np.array([np.sqrt(((cluster_loc_ - gcp)**2).sum()) for cluster_loc_ in cluster_loc])
                
                if frame_ref00 is not None: 
                    change_hist = []
                    [change_hist.append( np.abs(frame_ref00.cf_hist[icf]-hist_).sum()) for hist_ in cluster_hist]
                
                    idx = np.where( (dist_to_cluster<acceptable_dist_to_cluster) & (np.array(change_hist)<=acceptable_hist_change) )
                    if len(idx[0])==0:
                        cf_status_ = 'no easy cluster, min dist {:.2f}, min change_hist = {:.2f}'.format(dist_to_cluster.min(), 
                                                                                                             np.array(change_hist).min()) 
                else:
                    idx = np.where(dist_to_cluster<acceptable_dist_to_cluster)
                    if len(idx[0])==0:
                        cf_status_ = 'no easy cluster, min dist {:.2f}'.format(dist_to_cluster.min())
                
                if len(idx[0])==0:
                    cf_status.append(cf_status_)
                    continue

                elif len(idx[0])==1: 
                    cf_on_img.append(cluster_loc[idx[0][0]])
                    cf_T.append(cluster_temp[idx[0][0]])
                    cf_hist.append(cluster_hist[idx[0][0]])
                    cf_status.append('ok') 
                
                else :
                    cf_on_img.append( [cluster_loc[idx_]  for idx_ in idx[0]] )
                    cf_T.append(      [cluster_temp[idx_] for idx_ in idx[0]] ) 
                    cf_hist.append(   [cluster_hist[idx_] for idx_ in idx[0]] )
                    cf_status.append('too many candidate') 
                    
        else: 
            cf_status.append('out')
  

    '''
    if (len(cf_on_img)==gcps_img.shape[0]) & (''.join(cf_status) != 'ok'*4) & (frame_ref00 is not None): 
        for icf in range(gcps_img.shape[0]):
            if cf_status[icf] == 'ok': continue
            
            change_hist = []
            [change_hist.append( np.abs(frame_ref00.cf_hist[icf]-hist_).sum()) for hist_ in cf_hist[icf]]
            
            idx = np.array(change_hist).argmin()
            cf_on_img[icf] = cf_on_img[icf][idx]
            cf_T[icf]      = cf_T[icf][idx]
            cf_hist[icf]   = cf_hist[icf][idx]
            cf_status[icf] = 'ok'
    '''
    #for ref00
    if (len(cf_on_img)==gcps_img.shape[0]) & (''.join(cf_status) != 'ok'*4) & (frame_ref00 is None): 
        for icf in range(gcps_img.shape[0]):
            if cf_status[icf] == 'ok': continue
            
            dist_to_cluster = np.array([np.sqrt(((cluster_loc_ - gcp)**2).sum()) for cluster_loc_ in cf_on_img[icf]])
            
            idx = np.array(dist_to_cluster).argmin()
            cf_on_img[icf] = cf_on_img[icf][idx]
            cf_T[icf]      = cf_T[icf][idx]
            cf_hist[icf]   = cf_hist[icf][idx]
            cf_status[icf] = 'ok'

    return cf_status, cf_on_img, cf_T, cf_hist 


#################################################
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    im = np.array(im,dtype=np.float32)
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    
    mag, angle = cv2.cartToPolar(grad_x,grad_y)
    return grad, mag, angle


def local_pause_4_figure(interval, focus_figure=True):
    backend = mpl.rcParams['backend']
    if backend in _interactive_bk:
        figManager = _pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            if focus_figure:
                plt.show(block=False)
            canvas.start_event_loop(interval)
            return

    # No on-screen figure is active, so sleep() is all we need.
    import time
    time.sleep(interval)


################################################
def string_2_bool(string):
    if  string in ['true', 'TRUE' , 'True' , '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
        return  True
    else:
        return False


################################################
def cpu_count():
    try:
        return int(os.environ['ntask'])
    except:
        print('env variable ntask is not defined')
        sys.exit() 
        #return multiprocessing.cpu_count()


################################################
def get_best_plane(xs,ys,zs, flag_plot=False, dir_out=None, dimension=None, maskPlot=None):


    # do fit copied from https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    #fit = ((A.T.dot(A)).I).dot(A.T).dot(b)
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    '''
    print "solution:"
    print "%f x + %f y + %f = z" % (fit[0], fit[1], fit[2])
    print "errors:"
    print errors
    print "residual:"
    print residual
    '''

    # plot raw data
    '''
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(xs[::100], ys[::100], zs[::100], color='b')

    # plot plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                      np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    ax.plot_wireframe(X,Y,Z, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    '''
    if flag_plot: 
        tresh = np.array(np.where(maskPlot==2,255,0),dtype=np.uint8)
        contours, hierarchy = cv2.findContours(tresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        polygon =[ tuple( [pt[0][1],pt[0][0]] ) for pt in contours[0] ]
        img = Image.new('L', maskPlot.shape , 0)
        ImageDraw.Draw(img).polygon(polygon, 0, 1)
        maskPlot_ct = np.copy(img).T

        mpl.rcdefaults()
        mpl.rcParams['text.usetex'] = True
        #mpl.rcParams['font.family'] = 'Comic Sans MS'
        mpl.rcParams['font.size'] = 16.
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['axes.labelsize'] = 14.
        mpl.rcParams['xtick.labelsize'] = 14.
        mpl.rcParams['ytick.labelsize'] = 14.
        mpl.rcParams['figure.subplot.left'] = .1
        mpl.rcParams['figure.subplot.right'] = .9
        mpl.rcParams['figure.subplot.top'] = .9
        mpl.rcParams['figure.subplot.bottom'] = .1
        mpl.rcParams['figure.subplot.hspace'] = 0.02
        mpl.rcParams['figure.subplot.wspace'] = 0.5
        fig = plt.figure(3, figsize=(10,5)) 
   
        ax = plt.subplot(121)
        im = ax.imshow(zs.reshape(dimension).T, origin='lower')
        ax.imshow(np.ma.masked_where(maskPlot_ct==0,maskPlot_ct).T,origin='lower',cmap=mpl.cm.Greys_r)
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im ,cax = cbaxes,orientation='vertical')
        cbar.set_label('terrain (m)')
 
        ax = plt.subplot(122)
        vmin = -1 * np.where(errors<0, -1*errors, errors).max()
        vmax = -1 * vmin
        im = ax.imshow(errors.reshape(dimension).T, origin='lower',cmap=mpl.cm.bwr,
                       vmin=vmin,vmax=vmax)
        ax.imshow(np.ma.masked_where(maskPlot_ct==0,maskPlot_ct).T,origin='lower',cmap=mpl.cm.Greys_r)
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im ,cax = cbaxes,orientation='vertical')
        cbar.set_label('error (m)')
    
        fig.savefig(dir_out+'corrected_terrain_simpleHomography.png')
        plt.close(fig)

    return A * fit



#################################################
def set_frame_using_homography(frame, frame_ref00, frame_ref00_init, framesID_ref, info_loop,
                               camera,
                               params_camera, params_georef, flag_parallel,
                               gcps_cam,
                               win_size_ssim, lk_params,
                               dir_out_frame,
                               nbre_frame_availale_since_last_anchor,add_more_frame):

    last_ref_frame = len(framesID_ref)
    best_result_frame_saved_over_iteration = frame.copy() 
    nbre_of_call_warp_frame = 0
    results_feature =  [None, None]

    if params_camera['warp_on_prev_first']:

        #use previous frame first to warp the current image. if corr to ref00 is acceptable then it is set as the best frame 
        #-------------------------------------
        last_idx_frame_above_corr_ref00_good2 = np.where( info_loop.corr_ref00 >  params_camera['energy_good_2'])[0][-1]
        frame_last = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(info_loop.id[last_idx_frame_above_corr_ref00_good2])) 
        frames_ref_number = 1 
        frame_, _ = warp_frame_on_prev_frame(camera, params_camera, params_georef, frame.copy(), frame_last, 
                                                           frame_ref00, frame_ref00_init,
                                                           flag_parallel, win_size_ssim, lk_params, dir_out_frame)

        if  frame_.corr_ref00 >= best_result_frame_saved_over_iteration.corr_ref00: 
            best_result_frame_saved_over_iteration = frame_.copy()
 
    if best_result_frame_saved_over_iteration.corr_ref00 < params_camera['energy_good_2']: 
        print(''); print('            ', end=' ')
        frames_ref_number = min([nbre_frame_availale_since_last_anchor,add_more_frame])
        # then we try to find a better reference in last selected ref
        # while loop with increasing number of reference frame from where to get matching feature and compute homography matrix            
        #-------------------------------------
        if frame.type == 'lwir':
            nbre_call_limit = 20./params_georef['#frames_history_tail_lwir'] #MERDEMERDE
        elif frame.type == 'visible':
            nbre_call_limit = 100./params_georef['#frames_history_tail_visible'] #MERDEMERDE
        
        while (frames_ref_number<= nbre_frame_availale_since_last_anchor ):#MERDEONE
      
            #frame.set_trange(params_georef['trange']) 
            #frame.set_img()
            frame_, results_feature = warp_frame(camera, params_camera, params_georef, frame.copy(), framesID_ref[-1*frames_ref_number:last_ref_frame], 
                                                 frame_ref00, frame_ref00_init,
                                                 flag_parallel, win_size_ssim, lk_params, dir_out_frame, feature_last_call=results_feature)

            if  frame_.corr_ref00 >= best_result_frame_saved_over_iteration.corr_ref00: 
                best_result_frame_saved_over_iteration = frame_.copy()
            
            if frame_.corr_ref00 > params_camera['energy_good_2']: 
                break
            
            if (nbre_of_call_warp_frame > nbre_call_limit): 
                break
            elif (frames_ref_number+add_more_frame<nbre_frame_availale_since_last_anchor): 
                print('ref frame # up')
                print('            ', end=' ')
                frames_ref_number += add_more_frame
            elif (frames_ref_number < nbre_frame_availale_since_last_anchor):
                print('ref frame # up')
                print('            ', end=' ')
                frames_ref_number = nbre_frame_availale_since_last_anchor
            else:
                break
            
            nbre_of_call_warp_frame += 1
        #end loop

    # Select best frame            
    #-------------------------------------
    if (best_result_frame_saved_over_iteration.corr_ref00 > params_camera['energy_good_2']): 
        frame = best_result_frame_saved_over_iteration.copy()
        frame.save_number_ref_frames_used(len(framesID_ref[-1*frames_ref_number:last_ref_frame]))
    
        
    else: # use optimize_homography to get close the the prev image
        last_idx_frame_above_corr_ref00_good2 = np.where( info_loop.corr_ref00 >  params_camera['energy_good_2'])[0][-1]
        frame_last = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(info_loop.id[last_idx_frame_above_corr_ref00_good2])) 
       
        frame_here = frame.copy()

        if (best_result_frame_saved_over_iteration.corr_ref00 > params_camera['energy_good_2']-0.1): 
            frame_here = best_result_frame_saved_over_iteration.copy()
            frame_here.save_number_ref_frames_used(len(framesID_ref[-1*frames_ref_number:last_ref_frame]))
        else:
            frame_here.set_homography_to_ref( frame_last.H2Ref ) 
            frame_here.set_warp(     cv2.warpPerspective(frame.img,      frame_last.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ) )
            frame_here.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame_last.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST ) )
            frame_here.set_id_best_ref(-2)

        if frame_here.type == 'lwir'   : print('\n          '+ 56*' ' + '|', end=' ') 
        if frame_here.type == 'visible': print('\n          '+ 44*' ' + '|', end=' ')
        sys.stdout.flush()
        
        #frame.set_homography_to_ref( np.eye(3, 3, dtype=np.float32) )
        #frame.set_warp(frame.img)
        #frame.set_maskWarp(frame.mask_img)
        frame_here.optimize_homography( params_georef, params_camera, 
                                   frame_ref00,frame_ref00_init,
                                   win_size_ssim, 
                                   flag='coarse',frame_ref=frame_last)
        print(' {:5.3f} {:04d}'.format(frame_here.corr_ref00, frame_here.id_best_ref), end=' ') 
        sys.stdout.flush()

        if frame_here.corr_ref00 > best_result_frame_saved_over_iteration.corr_ref00 : 
            print(' ecc  ', end=' ') 
            frame = frame_here.copy()
        else: 
            print(' sift ', end=' ') 
            frame = best_result_frame_saved_over_iteration.copy()
            frame.save_number_ref_frames_used(len(framesID_ref[-1*frames_ref_number:last_ref_frame]))

    if frame.corr_ref == 0:
        frame.set_warp(np.zeros_like(frame.img))
        frame.set_maskWarp(np.zeros_like(frame.mask_img))

    #cf tracking method was removed
    #-------------------------------------
    frame.set_flag_cfMode('trackpts')

    #optimize homography using findTransformECC
    #-------------------------------------
    if  (params_georef['run_opti']) & ('H2Ref' in frame.__dict__): 
        
        if (frame.type == 'lwir') & (frame.id_best_ref != -1):
            frame_best_ref = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(frame.id_best_ref))
            frame.optimize_homography( params_georef, params_camera, 
                                       frame_ref00, frame_ref00_init,
                                       win_size_ssim, 
                                       flag='refine',frame_ref=frame_best_ref)
       
        frame.optimize_homography(params_georef, params_camera,                                                       \
                                  frame_ref00, frame_ref00_init,                                                                      \
                                  win_size_ssim)
      
    
        
        #plt.figure() 
        #temp_warp_ = cv2.warpPerspective(frame_ref00.temp, frame_ref00.H2Ref, frame_ref00.img.shape[::-1], flags=cv2.INTER_LINEAR)
        #plt.imshow(np.ma.masked_where(frame_ref00.mask_warp==0,temp_warp_).T,origin='lower', vmax=70)
        
        #plt.figure() 
        #temp_warp_ = cv2.warpPerspective(frame.temp,       frame.H2Ref,       frame.img.shape[::-1],       flags=cv2.INTER_LINEAR)
        #plt.imshow(np.ma.masked_where(frame.mask_warp==0,      temp_warp_).T,origin='lower', vmax=70)
        
        #
        if params_camera['final_opti_threshold'] < 1.:
            id_frame = frame_ref00.id_ref00
            final_correction_on_corr_ref00_init = np.nan
            id_frames = []
            while id_frame >= frame_ref00_init.id: 
                frame_ref = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(id_frame))
                id_frames.append(id_frame) 
                id_frame = frame_ref.id_ref00
              
            nbre_corr_opti_done = 0
            for id_frame in id_frames[::-1]:
                if np.isnan(final_correction_on_corr_ref00_init): final_correction_on_corr_ref00_init=0
                frame_ref = camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(id_frame))
                final_correction_on_corr_ref00_init_, test_val = frame.optimize_homography( params_georef, params_camera, 
                                                                                  frame_ref00, frame_ref00_init,
                                                                                  win_size_ssim,
                                                                                  flag='final',frame_ref=frame_ref)
                if final_correction_on_corr_ref00_init_>0: nbre_corr_opti_done+=1
                final_correction_on_corr_ref00_init += final_correction_on_corr_ref00_init_

            if not np.isnan(final_correction_on_corr_ref00_init):
                print(' opt{:1d} (d={:6.3f}, x{:2d}) '.format( 4, final_correction_on_corr_ref00_init,nbre_corr_opti_done), end=' ')
                
    #plt.imshow(np.ma.masked_where(frame_ref00.mask_warp==0,frame_ref00.warp).T,origin='lower')
    #plt.imshow(np.ma.masked_where(frame.mask_warp==0,frame.warp).T,origin='lower',cmap=mpl.cm.Greys_r, alpha=.5); plt.show()
    #pdb.set_trace()
    
    #plt.figure() 
    #temp_warp_ = cv2.warpPerspective(frame.temp,       frame.H2Ref,       frame.img.shape[::-1],       flags=cv2.INTER_LINEAR)
    #plt.imshow(np.ma.masked_where(frame.mask_warp==0,      temp_warp_).T,origin='lower', vmax=70)
    #plt.show()
    #pdb.set_trace()

    #Check if cf can be found on the image
    #-------------------------------------
    if (frame.type == 'lwir') & (params_georef['look4cf']) & ('H2Ref' in frame.__dict__): 
        
        # anchor to corner fire if they are close to expected postion
        gcps_img = cv2.perspectiveTransform( (gcps_cam+old_div(frame.bufferZone,2)).reshape(-1,1,2), np.linalg.inv(frame.H2Ref) )[:,0,:]
        
        cf_status, cf_on_img, cf_T, cf_hist =  get_stat_info_cluster(gcps_img, frame, frame_ref00, params_georef)
   
        if (len(cf_on_img)==gcps_img.shape[0]) & (''.join(cf_status) != 'ok'*4): 
            for icf in range(gcps_img.shape[0]):
                if cf_status[icf] == 'ok': continue
                
                dist_to_cluster = np.array([np.sqrt(((cluster_loc_ - gcps_img[icf])**2).sum()) for cluster_loc_ in cf_on_img[icf]])
                
                idx = np.array(dist_to_cluster).argmin()
                cf_on_img[icf] = cf_on_img[icf][idx]
                cf_T[icf]      = cf_T[icf][idx]
                cf_hist[icf]   = cf_hist[icf][idx]
                cf_status[icf] = 'ok'
        
        if (''.join(cf_status) == 'ok'*4):
            frame.set_cf_on_img(np.dstack((cf_on_img))[0].T, np.dstack((cf_hist))[0].T)
       
        print('cf{:02d}'.format(len(cf_on_img)), end=' ') 

    
    # in case the frame warping did not work
    #-------------------------------------
    if ('H2Ref' not in frame.__dict__ ) :
        frame.set_correlation(0, 0., 0., 0.)


    return frame


#################################################
def set_frame_gcps(frame, framesID_ref, 
                   camera,
                   params_camera, params_georef, flag_parallel,
                   dir_out_frame, wkdir,
                   win_size_ssim, lk_params):

    
    frames_ref = []
    for id in framesID_ref[::-1]:
        frames_ref.append(camera.load_existing_file(params_camera, dir_out_frame+'frame{:06d}.nc'.format(id)))

    gcp_cam = []
    gcp_world = []
    for frame_ref in frames_ref:
        feature_on_frame, feature_on_ref, nbrept_badLoc_, nbrept_badTemp2_ = get_matching_feature_SIFT(frame,frame_ref, lk_params, flag='use for Wt')

        #get world locatoin of feature_on_ref
        pts_img_xyz_map_ref   = np.load( wkdir+'frame{:06d}_2d23d.npy'.format(frame_ref.id) )   
        idx = np.array(feature_on_ref-old_div(frame.bufferZone,2),dtype=int)
        world_on_ref = pts_img_xyz_map_ref[idx[:,1],idx[:,0],:]

        for (feature, pt)  in zip(feature_on_frame,world_on_ref):
            if np.prod(pt,axis=0)!=0:
                gcp_cam.append( feature ) 
                gcp_world.append( pt )

    gcp_cam = np.array(gcp_cam,dtype=np.float32).reshape(-1,2) - old_div(frame.bufferZone,2)
    gcp_world = np.array(gcp_world)

    #set all feature and let solvePnP do his best
    frame.set_matching_feature(gcp_cam,gcp_world)

    return frame


#################################################
def apply_translation_img(warp, mask, trans_vec, ref, mask_ref, communArea_limit=None):

    mat_trans = np.eye(2, 3, dtype=np.float32)
    mat_trans[:,-1] = trans_vec
    warp_trans  = cv2.warpAffine(warp, mat_trans, warp.shape[::-1], flags=cv2.INTER_LINEAR   )
    mask_trans  = cv2.warpAffine(mask, mat_trans, mask.shape[::-1], flags=cv2.INTER_NEAREST  ) 
   
    if communArea_limit is None: 
        EP08_trans = get_EP08_from_img(warp_trans, ref, inputMask=mask_trans, inputMask_ref=mask_ref )
        return EP08_trans
    
    else:
        mask_both = np.zeros_like(mask)
        mask_both[np.where(mask_trans ==1)] += 1
        mask_both[np.where(mask_ref   ==1)] += 2
        if np.where(mask_trans==1)[0].shape[0] > 0: 
            communArea = old_div(1.*np.where(mask_both==3)[0].shape[0],\
                            min([np.where(mask_trans==1)[0].shape[0],np.where(mask_ref==1)[0].shape[0]]))
        else: 
            communArea = old_div(1.*np.where(mask_both==3)[0].shape[0],\
                                     np.where(mask_ref==1)[0].shape[0]) 
        
        if communArea > communArea_limit: 
            EP08_trans = get_EP08_from_img(warp_trans, ref, inputMask=mask_trans, inputMask_ref=mask_ref )
        else: 
            EP08_trans = 0

        return EP08_trans, communArea


#################################################
def findTransformECC_on_prev_frame(flag, frame, frame_ref, trans_len_limit=[80,40], ep08_limit=[.7,.6], mask_func=mask_EP08, mask_func_param=[1.e6,0]):
        
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-4)
    selected_field = 'templn' if frame.type=='lwir' else 'img'

    frame_ref_temp = cv2.warpPerspective(getattr(frame_ref,selected_field), frame_ref.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )  
    frame_mask_warp, frame_ref_mask = mask_func( frame, frame_ref, kernel_warp=frame.kernel_warp, kernel_plot=frame.kernel_plot, 
                                                    lowT       =mask_func_param[0],
                                                    kernel_lowT=mask_func_param[1] )
    frame_temp_warp = cv2.warpPerspective( getattr(frame,selected_field), frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )

    EP08_0 = get_EP08_from_img(frame_temp_warp, frame_ref_temp, inputMask=frame_mask_warp, inputMask_ref=frame_ref_mask )
    id_ecc = 0

    #TRANSLATION ON LOWER IMAGE RESOLUTION
    if trans_len_limit[0] >= 10: 
        
        temp0  = getattr(frame,selected_field)
        mask0 = frame.mask_img
        frame_ref_temp0 = getattr(frame_ref,selected_field)
        frame_ref_mask0 = frame_ref.mask_img

        res_factor = 2
        ______coarse    = np.zeros([old_div(temp0.shape[0],res_factor),old_div(temp0.shape[1],res_factor)]).shape
        temp0_coarse    = downgrade_resolution_4nadir(temp0,     ______coarse, flag_interpolation='conservative')
        mask0_coarse   = downgrade_resolution_4nadir(mask0, ______coarse, flag_interpolation='max')
        temp_ref_coarse0 = downgrade_resolution_4nadir(frame_ref_temp0, ______coarse, flag_interpolation='conservative')
        mask_ref_coarse0 = downgrade_resolution_4nadir(frame_ref_mask0, ______coarse, flag_interpolation='max')
       
        ep08 = 0.
        trans_len = 10
        info_prev = None
        while ep08 < ep08_limit[0]:
            trans_x = np.arange(-trans_len,trans_len+1)
            trans_y = np.arange(-trans_len,trans_len+1)
            nbre_iter = trans_x.shape[0]*trans_y.shape[0]
            info_ = np.zeros([nbre_iter,4])
            for ivec, vec in enumerate(itertools.product(trans_x, trans_y)) :
                if info_prev is not None:
                    idx_ = np.where( ((vec[0]-info_prev[:,0])**2 + (vec[1]-info_prev[:,1])**2)==0)
                    if len(idx_[0])!=0:
                        info_[ivec,:] = info_prev[idx_,:]
                        continue
                info_[ivec, :2] = vec
                info_[ivec, 2]  = apply_translation_img( temp0_coarse, mask0_coarse, vec, temp_ref_coarse0, mask_ref_coarse0 )

            idx2_ = info_[:,2].argmax()
            
            ep08       = info_[idx2_,2]
            trans_vec  = info_[idx2_,:2]

            trans_len += 10
            info_prev = np.copy(info_)
            #if (trans_len > trans_len_limit[0]) & (flag != 'final'): print '**',
            if (trans_len > trans_len_limit[0]): break
        
        #print trans_len-10,
        warp_matrix1_ = np.eye(2, 3, dtype=np.float32)
        warp_matrix1_[:,-1] = trans_vec * res_factor
        warp_matrix1 = np.eye(3, 3, dtype=np.float32)
        warp_matrix1[:2,:] = warp_matrix1_
       
        frame_ = frame.copy()
        frame_.set_homography_to_ref( frame_ref.H2Ref.dot(warp_matrix1 ))   
        frame_.set_warp(     cv2.warpPerspective(frame.img,      frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ))
        frame_.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST))
        frame_temp =         cv2.warpPerspective(getattr(frame,selected_field),     frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )

        EP08_1 = get_EP08_from_img(frame_temp, frame_ref_temp, inputMask=frame_.mask_warp, inputMask_ref=frame_ref_mask )
        
        if EP08_1 > EP08_0: 
            frame.set_homography_to_ref( frame_.H2Ref )
            frame.set_warp(     cv2.warpPerspective(frame.img,      frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ))
            frame.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST))
            id_ecc = 1
        else:
            EP08_1 = EP08_0
            warp_matrix1 = np.eye(3, 3, dtype=np.float32)

    else: 
        EP08_1 = EP08_0

    #print '1 ',  EP08_1, trans_vec

    #2nd TRANSLATION AT FULL RESOLUTION
    if trans_len_limit[1]>= 10:
        
        temp1 = cv2.warpPerspective(getattr(frame,selected_field), frame.H2Ref,                                
                                    frame.img.shape[::-1],
                                    flags=cv2.INTER_LINEAR )
        mask1 =  cv2.warpPerspective(frame.mask_img, frame.H2Ref,                                
                                    frame.img.shape[::-1],
                                    flags=cv2.INTER_NEAREST )
        frame_ref_temp1 = getattr(frame_ref,selected_field)
        frame_ref_mask1 = frame_ref.mask_img

        trans_len = 10
        ep08 = 0.
        info_prev = None
        while ep08 < ep08_limit[1]:
            trans_x = np.arange(-trans_len,trans_len+1)
            trans_y = np.arange(-trans_len,trans_len+1)
            nbre_iter = trans_x.shape[0]*trans_y.shape[0]
            info_ = np.zeros([nbre_iter,4])
            for ivec, vec in enumerate(itertools.product(trans_x, trans_y)) :
                if info_prev is not None:
                    idx_ = np.where( ((vec[0]-info_prev[:,0])**2 + (vec[1]-info_prev[:,1])**2)==0)
                    if len(idx_[0])!=0:
                        info_[ivec,:] = info_prev[idx_,:]
                        continue
                info_[ivec, :2] = vec
                info_[ivec, 2] = apply_translation_img( temp1, mask1, vec, frame_ref_temp1, frame_ref_mask1 )
            
            #idx_ = np.where(info_[:,2] >= .8* info_[:,2].max())
            #idx2_ = idx_[0][info_[ idx_[0][ np.where( info_[idx_[0],3] == info_[idx_,3].max()) ], 2].argmax()] 
            #if info_[:,2].max() > .8*ep08_limit[0]:
            #    idx_  = np.where(info_[:,2] >= .9* info_[:,2].max()) #first get the vec with corr > 90% of the max
            #    idx2_ = idx_[0][info_[ idx_[0][ np.where( info_[idx_[0],3] == info_[idx_,3].max()) ], 2].argmax()] # then get the vec that have max match and get max corr on those
            #else: 
            idx2_ = info_[:,2].argmax()
            ep08       = info_[idx2_,2]
            trans_vec  = info_[idx2_,:2]

            trans_len += 10
            info_prev = np.copy(info_)

            #if (trans_len > trans_len_limit[1]) & (flag != 'final'): print '**',
            if (trans_len > trans_len_limit[1]): break

        #print trans_len-10,  
        warp_matrix2_ = np.eye(2, 3, dtype=np.float32)
        warp_matrix2_[:,-1] = trans_vec
        warp_matrix2 = np.eye(3, 3, dtype=np.float32)
        warp_matrix2[:2,:] = warp_matrix2_
   
        #warp_matrix2 = np.linalg.inv(frame.H2Ref).dot(warp_matrix2)
        
        frame_ = frame.copy()
        frame_.set_homography_to_ref(frame.H2Ref.dot(warp_matrix2)) 
        frame_.set_warp(     cv2.warpPerspective(frame.img,      frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ))
        frame_.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST))
        frame_temp =         cv2.warpPerspective(getattr(frame,selected_field),     frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )

        EP08_2 = get_EP08_from_img(frame_temp, frame_ref_temp, inputMask=frame_.mask_warp, inputMask_ref=frame_ref_mask )
        
        if EP08_2 > EP08_1: 
            frame.set_homography_to_ref(frame_.H2Ref)
            frame.set_warp(     cv2.warpPerspective(frame.img,      frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ))
            frame.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST))
            id_ecc = 2
        else: 
            EP08_2 = EP08_1

    else:
        EP08_2 = EP08_1
 
    #print  '2 ', EP08_2,trans_vec
    
    '''
    mm = cv2.warpAffine(temp0, warp_matrix1_ ,temp0.shape[::-1], flags=cv2.INTER_LINEAR  )
    plt.imshow(frame_ref_temp.T,origin='lower')
    plt.imshow(mm.T,origin='lower', alpha=.5, cmap=mpl.cm.Greys_r)
    
    plt.figure()
    mm = cv2.warpPerspective(temp0, warp_matrix1.dot(warp_matrix2) ,frame.img.shape[::-1], flags=cv2.INTER_LINEAR  )
    plt.imshow(frame_ref_temp.T,origin='lower')
    plt.imshow(mm.T,origin='lower', alpha=.5, cmap=mpl.cm.Greys_r)
    plt.show()
    ''' 
    frame_ref_temp = cv2.warpPerspective(getattr(frame_ref,selected_field),     frame_ref.H2Ref, frame.img.shape[::-1],flags=cv2.INTER_LINEAR  )  
    frame_mask_warp, frame_ref_mask = mask_func( frame, frame_ref, kernel_warp=frame.kernel_warp, kernel_plot=frame.kernel_plot, 
                                                    lowT       =mask_func_param[0],
                                                    kernel_lowT=mask_func_param[1] )
    
    '''
    warp_matrix2_init = np.array(np.linalg.inv(frame.H2Ref),dtype=np.float32)
    img = cv2.warpPerspective(frame.temp, warp_matrix2_init, frame.img.shape[::-1], flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
    plt.imshow(np.ma.masked_where(frame_ref_mask==0,frame_ref_temp).T,origin='lower')
    plt.imshow(np.ma.masked_where(frame.mask_warp==0,img).T,origin='lower',alpha=.5,cmap=mpl.cm.Greys_r)
    plt.title('pre homography')
    plt.show()
    '''
       
    #frame_temp =         cv2.warpPerspective(frame.temp,     frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )
    #print get_EP08_from_img(frame_temp, frame_ref_temp, inputMask= np.where( (frame.mask_warp==1) & (frame_ref_mask==1) ,1,0) )

    #HOMOGRAPHY
    frame_mask_img = cv2.warpPerspective(frame_mask_warp, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP )
    warp_matrix3_init = np.array(np.linalg.inv(frame.H2Ref),dtype=np.float32)

    try:
        (cc, warp_matrix3) = cv2.findTransformECC(frame_ref_temp, getattr(frame,selected_field), warp_matrix3_init,  cv2.MOTION_HOMOGRAPHY, criteria, 
                                                  inputMask    = np.array(frame_mask_img,dtype=np.uint8) , 
                                                  templateMask = np.array(frame_ref_mask,dtype=np.uint8) )
    except: 
        warp_matrix3 = np.array(np.linalg.inv(frame.H2Ref),dtype=np.float32)

    frame_ = frame.copy()
    frame_.set_homography_to_ref(np.linalg.inv(warp_matrix3))    
    frame_.set_warp(     cv2.warpPerspective(frame.img,      frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ))
    frame_.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST))
    frame_temp =         cv2.warpPerspective(getattr(frame,selected_field),     frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )
    
    EP08_3 = get_EP08_from_img(frame_temp, frame_ref_temp, inputMask=frame_.mask_warp, inputMask_ref=frame_ref_mask )
    
    '''
    img_      = cv2.warpPerspective(frame.temp,     frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR)
    mask_img_ = cv2.warpPerspective(frame.mask_img, frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)
    plt.imshow(np.ma.masked_where(frame_ref_mask==0,frame_ref_temp).T,origin='lower')
    plt.imshow(np.ma.masked_where(mask_img_==0,img_).T,origin='lower',alpha=.7,cmap=mpl.cm.Greys_r)
    plt.show()
    '''
    
    if EP08_3 > EP08_2: 
        frame.set_homography_to_ref(frame_.H2Ref)
        id_ecc = 3
    else: 
        EP08_3 = EP08_2
   
    return id_ecc, frame.H2Ref
   

#################################################
def findTransformECC_on_ref_frame(flag, frame_in, frame_ref, trans_len_limit=[80,40], ep08_limit=[.7,.6], mask_func=mask_EP08, mask_func_param=[1.e6,0]):
    
    
    frame = frame_in.copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-4)
    selected_field = 'templn' if frame.type=='lwir' else 'img'

    frame_ref_temp = cv2.warpPerspective(getattr(frame_ref,selected_field) ,     frame_ref.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )  

    #add plot mask in frame_ref mask
    frame_mask_warp, frame_ref_mask = mask_func( frame, frame_ref, kernel_warp=frame.kernel_warp, kernel_plot=frame.kernel_plot, 
                                                    lowT       =mask_func_param[0],
                                                    kernel_lowT=mask_func_param[1] )
    frame_mask_img = cv2.warpPerspective(frame_mask_warp, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP )

    if frame.type == 'lwir':
        EP08_0 = get_EP08(frame, frame_ref, inputMask_function=mask_func, inputMask_param=mask_func_param)
        #diff_temp_sum_0 = np.abs(frame.temp - frame_ref_temp)[np.where((frame_mask_img==1)&(frame_ref_mask==1))].sum()

        #print ''
        #print '0 ', EP08_0
        id_ecc = 0

        #TRANSLATION ON LOWER IMAGE RESOLUTION
        if ((flag=='refine') | (flag=='coarse')) & (trans_len_limit[0] >= 10): 
            
            temp0 = cv2.warpPerspective(getattr(frame,selected_field), frame.H2Ref,                                
                                        frame.img.shape[::-1],
                                        flags=cv2.INTER_LINEAR )
            mask0 =  frame_mask_warp

            res_factor = 2
            ______coarse    = np.zeros([old_div(temp0.shape[0],res_factor),old_div(temp0.shape[1],res_factor)]).shape
            temp0_coarse    = downgrade_resolution_4nadir(temp0,     ______coarse, flag_interpolation='conservative')
            mask0_coarse    = downgrade_resolution_4nadir(mask0, ______coarse, flag_interpolation='max')
            temp_ref_coarse = downgrade_resolution_4nadir(frame_ref_temp, ______coarse, flag_interpolation='conservative')
            mask_ref_coarse = downgrade_resolution_4nadir(frame_ref_mask, ______coarse, flag_interpolation='max')
           
            ep08 = 0.
            trans_len = 10
            info_prev = None
            while ep08 < ep08_limit[0]:
                trans_x = np.arange(-trans_len,trans_len+1)
                trans_y = np.arange(-trans_len,trans_len+1)
                nbre_iter = trans_x.shape[0]*trans_y.shape[0]
                info_ = np.zeros([nbre_iter,4])
                for ivec, vec in enumerate(itertools.product(trans_x, trans_y)) :
                    if info_prev is not None:
                        idx_ = np.where( ((vec[0]-info_prev[:,0])**2 + (vec[1]-info_prev[:,1])**2)==0)
                        if len(idx_[0])!=0:
                            info_[ivec,:] = info_prev[idx_,:]
                            continue
                    info_[ivec, :2] = vec
                    info_[ivec, 2:] = apply_translation_img( temp0_coarse, mask0_coarse, vec, temp_ref_coarse, mask_ref_coarse )
                
                #if info_[:,2].max() > .8*ep08_limit[0]:
                #    idx_  = np.where(info_[:,2] >= .9* info_[:,2].max()) #first get the vec with corr > 90% of the max
                #    idx2_ = idx_[0][info_[ idx_[0][ np.where( info_[idx_[0],3] == info_[idx_,3].max()) ], 2].argmax()] # then get the vec that have max match and get max corr on those
                #else: 
                idx2_ = info_[:,2].argmax()
                
                ep08       = info_[idx2_,2]
                trans_vec  = info_[idx2_,:2]

                trans_len += 10
                info_prev = np.copy(info_)
                #if (trans_len > trans_len_limit[0]) & (flag != 'final'): print '**',
                if (trans_len > trans_len_limit[0]): break
            
            #print trans_len-10,
            warp_matrix1_ = np.eye(2, 3, dtype=np.float32)
            warp_matrix1_[:,-1] = trans_vec * res_factor
            warp_matrix1 = np.eye(3, 3, dtype=np.float32)
            warp_matrix1[:2,:] = warp_matrix1_
          
            frame_ = frame.copy()
            frame_.set_homography_to_ref( frame_.H2Ref.dot(warp_matrix1) )
            frame_.set_warp(     cv2.warpPerspective(frame.img,      frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ))
            frame_.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST))
            frame_temp =         cv2.warpPerspective(getattr(frame,selected_field),     frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )
            frame_mask_temp = cv2.warpPerspective(frame_mask_img,    frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)

            EP08_1 = get_EP08(frame_, frame_ref, inputMask_function=mask_func, inputMask_param=mask_func_param)
            #diff_temp_sum_1 = np.abs(frame_temp - frame_ref_temp)[np.where((frame_mask_temp==1)&(frame_ref_mask==1))].sum()
            
            if EP08_1 > EP08_0: 
                frame.set_homography_to_ref(frame_.H2Ref)
                frame.set_warp(     cv2.warpPerspective(frame.img,      frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ))
                frame.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST))
                id_ecc = 1
            else:
                EP08_1 = EP08_0
                #diff_temp_sum_1 = diff_temp_sum_0
                warp_matrix1 = np.eye(3, 3, dtype=np.float32)
        else: 
            EP08_1 = EP08_0
            #diff_temp_sum_1 = diff_temp_sum_0
            warp_matrix1 = np.eye(3, 3, dtype=np.float32)

        #print '1 ',  EP08_1, trans_vec

        #2nd TRANSLATION AT FULL RESOLUTION
        if trans_len_limit[1]>= 10:
            
            temp1 = cv2.warpPerspective(getattr(frame,selected_field), frame.H2Ref,                                
                                        frame.img.shape[::-1],
                                        flags=cv2.INTER_LINEAR )
            mask1 =  cv2.warpPerspective(frame_mask_warp, warp_matrix1,                                
                                        frame.img.shape[::-1],
                                        flags=cv2.INTER_NEAREST )
            
            trans_len = 10
            ep08 = 0.
            info_prev = None
            while ep08 < ep08_limit[1]:
                trans_x = np.arange(-trans_len,trans_len+1)
                trans_y = np.arange(-trans_len,trans_len+1)
                nbre_iter = trans_x.shape[0]*trans_y.shape[0]
                info_ = np.zeros([nbre_iter,4])
                for ivec, vec in enumerate(itertools.product(trans_x, trans_y)) :
                    if info_prev is not None:
                        idx_ = np.where( ((vec[0]-info_prev[:,0])**2 + (vec[1]-info_prev[:,1])**2)==0)
                        if len(idx_[0])!=0:
                            info_[ivec,:] = info_prev[idx_,:]
                            continue
                    info_[ivec, :2] = vec
                    info_[ivec, 2:] = apply_translation_img( temp1, mask1, vec, frame_ref_temp, frame_ref_mask )
                    #print(vec, info_[ivec, 2:],trans_len, trans_len_limit[1] ) 
                #idx_ = np.where(info_[:,2] >= .8* info_[:,2].max())
                #idx2_ = idx_[0][info_[ idx_[0][ np.where( info_[idx_[0],3] == info_[idx_,3].max()) ], 2].argmax()] 
                #if info_[:,2].max() > .8*ep08_limit[0]:
                #    idx_  = np.where(info_[:,2] >= .9* info_[:,2].max()) #first get the vec with corr > 90% of the max
                #    idx2_ = idx_[0][info_[ idx_[0][ np.where( info_[idx_[0],3] == info_[idx_,3].max()) ], 2].argmax()] # then get the vec that have max match and get max corr on those
                #else: 
                idx2_ = info_[:,2].argmax()
                ep08       = info_[idx2_,2]
                trans_vec  = info_[idx2_,:2]

                trans_len += 10
                info_prev = np.copy(info_)

                #if (trans_len > trans_len_limit[1]) & (flag != 'final'): print '**',
                if (trans_len > trans_len_limit[1]): break

            #print trans_len-10,  
            warp_matrix2_ = np.eye(2, 3, dtype=np.float32)
            warp_matrix2_[:,-1] = trans_vec
            warp_matrix2 = np.eye(3, 3, dtype=np.float32)
            warp_matrix2[:2,:] = warp_matrix2_
       
            #warp_matrix2 = np.linalg.inv(frame.H2Ref).dot(warp_matrix2)
            
            frame_ = frame.copy()
            frame_.set_homography_to_ref(frame_.H2Ref.dot(warp_matrix2)) 
            frame_.set_warp(     cv2.warpPerspective(frame.img,      frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ))
            frame_.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST))
            frame_temp =         cv2.warpPerspective(getattr(frame,selected_field),     frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )
            frame_mask_temp = cv2.warpPerspective(frame_mask_img,    frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)

            EP08_2 = get_EP08(frame_, frame_ref, inputMask_function=mask_func, inputMask_param=mask_func_param)
            #diff_temp_sum_2 = np.abs(frame_temp - frame_ref_temp)[np.where((frame_mask_temp==1)&(frame_ref_mask==1))].sum()
            
            if EP08_2 > EP08_1: 
                frame.set_homography_to_ref(frame_.H2Ref)
                frame.set_warp(     cv2.warpPerspective(frame.img,      frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ))
                frame.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST))
                id_ecc = 2
            else: 
                EP08_2 = EP08_1
                #diff_temp_sum_2 = diff_temp_sum_1
        else:
            EP08_2 = EP08_1
            #diff_temp_sum_2 = diff_temp_sum_1
     
        #print  '2 ', EP08_2,trans_vec
      
    else:
        EP08_2 = frame.corr_ref00 # get_EP08(frame, frame_ref, inputMask_function=mask_func, inputMask_param=mask_func_param)
        frame_temp      = cv2.warpPerspective(getattr(frame,selected_field),     frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )
        frame_mask_temp = cv2.warpPerspective(frame_mask_img,    frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)
        id_ecc = 0

    frame_temp =         cv2.warpPerspective(getattr(frame,selected_field),     frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )
    #print ''
    #print 'beofre ', get_EP08(frame, frame_ref, inputMask_function=mask_func, inputMask_flag='warp', inputMask_param=mask_func_param)

    frame_mask_warp, frame_ref_mask = mask_func( frame, frame_ref, kernel_warp=frame.kernel_warp, kernel_plot=frame.kernel_plot, 
                                                    lowT       =mask_func_param[0],
                                                    kernel_lowT=mask_func_param[1] )
    frame_mask_img = cv2.warpPerspective(frame_mask_warp, frame.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP )
    
    #HOMOGRAPHY
    warp_matrix3_init = np.array(np.linalg.inv(frame.H2Ref),dtype=np.float32)
    img_pre = cv2.warpPerspective(getattr(frame,selected_field),          warp_matrix3_init, frame.img.shape[::-1], flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
    mask_img_pre = cv2.warpPerspective(frame_mask_img, warp_matrix3_init, frame.img.shape[::-1], flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP)
    pre_diff =  (img_pre * frame_ref_temp)[np.where((frame_mask_temp==1)&(mask_img_pre==1))].sum()

    flag_plot_ = False 
    if flag_plot_:
        plt.figure()
        plt.imshow(frame_ref_temp.T,origin='lower',vmax=70)
        plt.contour(np.ma.masked_where(frame.mask_warp==0,img_pre).T,origin='lower',cmap=mpl.cm.Greys_r,levels=[25,35,100])
        plt.title('pre homography')
        
    #print ''
    #print EP08_0, EP08_2
    #mm = np.where( (mask_img_pre==1) & (frame_ref_mask==1),1,0) 
    #print '    ',  get_EP08_from_img(img_pre,frame_ref_temp , inputMask=mask_img_pre, inputMask_ref=frame_ref_mask )

    #if flag == 'final': 
    #    template = get_gradient(frame_ref_temp)[0]
    #    image    = get_gradient(frame.temp)[0]
    #else:
    template = frame_ref_temp
    image    = getattr(frame,selected_field)
    
    #if frame.type == 'visible':
    #    image = 255- image
    #plt.imshow(np.ma.masked_where(np.where(frame.plotMask_withBuffer_ring!=2,0,frame_mask_img)==0,image).T,origin='lower'); plt.show()
    #pdb.set_trace()

    try: 
        #(cc, warp_matrix3) = cv2.findTransformECC(get_gradient(frame_ref_temp)[0], get_gradient(frame.temp)[0], 
        (cc, warp_matrix3) = cv2.findTransformECC(template, image, 
                                                  warp_matrix3_init,  cv2.MOTION_HOMOGRAPHY, criteria, 
                                                  inputMask    = np.array( np.where(frame.plotMask_withBuffer_ring!=2,0,frame_mask_img) ,dtype=np.uint8),
                                                  templateMask = np.array(frame_ref_mask,dtype=np.uint8) )
                                                  #inputMask    = np.array(np.where((frame_mask_img==1)&(frame_ref_mask==1),1,0),dtype=np.uint8) )
    except: 
        warp_matrix3 = np.array(np.linalg.inv(frame.H2Ref),dtype=np.float32)

    frame_ = frame.copy()
    frame_.set_homography_to_ref(np.linalg.inv(warp_matrix3))    
    frame_.set_warp(     cv2.warpPerspective(frame.img,      frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR ))
    frame_.set_maskWarp( cv2.warpPerspective(frame.mask_img, frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST))
    #frame_temp =         cv2.warpPerspective(getattr(frame,selected_field),     frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR )
    #frame_mask_temp = cv2.warpPerspective(frame_mask_img,    frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)
    
    EP08_3 = get_EP08(frame_, frame_ref, inputMask_function=mask_func, inputMask_param=mask_func_param)
    
    #diff_temp_sum_3 = np.abs(frame_temp - frame_ref_temp)[np.where((frame_mask_temp==1)&(frame_ref_mask==1))].sum()
    #print '##############after ', EP08_3, EP08_0, EP08_3-EP08_0

    if flag_plot_:
        img_      = cv2.warpPerspective(getattr(frame_,selected_field),     frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR)
        mask_img_ = cv2.warpPerspective(frame_mask_img, frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)
        
        plt.figure()
        plt.imshow(frame_ref_temp.T,origin='lower',vmax=70)
        plt.contour(img_.T,origin='lower',cmap=mpl.cm.Greys_r,levels=[25,35,100])
        plt.title('post homography')
        
        mm = np.where( (mask_img_==1) & (frame_ref_mask==1),1,0) 

        #print 'post ',  (img_ * frame_ref_temp)[np.where((frame_mask_temp==1)&(mask_img_==1))].sum()
        print(get_EP08_from_img(img_,frame_ref_temp , inputMask=mask_img_, inputMask_ref=frame_ref_mask ))
        
        plt.show()
        pdb.set_trace()
   
    #img_      = cv2.warpPerspective(frame_.temp,     frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_LINEAR)
    #mask_img_ = cv2.warpPerspective(frame_mask_img, frame_.H2Ref, frame.img.shape[::-1], flags=cv2.INTER_NEAREST)
    #print  '|', EP08_3, get_EP08_from_img(img_,frame_ref_temp , inputMask=mask_img_, inputMask_ref=frame_ref_mask ), '|', EP08_3-EP08_0

    if (EP08_3 > EP08_2 ) :# & (diff_temp_sum_3 < diff_temp_sum_2): 
        frame.set_homography_to_ref(frame_.H2Ref)
        id_ecc = 3
    else: 
        EP08_3 = EP08_2
   
    return id_ecc, frame.H2Ref


#############################################################################
def get_covergaeOfExtendedMaskPlot(frame,inputConfig,delatationKernel=81):
    '''
    this is a measure of how much the warped image is covering the plot and the aera around it
    '''
    if frame.type == 'visible': delatationKernel_ = old_div(int(old_div(delatationKernel,inputConfig.params_vis_camera['shrink_factor'])),2) * 2 + 1
    else:                       delatationKernel_ = old_div(int(delatationKernel                                               ),2) * 2 + 1
    kernel = np.ones((delatationKernel_,delatationKernel_),np.uint8)
    mask_plot = np.where( (frame.plotMask_withBuffer ==2), np.ones_like(frame.warp), np.zeros_like(frame.warp) )
    img_ = np.array(mask_plot,dtype=np.uint8)*255
    mask_plot = cv2.dilate(img_, kernel, iterations = 1)
    mask_plot = np.where((mask_plot==255) & (img_ != 255), np.ones_like(mask_plot),np.zeros_like(mask_plot))


    ring_ok =  old_div(1.*( np.where( (mask_plot==1) ,1,0) - np.where( (frame.mask_warp==1) & (mask_plot==1) ,1,0) ).sum(), np.where( (mask_plot==1) ,1,0).sum()) 
   
    '''
    if frame.id>973:
        plt.figure()
        plt.imshow(np.where( (mask_plot==1) ,1,0).T,origin='lower')
        plt.figure()
        plt.imshow(np.where( (frame.mask_warp==1) & (mask_plot==1) ,1,0).T,origin='lower'); plt.show()
        pdb.set_trace()
    '''

    return ring_ok #1.*np.where( (frame.mask_warp==1) & (mask_plot==1) ,1,0).sum() / mask_plot.sum()


#############################################################################
def get_bad_idLwir(dirIn):

    filenames = glob.glob(dirIn+'badFrameID*')
    badId_lwir = []
    for filename in filenames: 
        badId_lwir_ = np.load(filename)
        [badId_lwir.append(id) for id in badId_lwir_ ]

    return sorted(badId_lwir)


###################################################################
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


#############################################################################
def interpolation2d(maps_fire, inputArray, flag_interpolation_type='rbf', distance_interaction=60, mask=None):

    print('interpolation is :',  flag_interpolation_type)
    
    plotMask = maps_fire.mask
    resolution = maps_fire.grid_e[1,1]-maps_fire.grid_e[0,0]
    xi, yi = maps_fire.grid_e, maps_fire.grid_n
    xi = xi + .5*resolution 
    yi = yi + .5*resolution

    zi = np.copy(inputArray) # np.zeros_like(xi)

    #plt.imshow(zi.T,origin='lower',interpolation='nearest')
    #plt.show()
    #pdb.set_trace()

    mask_nodata = np.where(inputArray<0, np.ones_like(inputArray), np.zeros_like(inputArray))
    s = [[1,1,1], \
         [1,1,1], \
         [1,1,1]]
    mask_nodata_labled, num_cluster = ndimage.label(mask_nodata, structure=s )
    dimension_cluster_nodata = np.zeros(num_cluster)
    for i_cluster in range(num_cluster):
        idx_ = np.where(mask_nodata_labled==i_cluster+1) 
        pts = np.array(np.dstack( [idx_[0],idx_[1]] )[0],dtype=np.float)
        (center_x, center_y), (width, height), angle  = cv2.minAreaRect(np.array(pts,dtype=np.float32))
        dimension_cluster_nodata[i_cluster] = min([width,height])

    #interpolate the data with the matlab griddata function
    # try first griddata if fit the request, if point are missing (most probably value on the bndf), then we fall back to rbf 
    if flag_interpolation_type == 'griddata':
        idx = np.where(inputArray>=0)
        x = xi[idx]
        y = yi[idx]
        z = inputArray[idx]
        coord_pts = np.vstack((x, y)).T
        data      = z.flatten()
        fill_val  = -999
        #method    = 'cubic'
        #zi = interpolate.griddata(coord_pts , data, (xi,yi), fill_value=fill_val, method=method)
        
        interp = interpolate.LinearNDInterpolator(coord_pts, data, fill_value=fill_val, rescale=True)
        zi = interp(xi, yi)
        
        plt.imshow(zi.T,origin='lower',interpolation='nearest',vmin=0,vmax=1000)
        plt.show()
        pdb.set_trace()

        if np.where( (zi<=0) &  (mask==1) )[0].shape[0] != 0:  
            flag_interpolation_type = 'rbf' 

    
    if (flag_interpolation_type == 'rbf') : 
        time_now = datetime.datetime.now()
        
        possible_subset = []
        list1 = factor.get_factor(plotMask.shape[0])
        list2 = factor.get_factor(plotMask.shape[1])
        for ii in list1 :
            for jj in list2 :
            
                nrows = old_div(plotMask.shape[0],ii)
                ncols = old_div(plotMask.shape[1],jj)

                mem= psutil.virtual_memory()

                if (2*((3*nrows)*(3*ncols))**2*8 < mem.available)     &\
                   (nrows*resolution>distance_interaction)            &\
                   (ncols*resolution>distance_interaction)              : 
                       possible_subset.append([ii,jj])
       
        if len(possible_subset) == 0:
            print('cannot call rbf interpolation')
            print('no set up found for available mem and distance =', distance_interaction)
            print('mem   =',  mem)
            print('list1 =', list1)
            print('list2 =', list2)
            pdb.set_trace()
            sys.exit()

        print('  possible split of the domain for rbf call')
        print(possible_subset)
        print('  select')
        if len(possible_subset)==1:
            id_select = 0
        else:
            diff = []; mean = []
            for id_ in range(len(possible_subset)):
                diff.append(abs(possible_subset[id_][0]-possible_subset[id_][1]))
                mean.append(.5*(possible_subset[id_][0]+possible_subset[id_][1]))
            diff = np.array(diff);mean = np.array(mean)
            id_select_1 = np.where(diff <= 1.1*diff.min())
            id_select_2  = mean[id_select_1].argmin()
            id_select = id_select_1[0][id_select_2]

        #possible_subset[id_select] = [50,50]
        subset = possible_subset[id_select]
        print(subset)

        nrows = old_div(plotMask.shape[0],subset[0])
        ncols = old_div(plotMask.shape[1],subset[1])
        print('  sub domain = {:3.1f} x {:3.1f}'.format(nrows*resolution,ncols*resolution)) 
        nx,ny = xi.shape
        ii_sub = blockshaped(np.arange(xi.size).reshape(xi.shape), nrows, ncols) 

        #count number of point in each zone to sort the order to run interpolation
        density_point = np.zeros(ii_sub.shape[0])
        for i_sub in range(ii_sub.shape[0]):
            i_min_zz, j_min_zz = np.unravel_index(ii_sub[i_sub].min(),xi.shape) # zone we want
            i_max_zz, j_max_zz = np.unravel_index(ii_sub[i_sub].max(),xi.shape)
            
            i_min = i_min_zz-nrows # zone we want + extra point around
            i_max = i_max_zz+nrows
            j_min = j_min_zz-ncols
            j_max = j_max_zz+ncols
            
            #print max([i_min,0]),min([i_max,nx]),max([j_min,0]),min([j_max,ny])
            xi_tmp = xi[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
            yi_tmp = yi[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
            

            flag_zz = np.zeros_like(xi)
            flag_zz[max([i_min_zz,0]):min([i_max_zz,nx-1])+1,max([j_min_zz,0]):min([j_max_zz,ny-1])+1] = 1
            flag_zz_tmp = flag_zz[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]

            inputArray_tmp = inputArray[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
            plotMask_tmp = plotMask[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
        
            idx = np.where(inputArray_tmp>0)
            
            if (len(idx[0]) == 0) :
                density_point[i_sub] = -999
            else: 
                density_point[i_sub] = old_div(1.*len(idx[0]),(inputArray_tmp.shape[0]*inputArray_tmp.shape[1]))
           

        for i_loop, i_sub in enumerate(np.argsort(density_point)[::-1]):

            if i_sub < 0:
                continue

            i_min_zz, j_min_zz = np.unravel_index(ii_sub[i_sub].min(),xi.shape) # zone we want
            i_max_zz, j_max_zz = np.unravel_index(ii_sub[i_sub].max(),xi.shape)
            
            i_min = i_min_zz-nrows # zone we want + extra point around
            i_max = i_max_zz+nrows
            j_min = j_min_zz-ncols
            j_max = j_max_zz+ncols
            
            #print max([i_min,0]),min([i_max,nx]),max([j_min,0]),min([j_max,ny])
            xi_tmp = xi[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
            yi_tmp = yi[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
            

            flag_zz = np.zeros_like(xi)
            flag_zz[max([i_min_zz,0]):min([i_max_zz,nx-1])+1,max([j_min_zz,0]):min([j_max_zz,ny-1])+1] = 1
            flag_zz_tmp = flag_zz[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]

            inputArray_tmp = inputArray[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
            plotMask_tmp = plotMask[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
        
            idx = np.where(inputArray_tmp>0)
            if (len(idx[0]) == 0) :
                zi[max([i_min_zz,0]):min([i_max_zz,nx-1])+1,max([j_min_zz,0]):min([j_max_zz,ny-1])+1] = -999.
                continue
            
            idx_pt_to_interp = np.where((plotMask_tmp==2) & (inputArray_tmp<0 ))
            if len(idx_pt_to_interp[0]) == 0 : # if we only have 1 point with data we continue
                continue

            x = xi_tmp[idx]
            y = yi_tmp[idx]
            z = inputArray_tmp[idx]
            z_init = np.copy(inputArray_tmp)

            try: 
                interp = interpolate.Rbf(x, y, z, function='multiquadric',epsilon=resolution)
            except : 
                pdb.set_trace()
            #print 'rbf interp 1 elapse time (h):', (datetime.datetime.now()-time_now).total_seconds() / 3600
            sys.stdout.flush()
            zi_tmp = np.array([interp(xi_tmp.flatten()[i], yi_tmp.flatten()[i]) for i in range(xi_tmp.flatten().shape[0])]).reshape(xi_tmp.shape)
            print('  {:3d}% | elapse time (min): {:4.1f} |  point density: {:3.2} | mem availaible = {:3.1f}'.format(int(old_div(100.*i_loop,len(np.where(density_point>=0)[0]))), old_div((datetime.datetime.now()-time_now).total_seconds(), 60), density_point[i_sub], old_div(100.*psutil.virtual_memory().available,psutil.virtual_memory().total) ))

            zi_tmp2 = np.copy(zi_tmp)

            idx = np.where(plotMask_tmp!=2)
            zi_tmp[idx] = -999
  
            idx = np.where(flag_zz_tmp==1)
            zi[max([i_min_zz,0]):min([i_max_zz,nx-1])+1,max([j_min_zz,0]):min([j_max_zz,ny-1])+1] = zi_tmp[idx].reshape((nrows,ncols))
           
            '''
            fig = plt.figure(figsize=(15,8))
            ax=plt.subplot(141)
            plt.imshow(np.ma.masked_where(zi<=0,zi).T,origin='lower',interpolation='nearest')
            ax=plt.subplot(142)
            plt.imshow(np.ma.masked_where(z_init<=0,z_init).T,origin='lower',interpolation='nearest')
            ax=plt.subplot(143)
            plt.imshow(np.ma.masked_where(zi_tmp<=0,zi_tmp).T,origin='lower',interpolation='nearest')
            ax=plt.subplot(144)
            plt.imshow(np.ma.masked_where(zi_tmp2<=0,zi_tmp2).T,origin='lower',interpolation='nearest')
            plt.show()
            '''
            zi_tmp = None
            interp = None
            inputArray_tmp = None
            plotMask_tmp = None
            gc.collect() # collect memory carbage

            #plt.imshow(np.ma.masked_where(zi<0,zi).T,origin='lower',interpolation='nearest')
            #plt.show()
            #pdb.set_trace()
            
    
    #final step, set point outside the plot to -999
    idx = np.where(mask==0)
    zi[idx]=-999

    return zi


#############################################    
def group_consecutives(vals, step=1, vals_mask=None, flag_output='val'):
    """Return list of consecutive lists of numbers from vals (number list)."""
   
    try:
        if vals_mask == None: 
            vals_mask = np.zeros_like(vals)
            vals = np.ma.array(vals, mask=np.zeros_like(vals))
    except ValueError: 
        vals = np.ma.array(vals, mask=vals_mask)

    if np.ma.isMaskedArray(vals) is False: 
        print('error in group_consecutives: input array is not masked')
        sys.exit()

    run = []
    result = [run]
    expect = None
    flag_first_unMask = 0
    for iv,(v,vmask) in enumerate(zip(vals.data, vals.mask)):
        if not vmask:
            if (v == expect) or (expect is None):
               
                if flag_output == 'val':
                    run.append(v)
                elif flag_output == 'idx':
                    run.append(iv)

            else:
                if flag_output == 'val':
                    if len(run)==0:
                        run.append(v)
                    else:
                        run = [v]
                elif flag_output == 'idx':
                    if len(run)==0:
                        run.append(iv)
                    else:
                        run = [iv]
                if flag_first_unMask != 0: 
                    result.append(run)
            
            flag_first_unMask = 1

        expect = v + step
    
    return result


############################################
def load_good_lwir_frame_selection(flag_restart, dir_out_npy, georefMode='SH'):

    if (not(os.path.isfile(dir_out_npy+'../lwir_time_info.npy'))) | (flag_restart == False):
        print('')
        print('loop over lwir ... ', end=' ')
        sys.stdout.flush()
        
        lwir_goeref00    = sorted(glob.glob(dir_out_npy+'*'+georefMode+'*.npy'))
        
        lwir_time = []
        lwir_id = []  

        badId_lwir = get_bad_idLwir(dir_out_npy) 
        lwir_goeref = []
        for file_ in lwir_goeref00:
            id_lwir =  int( os.path.basename(file_).split('.')[0].split('_')[-2] )
            if id_lwir in badId_lwir: continue
            lwir_goeref.append(file_)
            lwir_time.append( np.load(file_, allow_pickle=True, encoding='latin1')[0][1] )
            lwir_id.append( id_lwir)
       
        lwir_time = np.array(lwir_time)
        lwir_id = np.array(lwir_id,dtype=int)    

        np.save(dir_out_npy+'../lwir_time_info.npy',[lwir_time,lwir_id])
        np.save(dir_out_npy+'../lwir_selection_filename.npy',lwir_goeref)  
    
    else:
        lwir_time, lwir_id = np.load(dir_out_npy+'../lwir_time_info.npy')
        lwir_goeref_        = np.load(dir_out_npy+'../lwir_selection_filename.npy')
        
        lwir_goeref = []
        for ii,lwir_goeref__ in enumerate(lwir_goeref_):
            try:
                lwir_goeref.append( lwir_goeref__ )
            except:
                lwir_goeref.append( lwir_goeref__.decode() )
        lwir_goeref = np.array(lwir_goeref)

    print('done ')

    return np.array(lwir_id,dtype=int), np.array(lwir_time), np.array(lwir_goeref)



############################################3
def load_good_mir_frame_selection(flag_restart, dir_out_npy, plotname, path_data, georefMode='SH'):

    if (not(os.path.isfile(dir_out_npy+'../mir_time_info.npy'))) | (flag_restart == False):
        print('')
        print('loop over mir ... ', end=' ')
        sys.stdout.flush()
    
        mir_goeref00    = sorted(glob.glob(dir_out_npy+'*'+georefMode+'*.npy'))
        
        mir_time = []
        mir_id = []  

        try:
            [badId_mir,badMirTime],[goodMirId,goodMirTime] = np.load(dir_out_npy+'{:s}_badgoodMirIdTime.npy'.format(plotname),allow_pickle=True)
        except: 
            [badId_mir,badMirTime],[goodMirId,goodMirTime] = np.load(dir_out_npy+'{:s}_badgoodMirIdTime.npy'.format(plotname),allow_pickle=True, encoding='latin1')

        mir_goeref = []
        for file_ in mir_goeref00:
            id_mir =  int( os.path.basename(file_).split('.')[0].split('_')[-2] )
            if id_mir in badId_mir: continue
            mir_goeref.append(file_)
            try:
                mir_time.append( np.load(file_, allow_pickle=True)[0][1] )
            except: 
                mir_time.append( np.load(file_, allow_pickle=True, encoding='latin1')[0][1] )

            mir_id.append( id_mir)
       
        mir_time = np.array(mir_time)
        mir_id = np.array(mir_id,dtype=int)    

        np.save(dir_out_npy+'../mir_time_info.npy',[mir_time,mir_id])
        np.save(dir_out_npy+'../mir_selection_filename.npy',mir_goeref)
    
    else:
        mir_time, mir_id = np.load(dir_out_npy+'../mir_time_info.npy')
        mir_goeref_       = np.load(dir_out_npy+'../mir_selection_filename.npy')
        
        mir_goeref = []
        for ii,mir_goeref__ in enumerate(mir_goeref_):
            try:
                mir_goeref.append( mir_goeref__.replace('/media/paugam/goulven/data/', path_data) )
            except: 
                mir_goeref.append( mir_goeref__.decode().replace('/media/paugam/goulven/data/', path_data) )
        mir_goeref = np.array(mir_goeref)

    print('done')

    return np.array(mir_id,dtype=int), np.array(mir_time), np.array(mir_goeref)
