from __future__ import print_function
from __future__ import division
from past.utils import old_div
from PIL import Image
import datetime
import os, sys, glob 
import numpy as np
from PIL.ExifTags import TAGS
from scipy import io,ndimage,stats,signal,optimize 
import pdb 


#homebrewed 
import tools

#######################################
def get_time_shift_vis_lwir_mir(params):

    root                 = params['root']
    time_calibration_dir = params['dir_time_calibration']
    time_file_vis_       = params['time_calibration_vis']
    time_file_mir_       = params['time_calibration_mir']
    time_file_lwir_      = params['time_calibration_lwir']

    out_dir = params['root_postproc'] + '/Time_pixel_correction/'
    time_path_in = root + time_calibration_dir

    time_file_vis  = time_path_in + time_file_vis_
    time_file_mir  = None if time_file_mir_== None else time_path_in + time_file_mir_
    time_file_lwir = time_path_in + time_file_lwir_

    tools.ensure_dir(out_dir)


    #get time difference
    ####################
    #gopro
    if os.path.isfile(time_file_vis):
        img = Image.open(time_file_vis)
        exif_data = img._getexif()
        img = None
        vis_time = datetime.datetime.strptime(get_field(exif_data,'DateTime'), "%Y:%m:%d %H:%M:%S")
    else: 
        #vis_time=datetime.datetime(2014,1,1) # set time as this would be use to make difference between delta_t_lwir and delta_t_mir
        print() 
        'no time correction applied'
        return 0.,0.

    #optris
    if os.path.isfile(time_file_lwir) :
        try: 
            lwir_time = np.load(time_file_lwir, allow_pickle=True)[0][0]
        except: 
            lwir_time = datetime.datetime.strptime(os.path.basename(time_file_lwir).split('.')[0].split('d_')[1], "%Y-%m-%d_%H-%M-%S")
    else: 
        lwir_time = None

    #agema
    if time_file_mir == None: 
        mir_time = None
    elif os.path.isfile(time_file_mir):
        MIR_name = os.path.basename(time_file_mir).split('.')[0]
        res = io.loadmat(time_file_mir)
        in_ = np.array(res[MIR_name+'_DateTime'][0],dtype=np.int)
        in_[-1] = 1.e3*in_[-1]
        mir_time = datetime.datetime(*in_)
    else: 
        mir_time = None

    delta_t_lwir = (vis_time - lwir_time).total_seconds() 
    
    if mir_time!=None:  
        delta_t_mir  = (vis_time - mir_time).total_seconds() 
    else: 
        delta_t_mir = None
    print('Vis  ', vis_time)
    print('lwir ', datetime.datetime.strftime(lwir_time, "%Y-%m-%d_%H-%M-%S.%f"), delta_t_lwir)
    if mir_time!=None: print('mir  ', datetime.datetime.strftime(mir_time, "%Y-%m-%d_%H-%M-%S.%f"),  delta_t_mir)

    np.save(out_dir+params['plotname']+'_lwir2vis_time',[delta_t_lwir])
    if mir_time!=None: np.save(out_dir+params['plotname']+'_mir2vis_time',[delta_t_mir])


    return delta_t_lwir, delta_t_mir



#######################################
def get_field (exif,field) :
    for (k,v) in exif.items():
        #print TAGS.get(k)
        if TAGS.get(k) == field:
            return v



######################################################
def get_camera_intrinsec_distortion_matrix(params_camera):

    '''
    use prescribed constant for now. Could set up a claibration later. 
    '''

    try: 
        return np.load('../data_static/Camera/' + params_camera['camera_name'].split('_')[0]
                                                + '/Geometry/'
                                                + params_camera['camera_name']
                                                + '_cameraMatrix_DistortionCoeff.npy', allow_pickle=True, encoding='latin1')
    except IOError:

        print('** ideal camera matrix is set with **')
        pix_width = params_camera['nbrePix_width']
        pix_height = params_camera['nbrePix_height']
        aspect_ratio = old_div(pix_width,pix_height)
        # Focal length, sensor size (mm and px)
        if ('focal' in list(params_camera.keys())): 
            focal         = params_camera['focal']
            sensor_width  = 2 * focal * np.tan( .5*params_camera['cameraLens'] * 3.14/180) 
            sensor_height = old_div(sensor_width, aspect_ratio)
        elif ('sensor_width' in list(params_camera.keys())): 
            sensor_width  = params_camera['sensor_width']
            sensor_height = old_div(sensor_width, aspect_ratio)
            focal = old_div((.5*sensor_width),np.tan(old_div((3.14/180) * params_camera['cameraLens'],2)))
        else: 
            print('  ** missing info in camera info to compute K')
            print('  ** stop here')
            sys.exit()
        print('    camera lens = ',  params_camera['cameraLens'])
        print('    camera nx = ',    sensor_width)
        print('    camera ny = ',    sensor_height)
        print('    camera focal = ', focal)
        print('  ')
 
        # set center pixel
        u0 = int(pix_width / 2.0)
        v0 = int(pix_height / 2.0)

        # determine values of camera-matrix
        mu = old_div(pix_width, sensor_width) # px/mm
        alpha_u = focal * mu # px

        mv = old_div(pix_height, sensor_height) # px/mm
        alpha_v = focal * mv # px
        
        #print 'sensor width  = ', sensor_height
        #print 'sensor height = ', sensor_width # width and height are inverse 

        # Distortion coefs 
        D = np.array([[0.0, 0.0, 0.0, 0.0, 0.e0]])

        # Camera matrix
        K = np.array([[alpha_v, 0.0, v0],
                      [0.0, alpha_u, u0],
                      [0.0, 0.0, 1.0]])

        return K, D

