from __future__ import print_function
from __future__ import division
from builtins import input
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np
import cv2
import img_scale
import glob 
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt 
import sys
from skimage import feature, measure
import tools 
import os 
import pdb 
import scipy
import transformation 
import pandas 
import shapefile
from osgeo import gdal,osr,ogr
import asciitable
import multiprocessing
import pickle
import matplotlib.path as mpath
from matplotlib.path import Path as mpPath
import matplotlib.patches as mpatches
import shapely
from shapely.geometry import Polygon
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
import datetime
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS
from scipy import io 
import copy 
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import shutil
from netCDF4 import Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from importlib import reload

#homebrewed 
import spectralTools
#import runingNeighbourImage
import tools
sys.path.append('../../RuningAverageImage/')
import runingNeighbourImage
import hist_matching
import camera_tools as cameraTools 
reload(hist_matching)


#################################################
def residual(x, frame, feature_params):
    
    frame.set_trange(x)
    frame.set_img()
    frame.set_feature(feature_params)

    if frame.feature.shape[0] != 0: 
        return 1./frame.feature.shape[0]
    else: 
        return 1.e6


#################################################
class loadFrame(object):
    
    def __init__(self, params_camera): 
        self.type = 'lwir'
        #self.energy = 0
        #self.shrink_factor=params_camera['shrink_factor']
        #self.bufferZone = 100
        #self.grayZone   = 1
            
        self.corr_ref        = 0
        self.corr_ref00      = 0
        self.corr_ref00_init = 0
        self.inRefList = 'no'
        self.shrink_factor=params_camera['shrink_factor']
        self.bufferZone = 100
        self.grayZone   = 1

    def init(self, id_file, frame_ref00_, filenames, timelookupTable, K, D, inputConfig, grid_shape, feature_params=None, flag_blur=None, image_pdf_reference_tools=None ):

        #load image
        time_date, time_igni, tempRaw_ = np.load(filenames[id_file], allow_pickle=True)
        self.id   = id_file
        self.clahe_clipLimit = inputConfig.params_lwir_camera['clahe_clipLimit']
        self.time_igni = timelookupTable.time[np.where(timelookupTable.name==os.path.basename(filenames[id_file]))][0]
        #self.time_igni = time_igni
        self.time_date = time_date
        self.set_flag_inRefList('no') # default
        
        self.kernel_plot = inputConfig.params_lwir_camera['kernel_plot']
        self.kernel_warp = inputConfig.params_lwir_camera['kernel_warp']
        self.grid_shape  = tuple(grid_shape)
        self.lowT_param = inputConfig.params_lwir_camera['lowT_param']


        #apply distrotion correction
        h, w  = tempRaw_.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))
        # undistort
        tempRaw_undistorted = cv2.undistort(tempRaw_, K, D, None, newcameramtx)
        # crop the image
        x,y,w,h = roi
        tempRaw_crop = tempRaw_undistorted[y:y+h, x:x+w]
      
        if tempRaw_crop.shape[0] % 2 !=0: 
            tempRaw_crop = tempRaw_crop[:-1,:]
        if tempRaw_crop.shape[1] % 2 !=0: 
            tempRaw_crop = tempRaw_crop[:,:-1]

        #save camera matrix
        self.K = newcameramtx
        # D is now 0 as we undistrot the image
        
        #save camera matrix
        self.K_raw         = K
        self.K_undistorted = newcameramtx
        if self.shrink_factor > 1: 
            a = 1./self.shrink_factor
            self.K_undistorted_imgRes = np.dot(np.array([[a,0,0],[0,a,0],[0,0,1.]]),newcameramtx)
        else:
            self.K_undistorted_imgRes = newcameramtx

        if frame_ref00_ is not None:
            self.set_id_ref00(frame_ref00_.id)
            self.set_bareGroundMask(frame_ref00_.bareGroundMask_withBuffer)
        else:
            self.id_ref00 = -1

        #downgrade
        if self.shrink_factor > 1: 
            ni_, nj_ = old_div(tempRaw_crop.shape[0],self.shrink_factor), old_div(tempRaw_crop.shape[1],self.shrink_factor) 
            tempRaw = tools.downgrade_resolution_4nadir( tempRaw_crop[:,:], np.zeros([ni_,nj_]) , flag_interpolation='conservative' )
        else:
            tempRaw = tempRaw_crop

        self.ni = tempRaw.shape[0]
        self.nj = tempRaw.shape[1]

        temp = np.zeros([self.ni+self.bufferZone,self.nj+self.bufferZone])
        temp[old_div(self.bufferZone,2):old_div(-self.bufferZone,2),old_div(self.bufferZone,2):old_div(-self.bufferZone,2)] = tempRaw
       
        self.temp           = temp 
        self.imgRaw         = convert_2_uint8(tempRaw)
        
        #self.mask_img = mask_helico_leg(self)
        self.mask_img = np.zeros_like(temp) 
        self.mask_img[old_div(self.bufferZone,2):old_div(-self.bufferZone,2),old_div(self.bufferZone,2):old_div(-self.bufferZone,2)] = 1
                

        if feature_params is not None:
        
            #get thetrange that gives the largest number of features
            '''self.brute_opti_scan_nbre=10
            self.set_trange( scipy.optimize.brute(residual,\
                                                  ranges=((30,40),(40,50)),\
                                                  args=(self, feature_params),\
                                                  Ns=self.brute_opti_scan_nbre,
                                                  finish='None') )'''
         
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clipLimit, tileGridSize=(10,10)) # sha3
            temp_max_arr = np.arange(self.temp[old_div(self.bufferZone,2):old_div(-self.bufferZone,2),old_div(self.bufferZone,2):old_div(-self.bufferZone,2)].min()+10,
                                     self.temp[old_div(self.bufferZone,2):old_div(-self.bufferZone,2),old_div(self.bufferZone,2):old_div(-self.bufferZone,2)].min()+70,1)
            feature_nbre = np.zeros_like(temp_max_arr)
            for ii, temp_max in enumerate(temp_max_arr):
                img = clahe.apply(convert_2_uint8(self.temp[old_div(self.bufferZone,2):old_div(-self.bufferZone,2),old_div(self.bufferZone,2):old_div(-self.bufferZone,2)],
                                                  trange=(self.temp[old_div(self.bufferZone,2):old_div(-self.bufferZone,2),old_div(self.bufferZone,2):old_div(-self.bufferZone,2)].min(),temp_max)
                                                 ))
                p00      = cv2.goodFeaturesToTrack(img,      mask = None, **feature_params)
                try: 
                    feature_nbre[ii] = len(p00)
                except: 
                    pass
            #self.set_trange((29,47))
            idx_ = np.where(feature_nbre == feature_nbre.max())
            idx_ = idx_[0].max()
            self.set_trange((self.temp[old_div(self.bufferZone,2):old_div(-self.bufferZone,2),old_div(self.bufferZone,2):old_div(-self.bufferZone,2)].min(),temp_max_arr[idx_]))
            self.set_img()
            self.set_feature(feature_params)
        
            self.blurred = False
            # check fro blurry image
            #idx = np.where( flag_blur.filename == os.path.basename(filenames[id_file]))
            #self.blurred = True if flag_blur.blurred[idx] == 'yes' else False
        
        else:
            self.blurred = False
            self.set_img()
        



    def set_img(self):
        img_            = convert_2_uint8(self.temp, self.trange)
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10)) # sku6 sha1
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clipLimit, tileGridSize=(10,10)) # sha3

        self.img = clahe.apply(img_)
            
        if 'trange2' in self.__dict__: 
            img_            = convert_2_uint8(self.temp, self.trange2)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clipLimit, tileGridSize=(10,10)) 
            self.img2 = clahe.apply(img_)


    '''def update_img(self,feature_params):
            
        #get thetrange that gives the largest number of features
        self.set_trange( scipy.optimize.brute(residual,\
                                              ranges=((25,60),(60,150)),\
                                              args=(self, feature_params),\
                                              Ns=self.brute_opti_scan_nbre,\
                                              finish='None')  )
        self.set_img()
        self.set_feature(feature_params)

        #img_ = convert_2_uint8(self.temp,self.trange)
        #if self.trange[1]>=100:
        #    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
        #    self.img = clahe.apply(img_)
        #else:
        #    self.img = img_
        #self.set_feature(feature_params)
    '''

    def return_img(self,trange):
        img_ = convert_2_uint8(self.temp,trange)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clipLimit, tileGridSize=(10,10))
        return clahe.apply(img_)


    def set_feature(self,feature_params):
        self.feature, self.nbrept_badTemp, self.nbrept_helico = get_feature(self,feature_params)
        if 'trange2' in self.__dict__:
            self.feature2, self.nbrept_badTemp2, self.nbrept_helico2 = get_feature(self,feature_params,input_img=['img2','trange2'])


    def set_warp(self,input_):
        self.warp = input_

    def return_warp(self,trange):
        nx,ny = self.img.shape
        img_warp = cv2.warpPerspective(self.return_img(trange), self.H2Ref, \
                                       (ny,nx),                             \
                                       borderValue=0,flags=cv2.INTER_NEAREST)
        return img_warp

    def set_homography_to_ref(self,input_):
        self.H2Ref = input_
    
    def set_homography_to_grid(self,input_):
        self.H2Grid = input_

    def set_pause(self,rvec,tvec):
        self.rvec = rvec
        self.tvec = tvec
    
    #def set_plotMask(self,input_):
    #    self.plotMask_withBuffer = input_
    def set_plotMask(self,input_, input2_):
        self.plotMask_withBuffer      = input_
        self.plotMask_withBuffer_ring = input2_ 

    def set_bareGroundMask(self,input_):
        self.bareGroundMask_withBuffer = input_

    def set_maskWarp(self,input_):
        self.mask_warp = input_
    
    def set_correlation(self,corr0,corr1,corr2,ssim):
        if corr0 is not None: 
            self.corr_ref        = corr0
        if corr1 is not None: 
            self.corr_ref00      = corr1
        if corr2 is not None: 
            self.corr_ref00_init = corr2
        if ssim is not None:
            self.ssim   = ssim
    
    #def set_energy(self,energy_):
    #    self.energy = energy_

    def set_id_best_ref(self, input_):
        self.id_best_ref = input_
    
    def set_id_ref00(self, input_):
        self.id_ref00 = input_
    
    def set_similarity_info(self,ssim_2d,mask_ssim):
        self.ssim_2d = ssim_2d
        self.mask_ssim = mask_ssim

    def save_feature_old_new(self, good_new_4plot,good_new,good_old):
        self.good_new_4plot = good_new_4plot
        self.good_new = good_new
        self.good_old = good_old
    
    def save_number_ref_frames_used(self, input_):
        self.number_ref_frames_used = input_

    def save_georef_radiance(self, input_):
        self.georef_radiance = input_

    def save_georef_temperature(self, georef_radiance, lookupTable_TTRad):
        self.georef_temp = spectralTools.conv_Rad2Temp(georef_radiance, lookupTable_TTRad)

    def copy(self):
        return copy.deepcopy(self)

    def set_trange(self, trange_):
        self.trange = trange_
        if len(trange_)>2: 
            self.trange2 = trange_[2:]

    def create_backgrdimg(self):
        self.backgrdimg = np.copy(self.img)
        kernel = np.ones((3,3),np.uint8)
        mask_eroded = cv2.erode(self.mask_img, kernel,iterations = self.grayZone)
        idx = np.where(mask_eroded == 0)
        self.mask_backgrdimg = np.ones_like(self.backgrdimg)
        self.mask_backgrdimg[idx] = 0 
        return 'init'

    def update_backgrdimg(self,frame_):
        
        kernel = np.ones((3,3),np.uint8)
        newframe_mask_eroded = cv2.erode( frame_.mask_warp, kernel,iterations = self.grayZone)
       
        idx = np.where( (self.mask_backgrdimg == 0) & (frame_.warp > 0) & (newframe_mask_eroded==1) )
        
        self.backgrdimg[idx] = frame_.return_warp(self.trange)[idx]
        self.mask_backgrdimg[idx] = 1

        '''plt.gcf().clear()
        ax = plt.subplot(121)
        ax.imshow(self.backgrdimg.T,origin='lower')
        ax = plt.subplot(122)
        mm = np.zeros_like(self.backgrdimg)
        mm[idx] = 1
        ax.imshow(mm.T,origin='lower')
        plt.draw()
        plt.pause(.1)
        #pdb.set_trace()
        '''
        if np.where(self.mask_backgrdimg==0)[0].shape[0] > 0: 
            return 'not finished'
        else:
            return 'done'

    def pass_refKit(self,frame_):
        #self.backgrdimg      = frame_.backgrdimg
        #self.mask_backgrdimg = frame_.mask_backgrdimg
        self.bareGroundMask_withBuffer = frame_.bareGroundMask_withBuffer

    def set_matching_feature(self,gcp_cam,gcp_world):
        self.gcps_cam = gcp_cam
        self.gcps_world = gcp_world
    
    def dump(self,filename):
        ncfile = Dataset(filename,'w')
    
        ncfile.description = 'lwir frame generated by GeoRefCam'
   
        # Global attributes
        setattr(ncfile, 'created', 'R. Paugam') 
        setattr(ncfile, 'title', 'lwir frame')
        setattr(ncfile, 'Conventions', 'CF')

        setattr(ncfile,'type', self.type)
        setattr(ncfile,'id', self.id)
        setattr(ncfile,'id_ref00', self.id_ref00)
        setattr(ncfile, 'correlation ref',        self.corr_ref)
        setattr(ncfile, 'correlation ref00',      self.corr_ref00)
        setattr(ncfile, 'correlation ref00 init', self.corr_ref00_init)
        setattr(ncfile, 'ssim', self.ssim)
        setattr(ncfile,'time since ignition', self.time_igni)
        setattr(ncfile,'date', self.time_date.strftime("%Y-%m-%d %H:%M:%S"))
        setattr(ncfile,'shrink_factor', self.shrink_factor)
        setattr(ncfile,'cfMode', self.cfMode)
        setattr(ncfile,'inRefList', self.inRefList)
        setattr(ncfile,'blurred', int(self.blurred))
        setattr(ncfile,'clahe_clipLimit', self.clahe_clipLimit)
        setattr(ncfile,'kernel_plot', self.kernel_plot)
        setattr(ncfile,'kernel_warp', self.kernel_warp)
        setattr(ncfile,'grid_shape', self.grid_shape)
        setattr(ncfile,'lowT_param', self.lowT_param)


        setattr(ncfile,'img width',  self.ni)
        setattr(ncfile,'img height', self.nj)
        if 'id_best_ref'   in self.__dict__         : setattr(ncfile,'id_best_ref', self.id_best_ref)
        if 'number_ref_frames_used' in self.__dict__: setattr(ncfile,'nbre reference frames used', self.number_ref_frames_used)
        if 'rvec' in self.__dict__                  : setattr(ncfile,'rvec', self.rvec.flatten())
        if 'tvec' in self.__dict__                  : setattr(ncfile,'tvec', self.tvec.flatten())
        if 'K_raw'in self.__dict__                  : setattr(ncfile,'K_raw', self.K_raw.flatten())
        if 'K_undistorted'in self.__dict__          : setattr(ncfile,'K_undistorted'       , self.K_undistorted.flatten())
        if 'K_undistorted_imgRes'in self.__dict__   : setattr(ncfile,'K_undistorted_imgRes', self.K_undistorted_imgRes.flatten())
        if 'trange'in self.__dict__                 : setattr(ncfile,'trange', self.trange)
        
        # dimensions
        ncfile.createDimension('imgi',self.img.shape[0])
        ncfile.createDimension('imgj',self.img.shape[1])
        ncfile.createDimension('FeatureNbre',None)
        ncfile.createDimension('FeatureDim',2)
        ncfile.createDimension('CoordDim',3)
        ncfile.createDimension('FeatureDimHist',53)
        ncfile.createDimension('mtxi',3)
        ncfile.createDimension('mtxj',3)


        # set dimension
        ncimgi = ncfile.createVariable('i', 'f8', ('imgi',))
        setattr(ncimgi, 'long_name', 'image width + buffer zone')
        setattr(ncimgi, 'standard_name', 'imgi')
        setattr(ncimgi, 'units','-')

        ncimgj = ncfile.createVariable('j', 'f8', ('imgj',))
        setattr(ncimgj, 'long_name', 'image height + buffer zone')
        setattr(ncimgj, 'standard_name', 'imgj')
        setattr(ncimgj, 'units','-')
       
        ncfeatnbre = ncfile.createVariable('FeatureNbre', 'f8', ('FeatureNbre',))
        setattr(ncfeatnbre, 'long_name', 'number of feature')
        setattr(ncfeatnbre, 'standard_name', 'number of feature')
        setattr(ncfeatnbre, 'units','-')
        
        ncfeatdim = ncfile.createVariable('FeatureDim', 'f8', ('FeatureDim',))
        setattr(ncfeatdim, 'long_name', 'feature dimension')
        setattr(ncfeatdim, 'standard_name', 'feature dimension')
        setattr(ncfeatdim, 'units','-')

        ncmtxi = ncfile.createVariable('mtxi', 'f8', ('mtxi',))
        setattr(ncmtxi, 'long_name', 'image height + buffer zone')
        setattr(ncmtxi, 'standard_name', 'mtxi')
        setattr(ncmtxi, 'units','-')
        
        ncmtxj = ncfile.createVariable('mtxj', 'f8', ('mtxi',))
        setattr(ncmtxj, 'long_name', 'image height + buffer zone')
        setattr(ncmtxj, 'standard_name', 'mtxj')
        setattr(ncmtxj, 'units','-')
       

        # set Variables
        ncimg    = ncfile.createVariable('img','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncimg, 'long_name', 'gray image used to track feature') 
        setattr(ncimg, 'standard_name', 'img') 
        setattr(ncimg, 'units', '-') 
        
        ncmask_img    = ncfile.createVariable('mask_img','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncmask_img, 'long_name', 'mask for img pixel which are outside the helico leg or the bufferzone') 
        setattr(ncmask_img, 'standard_name', 'mask_img') 
        setattr(ncmask_img, 'units', '-') 
        
        nctemp    = ncfile.createVariable('temp','float32', (u'imgi',u'imgj',), fill_value=0.)
        setattr(nctemp, 'long_name', 'undistorted temperature') 
        setattr(nctemp, 'standard_name', 'temp') 
        setattr(nctemp, 'units', 'C') 
        
        ncwarp    = ncfile.createVariable('warp','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncwarp, 'long_name', 'warp of gray image on ref frame') 
        setattr(ncwarp, 'standard_name', 'warp') 
        setattr(ncwarp, 'units', '-') 

        ncmask_warp    = ncfile.createVariable('mask_warp','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncmask_warp, 'long_name', 'mask of the warp img') 
        setattr(ncmask_warp, 'standard_name', 'mask_warp') 
        setattr(ncmask_warp, 'units', '-') 

        ncplotMask    = ncfile.createVariable('plotMask','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncplotMask, 'long_name', 'plot Mask with biffer zone') 
        setattr(ncplotMask, 'standard_name', 'plotMask') 
        setattr(ncplotMask, 'units', '-') 
        
        ncplotMask_ring    = ncfile.createVariable('plotMask_ring','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncplotMask_ring, 'long_name', 'plot Mask with biffer zone') 
        setattr(ncplotMask_ring, 'standard_name', 'plotMask_ring') 
        setattr(ncplotMask_ring, 'units', '-') 
        
        ncbareGroundMask    = ncfile.createVariable('bareGroundMask','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncbareGroundMask, 'long_name', 'bare ground mask with biffer zone') 
        setattr(ncbareGroundMask, 'standard_name', 'bareGroundMask') 
        setattr(ncbareGroundMask, 'units', '-') 

        ncssim    = ncfile.createVariable('ssim','float32', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncssim, 'long_name', 'structural similarity between img and ref') 
        setattr(ncssim,'standard_name', 'ssim') 
        setattr(ncssim, 'units', '-') 
        
        ncmask_ssim    = ncfile.createVariable('mask_ssim','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncmask_ssim, 'long_name', 'mask of the structural similarity between img and ref') 
        setattr(ncmask_ssim, 'standard_name', 'mask_ssim') 
        setattr(ncmask_ssim, 'units', '-') 

        ncH2Ref    = ncfile.createVariable('H2Ref','float32', (u'mtxi',u'mtxj',), fill_value=None)
        setattr(ncH2Ref, 'long_name', 'homography matrix to Reference image') 
        setattr(ncH2Ref, 'standard_name', 'H2Ref') 
        setattr(ncH2Ref, 'units', '-') 
        
        ncH2Grid    = ncfile.createVariable('H2Grid','float32', (u'mtxi',u'mtxj',), fill_value=None)
        setattr(ncH2Grid, 'long_name', 'homography matrix to Grid') 
        setattr(ncH2Grid, 'standard_name', 'H2Ref') 
        setattr(ncH2Grid, 'units', '-') 

        ncfeat0 = ncfile.createVariable('features_img','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
        setattr(ncfeat0, 'long_name', 'original feature selected on img') 
        setattr(ncfeat0, 'standard_name', 'original features') 
        setattr(ncfeat0, 'units', '-') 

        ncfeat1 = ncfile.createVariable('good_features_img','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
        setattr(ncfeat1, 'long_name', 'selected good features on img from all ref images') 
        setattr(ncfeat1, 'standard_name', 'good features 1') 
        setattr(ncfeat1, 'units', '-') 
        
        ncfeat2 = ncfile.createVariable('good_features_4plot','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
        setattr(ncfeat2, 'long_name', 'selected good features on img ') 
        setattr(ncfeat2, 'standard_name', 'good features 2') 
        setattr(ncfeat2, 'units', '-') 

        try: 
            if self.good_old is not None:
                ncfeat3 = ncfile.createVariable('good_features_ref','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
                setattr(ncfeat3, 'long_name', 'selected good features on all ref images ') 
                setattr(ncfeat3, 'standard_name', 'good features 3') 
                setattr(ncfeat3, 'units', '-') 
        except: 
            pass
        
        #Wt
        try: 
            if self.gcps_cam is not None:
                ncfeat_gcpscam = ncfile.createVariable('gcps_cam','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
                setattr(ncfeat_gcpscam, 'long_name', 'gcps on cam to use in Wt') 
                setattr(ncfeat_gcpscam, 'standard_name', 'gcps_cam') 
                setattr(ncfeat_gcpscam, 'units', '-') 
                
                ncfeat_gcpsW = ncfile.createVariable('gcps_world','float32', (u'FeatureNbre',u'CoordDim',), fill_value=0.)
                setattr(ncfeat_gcpsW, 'long_name', 'gcps on cam to use in Wt') 
                setattr(ncfeat_gcpsW, 'standard_name', 'gcps_world') 
                setattr(ncfeat_gcpsW, 'units', '-') 
        except: 
            pass
        
        
        if 'cf_on_img'      in self.__dict__:
            nc_cf_loc = ncfile.createVariable('cf_loc','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
            setattr(nc_cf_loc, 'long_name', 'cf on img with buffer zone') 
            setattr(nc_cf_loc, 'standard_name', 'cf_on_img') 
            setattr(nc_cf_loc, 'units', '-')

            nc_cf_hist = ncfile.createVariable('cf_hist','float32', (u'FeatureNbre',u'FeatureDimHist',), fill_value=0.)
            setattr(nc_cf_hist, 'long_name', 'histogram of cf') 
            setattr(nc_cf_hist, 'standard_name', 'cf_hist') 
            setattr(nc_cf_hist, 'units', '-')

        #write grid
        ncimgi[:]    = np.arange(self.img.shape[0])
        ncimgj[:]    = np.arange(self.img.shape[1])
        ncmtxi[:]    = np.arange(3)
        ncmtxj[:]    = np.arange(3)
        ncfeatdim[:] = np.arange(2)


        #write data  
        if 'img'                 in self.__dict__: ncimg[:,:]        = self.img
        if 'mask_img'            in self.__dict__: ncmask_img[:,:]   = self.mask_img
        if 'temp'                in self.__dict__: nctemp[:,:]        = self.temp
        if 'warp'                in self.__dict__: ncwarp[:,:]       = self.warp
        if 'mask_warp'           in self.__dict__: ncmask_warp[:,:]  = self.mask_warp
        if 'plotMask_withBuffer' in self.__dict__: ncplotMask[:,:]   = self.plotMask_withBuffer
        if 'plotMask_withBuffer_ring' in self.__dict__: ncplotMask_ring[:,:]   = self.plotMask_withBuffer_ring
        if 'bareGroundMask_withBuffer' in self.__dict__: ncbareGroundMask[:,:]   = self.bareGroundMask_withBuffer
        if ('ssim_2d'             in self.__dict__) :
            if (self.ssim_2d is not None)  : ncssim[:,:]       = self.ssim_2d
        if ('mask_ssim'           in self.__dict__) :
            if (self.mask_ssim is not None): ncmask_ssim[:,:]  = self.mask_ssim


        if 'H2Ref'   in self.__dict__: ncH2Ref[:,:]  = self.H2Ref
        if 'H2Grid'  in self.__dict__: ncH2Grid[:,:] = self.H2Grid
        
        if 'feature'        in self.__dict__: ncfeat0[:,:]  = self.feature
        if 'good_new'       in self.__dict__: ncfeat1[:,:] = self.good_new
        try: 
            if 'good_old'       in self.__dict__: ncfeat3[:,:] = self.good_old
        except: 
            pass
        if 'good_new_4plot' in self.__dict__: ncfeat2[:,:] = self.good_new_4plot
        try: 
            if 'gcps_cam'       in self.__dict__: ncfeat_gcpscam[:,:] = self.gcps_cam
            if 'gcps_world'     in self.__dict__: ncfeat_gcpsw[:,:]   = self.gcps_world
        except: 
            pass
        
        if 'cf_on_img'      in self.__dict__: nc_cf_loc[:,:] = self.cf_on_img 
        if 'cf_hist'        in self.__dict__: nc_cf_hist[:,:] = self.cf_hist

        #close file
        ncfile.close()

        return 0


    def loadFromFile2(self,filename):
        ncfile = Dataset(filename,'r')

        if self.type != ncfile.getncattr('type') : 
            print('error when load file', filename)
            sys.exit()
        self.id        = ncfile.getncattr('id')
        self.corr_ref        = ncfile.getncattr('correlation ref')
        self.corr_ref00      = ncfile.getncattr('correlation ref00')
        self.corr_ref00_init = ncfile.getncattr('correlation ref00 init')
        if 'ssim'            in ncfile.ncattrs():    self.ssim = ncfile.getncattr('ssim')
        self.time_igni = ncfile.getncattr('time since ignition')
        self.time_date = datetime.datetime.strptime(ncfile.getncattr('date'),"%Y-%m-%d %H:%M:%S")
        self.ni        = ncfile.getncattr('img width')
        self.nj        = ncfile.getncattr('img height')
        self.shrink_factor = ncfile.getncattr('shrink_factor')
        self.cfMode        = ncfile.getncattr('cfMode')
        self.inRefList     = ncfile.getncattr('inRefList')
        self.id_ref00      = ncfile.getncattr('id_ref00')
        self.blurred       = bool(ncfile.getncattr('blurred'))
        if 'clahe_clipLimit' in ncfile.ncattrs(): self.clahe_clipLimit = ncfile.getncattr('clahe_clipLimit')
        if 'kernel_plot'     in ncfile.ncattrs(): self.kernel_plot = ncfile.getncattr('kernel_plot')
        if 'kernel_warp'     in ncfile.ncattrs(): self.kernel_warp = ncfile.getncattr('kernel_warp')
        if 'grid_shape'     in ncfile.ncattrs(): self.grid_shape = tuple(ncfile.getncattr('grid_shape'))
        if 'lowT_param'     in ncfile.ncattrs(): self.lowT_param = tuple(ncfile.getncattr('lowT_param'))


    def loadFromFile(self,filename):
        ncfile = Dataset(filename,'r')

        if self.type != ncfile.getncattr('type') : 
            print('error when load file', filename)
            sys.exit()
        self.id        = ncfile.getncattr('id')
        self.corr_ref        = ncfile.getncattr('correlation ref')
        self.corr_ref00      = ncfile.getncattr('correlation ref00')
        self.corr_ref00_init = ncfile.getncattr('correlation ref00 init')
        if 'ssim'            in ncfile.ncattrs():    self.ssim = ncfile.getncattr('ssim')
        self.time_igni = ncfile.getncattr('time since ignition')
        self.time_date = datetime.datetime.strptime(ncfile.getncattr('date'),"%Y-%m-%d %H:%M:%S")
        self.ni        = ncfile.getncattr('img width')
        self.nj        = ncfile.getncattr('img height')
        self.shrink_factor = ncfile.getncattr('shrink_factor')
        self.cfMode        = ncfile.getncattr('cfMode')
        self.inRefList     = ncfile.getncattr('inRefList')
        self.id_ref00      = ncfile.getncattr('id_ref00')
        self.blurred       = bool(ncfile.getncattr('blurred'))
        if 'clahe_clipLimit'            in ncfile.ncattrs(): self.clahe_clipLimit = ncfile.getncattr('clahe_clipLimit')
        if 'kernel_plot'     in ncfile.ncattrs(): self.kernel_plot = ncfile.getncattr('kernel_plot')
        if 'kernel_warp'     in ncfile.ncattrs(): self.kernel_warp = ncfile.getncattr('kernel_warp')
        if 'grid_shape'     in ncfile.ncattrs(): self.grid_shape = tuple(ncfile.getncattr('grid_shape'))
        if 'lowT_param'     in ncfile.ncattrs(): self.lowT_param = tuple(ncfile.getncattr('lowT_param'))
  
        if 'id_best_ref'            in ncfile.ncattrs(): self.id_best_ref = ncfile.getncattr('id_best_ref')
        if 'nbre reference frames used' in ncfile.ncattrs(): self.number_ref_frames_used = ncfile.getncattr('nbre reference frames used')
        if 'rvec' in ncfile.ncattrs(): self.rvec = ncfile.getncattr('rvec')
        if 'tvec' in ncfile.ncattrs(): self.tvec = np.matrix(ncfile.getncattr('tvec')).T
        if 'K_raw'in ncfile.ncattrs():                 self.K_raw                =  ncfile.getncattr('K_raw').reshape([3,3])
        if 'K_undistorted' in ncfile.ncattrs():        self.K_undistorted        = ncfile.getncattr('K_undistorted').reshape([3,3])
        if 'K_undistorted_imgRes' in ncfile.ncattrs(): self.K_undistorted_imgRes = ncfile.getncattr('K_undistorted_imgRes').reshape([3,3])
        if 'trange' in ncfile.ncattrs():               self.trange = ncfile.getncattr('trange')

        self.img                 = np.ma.filled(ncfile.variables['img'][:,:])
        self.warp                = np.ma.filled(ncfile.variables['warp'][:,:])
        self.mask_warp           = np.ma.filled(ncfile.variables['mask_warp'][:,:])
        self.plotMask_withBuffer = np.ma.filled(ncfile.variables['plotMask'][:,:])
        self.plotMask_withBuffer_ring = np.ma.filled(ncfile.variables['plotMask_ring'][:,:])
        self.bareGroundMask_withBuffer = np.ma.filled(ncfile.variables['bareGroundMask'][:,:])
        if 'mask_img'  in ncfile.variables : self.mask_img         = np.ma.filled(ncfile.variables['mask_img'][:,:])
        if 'temp'      in ncfile.variables : self.temp             = np.ma.filled(ncfile.variables['temp'][:,:])
        if 'ssim'      in ncfile.variables : self.ssim_2d          = np.ma.filled(ncfile.variables['ssim'][:,:])
        if 'mask_ssim' in ncfile.variables : self.mask_ssim        = np.ma.filled(ncfile.variables['mask_ssim'][:,:])


        if 'H2Ref'  in ncfile.variables: self.H2Ref  = np.ma.filled(ncfile.variables['H2Ref'][:,:])
        if 'H2Grid' in ncfile.variables: self.H2Grid = np.ma.filled(ncfile.variables['H2Grid'][:,:])
       
        if 'features_img'        in ncfile.variables: self.feature        = np.ma.filled(ncfile.variables['features_img'])
        if 'good_features_img'   in ncfile.variables: self.good_new       = np.ma.filled(ncfile.variables['good_features_img'])
        if 'good_features_4plot' in ncfile.variables: self.good_new_4plot = np.ma.filled(ncfile.variables['good_features_4plot'])
        if 'good_features_ref'   in ncfile.variables: self.good_old       = np.ma.filled(ncfile.variables['good_features_ref'])

        if 'gcps_cam'            in ncfile.variables: self.gcps_cam       = np.ma.filled(ncfile.variables['gcps_cam'])
        if 'gcps_world'            in ncfile.variables: self.gcps_world   = np.ma.filled(ncfile.variables['gcps_world'])

        if 'cf_loc'              in ncfile.variables: self.cf_on_img      = np.ma.filled(ncfile.variables['cf_loc'])
        if 'cf_hist'             in ncfile.variables: self.cf_hist        = np.ma.filled(ncfile.variables['cf_hist'])
        
        if 'cf_loc'              in ncfile.variables:
            idx = np.where( (self.cf_on_img[:,0]*self.cf_on_img[:,1] == 0) & (self.cf_on_img[:,0]+self.cf_on_img[:,1] == 0) )[0].min()
            self.cf_on_img = self.cf_on_img[:idx,:]    
            self.cf_hist = self.cf_hist[:idx,:]    

        ncfile.close()

    '''
    def dump(self,filename):
        ncfile = Dataset(filename,'w')
    
        ncfile.description = 'lwir frame generated by GeoRefCam'
   
        # Global attributes
        setattr(ncfile, 'created', 'R. Paugam') 
        setattr(ncfile, 'title', 'lwir frame')
        setattr(ncfile, 'Conventions', 'CF')

        setattr(ncfile,'type', self.type)
        setattr(ncfile,'id', self.id)
        setattr(ncfile,'id_ref00', self.id_ref00)
        setattr(ncfile, 'energy', self.energy)
        setattr(ncfile,'time since ignition', self.time_igni)
        setattr(ncfile,'date', self.time_date.strftime("%Y-%m-%d %H:%M:%S"))
        setattr(ncfile,'shrink_factor', self.shrink_factor)
        setattr(ncfile,'cfMode', self.cfMode)
        setattr(ncfile,'inRefList', self.inRefList)
        setattr(ncfile,'blurred', int(self.blurred))


        setattr(ncfile,'img width',  self.ni)
        setattr(ncfile,'img height', self.nj)
        if 'id_best_ref'   in self.__dict__         : setattr(ncfile,'id_best_ref', self.id_best_ref)
        if 'number_ref_frames_used' in self.__dict__: setattr(ncfile,'nbre reference frames used', self.number_ref_frames_used)
        if 'rvec' in self.__dict__                  : setattr(ncfile,'rvec', self.tvec.flatten())
        if 'tvec' in self.__dict__                  : setattr(ncfile,'tvec', self.tvec.flatten())
        if 'K_raw'in self.__dict__                  : setattr(ncfile,'K_raw', self.K_raw.flatten())
        if 'K_undistorted'in self.__dict__          : setattr(ncfile,'K_undistorted'       , self.K_undistorted.flatten())
        if 'K_undistorted_imgRes'in self.__dict__   : setattr(ncfile,'K_undistorted_imgRes', self.K_undistorted_imgRes.flatten())
        if 'trange'in self.__dict__                 : setattr(ncfile,'trange', self.trange)
        
        # dimensions
        ncfile.createDimension('imgi',self.img.shape[0])
        ncfile.createDimension('imgj',self.img.shape[1])
        ncfile.createDimension('FeatureNbre',None)
        ncfile.createDimension('FeatureDim',2)
        ncfile.createDimension('FeatureDimHist',53)
        ncfile.createDimension('mtxi',3)
        ncfile.createDimension('mtxj',3)


        # set dimension
        ncimgi = ncfile.createVariable('i', 'f8', ('imgi',))
        setattr(ncimgi, 'long_name', 'image width + buffer zone')
        setattr(ncimgi, 'standard_name', 'imgi')
        setattr(ncimgi, 'units','-')

        ncimgj = ncfile.createVariable('j', 'f8', ('imgj',))
        setattr(ncimgj, 'long_name', 'image height + buffer zone')
        setattr(ncimgj, 'standard_name', 'imgj')
        setattr(ncimgj, 'units','-')
       
        ncfeatnbre = ncfile.createVariable('FeatureNbre', 'f8', ('FeatureNbre',))
        setattr(ncfeatnbre, 'long_name', 'number of feature')
        setattr(ncfeatnbre, 'standard_name', 'number of feature')
        setattr(ncfeatnbre, 'units','-')
        
        ncfeatdim = ncfile.createVariable('FeatureDim', 'f8', ('FeatureDim',))
        setattr(ncfeatdim, 'long_name', 'feature dimension')
        setattr(ncfeatdim, 'standard_name', 'feature dimension')
        setattr(ncfeatdim, 'units','-')

        ncmtxi = ncfile.createVariable('mtxi', 'f8', ('mtxi',))
        setattr(ncmtxi, 'long_name', 'image height + buffer zone')
        setattr(ncmtxi, 'standard_name', 'mtxi')
        setattr(ncmtxi, 'units','-')
        
        ncmtxj = ncfile.createVariable('mtxj', 'f8', ('mtxi',))
        setattr(ncmtxj, 'long_name', 'image height + buffer zone')
        setattr(ncmtxj, 'standard_name', 'mtxj')
        setattr(ncmtxj, 'units','-')
       

        # set Variables
        ncimg    = ncfile.createVariable('img','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncimg, 'long_name', 'gray image used to track feature') 
        setattr(ncimg, 'standard_name', 'img') 
        setattr(ncimg, 'units', '-') 
        
        ncmask_img    = ncfile.createVariable('mask_img','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncmask_img, 'long_name', 'mask for img pixel which are outside the helico leg or the bufferzone') 
        setattr(ncmask_img, 'standard_name', 'mask_img') 
        setattr(ncmask_img, 'units', '-') 
        
        nctemp    = ncfile.createVariable('temp','float32', (u'imgi',u'imgj',), fill_value=0.)
        setattr(nctemp, 'long_name', 'undistorted temperature') 
        setattr(nctemp, 'standard_name', 'temp') 
        setattr(nctemp, 'units', 'C') 
        
        ncwarp    = ncfile.createVariable('warp','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncwarp, 'long_name', 'warp of gray image on ref frame') 
        setattr(ncwarp, 'standard_name', 'warp') 
        setattr(ncwarp, 'units', '-') 

        ncmask_warp    = ncfile.createVariable('mask_warp','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncmask_warp, 'long_name', 'mask of the warp img') 
        setattr(ncmask_warp, 'standard_name', 'mask_warp') 
        setattr(ncmask_warp, 'units', '-') 

        ncplotMask    = ncfile.createVariable('plotMask','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncplotMask, 'long_name', 'plot Mask with biffer zone') 
        setattr(ncplotMask, 'standard_name', 'plotMask') 
        setattr(ncplotMask, 'units', '-') 
        
        ncbareGroundMask    = ncfile.createVariable('bareGroundMask','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncbareGroundMask, 'long_name', 'bare ground mask with biffer zone') 
        setattr(ncbareGroundMask, 'standard_name', 'bareGroundMask') 
        setattr(ncbareGroundMask, 'units', '-') 

        ncssim    = ncfile.createVariable('ssim','float32', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncssim, 'long_name', 'structural similarity between img and ref') 
        setattr(ncssim,'standard_name', 'ssim') 
        setattr(ncssim, 'units', '-') 
        
        ncmask_ssim    = ncfile.createVariable('mask_ssim','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncmask_ssim, 'long_name', 'mask of the structural similarity between img and ref') 
        setattr(ncmask_ssim, 'standard_name', 'mask_ssim') 
        setattr(ncmask_ssim, 'units', '-') 

        ncH2Ref    = ncfile.createVariable('H2Ref','float32', (u'mtxi',u'mtxi',), fill_value=0.)
        setattr(ncH2Ref, 'long_name', 'homography matrix to Reference image') 
        setattr(ncH2Ref, 'standard_name', 'H2Ref') 
        setattr(ncH2Ref, 'units', '-') 
        
        ncH2Grid    = ncfile.createVariable('H2Grid','float32', (u'mtxi',u'mtxj',), fill_value=0.)
        setattr(ncH2Grid, 'long_name', 'homography matrix to Grid') 
        setattr(ncH2Grid, 'standard_name', 'H2Ref') 
        setattr(ncH2Grid, 'units', '-') 


        ncfeat0 = ncfile.createVariable('features_img','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
        setattr(ncfeat0, 'long_name', 'original feature selected on img') 
        setattr(ncfeat0, 'standard_name', 'original features') 
        setattr(ncfeat0, 'units', '-') 

        ncfeat1 = ncfile.createVariable('good_features_img','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
        setattr(ncfeat1, 'long_name', 'selected good features on img from all ref images') 
        setattr(ncfeat1, 'standard_name', 'good features 1') 
        setattr(ncfeat1, 'units', '-') 
        
        ncfeat2 = ncfile.createVariable('good_features_4plot','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
        setattr(ncfeat2, 'long_name', 'selected good features on img ') 
        setattr(ncfeat2, 'standard_name', 'good features 2') 
        setattr(ncfeat2, 'units', '-') 

        ncfeat3 = ncfile.createVariable('good_features_ref','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
        setattr(ncfeat3, 'long_name', 'selected good features on all ref images ') 
        setattr(ncfeat3, 'standard_name', 'good features 3') 
        setattr(ncfeat3, 'units', '-') 

        nc_cf_loc = ncfile.createVariable('cf_loc','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
        setattr(ncfeat3, 'long_name', 'cf on img with buffer zone') 
        setattr(ncfeat3, 'standard_name', 'cf_on_img') 
        setattr(ncfeat3, 'units', '-')

        nc_cf_hist = ncfile.createVariable('cf_hist','float32', (u'FeatureNbre',u'FeatureDimHist',), fill_value=0.)
        setattr(ncfeat3, 'long_name', 'histogram of cf') 
        setattr(ncfeat3, 'standard_name', 'cf_hist') 
        setattr(ncfeat3, 'units', '-')

        #write grid
        ncimgi[:]    = np.arange(self.img.shape[0])
        ncimgj[:]    = np.arange(self.img.shape[1])
        ncmtxi[:]    = np.arange(3)
        ncmtxj[:]    = np.arange(3)
        ncfeatdim[:] = np.arange(2)


        #write data  
        if 'img'                 in self.__dict__: ncimg[:,:]        = self.img
        if 'mask_img'            in self.__dict__: ncmask_img[:,:]   = self.mask_img
        if 'temp'                in self.__dict__: nctemp[:,:]        = self.temp
        if 'warp'                in self.__dict__: ncwarp[:,:]       = self.warp
        if 'mask_warp'           in self.__dict__: ncmask_warp[:,:]  = self.mask_warp
        if 'plotMask_withBuffer' in self.__dict__: ncplotMask[:,:]   = self.plotMask_withBuffer
        if 'bareGroundMask_withBuffer' in self.__dict__: ncbareGroundMask[:,:]   = self.bareGroundMask_withBuffer
        if 'ssim_2d'             in self.__dict__: ncssim[:,:]       = self.ssim_2d
        if 'mask_ssim'           in self.__dict__: ncmask_ssim[:,:]  = self.mask_ssim


        if 'H2Ref'   in self.__dict__: ncH2Ref[:,:]  = self.H2Ref
        if 'H2Grid'  in self.__dict__: ncH2Grid[:,:] = self.H2Grid
        
        if 'feature'        in self.__dict__: ncfeat0[:,:] = self.feature
        if 'good_new'       in self.__dict__: ncfeat1[:,:] = self.good_new
        if 'good_old'       in self.__dict__: ncfeat3[:,:] = self.good_old
        if 'good_new_4plot' in self.__dict__: ncfeat2[:,:] = self.good_new_4plot
       
        if 'cf_on_img'      in self.__dict__: nc_cf_loc[:,:] = self.cf_on_img 
        if 'cf_hist'        in self.__dict__: nc_cf_hist[:,:] = self.cf_hist

        #close file
        ncfile.close()

        return 0


    def loadFromFile(self,filename):
        ncfile = Dataset(filename,'r')

        if self.type != ncfile.getncattr('type') : 
            print('error when load file', filename)
            sys.exit()
        self.id        = ncfile.getncattr('id')
        self.energy    = ncfile.getncattr('energy')
        self.time_igni = ncfile.getncattr('time since ignition')
        self.time_date = datetime.datetime.strptime(ncfile.getncattr('date'),"%Y-%m-%d %H:%M:%S")
        self.ni        = ncfile.getncattr('img width')
        self.nj        = ncfile.getncattr('img height')
        self.shrink_factor = ncfile.getncattr('shrink_factor')
        self.cfMode        = ncfile.getncattr('cfMode')
        self.inRefList     = ncfile.getncattr('inRefList')
        self.blurred     = bool(ncfile.getncattr('blurred'))
        self.id_ref00  = ncfile.getncattr('id_ref00')

        if 'id_best_ref'            in ncfile.ncattrs(): self.id_best_ref = ncfile.getncattr('id_best_ref')
        if 'nbre reference frames used' in ncfile.ncattrs(): self.number_ref_frames_used = ncfile.getncattr('nbre reference frames used')
        if 'rvec' in ncfile.ncattrs(): self.rvec = ncfile.getncattr('rvec')
        if 'tvec' in ncfile.ncattrs(): self.tvec = ncfile.getncattr('tvec')
        if 'K_raw'in ncfile.ncattrs():                 self.K_raw                =  ncfile.getncattr('K_raw').reshape([3,3])
        if 'K_undistorted' in ncfile.ncattrs():        self.K_undistorted        = ncfile.getncattr('K_undistorted').reshape([3,3])
        if 'K_undistorted_imgRes' in ncfile.ncattrs(): self.K_undistorted_imgRes = ncfile.getncattr('K_undistorted_imgRes').reshape([3,3])
        if 'trange' in ncfile.ncattrs():               self.trange = ncfile.getncattr('trange')

        self.img                 = np.ma.filled(ncfile.variables['img'][:,:])
        self.warp                = np.ma.filled(ncfile.variables['warp'][:,:])
        self.mask_warp           = np.ma.filled(ncfile.variables['mask_warp'][:,:])
        self.plotMask_withBuffer = np.ma.filled(ncfile.variables['plotMask'][:,:])
        self.bareGroundMask_withBuffer = np.ma.filled(ncfile.variables['bareGroundMask'][:,:])
        if 'mask_img'  in ncfile.variables : self.mask_img         = np.ma.filled(ncfile.variables['mask_img'][:,:])
        if 'temp'      in ncfile.variables : self.temp             = np.ma.filled(ncfile.variables['temp'][:,:])
        if 'ssim'      in ncfile.variables : self.ssim_2d          = np.ma.filled(ncfile.variables['ssim'][:,:])
        if 'mask_ssim' in ncfile.variables : self.mask_ssim        = np.ma.filled(ncfile.variables['mask_ssim'][:,:])


        if 'H2Ref'  in ncfile.variables: self.H2Ref  = np.ma.filled(ncfile.variables['H2Ref'][:,:])
        if 'H2Grid' in ncfile.variables: self.H2Grid = np.ma.filled(ncfile.variables['H2Grid'][:,:])
        
        if 'features_img'        in ncfile.variables: self.feature        = np.ma.filled(ncfile.variables['features_img'])
        if 'good_features_img'   in ncfile.variables: self.good_new       = np.ma.filled(ncfile.variables['good_features_img'])
        if 'good_features_4plot' in ncfile.variables: self.good_new_4plot = np.ma.filled(ncfile.variables['good_features_4plot'])
        if 'good_features_ref'   in ncfile.variables: self.good_old       = np.ma.filled(ncfile.variables['good_features_ref'])
        
        if 'cf_loc'              in ncfile.variables: self.cf_on_img      = np.ma.filled(ncfile.variables['cf_loc'])
        if 'cf_hist'             in ncfile.variables: self.cf_hist        = np.ma.filled(ncfile.variables['cf_hist'])

        ncfile.close()
    '''

    def set_flag_cfMode(self, input):
        self.cfMode = input


    def set_flag_inRefList(self, input):
        self.inRefList = input

    def set_cf_on_img(self, loc, hist):
        self.cf_on_img = loc
        self.cf_hist = hist


    def optimize_homography(self, params_georef, params_camera, frame_ref00, frame_ref00_init, 
                            win_size_ssim, flag='firstCall', frame_ref=None ):
      
        frame_ = self.copy()
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1.e-3)
        mask_func_param1 = self.lowT_param  #params_camera['lowT_param'] #[150,3]
        mask_func_param2 = self.lowT_param  #params_camera['lowT_param'] #[150,3] # lowT, kernel_lowT
        mask_func_param3 = [1.e6, 0]
        mask_func_param4 = self.lowT_param  #params_camera['lowT_param'] #[150, 3]

        if flag == 'firstCall':
            flag_opt  = 1
            frame_ref_selected = frame_ref00
            mask_func = tools.mask_EP08
            mask_func_param = mask_func_param1
            trans_len_limit = [40,40]
            #corr_to_compare_with = frame_.corr_ref00

        elif flag == 'refine': 
            flag_opt       = 2
            frame_ref_selected = frame_ref
            mask_func       = tools.mask_lowT
            mask_func_param = mask_func_param2
            trans_len_limit = [40,40]
            #corr_to_compare_with = frame_.corr_ref00

        elif flag == 'coarse': 
            flag_opt       = 3
            frame_ref_selected = frame_ref
            mask_func = tools.mask_onlyImageMask
            mask_func_param = mask_func_param3
            trans_len_limit = [50,40]
            #corr_to_compare_with = None 
        
        elif flag == 'final': 
            flag_opt       = 4
            frame_ref_selected = frame_ref
            mask_func       = tools.mask_EP08
            mask_func_param = mask_func_param4
            trans_len_limit = [10,10]
            #corr_to_compare_with = frame_.corr_ref00_init
            #self_energy = tools.star_get_costFunction(['EP08', frame_, frame_ref00])# here frame_ref00 is the initial frame
        
        else: 
            print('bad flag in optimize_homography')
            pdb.set_trace()
       

        #print 
        #print '--'
        #print corr_to_compare_with
        #call  ECC
        #############
        #try:
        if flag == 'coarse':
            id_ecc, warp_matrix_frame2ref = tools.findTransformECC_on_prev_frame(flag, 
                                                                                 self, frame_ref_selected, 
                                                                                 trans_len_limit = trans_len_limit, 
                                                                                 ep08_limit      = [.7,params_camera['energy_good_2']],
                                                                                 mask_func=mask_func,
                                                                                 mask_func_param=mask_func_param3)
            #warp_matrix_frame2ref = frame_ref.H2Ref.dot(warp_matrix_frame2ref)
            frame_.set_homography_to_ref( warp_matrix_frame2ref )
            frame_.set_warp(    cv2.warpPerspective (frame_.img,      frame_.H2Ref, frame_.warp.shape[::-1], flags=cv2.INTER_LINEAR ))
            frame_.set_maskWarp(cv2.warpPerspective (frame_.mask_img, frame_.H2Ref, frame_.warp.shape[::-1], flags=cv2.INTER_NEAREST)) 
            if win_size_ssim != 0 :  
                mask_ssim, ssim_2d, ssim  = tools.star_get_costFunction(['ssim', frame_, frame_ref00, win_size_ssim ])
                frame_.set_similarity_info(ssim_2d, mask_ssim)
            else: 
                ssim = -999
            frame_.set_correlation(tools.star_get_costFunction(['EP08', frame_, frame_ref_selected, tools.mask_lowT, mask_func_param2]),
                                   tools.star_get_costFunction(['EP08', frame_, frame_ref00,        tools.mask_EP08, mask_func_param1]),
                                   tools.star_get_costFunction(['EP08', frame_, frame_ref00_init,   tools.mask_EP08, mask_func_param4]),
                                   ssim)
            frame_.set_id_best_ref(-1)
            
            if frame_.corr_ref > self.corr_ref: 
                print('*', end=' ')
                self.set_homography_to_ref(frame_.H2Ref)
                self.set_warp(frame_.warp)
                self.set_maskWarp(frame_.mask_warp)
                self.set_correlation(frame_.corr_ref, frame_.corr_ref00, frame_.corr_ref00_init, frame_.ssim)
                if win_size_ssim != 0: self.set_similarity_info(frame_.ssim_2d, frame_.mask_ssim)
                self.set_id_best_ref(frame_.id_best_ref)

            #plt.imshow(img_ref.T,origin='lower')
            #plt.imshow(img.T,origin='lower',cmap=mpl.cm.Greys_r,alpha=.5); plt.show()   
            #plt.imshow(frame_ref.warp.T,origin='lower')
            #plt.imshow(self.warp.T,origin='lower',cmap=mpl.cm.Greys_r,alpha=.5); plt.show()
            #pdb.set_trace()
            print('eopt{:1d} cc{:1d} '.format(flag_opt, id_ecc), self.id_best_ref, end=' ') 
            return 
      
        id_ecc, warp_matrix_frame2ref = tools.findTransformECC_on_ref_frame(flag, 
                                                                            self, frame_ref_selected, 
                                                                            trans_len_limit = trans_len_limit, 
                                                                            ep08_limit      = [.7,params_camera['energy_good_2']],
                                                                            mask_func=mask_func,
                                                                            mask_func_param=mask_func_param) 

        frame_.set_warp(    cv2.warpPerspective (frame_.img,      warp_matrix_frame2ref, frame_.warp.shape[::-1], flags=cv2.INTER_LINEAR ))
        frame_.set_maskWarp(cv2.warpPerspective (frame_.mask_img, warp_matrix_frame2ref, frame_.warp.shape[::-1], flags=cv2.INTER_NEAREST)) 
        frame_.set_homography_to_ref( warp_matrix_frame2ref )
        
        if win_size_ssim != 0 :  
            mask_ssim, ssim_2d, ssim  = tools.star_get_costFunction([ 'ssim', frame_, frame_ref00, win_size_ssim ])
            frame_.set_similarity_info(ssim_2d, mask_ssim)
        else:
            ssim = -999 
        frame_.set_correlation(tools.star_get_costFunction(['EP08', frame_, frame_ref, tools.mask_lowT, mask_func_param2]) \
                                                          if ((frame_ref is not None)&(flag!='final')) else None,
                               tools.star_get_costFunction(['EP08', frame_, frame_ref00,      tools.mask_EP08, mask_func_param1]) ,
                               tools.star_get_costFunction(['EP08', frame_, frame_ref00_init, tools.mask_EP08, mask_func_param4]),
                               ssim)

        #if flag=='refine': pdb.set_trace()
        
        if flag == 'firstCall':
            #print frame_.corr_ref00, self.corr_ref00,
            test_val = frame_.corr_ref00 - self.corr_ref00
        elif flag == 'refine':
            #print frame_.corr_ref, 
            test_val = frame_.corr_ref - self.corr_ref
        #elif flag == 'coarse': 
        #     
        elif flag == 'final': 
            if old_div((frame_.corr_ref00 - self.corr_ref00),self.corr_ref00) > params_camera['final_opti_threshold'] : 
                test_val = frame_.corr_ref00_init - self.corr_ref00_init
            else:
                test_val = 0

        else: 
            print('bad flag in optimize_homography')
            pdb.set_trace()

        test_ok = test_val> 0
        
        '''
        print energy, test_val
        frame__temp_warp2 = cv2.warpPerspective (frame__temp_warp,      warp_matrix, frame_.warp.shape[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        a x = plt.subplot(131)
        ax.imshow( (frame__temp_warp2-frame__temp_warp).T,origin='lower',vmin=0,vmax=10)
        ax = plt.subplot(132)
        ax.imshow( (frame_.warp-self.warp).T,origin='lower',vmin=-10,vmax=10)
        ax = plt.subplot(133)
        ax.imshow(inputMask.T,origin='lower')
        plt.show()    
        pdb.set_trace()
        '''
        if test_ok:
            self.set_homography_to_ref(frame_.H2Ref)
            self.set_warp(             frame_.warp)
            self.set_maskWarp(         frame_.mask_warp)
            if frame_.ssim != -999: self.set_similarity_info(  frame_.ssim_2d, frame_.mask_ssim)
            self.set_correlation(frame_.corr_ref,
                                 frame_.corr_ref00, 
                                 frame_.corr_ref00_init, 
                                 frame_.ssim)

        if flag != 'final': 
            print('opt{:1d} ecc{:1d} (d={:6.3f}) '.format( flag_opt, id_ecc, test_val), end=' ') 
        else: 
            return max([test_val,0]), test_val

    
    def getTemp(self,flag,gridShape=None):
        if flag == 'frame':
            return self.temp
        elif flag == 'grid':
            return cv2.warpPerspective(self.temp, self.H2Grid, gridShape[::-1], flags=cv2.INTER_LINEAR) + 273.14
        else: 
            print('bad flag in fct agema.loadFrame.getTemp. stop here')
            sys.exit()


    
#################################################
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad



#################################################
def load_existing_file(params_camera, filename):
    frame = loadFrame(params_camera)
    frame.loadFromFile(filename)
    return frame



#################################################
def return_radiance(frame, srf_file, wavelength_resolution=0.01):
    radiance = spectralTools.conv_temp2Rad(frame.temp[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)], srf_file, wavelength_resolution=wavelength_resolution)
    radiance_withBuffer = np.zeros([frame.ni+frame.bufferZone,frame.nj+frame.bufferZone])
    radiance_withBuffer[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)]=radiance 
    return radiance_withBuffer



#################################################
def get_feature(frame, feature_params):
    
    gray = frame.img
    temp = frame.temp
    
    tmin_bg,tmax_bg = frame.trange

    #MERDE3
    p00 = cv2.goodFeaturesToTrack(gray[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)],
                                  mask = None, **feature_params)
    #dst = cv2.cornerHarris(gray[50:-50,50:-50],2,3,0.04)
    #mm = np.where(dst>0.1*dst.max())
    #p00 = np.zeros([len(mm[0]),1,2],dtype='float32')
    #p00[:,0,0] = 50+mm[1]
    #p00[:,0,1] = 50+mm[0]

    p00[:,0,0] += old_div(frame.bufferZone,2)
    p00[:,0,1] += old_div(frame.bufferZone,2)

    '''
    plt.clf()
    plt.imshow(gray.T,origin='lower',cmap=mpl.cm.Greys_r)
    plt.scatter(p00[:,0,1],p00[:,0,0])
    plt.show()
    pdb.set_trace()
    '''

    #only keep point in good tempreature range
    idx = np.array(p00[:,0,:],dtype=np.int64)
    temp_ = temp[(idx[:,1],idx[:,0])]
    #idx_temp_ok = np.where( (temp_>=tmin_bg) & (temp_<=tmax_bg) )[0]
    idx_temp_ok = np.where( (temp_>=15) & (temp_<=tmax_bg) )[0]
    nbrept_remove_badTemp = p00.shape[0]-len(idx_temp_ok)
    #print 'remove bad temp pts :', p00.shape[0]-len(idx_temp_ok)
    p00 = p00[idx_temp_ok,:,:]

    #remove point to close from the helico legs
    nbrept_remove_helico = 0
    idx_helico = np.where(frame.mask_img==0)
    if len(idx_helico[0]) != 0: 
        tree_neighbour    = scipy.spatial.cKDTree(list(zip(idx_helico[1],idx_helico[0]))) # all point tree
        flag_pt_ok = np.zeros(p00.shape[0])
        for i_pt in range(p00.shape[0]):
            pt = p00[i_pt,0,:]
            d_, inds_ = tree_neighbour.query(pt, k = 3)
            if min(d_) < 5:
                flag_pt_ok[i_pt] = 1

        idx_helico_ok = np.where(flag_pt_ok==0)[0]
        nbrept_remove_helico = p00.shape[0]-len(idx_helico_ok)
        #print 'remove pts near helico legs:',p00.shape[0]-len(idx_helico_ok)
        p00 = p00[idx_helico_ok,:,:]


    return p00, nbrept_remove_badTemp, nbrept_remove_helico



#################################################
def convert_2_uint8(x,trange=None,flag_sqrtScale=True):
    x = np.array(x,dtype=np.float)
    if  trange is None: 
        xmin, xmax = x.min(), x.max()
    else:
        xmin, xmax = trange
    m = xmax-xmin
    p = xmin
    
    if flag_sqrtScale:
        x_01 = img_scale.sqrt( old_div((x-p),m)  , scale_min=0, scale_max=1)
    else: 
        x_01 = old_div((x-p),m)

    return np.array(np.round(x_01*255,0),dtype=np.uint8)



#################################################
def color_clustering(img_in,nbre_color,flag=None):
    #from http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    
    if flag=='gray':
        img = cv2.cvtColor(img_in,cv2.COLOR_GRAY2RGB)
    else:
        img = img_in

    Z = img.reshape((-1,3))
    
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = nbre_color
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    if flag=='gray':    
        res2 = cv2.cvtColor(res2,cv2.COLOR_RGB2GRAY)
   
    return res2


###########################################################33
def get_cluster_from_segment(img_raw,temp_raw,temp_raw_clip=25,sigma=2):

    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
    mm1band = clahe.apply(img_raw)
    mm  = np.repeat(mm1band[:,:,np.newaxis],3,axis=2) 
    segments = slic(mm, n_segments = 1200, sigma = sigma) # MERDE4 
    
    seg_mean = np.zeros(segments.shape)
    seg_var = np.zeros_like(seg_mean)
    for i_seg in np.arange(segments.max()+1):
        idx = np.where(segments==i_seg)
        seg_mean[idx] = temp_raw[idx].mean()
        seg_var[idx] = temp_raw[idx].std()

    nbre_pt_in_mask_arr = []
    temp_threshold_arr = np.arange(seg_mean.min(),seg_mean.max(),1.)
    for seg_mean_threshold in temp_threshold_arr:
        seg_mask = np.where(seg_mean<seg_mean_threshold,np.ones_like(seg_mean),np.zeros_like(seg_mean)) # MERDE4
        
        #s = [[1,1,1], \
        #     [1,1,1], \
        #     [1,1,1]] # for diagonal
        
        #seg_cluster, seg_clusterNbre =  ndimage.label(seg_mask, structure=s )
        
        #for i_cluster in range(seg_clusterNbre):
        #    idx = np.where(seg_cluster==i_cluster+1)
        #    if (len(idx[0])< 3*temp_raw.flatten().shape[0]/1200) | (idx[1].min() != 0): 
        #        seg_mask[idx] = 0
        
        nbre_pt_in_mask_arr.append( old_div(1.*np.where(seg_mask==1)[0].shape[0], seg_mask.size))
   
    xx = temp_threshold_arr
    yy = np.array(nbre_pt_in_mask_arr)
    d1 = old_div(np.diff(yy), np.diff(xx)) 

    i = d1.argmax()
    while( d1[i] >= d1[i-4:i+1].mean() + .3*d1[i-4:i+1].std()):
        #print d1[i],  d1[i-3:i+1].mean() + .3*d1[i-3:i+1].std()
        i -=1 
        if i < 4: break
   
    if i == 3:
        return np.zeros(img_raw.shape)

    seg_mask = np.where(seg_mean<temp_threshold_arr[i],np.ones_like(seg_mean),np.zeros_like(seg_mean)) # MERDE4
    #clean mask
    s = [[0,1,0], \
         [1,1,1], \
         [0,1,0]] # for diagonal
    seg_cluster, seg_clusterNbre =  ndimage.label(seg_mask, structure=s )
    #print seg_clusterNbre,  mm1band.min()
    for i_cluster in range(seg_clusterNbre):
        idx = np.where(seg_cluster == i_cluster+1)
        #print  '  ',i_cluster, 1.*len(idx[0])/seg_mask.size, mm1band[idx].min()
        #remove cluster that do not contain min 
        if mm1band.min() not in mm1band[idx]:
            seg_mask[idx] = 0
        #remove small cluster
        if old_div(1.*len(idx[0]),seg_mask.size) < .005: 
            seg_mask[idx] = 0


    #ax = plt.subplot(121)
    #ax.scatter(xx[:-1], d1)
    
    #ax = plt.subplot(122)
    #ax.imshow(np.ma.masked_where(seg_mask==1,img_raw).T,origin='lower')
    

    '''
    from scipy import interpolate
    xx = np.arange(seg_mean.min(),seg_mean.max(),1)
    yy = np.array(nbre_pt_in_mask)
    spl = interpolate.splrep(xx, yy, s=0)
    d1 = np.diff(yy) / np.diff(xx) 

    z = np.polyfit(xx[:-1], d1, deg=9)
   
    pdb.set_trace()
    plt.clf()
    plt.scatter(xx[:-1], d1)

    plt.plot(xx,np.poly1d(z)(xx) )
    plt.plot(xx,np.poly1d(z).deriv(m=2)(xx) )
    '''
    #plt.show()

    #pdb.set_trace()
    
    #return ndimage.label(seg_mask, structure=s ), seg_mask
    return seg_mask


###########################################################33
def mask_helico_leg(frame_in,\
                    flag=None,filenames=None,):



    if flag == 'test':
        print('')
        print('check parameters for masking helico leg')
        loop_items = filenames
        
        time_date, time_igni, temp_raw = np.load(ii_item)
        print('{:10s} {:6.2f} % {:6.2f} C\r'.format(os.path.basename(ii_item),old_div(100.*ii,len(filenames)),temp_raw.mean()), end=' ') 
        sys.stdout.flush()
        if ii == 0 : mask_all = np.zeros_like(temp_raw)
    else:
        loop_items = [frame_in]
        

    idx_mesh = None
    temp_old = None

    for ii, ii_item in enumerate(loop_items):

        img_raw = ii_item.imgRaw
        temp_raw = ii_item.temp[old_div(frame_in.bufferZone,2):old_div(-frame_in.bufferZone,2),old_div(frame_in.bufferZone,2):old_div(-frame_in.bufferZone,2)]

        #seg_clusterNbre = 1000
        #while (seg_clusterNbre>2): 
            #[seg_cluster, seg_clusterNbre], seg_mask = get_cluster_from_segment(img_raw,temp_raw,sigma=5) 
        seg_mask = get_cluster_from_segment(img_raw,temp_raw,sigma=5) 
        

        img_final = convert_2_uint8(temp_raw)
        img_final[np.where(seg_mask)] = 0

        #leg_cover = 1.*np.where(mask)[0].shape[0]/ mask.flatten().shape[0]
        #if  ((leg_cover>.15) & (np.unique(label_mask).shape[0]>3)) \
        #  | (theta_dist>.2)                                        :
        #    mask = np.zeros_like(mask_line)
        
        if flag == 'test':
            mask_all += seg_mask
            cv2.imshow('out',ndimage.zoom(img_final, 2, order=0).T[::-1])
            cv2.waitKey(100)
         
            #plt.clf()
            #plt.imshow(np.ma.masked_where(seg_mask==1,temp_raw).T,origin='lower'); plt.show()
            #pdb.set_trace()

    if flag == 'test':
        print('done     ')
        print('****')
        print('check that line are in the area where the helico leg is expected')
        print('only last frame is shown while line are for all mask')
        print('if not change the parameter sigma_canny in inputConfig.params_georef')
        print('  lower if two many lines around the plot')
        print('  higher if no line are showing')
        print('')
        print('press enter to continue')
        print('press q to exit and modify parameter')
        plt.clf()
        ax = plt.subplot(111)
        ax.imshow(np.ma.masked_where(mask_all==0,mask_all).T,origin='lower')
        plt.draw()
        plt.pause(0.01)
        if input('>>> ') == 'q':
            sys.exit()
        cv2.destroyWindow('out')
        cv2.waitKey(100)
        return mask_all
    
    else:
        mask_sinle_img = np.zeros_like(ii_item.temp) 
        mask_sinle_img[old_div(frame_in.bufferZone,2):old_div(-frame_in.bufferZone,2),old_div(frame_in.bufferZone,2):old_div(-frame_in.bufferZone,2)] = np.where(seg_mask==1,np.zeros_like(seg_mask),np.ones_like(seg_mask))
        
        #plt.imshow(mask_sinle_img.T,origin='lower'); plt.show()
        #pdb.set_trace()
        return mask_sinle_img



##########################################################
# below is code to read raw flir format and remove noise
##########################################################


##################################################
def read_flir(filename):
    key = os.path.basename(filename).split('.')[0]
    data = io.loadmat(filename)
    frame     = data['{:s}'.format(key)][::-1].T # K
    time_info = data['{:s}_DateTime'.format(key)][0]
    time = datetime.datetime.strptime('{:04.0f}-{:02.0f}-{:02.0f}_{:02.0f}:{:02.0f}:{:02.0f}:{:03.0f}'.format(*time_info),"%Y-%m-%d_%H:%M:%S:%f")
    return time, frame



####################################################
def plot_flir(rad,time_date,time_igni,dir_out,filename):
    
    mpl.rcdefaults()
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['figure.subplot.left'] = 0.02
    mpl.rcParams['figure.subplot.right'] = .93
    mpl.rcParams['figure.subplot.top'] = 1.
    mpl.rcParams['figure.subplot.bottom'] = .0
    mpl.rcParams['figure.subplot.hspace'] = 0.1
    mpl.rcParams['figure.subplot.wspace'] = 0.1

    vmin =  290
    vmax =  500
    fig = plt.figure(2)
    ax = plt.subplot(111)
    im = ax.imshow(rad.T,origin='lower',interpolation='nearest',vmin = vmin, vmax = vmax)
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im ,cax = cbaxes,orientation='vertical')
    cbar.set_label('L (W/m2/sr)')
    ax.set_axis_off()
    ax.set_title(time_date.strftime('%Y-%m-%d_%H:%M:%S:%f') + '   since igni = {:.2f} s'.format(time_igni) )

    tmp = os.path.basename(filename)
    fig.savefig(dir_out+tmp+'.png')
    plt.close(fig)

    return 0



#######################################
def get_field (exif,field) :
    for (k,v) in exif.items():
        #print TAGS.get(k)
        if TAGS.get(k) == field:
            return v



##########################################################
def processRawData(ignitionTime,params, params_camera, flag_restart):

    #input
    ###########
    dir_in       = params['root_data'] + params['root_data_DirLwir']
    dir_out_lwir =  params['root_postproc'] + params_camera['dir_input']
    tools.ensure_dir(dir_out_lwir)
    flag_save_npy  = True
    flag_applyMask = True



    #read time diff
    ############
    #load time difference 
    delta_t_vis_lwir, delta_t_vis_mir = 0,0#cameraTools.get_time_shift_vis_lwir_mir(params)
    delta_t_lwir_mir = delta_t_vis_lwir - delta_t_vis_mir


    if not(flag_restart):
        if os.path.isdir(dir_out_lwir + 'raw_data/'): shutil.rmtree(dir_out_lwir + 'raw_data/')


    dir_out_lwir_png = dir_out_lwir + 'raw_data/png/'
    dir_out_lwir_npy = dir_out_lwir + 'raw_data/npy/'
    tools.ensure_dir(dir_out_lwir_png)
    tools.ensure_dir(dir_out_lwir_npy)
    
    #get index
    filenames_ = sorted(glob.glob(dir_in+'*.MAT'))
    filenames = np.array(len(filenames_)*[('mm',0,0)],dtype=np.dtype([('name','U500'),('idx1',np.int),('idx2',np.int)])) 
    filenames = filenames.view(np.recarray)
    filenames.name = filenames_
    for ifile, filename in enumerate(filenames.name):
        #filenames.idx1[ifile] = int(os.path.basename(filename).split('.')[0].split('_')[0].split('R')[1])
        filenames.idx2[ifile] = int(os.path.basename(filename).split('.')[0].split('_')[1])
    

    out_time =  np.array(len(filenames_)*[('mm',0)],dtype=np.dtype([('name','U500'),('time',np.float)]))
    out_time = out_time.view(np.recarray)

    #save npy
    ############
    if flag_save_npy:
        print('read mat files: ')
        first_image_reached = False
        #MERDE load only 2000 first file
        time_igni_previous = -1.e6
        camera_name = params_camera['camera_name']
        srf_file = '../data_static/Camera/'+camera_name.split('_')[0]+'/SpectralResponseFunction/'+camera_name.split('_')[0]+'.txt'
        wavelength_resolution = 0.01
        for ifile, (filename, _, _) in enumerate(np.sort(filenames,order=['idx1','idx2'])): #[0::int(params['period_lwir'])]) :
            
            if filename == 'mm': continue

            #load_file
            time_date, flir_temp = read_flir(filename)
            time_igni = (time_date-ignitionTime).total_seconds() - delta_t_lwir_mir

            #flir_rad = spectralTools.conv_temp2Rad(flir_temp, srf_file, wavelength_resolution=wavelength_resolution)

            if time_igni < float(params['startTime_lwir']): continue
            if time_igni > float(params['endTime_lwir'])  : break

            if (time_igni - time_igni_previous) <  params['period_lwir']: continue
            
            print('{:40s}  {:6.2f} '.format(os.path.basename(filename), time_igni), end=' ')

            #skip file where img does not change
            if first_image_reached: 
                if (flir_temp_last-flir_temp).sum() == 0:
                    print(' skiped \r', end=' ')
                    continue
                print(' {:4.2f}    '.format((flir_temp_last-flir_temp).mean()), end=' ') 
            print('\r', end=' ')
            sys.stdout.flush() 
            

            #save
            prefix      = params['plotname']
            np.save(dir_out_lwir_npy+prefix+'_{:06d}'.format(ifile+1),np.array([time_date, time_igni, flir_temp],dtype=object))


            #plot
            plot_flir(flir_temp, time_date, time_igni, dir_out_lwir_png, prefix+'_{:06d}'.format(ifile+1))
            
            
            out_time.name[ifile] =  os.path.basename(dir_out_lwir_npy+prefix+'_{:06d}.npy'.format(ifile+1))
            out_time.time[ifile] =  time_igni
            time_igni_previous = time_igni

            flir_temp_last = flir_temp
            first_image_reached = True
    
    print('done processing raw data                                               ')
   
    idx = np.where(out_time.name != 'mm')
    out_time = out_time[idx]
    out_time = out_time.view(np.recarray)
    np.save(dir_out_lwir_npy+'/filename_time',out_time)

    return 0

