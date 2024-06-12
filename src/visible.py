from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys, glob
from PIL import Image
import datetime 
import pdb 
import shutil
import cv2 
import copy
from netCDF4 import Dataset
from skimage.segmentation import slic
from skimage import filters
from scipy import ndimage
import re
import itertools
import scipy 
import importlib 
import imp 

#homebrewed
import camera_tools as cameraTools 
import tools 

################################################
def get_img_fullReso(filename, K, D,):
    colorRaw_      = cv2.imread(filename)
    
    if D.sum()!=0:
        #apply distrotion correction
        h, w  = colorRaw_.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))
        # undistort
        colorRaw_undistorted = cv2.undistort(colorRaw_, K, D, None, newcameramtx)
        # crop the image
        x,y,w,h = roi
        colorRaw_crop = colorRaw_undistorted[y:y+h, x:x+w]
        colorRaw_crop = np.transpose(colorRaw_crop[::-1,:,:],[1,0,2])
    else: 
        colorRaw_crop = colorRaw_
        newcameramtx = K

    return colorRaw_crop


#################################################
class loadFrame(object):

    def __init__(self, params_camera): 
        self.type = 'visible'
        self.corr_ref        = 0
        self.corr_ref00      = 0
        self.corr_ref00_init = 0  
        self.inRefList = 'no'
        self.shrink_factor=params_camera['shrink_factor']
        self.bufferZone = 60 # 200
        self.grayZone   = 50 

    def init(self, id_file, frame_ref00_, filenames, timelookupTable, K, D, inputConfig, grid_shape, feature_params=None, flag_blur=None, image_pdf_reference_tools=None):
        #load image
        self.id        = id_file
        self.time_igni = timelookupTable.time[np.where(timelookupTable.name==os.path.basename(filenames[id_file]))][0]
        self.time_date = timelookupTable.datetime[np.where(timelookupTable.name==os.path.basename(filenames[id_file]))][0]
        colorRaw_      = cv2.imread(filenames[id_file])
      
        self.inputConfig = inputConfig

        self.set_flag_inRefList('no') # default
        self.kernel_plot = inputConfig.params_vis_camera['kernel_plot']
        self.kernel_warp = inputConfig.params_vis_camera['kernel_warp']
        self.grid_shape  = tuple(grid_shape)
      

        if D.sum()!=0:
            #apply distrotion correction
            h, w  = colorRaw_.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))
            # undistort
            colorRaw_undistorted = cv2.undistort(colorRaw_, K, D, None, newcameramtx)
            # crop the image
            x,y,w,h = roi
            colorRaw_crop = colorRaw_undistorted[y:y+h, x:x+w]
            colorRaw_crop = np.transpose(colorRaw_crop[::-1,:,:],[1,0,2])
        else: 
            colorRaw_crop = colorRaw_
            newcameramtx = K

        #save camera matrix
        self.K_raw         = K
        self.K_undistorted = newcameramtx
        if self.shrink_factor > 1: 
            a = 1./self.shrink_factor
            self.K_undistorted_imgRes = np.dot(np.array([[a,0,0],[0,a,0],[0,0,1.]]),newcameramtx)
        else:
            self.K_undistorted_imgRes = newcameramtx
        # D is now 0 as we undistrot the image
        
        if frame_ref00_ is not None:
            self.set_id_ref00(frame_ref00_.id)
            self.set_bareGroundMask(frame_ref00_.bareGroundMask_withBuffer)
        else:
            self.id_ref00 = -1
        
        #downgrade
        if self.shrink_factor > 1: 
            ni_, nj_ = old_div(colorRaw_crop.shape[0],self.shrink_factor), old_div(colorRaw_crop.shape[1],self.shrink_factor) 
            colorRaw = np.zeros([ni_, nj_,3])
            for iband in range(3):
                colorRaw[:,:,iband] = tools.downgrade_resolution_4nadir( colorRaw_crop[:,:,iband],      [ni_,nj_] , flag_interpolation='min' )
            colorRaw = np.array(colorRaw,dtype=np.uint8)
        else:
            colorRaw = colorRaw_crop
        
        self.ni = colorRaw.shape[0]
        self.nj = colorRaw.shape[1]
        
        #set img
        if 'inverseColor' in list(inputConfig.params_vis_camera.keys()):
            self.set_imggray(colorRaw,flag_inverseColor=inputConfig.params_vis_camera['inverseColor'])
        else:
            self.set_imggray(colorRaw)

        #set mask
        if image_pdf_reference_tools is not None :
            self.mask_img = build_mask(self,image_pdf_reference_tools)
        else:
            self.mask_img = np.zeros(self.img.shape)
            if 'fix_mask' in list(inputConfig.params_vis_camera.keys()):
                mask_ = 1./255 * np.array(Image.open(inputConfig.params_vis_camera['fix_mask']))[:,:,0][::-1].T
                self.mask_img[old_div(self.bufferZone,2):old_div(-self.bufferZone,2),old_div(self.bufferZone,2):old_div(-self.bufferZone,2)] = mask_
            else:
                self.mask_img[old_div(self.bufferZone,2):old_div(-self.bufferZone,2),old_div(self.bufferZone,2):old_div(-self.bufferZone,2)] = 1 


        #set feature
        if feature_params is not None:
            self.set_feature(feature_params)
       
        # check for blurry image
        if flag_blur is not None:
            idx = np.where( flag_blur.filename == os.path.basename(filenames[id_file]))
            self.blurred = True if flag_blur.blurred[idx] == 'yes' else False
        else: 
            self.blurred = False


    def set_imggray(self,colorRaw,flag_inverseColor=False):
        grayRaw = np.zeros([self.ni,self.nj],dtype=np.uint8)
        grayRaw = cv2.cvtColor(colorRaw, cv2.COLOR_BGR2GRAY)
        
        #self.img = np.zeros([self.ni+self.bufferZone,self.nj+self.bufferZone],dtype=np.uint8)
        #self.img[self.bufferZone/2:-self.bufferZone/2,self.bufferZone/2:-self.bufferZone/2] = grayRaw
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(11,11))
        self.img = np.zeros([self.ni+self.bufferZone,self.nj+self.bufferZone],dtype=np.uint8)
        self.img[old_div(self.bufferZone,2):old_div(-self.bufferZone,2),old_div(self.bufferZone,2):old_div(-self.bufferZone,2)] = clahe.apply(grayRaw)
        
        if flag_inverseColor: 
            self.img = 255-self.img

    def set_feature(self,feature_params):
        self.feature  = get_feature(self,feature_params)

    def set_warp(self,input_):
        self.warp = input_

    def return_warp(self,trange):
        nx,ny = self.img.shape
        img_warp = cv2.warpPerspective(self.img, self.H2Ref, \
                                       (ny,nx),\
                                       borderValue=0,flags=cv2.INTER_LINEAR)
        return img_warp

    def set_homography_to_ref(self,input_):
        self.H2Ref = input_
    
    def set_homography_to_grid(self,input_):
        self.H2Grid = input_

    def set_pause(self,rvec,tvec):
        self.rvec = rvec
        self.tvec = tvec

    def set_plotMask(self,input1_, input2_):
        self.plotMask_withBuffer = input1_
        self.plotMask_withBuffer_ring = input2_

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

    def copy(self):
        inputConfig_ = imp.new_module('inputConfig') 
        inputConfig_ = self.inputConfig
        delattr(self,'inputConfig')
        out = copy.deepcopy(self)
        out.inputConfig  = inputConfig_
        self.inputConfig = inputConfig_
        return out

    def set_trange(self, trange_):
        self.trange = trange_ # which is None here
    
    def create_backgrdimg(self):
        kernel = np.ones((3,3),np.uint8)
        mask_eroded = cv2.erode(self.mask_img, kernel,iterations = self.grayZone)
        idx = np.where(mask_eroded == 0)
        self.backgrdimg = np.copy(self.img)
        self.backgrdimg[idx] = 0 
        self.mask_backgrdimg = np.ones_like(self.backgrdimg)
        self.mask_backgrdimg[idx] = 0 
        return 'init'
    
    def set_bareGroundMask(self,input_):
        self.bareGroundMask_withBuffer = input_

    def update_backgrdimg(self,frame_):
        
        kernel = np.ones((3,3),np.uint8)
        newframe_mask_eroded = cv2.erode(frame_.mask_warp, kernel,iterations = 1)
       
        idx = np.where( (self.mask_backgrdimg == 0) & (frame_.warp > 0) & (newframe_mask_eroded==1) )
        self.backgrdimg[idx] = frame_.warp[idx]
        self.mask_backgrdimg[idx] = 1

        if np.where(self.mask_backgrdimg==0)[0].shape[0] > 0: 
            return 'not finished'
        else:
            return 'done'
    
    def save_backgrdimg(self, img, mask):
        self.backgrdimg      = img
        self.mask_backgrdimg = mask


    def set_plumeMask(self, input_):
        self.plumeMask = input_


    def dump(self,filename):
        
        if os.path.isfile(filename): 
            os.remove(filename)

        ncfile = Dataset(filename,'w')
        ncfile.description = 'visible frame generated by GeoRefCam'
   
        # Global attributes
        setattr(ncfile, 'created', 'R. Paugam') 
        setattr(ncfile, 'title', 'visible frame')
        setattr(ncfile, 'Conventions', 'CF')

        params = ncfile.createGroup('inputConfig')
        for params_ in list(self.inputConfig.__dict__.keys()):
            if '__' in params_: continue
            params2 = params.createGroup(params_)
            for k,v in list(self.inputConfig.__dict__[params_].items()):
                if type(v) == bool: v = int(v)
                if v == None: v = -999
                if type(k) == str:  k = k.replace('#','nbre_')
                try:
                    setattr(params2, k, v)
                except: 
                    pass
        
        setattr(ncfile,'type', self.type)
        setattr(ncfile,'id', self.id)
        setattr(ncfile,'id_ref00', self.id_ref00)
        setattr(ncfile, 'correlation ref',        self.corr_ref)
        setattr(ncfile, 'correlation ref00',      self.corr_ref00)
        setattr(ncfile, 'correlation ref00 init', self.corr_ref00_init)
        if 'ssim'     in self.__dict__         :setattr(ncfile, 'ssim', self.ssim)
        setattr(ncfile,'time since ignition', self.time_igni)
        setattr(ncfile,'date', self.time_date.strftime("%Y-%m-%d %H:%M:%S"))
        setattr(ncfile,'shrink_factor', self.shrink_factor)
        setattr(ncfile,'cfMode', self.cfMode)
        setattr(ncfile,'inRefList', self.inRefList)
        setattr(ncfile,'blurred', int(self.blurred))
        if 'clahe_clipLimit'   in self.__dict__: setattr(ncfile,'clahe_clipLimit', self.clahe_clipLimit)
        setattr(ncfile,'kernel_plot', self.kernel_plot)
        setattr(ncfile,'kernel_warp', self.kernel_warp)
        setattr(ncfile,'grid_shape', self.grid_shape)
        
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
        setattr(ncmask_img, 'long_name', 'mask of the gray img') 
        setattr(ncmask_img, 'standard_name', 'mask_img') 
        setattr(ncmask_img, 'units', '-') 
        
        ncbackgrdimg    = ncfile.createVariable('backgrdimg','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncbackgrdimg, 'long_name', 'gray background image made of stitch warp frame') 
        setattr(ncbackgrdimg, 'standard_name', 'backgrdimg') 
        setattr(ncbackgrdimg, 'units', '-') 
            
        ncmask_backgrdimg    = ncfile.createVariable('mask_backgrdimg','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncmask_backgrdimg, 'long_name', 'mask of the background img') 
        setattr(ncmask_backgrdimg, 'standard_name', 'mask_backgrdimg') 
        setattr(ncmask_backgrdimg, 'units', '-') 
        

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
        setattr(ncplotMask_ring, 'long_name', 'plot Mask_ring with biffer zone') 
        setattr(ncplotMask_ring, 'standard_name', 'plotMask_ring') 
        setattr(ncplotMask_ring, 'units', '-') 

        ncplumeMask    = ncfile.createVariable('plumeMask','uint8', (u'imgi',u'imgj',), fill_value=0.)
        setattr(ncplumeMask, 'long_name', 'plume Mask with biffer zone') 
        setattr(ncplumeMask, 'standard_name', 'plumeMask') 
        setattr(ncplumeMask, 'units', '-') 
        
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

        ncH2Ref    = ncfile.createVariable('H2Ref','float32', (u'mtxi',u'mtxi',), fill_value=None)
        setattr(ncH2Ref, 'long_name', 'homography matrix to Reference image') 
        setattr(ncH2Ref, 'standard_name', 'H2Ref') 
        setattr(ncH2Ref, 'units', '-') 
        
        ncH2Grid    = ncfile.createVariable('H2Grid','float32', (u'mtxi',u'mtxj',), fill_value=None)
        setattr(ncH2Grid, 'long_name', 'homography matrix to Grid') 
        setattr(ncH2Grid, 'standard_name', 'H2Grid') 
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
        
        if ('good_old' in self.__dict__):
            if self.good_old is not None:
                ncfeat3 = ncfile.createVariable('good_features_ref','float32', (u'FeatureNbre',u'FeatureDim',), fill_value=0.)
                setattr(ncfeat3, 'long_name', 'selected good features on all ref images ') 
                setattr(ncfeat3, 'standard_name', 'good features 3') 
                setattr(ncfeat3, 'units', '-') 

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
        if 'mask_img'           in self.__dict__: ncmask_img[:,:]  = self.mask_img
        
        
        if 'backgrdimg'          in self.__dict__: ncbackgrdimg[:,:] = self.backgrdimg
        if 'mask_backgrdimg'     in self.__dict__: ncmask_backgrdimg[:,:]  = self.mask_backgrdimg

        if 'warp'                in self.__dict__: ncwarp[:,:]       = self.warp
        if 'mask_warp'           in self.__dict__: ncmask_warp[:,:]  = self.mask_warp
        
        if 'plotMask_withBuffer' in self.__dict__: ncplotMask[:,:]   = self.plotMask_withBuffer
        if 'plotMask_withBuffer_ring' in self.__dict__: ncplotMask_ring[:,:]   = self.plotMask_withBuffer_ring
        if 'plumeMask'           in self.__dict__: ncplumeMask[:,:]   = self.plumeMask
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
        if 'clahe_clipLimit'            in ncfile.ncattrs(): self.clahe_clipLimit = ncfile.getncattr('clahe_clipLimit')
        if 'kernel_plot'     in ncfile.ncattrs(): self.kernel_plot = ncfile.getncattr('kernel_plot')
        if 'kernel_warp'     in ncfile.ncattrs(): self.kernel_warp = ncfile.getncattr('kernel_warp')
        if 'grid_shape'     in ncfile.ncattrs(): self.grid_shape = tuple(ncfile.getncattr('grid_shape'))



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
        if 'grid_shape'      in ncfile.ncattrs(): self.grid_shape = tuple(ncfile.getncattr('grid_shape'))

    
        inputConfig = imp.new_module('inputConfig')
        for gr in ncfile.groups:
            for gr2 in ncfile.groups[gr].groups:
                setattr(inputConfig,gr2,{})
                for attr in ncfile.groups[gr].groups[gr2].ncattrs():
                    k = attr
                    v= ncfile.groups[gr].groups[gr2].getncattr(attr)
                    if type(k) == str:  k = k.replace('nbre_', '#')
                    inputConfig.__dict__[gr2][k] = v 
        self.inputConfig = inputConfig


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
        self.plumeMask           = np.ma.filled(ncfile.variables['plumeMask'][:,:])
        self.bareGroundMask_withBuffer = np.ma.filled(ncfile.variables['bareGroundMask'][:,:])
        if 'mask_img'  in ncfile.variables : self.mask_img         = np.ma.filled(ncfile.variables['mask_img'][:,:])
        if 'ssim'      in ncfile.variables : self.ssim_2d          = np.ma.filled(ncfile.variables['ssim'][:,:])
        if 'mask_ssim' in ncfile.variables : self.mask_ssim        = np.ma.filled(ncfile.variables['mask_ssim'][:,:])
        
        if 'backgrdimg'          in ncfile.variables: self.backgrdimg       = np.ma.filled(ncfile.variables['backgrdimg'][:,:])
        if 'mask_backgrdimg'     in ncfile.variables: self.mask_backgrdimg  = np.ma.filled(ncfile.variables['mask_backgrdimg'][:,:])

        if 'H2Ref'  in ncfile.variables: self.H2Ref  = np.ma.filled(ncfile.variables['H2Ref'][:,:])
        if 'H2Grid' in ncfile.variables: self.H2Grid = np.ma.filled(ncfile.variables['H2Grid'][:,:])

        if 'features_img'        in ncfile.variables: self.feature        = np.ma.filled(ncfile.variables['features_img'])
        if 'good_features_img'   in ncfile.variables: self.good_new       = np.ma.filled(ncfile.variables['good_features_img'])
        if 'good_features_4plot' in ncfile.variables: self.good_new_4plot = np.ma.filled(ncfile.variables['good_features_4plot'])
        if 'good_features_ref'   in ncfile.variables: self.good_old       = np.ma.filled(ncfile.variables['good_features_ref'])

        if 'cf_loc'              in ncfile.variables: self.cf_on_img      = np.ma.filled(ncfile.variables['cf_loc'])
        if 'cf_hist'             in ncfile.variables: self.cf_hist        = np.ma.filled(ncfile.variables['cf_hist'])
        
        if 'cf_loc'              in ncfile.variables:
            if (self.cf_on_img.shape[0] > 4) : 
                idx = np.where( (self.cf_on_img[:,0]*self.cf_on_img[:,1] == 0) & (self.cf_on_img[:,0]+self.cf_on_img[:,1] == 0) )[0].max()
                self.cf_on_img = self.cf_on_img[:idx,:]    
                self.cf_hist = self.cf_hist[:idx,:]    

        ncfile.close()

    def set_flag_cfMode(self, input):
        self.cfMode = input

    def set_flag_inRefList(self, input):
        self.inRefList = input


    def optimize_homography(self, params_georef, params_camera, frame_ref00, frame_ref00_init, 
                            win_size_ssim, flag='firstCall', frame_ref=None ):
      
        frame_ = self.copy()
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1.e-3)
        mask_func_paramx = [1.e6, 0]

        if flag == 'firstCall':
            flag_opt  = 1
            frame_ref_selected = frame_ref00
            mask_func = tools.mask_EP08               #add plumemask
            #mask_func_param = mask_func_paramx
            trans_len_limit = [40,40]
            #corr_to_compare_with = frame_.corr_ref00

        elif flag == 'refine': 
            flag_opt       = 2
            frame_ref_selected = frame_ref
            mask_func       = tools.mask_onlyImageMask #add plumemask
            #mask_func_param = mask_func_paramx
            trans_len_limit = [40,40]
            #corr_to_compare_with = frame_.corr_ref00

        elif flag == 'coarse': 
            flag_opt       = 3
            frame_ref_selected = frame_ref
            mask_func = tools.mask_onlyImageMask
            #mask_func_param = mask_func_paramx
            trans_len_limit = [50,40]
            #corr_to_compare_with = None 
        
        elif flag == 'final': 
            flag_opt       = 4
            frame_ref_selected = frame_ref
            mask_func       = tools.mask_EP08
            #mask_func_param = mask_func_paramx
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
                                                                                 )
            #warp_matrix_frame2ref = frame_ref.H2Ref.dot(warp_matrix_frame2ref)
            frame_.set_homography_to_ref( warp_matrix_frame2ref )
            frame_.set_warp(    cv2.warpPerspective (frame_.img,      frame_.H2Ref, frame_.warp.shape[::-1], flags=cv2.INTER_LINEAR ))
            frame_.set_maskWarp(cv2.warpPerspective (frame_.mask_img, frame_.H2Ref, frame_.warp.shape[::-1], flags=cv2.INTER_NEAREST)) 
            if win_size_ssim != 0 :  
                mask_ssim, ssim_2d, ssim  = tools.star_get_costFunction(['ssim', frame_, frame_ref00, win_size_ssim,] )
                frame_.set_similarity_info(ssim_2d, mask_ssim)
            else: 
                ssim = -999
            frame_.set_correlation(
                                   tools.star_get_costFunction(['EP08', frame_, frame_ref_selected, tools.mask_onlyImageMask, mask_func_paramx]) \
                                                                                                     if (frame_ref_selected is not None) else None,
                                   tools.star_get_costFunction(['EP08', frame_, frame_ref00,        tools.mask_EP08,          mask_func_paramx])  ,
                                   tools.star_get_costFunction(['EP08', frame_, frame_ref00_init,   tools.mask_EP08,          mask_func_paramx])  ,
                                   ssim)
            frame_.set_id_best_ref(-1)
            
            if frame_.corr_ref > self.corr_ref: 
                self.set_homography_to_ref(frame_.H2Ref)
                self.set_warp(frame_.warp)
                self.set_maskWarp(frame_.mask_warp)
                if win_size_ssim != 0 :  
                    self.set_similarity_info(frame_.ssim_2d, frame_.mask_ssim)
                self.set_correlation(frame_.corr_ref, frame_.corr_ref00, frame_.corr_ref00_init, frame_.ssim)
                self.set_id_best_ref(frame_.id_best_ref)
                print('ecc{:1d} '.format(id_ecc), self.id_best_ref, end=' ') 

            #plt.imshow(img_ref.T,origin='lower')
            #plt.imshow(img.T,origin='lower',cmap=mpl.cm.Greys_r,alpha=.5); plt.show()   
            #plt.imshow(frame_ref.warp.T,origin='lower')
            #plt.imshow(self.warp.T,origin='lower',cmap=mpl.cm.Greys_r,alpha=.5); plt.show()
            #pdb.set_trace()
            return 
       
        id_ecc, warp_matrix_frame2ref = tools.findTransformECC_on_ref_frame(flag, 
                                                                            self, frame_ref_selected, 
                                                                            trans_len_limit = trans_len_limit, 
                                                                            ep08_limit      = [.7,params_camera['energy_good_2']],
                                                                            mask_func=mask_func,
                                                                            ) 

        frame_.set_warp(    cv2.warpPerspective (frame_.img,      warp_matrix_frame2ref, frame_.warp.shape[::-1], flags=cv2.INTER_LINEAR ))
        frame_.set_maskWarp(cv2.warpPerspective (frame_.mask_img, warp_matrix_frame2ref, frame_.warp.shape[::-1], flags=cv2.INTER_NEAREST)) 
        frame_.set_homography_to_ref( warp_matrix_frame2ref )
        
        if win_size_ssim != 0 :  
            mask_ssim, ssim_2d, ssim  = tools.star_get_costFunction([ 'ssim', frame_, frame_ref00, win_size_ssim ])
            frame_.set_similarity_info(ssim_2d, mask_ssim)
        else: 
            ssim = -999
        frame_.set_correlation(tools.star_get_costFunction(['EP08', frame_, frame_ref, tools.mask_lowT, mask_func_paramx]) \
                                                          if ((frame_ref is not None)&(flag!='final')) else None,
                               tools.star_get_costFunction(['EP08', frame_, frame_ref00,      tools.mask_EP08, mask_func_paramx]) ,
                               tools.star_get_costFunction(['EP08', frame_, frame_ref00_init, tools.mask_EP08, mask_func_paramx]),
                               ssim)

        
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
            if win_size_ssim != 0 : self.set_similarity_info(frame_.ssim_2d, frame_.mask_ssim)
            self.set_correlation(frame_.corr_ref,
                                 frame_.corr_ref00, 
                                 frame_.corr_ref00_init, 
                                 frame_.ssim)

        if flag != 'final': 
            print('opt{:1d} ecc{:1d} (d={:6.3f}) '.format( flag_opt, id_ecc, test_val), end=' ') 
        else: 
            return max([test_val,0]), test_val


#################################################
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


#################################################
def load_existing_file(params_camera,filename):
    frame = loadFrame(params_camera)
    frame.loadFromFile(filename)
    return frame



#################################################
def get_feature(frame, feature_params):
    
    
    p00 = cv2.goodFeaturesToTrack(frame.img, mask = None, **feature_params)
    
    #p00 = cv2.goodFeaturesToTrack(frame.img[500:-500,500:-500], mask = None, **feature_params)
    #p00 += 500

    #p00 = None

    '''
    plt.clf()
    plt.imshow(gray.T,origin='lower',cmap=mpl.cm.Greys_r)
    plt.scatter(p00[:,0,1]+500,p00[:,0,0]+500,c='r')
    plt.show()
    pdb.set_trace()
    ''' 

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
    

    return p00



###########################################################33
def get_cluster_from_segment(img_,img_raw,sigma=2):

    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
    mm1band = clahe.apply(img_)
    
    segments = slic(img_raw, n_segments = 400, sigma = sigma) # MERDE4 

    seg_mean = np.zeros(segments.shape)
    seg_var = np.zeros_like(seg_mean)
    for i_seg in np.arange(segments.max()):
        idx = np.where(segments==i_seg)
        seg_mean[idx] = img_[idx].mean()
        seg_var[idx] = img_[idx].std()

    
    nbre_pt_in_mask_arr = []
    img_threshold_arr = np.arange(seg_mean.min(),seg_mean.max(),1.)
    for seg_mean_threshold in img_threshold_arr:
        seg_mask = np.where(seg_mean<seg_mean_threshold,np.ones_like(seg_mean),np.zeros_like(seg_mean)) # MERDE4
        nbre_pt_in_mask_arr.append( old_div(1.*np.where(seg_mask==1)[0].shape[0], seg_mask.size))
   
    xx = img_threshold_arr
    yy = np.array(nbre_pt_in_mask_arr)
    d1 = old_div(np.diff(yy), np.diff(xx)) 

    i = d1.argmax()
    while( d1[i] >= d1[i-4:i+1].mean() + .3*d1[i-4:i+1].std()):
        i -=1 
        if i < 4: break
   
    if i == 3: # no helico leg
        return np.zeros(img_raw.shape)

    seg_mask = np.where(seg_mean<img_threshold_arr[i], np.ones_like(seg_mean), np.zeros_like(seg_mean)) #MERDE4
    #clean mask
    s = [[0,1,0], \
         [1,1,1], \
         [0,1,0]] # for diagonal
    seg_cluster, seg_clusterNbre =  ndimage.label(seg_mask, structure=s )
    #print seg_clusterNbre,  mm1band.min()
    for i_cluster in range(seg_clusterNbre):
        idx = np.where(seg_cluster == i_cluster+1)
        #remove cluster that do not contain min 
        #if mm1band.min() not in mm1band[idx]:
        #    seg_mask[idx] = 0
        #remove small cluster
        if old_div(1.*len(idx[0]),seg_mask.size) < .005: 
            seg_mask[idx] = 0

    #return ndimage.label(seg_mask, structure=s ), seg_mask
    pdb.set_trace()
    return seg_mask


###########################################################33
def mask_helico_leg(frame,\
                    flag=None,filenames=None,):

    img_raw = frame.img[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)]
    seg_mask = get_cluster_from_segment(img_raw,sigma=5) 
        
    mask_sinle_img = np.zeros_like(frame.img) 
    mask_sinle_img[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)] = np.where(seg_mask==1,np.zeros_like(seg_mask),np.ones_like(seg_mask))
    
    plt.imshow(frame.img.T,origin='lower')
    plt.imshow(np.ma.masked_where(mask_sinle_img!=1,mask_sinle_img).T,origin='lower',alpha=.5)
    plt.show()
    pdb.set_trace()

    return mask_sinle_img



###########################################################
def build_mask(frame,image_pdf_reference_tools):
    
    bins_ref,pdf_ref = image_pdf_reference_tools
    
    img_ = frame.img[old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2),old_div(frame.bufferZone,2):old_div(-frame.bufferZone,2)]
    blurred = filters.gaussian(img_, sigma=5.0)
    
    x,y = np.gradient(blurred)
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)
    theta = np.abs(np.where(theta>np.pi, theta - 2*np.pi, theta))
    
    shrink_factor = 4
    ni_, nj_ = old_div(frame.ni,shrink_factor), old_div(frame.nj,shrink_factor) 
    img   = tools.downgrade_resolution_4nadir(np.array(img_,dtype=float), [ni_,nj_] , flag_interpolation='average')
    theta = tools.downgrade_resolution_4nadir(theta, [ni_,nj_] , flag_interpolation='average')
    
    idx_subwin = np.arange(0,(old_div(img.shape[0],30))*(old_div(img.shape[1],30)))
    subwind = ndimage.zoom(idx_subwin.reshape(((old_div(img.shape[0],30)),(old_div(img.shape[1],30)))),30, order=0) 
    winsize = int(np.sqrt(np.where(subwind==0)[0].shape[0]))
   
    def test_func(values,bins,pdf_ref):
        hist = np.histogram(values[np.where(values>=0)].flatten(),bins)[0]
        pdf = old_div(np.array(hist,dtype=float),max([hist.sum(),np.spacing(1)]))
        return (pdf-pdf_ref)[np.where(pdf-pdf_ref>0)].sum()

    def padwithzeros(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = -999
        vector[-pad_width[1]:] = -999
        return vector

    footprint = np.ones([winsize,winsize])
   
    local_pdf_change = []
    for i, input_ in enumerate([img,theta]):
        x_padded = np.lib.pad(input_, winsize, padwithzeros)
        local_pdf_change.append( ndimage.generic_filter(x_padded, test_func, footprint=footprint, extra_arguments=(bins_ref[i],pdf_ref[i],))[winsize:-winsize,winsize:-winsize] )
    
    mask = np.where( (local_pdf_change[0]>0.1) & (local_pdf_change[1]>0.1), np.ones_like(img), np.zeros_like(img))
   
    cluster_label, cluster_number = ndimage.label(mask,structure=np.ones([3,3]))
    for i_cluster in range(cluster_number):
        idx = np.where(cluster_label == i_cluster+1)
        if (local_pdf_change[0][idx].max() < .9*local_pdf_change[0].max()) & (local_pdf_change[1][idx].max() < .9*local_pdf_change[1].max()):
            mask[idx] = 0
            
    mask_ = ndimage.zoom(mask,shrink_factor, order=0)

    mask = np.zeros_like(frame.img)
    mask[ (np.where(mask_==0)[0]+old_div(frame.bufferZone,2),np.where(mask_==0)[1]+old_div(frame.bufferZone,2))] = 1

    return mask 


#######################################################
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)



#######################################################
def load_img_as_in_loadFrame(filename, K, D, shrink_factor):
       
        colorRaw_      = cv2.imread(filename)
        if D.sum()!=0:
            #apply distrotion correction
            h, w  = colorRaw_.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))
            # undistort
            colorRaw_undistorted = cv2.undistort(colorRaw_, K, D, None, newcameramtx)
            # crop the image
            x,y,w,h = roi
            colorRaw_crop = colorRaw_undistorted[y:y+h, x:x+w]
            colorRaw_crop = np.transpose(colorRaw_crop[::-1,:,:],[1,0,2])
        else: 
            colorRaw_crop = colorRaw_
            newcameramtx = K
        
        #downgrade
        if shrink_factor > 1: 
            ni_, nj_ = old_div(colorRaw_crop.shape[0],shrink_factor), old_div(colorRaw_crop.shape[1],shrink_factor) 
            colorRaw = np.zeros([ni_, nj_,3])
            for iband in range(3):
                colorRaw[:,:,iband] = tools.downgrade_resolution_4nadir( colorRaw_crop[:,:,iband],   [ni_,nj_] , flag_interpolation='min' )
            colorRaw = np.array(colorRaw,dtype=np.uint8)
        else:
            colorRaw = colorRaw_crop
       
        #gray scale
        grayRaw = cv2.cvtColor(colorRaw, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5,5))
        img = clahe.apply(grayRaw)

        return img


#######################################################
def build_reference_pdf(filenames_, range_file_no_helico, K, D, shrink_factor):

    print('build referenced pdf ...', end=' ')
    sys.stdout.flush()
    bins_color = np.linspace(0,255,51)
    bins_theta = np.linspace(0,np.pi,51)
    bins = [bins_color,bins_theta]

    filenames = []
    for filename_ in filenames_:
        if (int(re.findall(r'\d+', os.path.basename(filename_).split('.')[0] )[0]) >= range_file_no_helico[0]) &\
           (int(re.findall(r'\d+', os.path.basename(filename_).split('.')[0] )[0]) <= range_file_no_helico[1]) :
            filenames.append(filename_)

    all_pdf = [[],[]]
    for filename in filenames:

        img = load_img_as_in_loadFrame(filename, K, D, shrink_factor)

        blurred = filters.gaussian(img, sigma=4.0)
        
        x,y = np.gradient(blurred)
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        theta = np.abs(np.where(theta>np.pi, theta - 2*np.pi, theta))
        
        ni, nj = img.shape
        shrink_factor_ = 4 
        ni_, nj_ = old_div(ni,shrink_factor_), old_div(nj,shrink_factor_) 
        img   = tools.downgrade_resolution_4nadir(img, [ni_,nj_] , flag_interpolation='average')
        theta = tools.downgrade_resolution_4nadir(theta, [ni_,nj_] , flag_interpolation='average')

        ss = 30
        idx_subwin = np.arange(0,(old_div(img.shape[0],ss))*(old_div(img.shape[1],ss)))
        subwind = ndimage.zoom(idx_subwin.reshape(((old_div(img.shape[0],ss)),(old_div(img.shape[1],ss)))),ss, order=0) 
        pdf_ = [[],[]]
       
        for i_idx_subwin in idx_subwin:
            idx = np.where(subwind==i_idx_subwin)
            
            for i, input_ in enumerate([img,theta]):
                hist = np.histogram(input_[idx].flatten(),bins=bins[i])[0]
                pdf_[i].append(old_div(np.array(hist,dtype=float),max([hist.sum(),np.spacing(1)])))
            
        all_pdf[0] += pdf_[0]; all_pdf[1] += pdf_[1]
       
    all_pdf_envelop_color = np.dstack(all_pdf[0])[0].max(axis=1)
    all_pdf_envelop_theta = np.dstack(all_pdf[1])[0].max(axis=1)
    all_pdf_envelop = [all_pdf_envelop_color,all_pdf_envelop_theta]

    print('done')
    return [bins,all_pdf_envelop]

#######################################################
def processRawData(ignitionTime, params, paramsCamera, flag_restart):
    
    #input
    ###########
    dir_in       = params['root_data']     + params['root_data_DirVis'] 
    dir_out_vis =  params['root_postproc'] + paramsCamera['dir_input']

    #read time diff
    ############
    #load time difference 
    #delta_t_vis_lwir, delta_t_vis_mir = cameraTools.get_time_shift_vis_lwir_mir(params)
    delta_t_vis_lwir, delta_t_vis_mir = 0 , 0
    if delta_t_vis_mir == None: 
        delta_t_vis = delta_t_vis_lwir
    else: 
        delta_t_vis = delta_t_vis_mir


    if not(flag_restart):
        if os.path.isdir(dir_out_vis + paramsCamera['dir_img_input']): shutil.rmtree(dir_out_vis + paramsCamera['dir_img_input'])
    tools.ensure_dir(dir_out_vis)
    tools.ensure_dir(dir_out_vis + paramsCamera['dir_img_input'])


    #loop over raw image and select the one in the time range
    filenames = sorted(glob.glob(dir_in+paramsCamera['dir_img_input_*']))
    print('create symlink image in ', dir_out_vis+'/'+ paramsCamera['dir_img_input'] + '  ...', end=' ')
    sys.stdout.flush()

    filenames_basename = []
    filenames_ignitime = []
    filenames_datetime = []
    for ifile, filename in enumerate(filenames):        
        #get time frame
        img = Image.open(filename)
        exif_data = img._getexif()
        img = None
        try: 
            time_frame = datetime.datetime.strptime(cameraTools.get_field(exif_data,'DateTime'), "%Y:%m:%d %H:%M:%S.%f")
        except:
            try: 
                time_frame = datetime.datetime.strptime(cameraTools.get_field(exif_data,'DateTime'), "%Y:%m:%d %H:%M:%S")
            except: 
                time_frame = datetime.datetime.strptime(exif_data[36867], "%Y:%m:%d %H:%M:%S") # whem using extractFramefromVideo.py

        time_sinceIgni = (time_frame - ignitionTime).total_seconds() - delta_t_vis
       
        filenames_ignitime.append(time_sinceIgni)
        filenames_datetime.append(time_frame)
        filenames_basename.append(os.path.basename(filename))
    print( '{:d} images found'.format(len(filenames)))      

    print('correct for same time image')
    #correct for images with same time
    data = list(np.insert( np.diff(filenames_ignitime), 0 , 1))
    grouped = (list(g) for _,g in itertools.groupby(enumerate(data), lambda t:t[1]))
    idxs = [(g[0][0], g[-1][0] + 1) for g in grouped if ( (len(g) >= 1) & (g[0][1]==0) )   ]

    filenames_ignitime_c = np.copy(filenames_ignitime)
    filenames_datetime_c = np.copy(filenames_datetime)
    for idx_s, idx_e in idxs:
        if  idx_e < len(filenames_ignitime):
            diffTime = filenames_ignitime[idx_e]- filenames_ignitime[idx_s-1] 
        else: 
            diffTime = 1. # this is for the last image, we assume 1 second diff with penultimate
        nbre_consecutive_value = idx_e-idx_s+1
        for ii in range(idx_s,idx_s-1+nbre_consecutive_value):
            filenames_ignitime_c[ii] = filenames_ignitime[idx_s-1] + old_div(diffTime,(idx_e-idx_s+1))
            filenames_datetime_c[ii] = filenames_datetime[idx_s-1] + datetime.timedelta(seconds=old_div(diffTime,(idx_e-idx_s+1)) )
       

    #create link in dir_out_vis
    filenames_basename2 = []
    filenames_ignitime2 = []
    filenames_datetime2 = []
    for ifile, filename in enumerate(filenames):        

        time_sinceIgni = filenames_ignitime_c[ifile]
        datetime_sinceIgni = filenames_datetime_c[ifile]
        if (time_sinceIgni >= params['startTimeVis']) & (time_sinceIgni <= params['endTimeVis']):

            filenames_basename2.append(filenames_basename[ifile])
            filenames_ignitime2.append(time_sinceIgni)
            filenames_datetime2.append(datetime_sinceIgni)

            if os.path.islink(dir_out_vis+'/'+ paramsCamera['dir_img_input']+os.path.basename(filenames_basename2[-1])): 
                os.remove(dir_out_vis+'/'+ paramsCamera['dir_img_input']+os.path.basename(filenames_basename2[-1]))
            os.symlink(filenames[ifile], dir_out_vis+'/'+ paramsCamera['dir_img_input']+os.path.basename(filenames_basename2[-1]))
       

    out_time = np.array(len(filenames_ignitime2)*[('mm',0,datetime.datetime(1970,1,1))],             \
                        dtype=np.dtype([('name','U100'),('time',float),('datetime',datetime.datetime)]))
    out_time = out_time.view(np.recarray)
    out_time.name = filenames_basename2
    out_time.time = filenames_ignitime2
    out_time.datetime = filenames_datetime2

    np.save(dir_out_vis+'/'+ paramsCamera['dir_img_input']+'filename_time',out_time)
    print(' done')
    
    return 

