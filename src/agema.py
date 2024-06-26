from __future__ import print_function
from builtins import range
from builtins import object
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import io, ndimage
import asciitable
import datetime 
import os
import sys
import glob 
import pdb 
import shutil
from netCDF4 import Dataset
import copy 

#homebrewed 
import tools


#################################################
class loadFrame(object):
    
    def __init__(self, ): 
        self.type = 'mir'
   

    def init(self, id_file, mir_frame_name, ignitionTime, K, D): 
        self.id = id_file
        self.mir_filename  = mir_frame_name
        
        mir_time, mir_temp = read_agema(mir_frame_name)
        time_igni = (mir_time-ignitionTime).total_seconds()

        self.time_igni = time_igni 
        self.time_date = mir_time 

        self.ni = mir_temp.shape[0]
        self.nj = mir_temp.shape[1]

        #apply distrotion correction
        h, w  = mir_temp.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))
        # undistort
        mir_temp_undistorted = cv2.undistort(mir_temp, K, D, None, newcameramtx)
        # crop the image
        x,y,w,h = roi
        mir_temp_crop = mir_temp_undistorted[y:y+h, x:x+w]
        #
        self.temp = mir_temp_crop

        #save camera matrix
        self.K_undistorted_imgRes = newcameramtx
        # D is now 0 as we undistrot the image
        

    def set_matching_lwir_info(self, id_lwir,lwir_filename):
        self.id_lwir = id_lwir
        self.lwir_filename = lwir_filename
    
    
    def set_homography_to_lwir(self,input_):
        self.H2lwir = input_


    def set_homography_to_grid(self,input_):
        self.H2Grid = input_
   

    def set_homography_to_mirRef(self,mirId, input_):
        self.H2mirRef = input_
        self.id_mirRef = mirId
   

    def set_pose(self,rvec,tvec):
        self.rvec = rvec
        self.tvec = tvec


    def dump(self,filename):

        ncfile = Dataset(filename,'w')
        ncfile.description = 'mir frame generated by GeoRefCam'
   
        # Global attributes
        setattr(ncfile, 'created', 'R. Paugam') 
        setattr(ncfile, 'title', 'mir frame')
        setattr(ncfile, 'Conventions', 'CF')

        setattr(ncfile,'type', self.type)
        setattr(ncfile,'id', self.id)
        if 'id_lwir' in self.__dict__ : setattr(ncfile,'id_lwir', self.id_lwir)
        setattr(ncfile,'time since ignition', self.time_igni)
        setattr(ncfile,'date', self.time_date.strftime("%Y-%m-%d %H:%M:%S"))

        setattr(ncfile,'mir_filename', self.mir_filename)
        setattr(ncfile,'lwir_filename', self.lwir_filename)

        setattr(ncfile,'img width',  self.ni)
        setattr(ncfile,'img height', self.nj)
        
        if 'id_mirRef' in self.__dict__             : setattr(ncfile,'id_mirRef', self.id_mirRef)
       
        #if 'rvec' in self.__dict__                  : setattr(ncfile,'rvec', self.tvec)
        #if 'tvec' in self.__dict__                  : setattr(ncfile,'tvec', self.tvec)
        #if 'K_undistorted_imgRes'in self.__dict__   : setattr(ncfile,'K_undistorted_imgRes', self.K_undistorted_imgRes)
        
        # dimensions
        ncfile.createDimension('imgi',self.temp.shape[0])
        ncfile.createDimension('imgj',self.temp.shape[1])
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
       
        ncmtxi = ncfile.createVariable('mtxi', 'f8', ('mtxi',))
        setattr(ncmtxi, 'long_name', 'homography dimx')
        setattr(ncmtxi, 'standard_name', 'mtxi')
        setattr(ncmtxi, 'units','-')
        
        ncmtxj = ncfile.createVariable('mtxj', 'f8', ('mtxi',))
        setattr(ncmtxj, 'long_name', 'homography dimy')
        setattr(ncmtxj, 'standard_name', 'mtxj')
        setattr(ncmtxj, 'units','-')
       

        # set Variables
        nctemp    = ncfile.createVariable('temp','float32', (u'imgi',u'imgj',), fill_value=0.)
        setattr(nctemp, 'long_name', 'mir brightness temperature') 
        setattr(nctemp, 'standard_name', 'temp') 
        setattr(nctemp, 'units', 'K') 
        
        ncH2Ref    = ncfile.createVariable('Hmir2lwir','float32', (u'mtxi',u'mtxj',), fill_value=None)
        setattr(ncH2Ref, 'long_name', 'homography matrix to Reference image') 
        setattr(ncH2Ref, 'standard_name', 'H2Ref') 
        setattr(ncH2Ref, 'units', '-') 
        
        ncH2MirRef    = ncfile.createVariable('Hmir2mir','float32', (u'mtxi',u'mtxj',), fill_value=None)
        setattr(ncH2MirRef, 'long_name', 'homography matrix to Reference Mir image') 
        setattr(ncH2MirRef, 'standard_name', 'H2MirRef') 
        setattr(ncH2MirRef, 'units', '-') 
        
        ncH2Grid    = ncfile.createVariable('H2Grid','float32', (u'mtxi',u'mtxj',), fill_value=None)
        setattr(ncH2Grid, 'long_name', 'homography matrix to Grid') 
        setattr(ncH2Grid, 'standard_name', 'H2Ref') 
        setattr(ncH2Grid, 'units', '-') 
        
        if 'K_undistorted_imgRes'in self.__dict__   :
            ncK_undistorted_imgRes  = ncfile.createVariable('K_undistorted_imgRes','float32', (u'mtxi',u'mtxj',), fill_value=None)
            setattr(ncK_undistorted_imgRes, 'long_name', 'camera matrix') 
            setattr(ncK_undistorted_imgRes, 'standard_name', 'K_undistorted_imgRes') 
            setattr(ncK_undistorted_imgRes, 'units', '-') 
        
        if 'rvec' in self.__dict__                  : 
            ncrvec  = ncfile.createVariable('rvec','float32', (u'mtxi',), fill_value=None)
            setattr(ncrvec, 'long_name', 'camera rotation vector') 
            setattr(ncrvec, 'standard_name', 'rvec') 
            setattr(ncrvec, 'units', '-') 
        
        if 'tvec' in self.__dict__                  : 
            nctvec  = ncfile.createVariable('tvec','float32', (u'mtxi',), fill_value=None)
            setattr(nctvec, 'long_name', 'camera translation vector') 
            setattr(nctvec, 'standard_name', 'tvec') 
            setattr(nctvec, 'units', '-') 

        #write grid
        ncimgi[:]    = np.arange(self.temp.shape[0])
        ncimgj[:]    = np.arange(self.temp.shape[1])
        ncmtxi[:]    = np.arange(3)
        ncmtxj[:]    = np.arange(3)

        #write data  
        if 'temp'    in self.__dict__: nctemp[:,:]        = self.temp

        if 'H2lwir'   in self.__dict__: ncH2Ref[:,:]  = self.H2lwir
        if 'H2mirRef' in self.__dict__: ncH2MirRef[:,:]  = self.H2mirRef
        if 'H2Grid'   in self.__dict__: ncH2Grid[:,:] = self.H2Grid
        if 'K_undistorted_imgRes'in self.__dict__  : ncK_undistorted_imgRes[:,:] = self.K_undistorted_imgRes 
        if 'rvec'in self.__dict__  : ncrvec[:] = self.rvec
        if 'tvec'in self.__dict__  : nctvec[:] = self.tvec
        
        
        #close file
        ncfile.close()

        return 0
    
    
    def loadFromFile(self,filename):
        ncfile = Dataset(filename,'r')

        if self.type != ncfile.getncattr('type') : 
            print('error when load file', filename)
            sys.exit()
        self.id        = ncfile.getncattr('id')
        self.time_igni = ncfile.getncattr('time since ignition')
        self.time_date = datetime.datetime.strptime(ncfile.getncattr('date'),"%Y-%m-%d %H:%M:%S")
        self.ni        = ncfile.getncattr('img width')
        self.nj        = ncfile.getncattr('img height')
        self.id_lwir      = ncfile.getncattr('id_lwir')
        
        if 'id_mirRef' in ncfile.ncattrs(): self.id_mirRef = ncfile.getncattr('id_mirRef')
        #if 'rvec' in ncfile.ncattrs()     : self.rvec = ncfile.getncattr('rvec')
        #if 'tvec' in ncfile.ncattrs()     : self.tvec = ncfile.getncattr('tvec')
        #if 'K_undistorted_imgRes' in ncfile.ncattrs(): self.K_undistorted_imgRes = ncfile.getncattr('K_undistorted_imgRes').reshape([3,3])
        if 'mir_filename' in ncfile.ncattrs()        : self.mir_filename = ncfile.getncattr('mir_filename')
        if 'lwir_filename' in ncfile.ncattrs()       : self.lwir_filename = ncfile.getncattr('lwir_filename')

        self.temp = np.ma.filled(ncfile.variables['temp'][:,:])

        if 'Hmir2lwir' in ncfile.variables: self.H2lwir  = np.ma.filled(ncfile.variables['Hmir2lwir'][:,:])
        if 'Hmir2mir'  in ncfile.variables: self.H2mirRef  = np.ma.filled(ncfile.variables['Hmir2mir'][:,:])
        if 'H2Grid'    in ncfile.variables: self.H2Grid = np.ma.filled(ncfile.variables['H2Grid'][:,:])
        if 'K_undistorted_imgRes'  in ncfile.variables: self.K_undistorted_imgRes = np.ma.filled(ncfile.variables['K_undistorted_imgRes'][:,:])
        if 'tvec'      in ncfile.variables: self.rvec = np.ma.filled(ncfile.variables['tvec'][:])
        if 'rvec'      in ncfile.variables: self.tvec = np.ma.filled(ncfile.variables['rvec'][:])
      
        ncfile.close()


    def copy(self):
        return copy.deepcopy(self)


    def getTemp(self,flag,gridShape=None):
        if flag == 'frame':
            return self.temp
        elif flag == 'grid':
            return cv2.warpPerspective(self.temp, self.H2Grid, gridShape[::-1], flags=cv2.INTER_LINEAR)
        else: 
            print('bad flag in fct agema.loadFrame.getTemp. stop here')
            sys.exit()


#################################################
def load_existing_file(filename):
    frame = loadFrame()
    frame.loadFromFile(filename)
    return frame


################################################
def processRawData(src_dir, ignitionTime, params, paramsCamera, params_flag, flag_restart):
    
    print(' process raw mir data ...', end=' ') 
    sys.stdout.flush()
    #input
    ###########
    if  params['root_data'][0] == '/':
        dir_in       = params['root_data']     + params['root_data_DirMir'] 
        dir_out_mir =  params['root_postproc'] + paramsCamera['dir_input']
    else: 
        dir_in       = src_dir+'/'+ params['root_data']     + params['root_data_DirMir'] 
        dir_out_mir =  src_dir+'/'+ params['root_postproc'] + paramsCamera['dir_input']

    if (params_flag['flag_mir_processRawData']) & (os.path.isdir(dir_out_mir+paramsCamera['dir_img_input'])) : shutil.rmtree(dir_out_mir+paramsCamera['dir_img_input'])
    if not os.path.isdir(dir_out_mir+paramsCamera['dir_img_input']): tools.ensure_dir(dir_out_mir+paramsCamera['dir_img_input'])
   
    #time for vis and lwir were set to match mir, so time we read here is the reference time

    mir_filenames = sorted(glob.glob(dir_in+'*.MAT'))
    out_time = np.array(len(mir_filenames)*[('mm',0)], dtype=np.dtype([('name','S100'),('time',float)]))
    out_time = out_time.view(np.recarray)
    #sort file in chronological order
    file_id = np.zeros(len(mir_filenames),dtype=np.int)
    for i_file, mir_file in enumerate(mir_filenames):
        file_id[i_file] = int(os.path.basename(mir_file).split('.')[0].split('_')[-1]) 
 
    file_base_name = os.path.basename(mir_filenames[0]).split('.')[0].split('_')[0]
    for i_file, mir_file in enumerate(np.array(mir_filenames)[np.argsort(file_id)]):
        mir_time,temp_mir = read_agema(mir_file)
        
        name_ = dir_out_mir+paramsCamera['dir_img_input']+file_base_name+'_{:06d}.MAT'.format(i_file)

        #create link on dir_out_mir
        os.symlink(mir_file, name_)

        out_time.time[i_file] = (mir_time-ignitionTime).total_seconds()
        out_time.name[i_file] = name_ 
        
    np.save(dir_out_mir+'/'+ paramsCamera['dir_img_input']+'filename_time',out_time)
    print(' done')


################################################
def read_agema(filename):
    res = io.loadmat(filename)
    MIR_name = list(res.keys())[np.array([len(string) for string in list(res.keys())]).argmin()]
    in_ = np.array(res[MIR_name+'_DateTime'][0],dtype=np.int) 
    in_[-1] = 1.e3 * in_[-1]
    mir_time = datetime.datetime(*in_)
    mir = res[MIR_name][::-1].T
    return mir_time, mir


################################################
def define_firstGuessWarp_mir_on_lwir(path_data, mir_name, lwir_name, cp_lwir_mir, flag_plot=False):

    #read mir
    ##########
    mir_file = path_data + mir_name
    mir_time, mir = read_agema(mir_file)
    
    #read lwir
    ##########
    lwir_file = path_data +  lwir_name
    reader = asciitable.NoHeader()
    reader.data.splitter.delimiter = ';'
    reader.data.splitter.process_line = None
    reader.data.start_line = 0
    data = reader.read(lwir_file)
    nx = len(data)
    ny = len(data[0][:-1])
    out = np.zeros([ny,nx])
    for i in range(nx):
        out[:,i] = data[i][:-1]
    lwir = out.T[::-1].T
    date_str, time_str = os.path.basename(lwir_file).split('.')[0].split('_')[1:3] 
    lwir_time = datetime.datetime.strptime(date_str+'_'+time_str,"%Y-%m-%d_%H-%M-%S")


    # Find the width and height of the color image
    height = lwir.shape[1]
    width = lwir.shape[0]

    # Allocate space for aligned image
    ir_aligned = np.zeros((width,height,2))
    ir_aligned[:,:,0] =  lwir

    # Define motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    lwir_gcps =  np.array(cp_lwir_mir[0]) + 50
    mir_gcps  =  np.array(cp_lwir_mir[1]) + 50
    src_pts = np.array((np.array(mir_gcps)[:,1],np.array(mir_gcps)[:,0])).T #; src_pts[:,1] =  mir.shape[1] - src_pts[:,1]
    dst_pts = np.array((np.array(lwir_gcps)[:,1],np.array(lwir_gcps)[:,0])).T #; src_pts[:,1] =  mir.shape[1] - src_pts[:,1]

    warp_matrix1, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    warp_matrix1 = np.array(warp_matrix1,dtype=np.float32)

    _, g_lwir, _ = tools.get_gradient(lwir)
   
    _, g_mir, _ = tools.get_gradient(mir)
    g_mir /= max([g_mir.max(),(-1*g_mir).max()]) 
    g_mir = np.copy(g_mir)
    g_mir_warp = cv2.warpPerspective(g_mir, warp_matrix1, (height,width))
    warp_mask = np.where( cv2.warpPerspective(np.ones_like(g_mir), warp_matrix1, (height,width)) == 1, np.ones(g_lwir.shape,dtype=np.uint8), 
                                                                                                      np.zeros(g_lwir.shape,dtype=np.uint8))
  
    g_lwir /=  max([g_lwir[np.where(warp_mask==1)].max(),(-1*g_lwir[np.where(warp_mask==1)]).max()]) 

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000,  1e-6)
    
    warp_matrix2_init = np.eye(3, 3, dtype=np.float32)
    (cc, warp_matrix2) = cv2.findTransformECC(g_lwir, g_mir_warp, warp_matrix2_init,  cv2.MOTION_HOMOGRAPHY, criteria, inputMask=warp_mask)
  

    g_mir_warp2 = cv2.warpPerspective (g_mir_warp,      warp_matrix2, g_lwir.shape[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    H_mir2lwir = np.dot(np.linalg.inv(warp_matrix2), warp_matrix1)

    if flag_plot:
        ax = plt.subplot(221)
        ax.imshow(g_lwir.T,origin='lower',interpolation='nearest',vmin=0,vmax=1)
        ax.scatter(dst_pts[:,1]-50,dst_pts[:,0]-50,marker='o',s=20,facecolors='none',edgecolors='r')
        
        ax = plt.subplot(223)
        ax.imshow(g_mir.T,origin='lower',interpolation='nearest',vmin=0,vmax=1)
        ax.scatter(src_pts[:,1]-50,src_pts[:,0]-50,marker='o',s=20,facecolors='none',edgecolors='r')
        mir_time, mir = read_agema(mir_file)


        ax = plt.subplot(222)
        ax.imshow(g_lwir.T,origin='lower',interpolation='nearest',vmin=0,vmax=1)
        ax.imshow(np.ma.masked_where(g_mir_warp==0,g_mir_warp).T,alpha=.5,cmap=mpl.cm.Greys_r,origin='lower')
        
        ax = plt.subplot(224)
        ax.imshow(g_lwir.T,origin='lower',interpolation='nearest',vmin=0,vmax=1)
        ax.imshow(np.ma.masked_where(g_mir_warp2==0,g_mir_warp2).T,alpha=.5,cmap=mpl.cm.Greys_r,origin='lower')
        
        plt.show()

        pdb.set_trace()
   
    return H_mir2lwir


