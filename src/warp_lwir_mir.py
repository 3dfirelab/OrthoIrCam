from __future__ import print_function
from __future__ import division
from builtins import input
from builtins import zip
from builtins import range
from past.utils import old_div
import cv2
import numpy as np
import matplotlib as mpl
import socket
if 'kraken' in socket.gethostname(): mpl.use('Agg')
if 'ibo'    in socket.gethostname(): mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import io, ndimage
import asciitable
import datetime 
import os
import sys
import glob 
import argparse
import importlib
import shutil 
from osgeo import gdal,osr,ogr
import pdb
import itertools
import socket
import shapely
import shapely.ops
import scipy 
import multiprocessing
import math 
from mpl_toolkits.axes_grid1 import make_axes_locatable

#homebrewed
import agema
import optris
import tools
import camera_tools
import spectralTools
import tools_georefWithTerrain as georefWT
import refine_lwir as refine

#################################################
def findTransformECC_stepbystep(img_g, img_ref_g, img_mask, img_mask_ref, 
                                trans_len_limit=[80,40], ep08_limit=[0.7,.6], 
                                first_guessed_H=None, flag_plot_=False, flag_print_=False, res_factor=[2], img_mask_ref2=None, flag_maskEdge=False ):

    plotFormat = '13' if not(flag_maskEdge) else '14' 
    
    communArea_limit_here = .5
    if flag_plot_ : print('')
    trans_vec_arr = []

    EP08_0 = tools.get_EP08_from_img(img_g, img_ref_g, inputMask=img_mask, inputMask_ref=img_mask_ref)
    mask_both = np.zeros_like(img_mask)
    mask_both[np.where(img_mask    ==1)] += 1
    mask_both[np.where(img_mask_ref==1)] += 2
    communArea = old_div(1.*np.where(mask_both==3)[0].shape[0],min([np.where(img_mask==1)[0].shape[0],np.where(img_mask_ref==1)[0].shape[0]]))
    if communArea < communArea_limit_here: 
        EP08_0 = 0

    if flag_plot_: print('0 ', EP08_0)
    id_ecc = 0
        
    '''
    ax = plt.subplot(121)
    plt.imshow(np.ma.masked_where(img_mask_ref==0, img_ref_g).T,origin='lower')
    ax = plt.subplot(122)
    plt.imshow(np.ma.masked_where(img_mask==0, img_g).T,origin='lower',alpha=.5,cmap=mpl.cm.Greys_r)
    plt.show()
    '''

    #criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-5)
    #EP08_1 = tools.get_EP08_from_img(img_g,img_ref_g,inputMask=img_mask, inputMask_ref=img_mask_ref)

    #TRANSLATION ON LOWER IMAGE RESOLUTION
    warp_matrix1  = np.eye(3, 3, dtype=np.float32)
    warp_matrix1_ = np.eye(2, 3, dtype=np.float32)
    img_g1 = img_g
    img_mask1 = img_mask
    EP08_1 = EP08_0
    trans_vec = 0
    
    for ires, res_factor_ in enumerate(res_factor): 
        
        '''
        probably missing a control on this loop to only accept improvement at each resolution
        '''
        
        ______coarse     = np.zeros([int(math.ceil(old_div(1.*img_g.shape[0],res_factor_))),int(math.ceil(old_div(1.*img_g.shape[1],res_factor_)))])
        img_g_coarse     =  tools.downgrade_resolution_4nadir(img_g1,     ______coarse.shape, flag_interpolation='conservative')
        img_ref_g_coarse =  tools.downgrade_resolution_4nadir(img_ref_g, ______coarse.shape, flag_interpolation='conservative')
        img_mask_coarse  =  tools.downgrade_resolution_4nadir(img_mask1, ______coarse.shape, flag_interpolation='max')
        img_mask_ref_coarse  =  tools.downgrade_resolution_4nadir(img_mask_ref, ______coarse.shape, flag_interpolation='max')
       
        if trans_len_limit[ires] >= 10: 
            trans_len = 10
            ep08 = 0.
            info_prev = None
            while ep08 < ep08_limit[ires]:
                trans_x = np.arange(-trans_len,trans_len+1)
                trans_y = np.arange(-trans_len,trans_len+1)
                nbre_iter = trans_x.shape[0]*trans_y.shape[0]
                info_ = np.zeros([nbre_iter,4])
                for ivec, vec in enumerate(itertools.product(trans_x, trans_y)) :
                    if info_prev is not None:
                        idx_ = np.where( ((vec[0]-info_prev[:,0])**2 + (vec[1]-info_prev[:,1])**2)==0)
                        if len(idx_[0])!=0:
                            info_[ivec,:] = info_prev[idx_,:]
                    info_[ivec, :2] = vec
                    info_[ivec, 2], info_[ivec, 3] = tools.apply_translation_img(img_g_coarse, img_mask_coarse, vec, img_ref_g_coarse, img_mask_ref_coarse,communArea_limit=0)

                #idx_ = np.where(info_[:,2] >= .8* info_[:,2].max()) # first get the vec with corr > 80% of the max
                #idx2_ = idx_[0][info_[ idx_[0][ np.where( info_[idx_[0],3] == info_[idx_,3].max()) ], 2].argmax()] # then get the vec that have max match and get max corr on those
                
                idx2__ = np.where(info_[:, 3]>communArea_limit_here)
                if len(idx2__[0])>0:
                    idx2_ = info_[idx2__,2].argmax()
                    idx2_ = idx2__[0][idx2_]
                else: 
                    idx2_ = np.where((info_[:, 0]==0) & (info_[:, 1]==0) )[0][0]

                ep08       = info_[idx2_,2]
                trans_vec  = info_[idx2_,:2]
                communAreas= info_[idx2_,3]
                
                trans_len += 10
                info_prev = np.copy(info_)
                if trans_len > trans_len_limit[ires]: 
                    if flag_print_: print('**', end=' ')
                    break
            
            #print trans_len-10,
            #warp_matrix1_ = np.eye(2, 3, dtype=np.float32)
            #warp_matrix1_[:,-1] = trans_vec * res_factor_
            warp_matrix1_[:,-1] = trans_vec * res_factor_ + warp_matrix1_[:,-1]
            warp_matrix1 = np.eye(3, 3, dtype=np.float32); warp_matrix1[:2,:] = warp_matrix1_
            img_g1 = cv2.warpAffine(img_g,      warp_matrix1_, img_ref_g.shape[::-1], flags=cv2.INTER_LINEAR  )
            img_mask1  = np.where( (cv2.warpAffine(img_mask, warp_matrix1_, img_ref_g.shape[::-1], 
                                                 flags=cv2.INTER_NEAREST ) == 1), np.ones(img_ref_g.shape ,dtype=np.uint8), 
                                                                                  np.zeros(img_ref_g.shape,dtype=np.uint8))
            EP08_1 = tools.get_EP08_from_img(img_g1,img_ref_g,inputMask=img_mask1, inputMask_ref=img_mask_ref)
            mask_both = np.zeros_like(img_mask1)
            mask_both[np.where(img_mask1    ==1)] += 1
            mask_both[np.where(img_mask_ref==1) ] += 2
            communArea = old_div(1.*np.where(mask_both==3)[0].shape[0],min([np.where(img_mask1==1)[0].shape[0],np.where(img_mask_ref==1)[0].shape[0]]))
            #if communArea < communArea_limit_here: 
            #    EP08_1 = 0
            
            #plt.imshow((2*img_mask_ref+img_mask1).T, origin='lower'); plt.show()
            #pdb.set_trace() 

        #else: 
        #    warp_matrix1 = np.eye(3, 3, dtype=np.float32)
        #    warp_matrix1_ = np.eye(2, 3, dtype=np.float32)
        #    img_g1 = img_g
        #    img_mask1 = img_mask
        #    EP08_1 = EP08_0
        #    trans_vec = 0

        trans_vec_arr.append(trans_vec * res_factor)

        if flag_plot_: 
            print('1-{:d} '.format(ires), EP08_1, trans_vec)
            ax = plt.subplot('{:s}1'.format(plotFormat))
            plt.imshow(np.ma.masked_where(img_mask_ref==0, img_ref_g).T,origin='lower')
            plt.imshow(np.ma.masked_where(img_mask1==0, img_g1).T,origin='lower',alpha=.5,cmap=mpl.cm.Greys_r)

    #2nd TRANSLATION AT FULL RESOLUTION
    if trans_len_limit[-1]>= 10:
        trans_len = 10
        ep08 = 0.
        info_prev = None
        while ep08 < ep08_limit[-1]:
            trans_x = np.arange(-trans_len,trans_len+1)
            trans_y = np.arange(-trans_len,trans_len+1)
            nbre_iter = trans_x.shape[0]*trans_y.shape[0]
            info_ = np.zeros([nbre_iter,4])
            for ivec, vec in enumerate(itertools.product(trans_x, trans_y)) :
                if info_prev is not None:
                    idx_ = np.where( ((vec[0]-info_prev[:,0])**2 + (vec[1]-info_prev[:,1])**2)==0)
                    if len(idx_[0])!=0:
                        info_[ivec,:] = info_prev[idx_,:]
                info_[ivec, :2] = vec
                #info_[ivec, 2:] = apply_translation(img_g1, vec, img_ref_g, img_mask1)
                info_[ivec, 2], info_[ivec, 3] = tools.apply_translation_img(img_g1, img_mask1, vec, img_ref_g, img_mask_ref, communArea_limit=0.)

            '''
            try: 
                idx_ = np.where(info_[:,2] >= .8* info_[:,2].max())
                idx2_ = idx_[0][info_[ idx_[0][ np.where( info_[idx_[0],3] == info_[idx_,3].max()) ], 2].argmax()] 
            except: 
                pdb.set_trace()
            '''
            #idx2_ = info_[:,2].argmax()
            idx2__ = np.where(info_[:, 3]>communArea_limit_here)
            if len(idx2__[0])>0:
                idx2_ = info_[idx2__,2].argmax()
                idx2_ = idx2__[0][idx2_]
            else: 
                idx2_ = np.where((info_[:, 0]==0) & (info_[:, 1]==0) )[0][0]
            
            
            ep08       = info_[idx2_,2]
            trans_vec  = info_[idx2_,:2]
            communAreas= info_[idx2_,3]

            trans_len += 10
            info_prev = np.copy(info_)

            if trans_len > trans_len_limit[-1]+10: 
                if flag_print_: print('**({:.3f},[{:3.1f},{:3.1f}])'.format(ep08,trans_vec[0],trans_vec[1]), end=' ') 
                break

        #print trans_len-10,  
        warp_matrix2_ = np.eye(2, 3, dtype=np.float32)
        warp_matrix2_[:,-1] = trans_vec + warp_matrix1_[:,-1]
        warp_matrix2 = np.eye(3, 3, dtype=np.float32); warp_matrix2[:2,:] = warp_matrix2_
        

        img_g2 = cv2.warpAffine(img_g,      warp_matrix2_, img_ref_g.shape[::-1], flags=cv2.INTER_LINEAR )
        img_mask2  = np.where( (cv2.warpAffine(img_mask, warp_matrix2_, img_ref_g.shape[::-1], 
                                         flags=cv2.INTER_NEAREST ) == 1), np.ones(img_ref_g.shape ,dtype=np.uint8), 
                                                                                                np.zeros(img_ref_g.shape,dtype=np.uint8))
        EP08_2 = tools.get_EP08_from_img(img_g2,img_ref_g,inputMask=img_mask2, inputMask_ref=img_mask_ref)
        #sys.stdout.flush()
    
    else:
        warp_matrix2 = warp_matrix1
        img_g2 = img_g1
        img_mask2 = img_mask1
        EP08_2 = EP08_1
        trans_vec = 0

    trans_vec_arr.append(trans_vec)


    if flag_plot_: print('2 ', EP08_2,trans_vec, end=' ')
    if EP08_2 < EP08_1:
        if flag_plot_: print('$')

        warp_matrix2_ = np.eye(2, 3, dtype=np.float32)
        warp_matrix2  = np.eye(3, 3, dtype=np.float32)
        EP08_2 = EP08_0
        img_g2 = cv2.warpAffine(img_g1,      warp_matrix2_, img_ref_g.shape[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP )
        img_mask2  = np.where( (cv2.warpAffine(img_mask1, warp_matrix2_, img_ref_g.shape[::-1], 
                                         flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) == 1), np.ones(img_ref_g.shape ,dtype=np.uint8), 
                                                                                                np.zeros(img_ref_g.shape,dtype=np.uint8))
    else: 
        if flag_plot_: print()
        id_ecc = 2

    if flag_plot_: 
        ax = plt.subplot('{:s}2'.format(plotFormat))
        plt.imshow(np.ma.masked_where(img_mask_ref==0, img_ref_g).T,origin='lower')
        plt.imshow(np.ma.masked_where(img_mask2==0, img_g2).T,origin='lower',alpha=.5,cmap=mpl.cm.Greys_r)

    '''
    #AFFINE
    warp_matrix3_init = np.copy( np.linalg.inv(warp_matrix2_.dot(warp_matrix1_)))
    pdb.set_trace()
    (cc, warp_matrix3_) = cv2.findTransformECC(img_ref_g, img_g, warp_matrix3_init,  cv2.MOTION_AFFINE, criteria, 
                                               inputMask    = img_mask,
                                               templateMask = img_mask_ref)
    warp_matrix3 = np.eye(3, 3, dtype=np.float32); warp_matrix3[:2,:] = warp_matrix3_
    img_g3     = cv2.warpAffine(img_g,      warp_matrix3_, img_ref_g.shape[::-1], flags=cv2.INTER_LINEAR  + cv2.WARP_INVERSE_MAP )
    img_mask3  = cv2.warpAffine(img_mask2,   warp_matrix3_, img_ref_g.shape[::-1], flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) 
    EP08_3 = tools.get_EP08_from_img(img_g3,img_ref_g,inputMask=img_mask3, inputMask_ref=img_mask_ref)
    if flag_plot_: print '3 ',  EP08_3,cc, 
    if EP08_3 < EP08_2:
        if flag_plot_: print '$'
        warp_matrix3 = np.copy( warp_matrix2.dot(warp_matrix1))
        EP08_3 = EP08_2
        img_g3     = cv2.warpAffine(img_g,    warp_matrix3_, img_ref_g.shape[::-1], flags=cv2.INTER_LINEAR  + cv2.WARP_INVERSE_MAP)
        img_mask3  = cv2.warpAffine(img_mask2, warp_matrix3_, img_ref_g.shape[::-1], flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
    else: 
        id_ecc = 3
        if flag_plot_: print
    
    if flag_plot_: 
        ax = plt.subplot(143)
        plt.imshow(np.ma.masked_where(img_mask_ref==0, img_ref_g).T,origin='lower')
        plt.imshow(np.ma.masked_where(img_mask3==0, img_g3).T,origin='lower',alpha=.5,cmap=mpl.cm.Greys_r)
    '''

    if flag_maskEdge:
        # ECC on mask + made-up edge
        mask_  = np.array(img_mask,dtype=np.float32)
        for i in range(res_factor[0]):
            mask__ = np.array(255*np.where(mask_>0,1,0),np.uint8)
            tmp_= cv2.dilate(mask__, np.ones((3,3),np.uint8) , iterations = 1)
            idx_ = np.where((tmp_==255)&(mask__==0))
            mask_[idx_] = 1- old_div(1.*(i+1),res_factor[0])
            #print i, res_factor[0],  1- 1.*(i+1)/res_factor[0]
            #plt.imshow(mask_.T, origin='lower'); plt.show()
            #pdb.set_trace()
        
        mask_ref  = np.array(img_mask_ref,dtype=np.float32)
        for i in range(res_factor[0]):
            mask__ = np.array(255*np.where(mask_ref>0,1,0),np.uint8)
            tmp_= cv2.dilate(mask__, np.ones((3,3),np.uint8) , iterations = 1)
            idx_ = np.where((tmp_==255)&(mask__==0))
            mask_ref[idx_] = 1- old_div(1.*(i+1),res_factor[0])
       
        #plt.figure()
        #plt.imshow( (mask_+mask_ref).T, origin='lower')

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-3)
        warp_matrix2_init =  np.copy( np.linalg.inv(warp_matrix2) ) 
        (cc, warp_matrix3) = cv2.findTransformECC(np.copy(mask_ref), np.copy(mask_), warp_matrix2_init,  cv2.MOTION_HOMOGRAPHY, criteria) 

        img_g3     = cv2.warpPerspective(img_g,    warp_matrix3, img_ref_g.shape[::-1], flags=cv2.INTER_LINEAR  + cv2.WARP_INVERSE_MAP)
        img_mask3  = cv2.warpPerspective(img_mask, warp_matrix3, img_ref_g.shape[::-1], flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        EP08_3 = tools.get_EP08_from_img(img_g3,img_ref_g,inputMask=img_mask3, inputMask_ref=img_mask_ref)
        if img_mask3.max() != 0: 
            Area_3 = tools.get_maskCommunAera(     img_g3,img_ref_g,inputMask=img_mask3, inputMask_ref=img_mask_ref)
        else:  
            Area_3 = 0.
        #plt.figure()
        #plt.imshow( (img_mask3+mask_ref).T, origin='lower')
        #plt.show()
        if flag_plot_: print('3 ', EP08_3, Area_3, end=' ') 
        
        if EP08_3 < EP08_2:
            if flag_plot_:print('$')
            #print '({:.5f})'.format(EP08_4-EP08_2), 
            warp_matrix3 = np.linalg.inv(warp_matrix2)
            Area_3 = 0.
            EP08_3 = EP08_2
        else: 
            if flag_plot_: print()
            id_ecc = 3 

        if flag_plot_:
            ax = plt.subplot(143)
            ax.imshow(np.ma.masked_where(img_mask_ref==0, img_ref_g).T,origin='lower')
            ax.imshow(np.ma.masked_where(img_mask3==0, img_g3).T,origin='lower',alpha=.5,cmap=mpl.cm.Greys_r)
    else:
        EP08_3 = EP08_2
        Area_3 = 0.
        warp_matrix3 = np.linalg.inv(warp_matrix2)


    #HOMOGRAPHY
    if img_mask_ref2 is None: 
        img_mask_ref_ = img_mask_ref
    else:
        img_mask_ref_ = img_mask_ref2

    #criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100,  1e-3)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-3)
    try:     
        warp_matrix4_init =  np.copy( warp_matrix3 ) 
        (cc, warp_matrix4) = cv2.findTransformECC(img_ref_g, img_g, warp_matrix4_init,  cv2.MOTION_HOMOGRAPHY, criteria, 
                                                  inputMask    = img_mask, 
                                                  templateMask = img_mask_ref_                                           )
        img_g4     = cv2.warpPerspective(img_g,    warp_matrix4, img_ref_g.shape[::-1], flags=cv2.INTER_LINEAR  + cv2.WARP_INVERSE_MAP)
        img_mask4  = cv2.warpPerspective(img_mask, warp_matrix4, img_ref_g.shape[::-1], flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        EP08_4 = tools.get_EP08_from_img(img_g4,img_ref_g,inputMask=img_mask4, inputMask_ref=img_mask_ref)
        Area_4 = tools.get_maskCommunAera(     img_g4,img_ref_g,inputMask=img_mask4, inputMask_ref=img_mask_ref)
    except: 
        EP08_4 = 0 
        Area_4 = 0
        warp_matrix4 = warp_matrix3
        cc = -999.
        #pdb.set_trace()
        #print warp_matrix4

    if flag_plot_:print('4 ', EP08_4, Area_4, cc, end=' ') 
    
    if (EP08_4 < EP08_3) | (Area_4 < Area_3):
        if flag_plot_:print('$')
        #print '({:.5f})'.format(EP08_4-EP08_2), 
        warp_matrix4 = warp_matrix3
    else: 
        if flag_plot_: print()
        id_ecc = 4
   
    if (flag_plot_) & (cc!=-999):
        ax = plt.subplot('{:s}3'.format(plotFormat)) if not(flag_maskEdge) else  plt.subplot('{:s}4'.format(plotFormat))
        ax = plt.subplot(144)
        plt.imshow(np.ma.masked_where(img_mask_ref_==0, img_ref_g).T,origin='lower')
        plt.imshow(np.ma.masked_where(img_mask4==0, img_g4).T,origin='lower',alpha=.5,cmap=mpl.cm.Greys_r)

    if (flag_plot_):
        plt.show()
        pdb.set_trace()

    return id_ecc, warp_matrix4, trans_vec_arr #(EP08_4-EP08_2)/EP08_2
    


############################################################
def apply_translation(g_mir_warp, trans_vec, g_lwir, warp_mask):

    warp_trans = np.eye(2, 3, dtype=np.float32)
    warp_trans[:,-1] = trans_vec
    g_mir_trans = cv2.warpAffine(g_mir_warp,      warp_trans, g_lwir.shape[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP )
    warp_mask_trans  = np.where( (cv2.warpAffine(warp_mask, warp_trans, g_lwir.shape[::-1], 
                                         flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) == 1), np.ones(g_lwir.shape ,dtype=np.uint8), 
                                                                                                np.zeros(g_lwir.shape,dtype=np.uint8))
    EP08_trans = get_EP08(g_mir_trans,g_lwir,inputMask=warp_mask_trans)
    return EP08_trans


############################################################
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=3)
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return np.array(grad,dtype=np.float32)


############################################################
def get_EP08(img, img_ref, inputMask=None):
  
    if inputMask is None:
        idx_mask_ = np.where( img > -.1e-6) 
    else: 
        idx_mask_ = np.where( inputMask==1 )

    iw = np.array( img[idx_mask_].flatten()     - img[idx_mask_].mean(), dtype=np.float64)
    ir = np.array( img_ref[idx_mask_].flatten() - img_ref[idx_mask_].mean(), dtype=np.float64)

    return old_div(np.dot(ir,iw),(np.linalg.norm(ir)*np.linalg.norm(iw)))



############################################################
def star_frp_per_pointGrid(param):
    return frp_per_pointGrid(*param)

#-----------------------------------------------------------
def frp_per_pointGrid(pts, H2Grid, burnplot): #frameMir, burnplot, poly_world_zone, radiance ):
        
    result = []

    img_polygons_ = shapely.geometry.Polygon(pts)

    pts_world = cv2.perspectiveTransform( np.array(pts[:,::-1],dtype=np.float32).reshape(-1,1,2), H2Grid)
    pts_world = pts_world[:,0,:][:,::-1]
    poly_world = shapely.geometry.Polygon(pts_world)
   
    if old_div(poly_world.intersection(poly_world_zone).area,poly_world.area) < 0.5: 
        return result

    x = np.arange(max([0,int(np.round(pts_world[:,0].min()-2,0))]), min([int(np.round(pts_world[:,0].max()+2,0)),burnplot.mask.shape[0]]) )
    y = np.arange(max([0,int(np.round(pts_world[:,1].min()-2,0))]), min([int(np.round(pts_world[:,1].max()+2,0)),burnplot.mask.shape[1]]) )

    xx, yy = np.meshgrid(x, y)
    grid_e_ = burnplot.grid_e[xx,yy]
    try:
        grid_e_interp = scipy.interpolate.interp2d(x, y, grid_e_ , kind='linear')
    except: 
        pdb.set_trace()
    grid_n_ = burnplot.grid_n[xx,yy]
    grid_n_interp = scipy.interpolate.interp2d(x, y, grid_n_ , kind='linear')
    
    terrain_ = burnplot.terrain[xx,yy]
    terrain_interp = scipy.interpolate.interp2d(x, y, terrain_ , kind='linear')

    #poly_world_area
    pts_world_world_e       = [ grid_e_interp(*pts_world[ii])  for ii in range(4) ]
    pts_world_world_n       = [ grid_n_interp(*pts_world[ii])  for ii in range(4) ]
    pts_world_world_terrain = [ terrain_interp(*pts_world[ii]) for ii in range(4) ]

    pts_world_world=np.dstack((pts_world_world_e, pts_world_world_n, pts_world_world_terrain))[:,0,:]
    #sum up area of 2 triangles
    poly_world_area =  \
        .5*np.linalg.norm(np.cross( (pts_world_world[1] -pts_world_world[0]), (pts_world_world[0] -pts_world_world[2]) )) + \
        .5*np.linalg.norm(np.cross( (pts_world_world[3] -pts_world_world[2]), (pts_world_world[2] -pts_world_world[0]) ))

    hlines = [((x1, yi), (x2, yi)) for x1, x2 in zip(x[:-1], x[1:]) for yi in y]
    vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(y[:-1], y[1:]) for xi in x]

    grids = list(shapely.ops.polygonize((shapely.geometry.MultiLineString(hlines + vlines))))
   
    result.append([pts[0],poly_world_area])
    for poly in grids: 
        intersection = poly_world.intersection(poly)
        if intersection.area!=0:
            ii, jj = int(poly.exterior.coords.xy[0][0]), int(poly.exterior.coords.xy[1][0])
            result.append([ ii, jj, old_div(intersection.area,poly_world.area) *poly_world_area* radiance[pts[0][0],pts[0][1]] ])

    return result

############################################################
def warpPerspective_p2p(flag_parallel_, frameMir, burnplot, param_set_radiance, flag_restart ):
    
    global radiance
    global poly_world_zone

    radiance = spectralTools.conv_temp2Rad(frameMir.temp, *param_set_radiance)
    background_radiance = radiance.min()

    nx, ny = burnplot.mask.shape
    poly_world_zone = shapely.geometry.Polygon([ [ 0 , 0  ], \
                                                 [ nx, 0  ], \
                                                 [ nx, ny ], \
                                                 [ 0 , ny ], \
                                               ]) 
    idxi = np.arange(0,radiance.shape[0]) 
    idxj = np.arange(0,radiance.shape[1]) 
    params = []
    for iii,jjj in itertools.product(idxi, idxj):
       
        #if radiance[iii,jjj] < 54: continue
        
        pts = [ [ iii  , jjj   ], \
                [ iii+1, jjj   ], \
                [ iii+1, jjj+1 ], \
                [ iii  , jjj+1 ], \
              ]
        pts = np.array(pts)
        params.append([pts,frameMir.H2Grid, burnplot]) #, frameMir, burnplot, poly_world_zone, radiance])

    #flag_parallel_ = False
    pixelSize = np.zeros_like(radiance)
    if flag_parallel_:
        # set up a pool to run the parallel processing
        cpus = tools.cpu_count()
        pool = multiprocessing.Pool(processes=cpus)

        # then the map method of pool actually does the parallelisation  
        result_ = pool.map(star_frp_per_pointGrid, params)
        
        result = []
        for res_ in result_ :
            if len(res_)==0: continue
            [result.append(res) for res in res_[1:] ] 
            pixelSize[res_[0][0][0],res_[0][0][1]] = res_[0][1]
        
        pool.close()
        pool.join()
       
    else:
        result = []
        for param in params:
            res_all = star_frp_per_pointGrid(param)
            if len(res_all)==0: continue
            [ result.append(res[1:]) for res in res_all[1:] ] 
            try:
                pixelSize[res_all[0][0][0],res_all[0][0][1]] = res_all[0][1]
            except: 
                pdb.set_trace()

    georef_radiance = np.zeros_like(burnplot.mask)
    for res in result:
        try: 
            georef_radiance[res[0],res[1]] += res[2]                 
        except: 
            pdb.set_trace()

    #divide by pixel size to get back to radiance
    georef_radiance /= burnplot.area 

    #plt.imshow(georef_radiance.T,origin='lower'); plt.show()
    #pdb.set_trace()

    return georef_radiance, pixelSize


#############################################
def star_stabilize_and_georef(param):
    return stabilize_and_georef(*param)

#-------------------------------------------
def stabilize_and_georef(flag_parallel, flag_restart,
                         imir, mir_frame_name, mir_info, badId_lwir, georefMode, 
                         path_, dir_out_mir_frame, dir_out_georef_npy_refine_2nd, dir_out_mir_georef_npy, dir_out_mir_georef_png,
                         params_mir_camera, plotname):
    
    mir_frame_name = mir_frame_name.replace('/scratch/globc/paugam/data/',path_)

    #if imir < 2970: continue

    try: 
        if mir_info.time[imir] < params_mir_camera['time_start']: return 'frame skipped, time<t0'
    except: 
        pass

    if (flag_restart) & (os.path.isfile(dir_out_mir_frame+'frameMIR{:06d}.nc'.format(imir))) :
        frameMir = agema.load_existing_file(dir_out_mir_frame+'frameMIR{:06d}.nc'.format(imir))
        frameLwir = optris.load_existing_file(params_lwir_camera, frameMir.lwir_filename)
        if not(flag_parallel):print('{:15s}'.format(' frame loaded'), end=' ')
    else:
        #print 'force init'
        frameMir = agema.loadFrame()
        frameMir.init(imir, mir_frame_name, ignitionTime, K, D)

        if np.abs(lwir_info.time_igni - frameMir.time_igni).min() > .3: 
            #print ' timediff = ', np.abs(lwir_info.time_igni - frameMir.time_igni).min(), ' skip' 
            #print '{:15s}'.format(' frame skipped ')
            return 'frame skipped'
        if not(flag_parallel):print('{:5.2f}'.format(frameMir.time_igni), end=' ')
        idx_lwir = np.abs(lwir_info.time_igni - frameMir.time_igni).argmin()
        frameLwir = optris.load_existing_file(params_lwir_camera, lwir_info.filename[idx_lwir])

        if (not(os.path.isfile(dir_out_georef_npy_refine_2nd + '{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,frameLwir.id,georefMode)))): 
            #print '{:s}'.format(' frame skipped: no refined lwir file found ')
            return 'frame skipped: no refined lwir file found'
        
        if idx_lwir in badId_lwir:
            #print '{:s}'.format(' frame skipped: lwir frame was removed by ssim_prev.py ')
            return 'frame skipped: lwir frame was removed by ssim_prev.py'

        if not(flag_parallel): print(frameLwir.id, end=' ') 

        homogra_mat = np.load(dir_out_georef_npy_refine_2nd + '{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,frameLwir.id,georefMode),allow_pickle=True)[1]
        lwirH2Grid = homogra_mat#.dot(frameLwir.H2Grid)
       
        #reso = params_lwir_camera['reso_refined']
        #scaling_mat = np.identity(3); scaling_mat[0,0] = reso; scaling_mat[1,1] = reso
        #lwirH2Grid = scaling_mat.dot(homogra_mat).dot(np.linalg.inv(scaling_mat)).dot(frameLwir.H2Grid)
        
        #if refined_info[2] != 'applied': 
        #    print '{:15s}'.format(' frame skipped ')
        #    continue

        frameMir.set_matching_lwir_info(frameLwir.id, lwir_info.filename[idx_lwir])

        #compute gradient
        ##_, g_lwir, _ = tools.get_gradient(frameLwir.temp[frameLwir.bufferZone/2:-frameLwir.bufferZone/2,frameLwir.bufferZone/2:-frameLwir.bufferZone/2])
        
        diskSize = 30
        
        temp_lwir = old_div(frameLwir.temp,frameLwir.temp.max())
        temp_lwir_filtered = np.where(frameLwir.temp +273.14 > 350, frameLwir.temp +273.14, 350)
        
        mask_lwir2  = np.array(np.where(temp_lwir_filtered>350, 1, 0), dtype=np.uint8)
        frameLwir_temp_normed = refine.local_normalization(temp_lwir_filtered , mask_lwir2,   diskSize=diskSize)
        frameLwir_temp_normed = np.where(mask_lwir2==0,0,frameLwir_temp_normed)
        
        
        temp_mir  = old_div(frameMir.temp,frameMir.temp.max())
        temp_mir_filtered = np.where(frameMir.temp > 480, frameMir.temp, 480)
        mask_mir   = np.array(np.ones_like(temp_mir), dtype=np.uint8)
        
        mask_mir2  = np.array(np.where(frameMir.temp>480, 1, 0), dtype=np.uint8)
        frameMir_temp_normed  = refine.local_normalization(temp_mir_filtered , mask_mir2 ,   diskSize=diskSize)
        frameMir_temp_normed = np.where(mask_mir2==0,0,frameMir_temp_normed)


        _, g_lwir, _ = tools.get_gradient(frameLwir_temp_normed)
        _, g_mir, _ = tools.get_gradient(frameMir_temp_normed)
        

        #define mask of mir on lwir frame
        temp_mir_warp = cv2.warpPerspective(frameMir.temp, H_mir2lwir_guess, g_lwir.shape[::-1])
        mask_mir_warp  = np.where( (cv2.warpPerspective(mask_mir, H_mir2lwir_guess, g_lwir.shape[::-1]) == 1), np.ones(g_lwir.shape ,dtype=np.uint8), 
                                                                                                               np.zeros(g_lwir.shape,dtype=np.uint8))
        mask_mir_warp2 = np.where( (cv2.warpPerspective(mask_mir2, H_mir2lwir_guess, g_lwir.shape[::-1]) == 1), np.ones(g_lwir.shape ,dtype=np.uint8), 
                                                                                                               np.zeros(g_lwir.shape,dtype=np.uint8))
         #normalize gradient
        g_mir /= max([g_mir.max(),(-1*g_mir).max()]) 
        g_mir = np.copy(g_mir)
        g_lwir /=  max([g_lwir[np.where(mask_mir_warp==1)].max(),(-1*g_lwir[np.where(mask_mir_warp==1)]).max()]) 
        
        #warp mir
        g_mir_warp = cv2.warpPerspective(g_mir, H_mir2lwir_guess, g_lwir.shape[::-1])
   
        
        #plt.imshow(g_lwir.T,origin='lower')
        #plt.figure()
        #plt.imshow(g_mir_warp.T,origin='lower'); plt.show()
        #plt.show()
        #pdb.set_trace() 
           
        
        #run EP08
        #flag_get_match = False
        #try:
        
        #EP080 = tools.get_EP08_from_img(temp_mir_warp, temp_lwir , inputMask=mask_mir_warp2, inputMask_ref=frameLwir.mask_img)
            
        #first opt on temperature
        #######
        res_factor = [5,2]
        id_ecc1, warp_matrix_homo_f, trans_vec = findTransformECC_stepbystep(g_mir_warp, g_lwir, mask_mir_warp, frameLwir.mask_img, 
                                                                             res_factor=res_factor, 
                                                                             trans_len_limit=[20,20,10], 
                                                                             ep08_limit=[.9,.9,0.6], flag_plot_=False, flag_maskEdge=False)
        
        if not(flag_parallel):print(' '.join([ 'res:{:02d} [{:d},{:d}]  '.format(
                                                                          res_factor_,int(trans_vec_[0]),int(trans_vec_[1])) 
                                                                          for (res_factor_,trans_vec_) in zip(res_factor+[1],trans_vec) ]), end=' ') 
        H_mir2lwir = np.linalg.inv(warp_matrix_homo_f).dot(H_mir2lwir_guess)
        #flag_get_match = True

        g_mir_warp = cv2.warpPerspective(g_mir, H_mir2lwir, g_lwir.shape[::-1])
        temp_mir_warp = cv2.warpPerspective(temp_mir, H_mir2lwir, g_lwir.shape[::-1])
        mask_mir_warp = np.where( (cv2.warpPerspective(mask_mir, H_mir2lwir, g_lwir.shape[::-1]) == 1), np.ones(g_lwir.shape ,dtype=np.uint8), 
                                                                                                        np.zeros(g_lwir.shape,dtype=np.uint8))
        mask_mir_warp2 = np.where( (cv2.warpPerspective(mask_mir2, H_mir2lwir, g_lwir.shape[::-1]) == 1), np.ones(g_lwir.shape ,dtype=np.uint8), 
                                                                                                               np.zeros(g_lwir.shape,dtype=np.uint8))

        EP081 = tools.get_EP08_from_img(temp_mir_warp, temp_lwir , inputMask=mask_mir_warp2, inputMask_ref=frameLwir.mask_img)

        '''
        #second opt on temperature
        #######
        if False:
            arrivalTime_lwir_ = cv2.warpPerspective(arrivalTime_lwir, frameLwir.H2Grid, g_lwir.shape[::-1], flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP, borderValue=-999) 
                                                                                                           
            frameLwir_mask_img = np.where( (frameLwir.mask_img==1) & (arrivalTime_lwir_<=frameLwir.time_igni) 
                                         & (arrivalTime_lwir_>=0),  np.ones(g_lwir.shape ,dtype=np.uint8), np.zeros(g_lwir.shape,dtype=np.uint8))
            plt.imshow(frameLwir_mask_img.T, origin='lower'); plt.show()
            pdb.set_trace()
            id_ecc2, warp_matrix_homo_f, _ = findTransformECC_stepbystep(g_mir_warp, g_lwir, mask_mir_warp, frameLwir_mask_img, 
                                                                         trans_len_limit=[20,10], ep08_limit=[.7,0.6], flag_plot_=True)
            
            H_mir2lwir2 = np.linalg.inv(warp_matrix_homo_f).dot(H_mir2lwir)
            #flag_get_match = True

            g_mir_warp = cv2.warpPerspective(g_mir, H_mir2lwir2, g_lwir.shape[::-1])
            temp_mir_warp = cv2.warpPerspective(temp_mir, H_mir2lwir2, g_lwir.shape[::-1])
            mask_mir_warp = np.where( (cv2.warpPerspective(mask_mir, H_mir2lwir2, g_lwir.shape[::-1]) == 1), np.ones(g_lwir.shape ,dtype=np.uint8), 
                                                                                                            np.zeros(g_lwir.shape,dtype=np.uint8))
            mask_mir_warp2 = np.where( (cv2.warpPerspective(mask_mir2, H_mir2lwir2, g_lwir.shape[::-1]) == 1), np.ones(g_lwir.shape ,dtype=np.uint8), 
                                                                                                               np.zeros(g_lwir.shape,dtype=np.uint8))

            
            #id_ecc2, warp_matrix_homo_f, _ = findTransformECC_stepbystep(temp_mir_warp, temp_lwir, mask_mir_warp2, frameLwir.mask_img, 
                                                                          trans_len_limit=[80,40], ep08_limit=[.7,0.6], flag_plot_=False)
            #H_mir2lwir = np.linalg.inv(warp_matrix_homo_f).dot(H_mir2lwir)
            #g_mir_warp = cv2.warpPerspective(g_mir, H_mir2lwir, g_lwir.shape[::-1])
            #temp_mir_warp = cv2.warpPerspective(temp_mir, H_mir2lwir, g_lwir.shape[::-1])
            #mask_mir_warp = np.where( (cv2.warpPerspective(mask_mir, H_mir2lwir, g_lwir.shape[::-1]) == 1), np.ones(g_lwir.shape ,dtype=np.uint8), 
            #                                                                                                       np.zeros(g_lwir.shape,dtype=np.uint8))
            #mask_mir_warp2 = np.where( (cv2.warpPerspective(mask_mir2, H_mir2lwir_guess, g_lwir.shape[::-1]) == 1), np.ones(g_lwir.shape ,dtype=np.uint8), 
            #                                                                                                       np.zeros(g_lwir.shape,dtype=np.uint8))
            
            EP082 = tools.get_EP08_from_img(temp_mir_warp, temp_lwir , inputMask=mask_mir_warp2, inputMask_ref=frameLwir.mask_img)

            if EP082 > EP081:
                print '2nd ECC', 
                H_mir2lwir = H_mir2lwir2
        '''     
        frameMir.set_homography_to_lwir(H_mir2lwir)
        frameMir.set_homography_to_grid(lwirH2Grid.dot(H_mir2lwir))
      

        #set pose
        #------
        mask_on_grid = cv2.warpPerspective (np.ones(frameMir.temp.shape), frameMir.H2Grid, burnplot.shape[::-1], flags=cv2.INTER_LINEAR )
        idx_plot = np.where(mask_on_grid==1)
        gcps_world_idx = np.dstack((idx_plot[1],idx_plot[0]))[0].reshape(-1,1,2)
        gcps_world_idx = np.array(gcps_world_idx,dtype=np.float32)

        #gcps_world = np.dstack((burnplot.grid_e[idx_plot],burnplot.grid_n[idx_plot],burnplot.terrain[idx_plot]))[0].reshape(-1,1,3)
        gcps_world = np.dstack((burnplot.grid_e[idx_plot],burnplot.grid_n[idx_plot],np.zeros(idx_plot[0].shape)))[0].reshape(-1,1,3)
        gcps_frame = cv2.perspectiveTransform( gcps_world_idx, np.linalg.inv(frameMir.H2Grid))
        
        flag, rvec, tvec = cv2.solvePnP(gcps_world, gcps_frame[:,0,:], frameMir.K_undistorted_imgRes, np.zeros(5) )
        frameMir.set_pose( rvec, tvec)
        #print tools.get_cam_loc_angle(rvec,tvec)[0].T, tools.get_cam_loc_angle(rvec,tvec)[1]

        #dump Frame info
        #------
        frameMir.dump(dir_out_mir_frame+'frameMIR{:06d}.nc'.format(imir))
        
        if not(flag_parallel): print(id_ecc1, end=' ') 
        #print EP080, EP081, EP082, 

        mpl.rcdefaults()
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['axes.labelsize'] = 16.
        mpl.rcParams['legend.fontsize'] = 'small'
        mpl.rcParams['legend.fancybox'] = True
        mpl.rcParams['font.size'] = 16.
        mpl.rcParams['xtick.labelsize'] = 16.
        mpl.rcParams['ytick.labelsize'] = 16.
        mpl.rcParams['figure.subplot.left'] = .05
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.top'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .05
        mpl.rcParams['figure.subplot.hspace'] = 0.05
        mpl.rcParams['figure.subplot.wspace'] = 0.15
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot(111)
        plt.imshow(np.ma.masked_where(frameLwir.mask_img==0, g_lwir).T,origin='lower')
        plt.contour(np.ma.masked_where(mask_mir_warp==0, g_mir_warp).T,origin='lower',alpha=.8,cmap=mpl.cm.Greys_r)
        ax.set_title(  r'lwirId {:04d}   mirId {:04d}   '.format(frameLwir.id,frameMir.id) 
                     + r'$\quad t_{Lwir}-t_{Mir}$='+'{:5.3f} s'.format(frameLwir.time_igni-frameMir.time_igni), pad=-10)
        ax.set_axis_off()

        fig.savefig(dir_out_mir_warp+plotname+'_{:06d}.png'.format(imir))
        plt.close(fig)
        

    npyFile = dir_out_mir_georef_npy+plotname+'_georef_{:06d}_{:s}.npy'.format(frameMir.id,georefMode)
   
    if (flag_restart) & (os.path.isfile(npyFile)):
        return 'georef already done'
    
    else:
        #georef 
        #------
        if georefMode == 'SH':
            
            if True:
                radiance = spectralTools.conv_temp2Rad(frameMir.temp, *param_set_radiance)
                georef_mir_rad = cv2.warpPerspective (radiance, frameMir.H2Grid, burnplot.shape[::-1], flags=cv2.INTER_LINEAR )
            
            else: # warping pixel to pixels
                georef_mir_rad = warpPerspective_p2p(not(flag_parallel), frameMir, burnplot, param_set_radiance, flag_restart)

            georef_mir_temp = spectralTools.conv_Rad2Temp(georef_mir_rad, param_set_temperature)

        else: 
            #print '  ** only SH is ready. stop here'
            return '  ** only SH is ready. stop here'

        
        #plot
        #------
        georef_lwir_temp = np.load(dir_out_georef_npy_refine_2nd+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,frameLwir.id,georefMode), allow_pickle=True)[3]
        georef_lwir_mask = np.load(dir_out_georef_npy_refine_2nd+'{:s}_georef2nd_{:06d}_{:s}.npy'.format(plotname,frameLwir.id,georefMode), allow_pickle=True)[2]
        _, g_georef_temp_mir, _  = tools.get_gradient(georef_mir_temp)
        _, g_georef_temp_lwir, _ = tools.get_gradient(georef_lwir_temp)

        levels_lwir=np.array([330,350,400,500,600,700])

        mpl.rcdefaults()
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(111)
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(np.ma.masked_where(georef_mir_temp<=0,georef_mir_temp).T,origin='lower',cmap=mpl.cm.jet,vmin=470,vmax=700)
        cbar = fig.colorbar(im ,cax = cbaxes)
        cbar.set_label('Brightness Temperature (K)',labelpad=10)
        ax.set_axis_off()
        cs_lwir1 = ax.contour(georef_lwir_temp.T, levels=levels_lwir,vmin=levels_lwir.min(),vmax=levels_lwir.max(),
                              origin='lower',cmap=mpl.cm.Greys,linewidths=1.5,alpha=.5)
        #ax.contour(np.ma.masked_where(burnplot.mask!=2,g_georef_temp_lwir).T,origin='lower',cmap=mpl.cm.Greys_r)
        fig.savefig(dir_out_mir_georef_png+plotname+'_{:06d}'.format(imir))
        plt.close(fig)
    
        #save npy
        #------
        np.save(npyFile, [[frameMir.time_date,                       \
                           frameMir.time_igni,                       \
                           frameMir.time_igni-frameLwir.time_igni,   \
                           frameMir.rvec,frameMir.tvec             ],\
                           georef_mir_temp,                          \
                           georef_mir_rad ]                          ) 
        return 'georef done'
    


#########################################
if __name__ == '__main__':
#########################################
   

    time_start_run = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='this is the driver of the GeorefCam Algo.')
    parser.add_argument('-i','--input', help='Input run name',required=True)
    parser.add_argument('-s','--newStart',  help='True then it uses existing data',required=False)
    parser.add_argument('-p','--parallel',  help='True then main loop is parallelized',required=False)
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
    runMode = 'mir'
    inputConfig = imp.load_source('config_'+runName,os.getcwd()+'/../input_config/config_'+runName+'.py')
    if socket.gethostname() == 'coreff':
        path_ = '/media/paugam/goulven/data/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/paugam/data/', path_)
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/paugam/data/', path_)
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/paugam/data/', path_)
    elif socket.gethostname() == 'ibo':
        path_ = '/space/paugam/data/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/paugam/data/',path_)
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/paugam/data/',path_)
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/paugam/data/',path_)

    # input parameters
    params_grid       = inputConfig.params_grid
    params_gps        = inputConfig.params_gps 
    params_mir_camera     = inputConfig.params_mir_camera
    params_lwir_camera     = inputConfig.params_lwir_camera
    params_rawData    = inputConfig.params_rawData
    params_georef     = inputConfig.params_georef

    # control flag
    flag_georef_mode = inputConfig.params_flag['flag_georef_mode']
    flag_parallel = inputConfig.params_flag['flag_parallel']
    flag_restart  = inputConfig.params_flag['flag_restart']
    
    if flag_georef_mode   == 'WithTerrain'     : 
        georefMode = 'WT'
        params_mir_camera['dir_input'] = params_mir_camera['dir_input'][:-1] + '_WT/'
    elif flag_georef_mode == 'SimpleHomography': 
        georefMode = 'SH'
    
    if (args.newStart is None) : 
        if not(flag_restart): 
            res = 'na'
            while res not in {'y','n'}:
                res = input('are you sure you want to delete the existing data in {:s}? (y/n)'.format(params_mir_camera['dir_input']))
            if res == 'n':
                print('stopped here')
                sys.exit()
    elif args.newStart is not None:
        flag_restart = tools.string_2_bool(args.newStart)

    if (args.parallel is None) : 
        flag_parallel = inputConfig.params_flag['flag_parallel']
    else: 
        flag_parallel = tools.string_2_bool(args.parallel)

    plotname          = params_rawData['plotname']
    root_postproc     = params_rawData['root_postproc']

    dir_out           = root_postproc + params_mir_camera['dir_input']
    dir_dem           = root_postproc + params_georef['dir_dem_input']
    
    #new
    dir_out_mir_raw      = dir_out + 'raw/'
    dir_out_mir_frame    = dir_out + 'Frames/'
    dir_out_mir_georef   = dir_out + 'Georef_{:s}/'.format(georefMode)
    dir_out_mir_georef_npy = dir_out + 'Georef_{:s}/npy/'.format(georefMode)
    dir_out_mir_georef_png = dir_out + 'Georef_{:s}/png/'.format(georefMode)
    dir_out_mir_georef_nc  = dir_out + 'Georef_{:s}/netcdf/'.format(georefMode)
    dir_out_mir_warp =  dir_out + 'Warping/'
    dir_out_wkdir      = dir_out + 'Wkdir/'
    
    #old
    dir_out_georef_npy_refine_2nd_reso = root_postproc + params_lwir_camera['dir_input'] + \
                                         'Georef_refined_reso{:02d}_{:s}/npy/'.format(int(params_lwir_camera['reso_refined']),georefMode)    
    dir_out_georef_npy_refine_2nd      = root_postproc + params_lwir_camera['dir_input'] + \
                                         'Georef_refined_{:s}/npy/'.format(georefMode)    
    reload(agema)
    reload(optris)
    reload(camera_tools)
    reload(tools)
    reload(spectralTools)
    reload(georefWT)

    if not(flag_restart):
        if os.path.isdir(dir_out_mir_frame): shutil.rmtree(dir_out_mir_frame)
        if os.path.isdir(dir_out_mir_georef): shutil.rmtree(dir_out_mir_georef)
        if os.path.isdir(dir_out_mir_warp): shutil.rmtree(dir_out_mir_warp)

    tools.ensure_dir(dir_out_mir_frame)
    tools.ensure_dir(dir_out_mir_georef)
    tools.ensure_dir(dir_out_mir_georef_npy)
    tools.ensure_dir(dir_out_mir_georef_png)
    tools.ensure_dir(dir_out_mir_georef_nc)
    tools.ensure_dir(dir_out_mir_warp)
    tools.ensure_dir(dir_out_wkdir)

    shutil.copy(os.getcwd()+'/../input_config/config_{0:s}.py'.format(runName), dir_out+'config_{0:s}_{1:s}.py'.format(runName,runMode))


    # load camera info
    #####################################
    #geometry
    K, D = camera_tools.get_camera_intrinsec_distortion_matrix(params_mir_camera) 
    #set parameter for radiance/temperature conversion 
    srf_file = '../data_static/Camera/'+params_mir_camera['camera_name'].split('_')[0]+'/SpectralResponseFunction/'+params_mir_camera['camera_name'].split('_')[0]+'.txt'
    wavelength_resolution = 0.01
    param_set_radiance = [srf_file, wavelength_resolution]
    param_set_temperature = spectralTools.get_tabulated_TT_Rad(srf_file, wavelength_resolution)


    #read ignition time
    ############
    file_ignition_time = params_rawData['root_data'] + 'ignition_time.dat'
    f = open(file_ignition_time,'r')
    lines = f.readlines()
    ignitionTime = datetime.datetime.strptime(params_rawData['fire_date']+'_'+lines[0].rstrip(), "%Y-%m-%d_%H:%M:%S")
    endTime = datetime.datetime.strptime(params_rawData['fire_date']+'_'+lines[1].rstrip(), "%Y-%m-%d_%H:%M:%S")
    fire_durationTime = (endTime-ignitionTime).total_seconds()


    #load grid 
    ###########################################
    burnplot = np.load(root_postproc+'grid_'+plotname+'.npy')
    burnplot = burnplot.view(np.recarray)
    grid_e, grid_n = burnplot.grid_e, burnplot.grid_n
    
    nx, ny = grid_e.shape
    prj_info = open(root_postproc+'grid_'+plotname+'.prj').readline()
    prj = osr.SpatialReference()
    prj.ImportFromWkt(prj_info)

    fit_planck_constant = 3.e-9
    boltzmann_constant = 5.67e-8
    dx = grid_e[1,1]-grid_e[0,0]
    dy = grid_n[1,1]-grid_n[0,0]
    psize = dx * dy 
   
    if os.path.isfile(dir_out_georef_npy_refine_2nd+'/mask_nodata_{:s}_georef.npy'.format(plotname)): 
        mask_burnNoburn = np.load(dir_out_georef_npy_refine_2nd+'/mask_nodata_{:s}_georef.npy'.format(plotname))
    else:
        mask_type_ = inputConfig.params_lwir_camera['mask_burnNobun_type_refined']
        mask_type_val = inputConfig.params_lwir_camera['mask_burnNobun_type_val_refined']
        print('#### mask_burnNoburn not found. now generated with mask type:', mask_type_)
        mask_burnNoburn =  get_mask_to_remove_missed_burn_area.get_mask(inputConfig, burnplot, georefMode, 
                                                                          dir_out_georef_npy_refine_2nd,
                                                                          mask_type_, mask_type_val, flag_plot=False)



   
    #triangulate terrain and map triangle
    ###########################################
    '''
    flag_restart_ = flag_restart
    if (not(flag_restart_)) | (not(os.path.isfile(dir_dem + 'tri_angles.npy'))):
        tri, points, triangles, triangles_grid, triangles_area = georefWT.set_triangle(burnplot, dir_dem, flag_restart_)
    else:
        print ''
        print 'load triangulation of the terrain'
        tri, points    = np.load(dir_dem + 'tri_angles.npy'    )
        triangles      = np.load(dir_dem + 'triangle_coord.npy').tolist()
        triangles_grid = np.load(dir_dem + 'triangle_grid.npy' ).tolist()
        triangles_area = np.load(dir_dem + 'triangle_aera.npy' ).tolist()
    
    grid_tools = [tri, points, triangles, triangles_grid, triangles_area, dir_out_wkdir] 
    '''

    #process RawData
    ###########################################
    if   (not os.path.isdir(params_rawData['root_postproc'] + params_mir_camera['dir_input']+params_mir_camera['dir_img_input'])) \
       | (inputConfig.params_flag['flag_mir_processRawData']):
        agema.processRawData(os.getcwd(),ignitionTime, params_rawData, params_mir_camera, inputConfig.params_flag, flag_restart)

    
    # get a first guess to warp mir on lwir using calibration image made just after the fire
    ###########################################
    H_mir2lwir_guess = agema.define_firstGuessWarp_mir_on_lwir(params_rawData['root_data']+'../', params_mir_camera['mir_name_ref'], 
                                                                   params_mir_camera['lwir_name_ref'], 
                                                                   params_mir_camera['cp_lwir_mir'], 
                                                                   flag_plot=False)

    #load lwir frame time info
    ############################################
    if   (not os.path.isfile(params_rawData['root_postproc']+params_lwir_camera['dir_input']+'Frames/frame_time_info.npy') )  \
       | (not inputConfig.params_flag['flag_restart']):
        lwir_frame_names = sorted(glob.glob(params_rawData['root_postproc']+params_lwir_camera['dir_input']+'Frames/*.nc' ))
        lwir_info = np.array([(0,0,0,'mm')]*len(lwir_frame_names),dtype=np.dtype([('time_igni',np.float),('ep08',np.float),('ssim',np.float),('filename','S600')]))
        lwir_info = lwir_info.view(np.recarray)    
        for ifile, lwir_frame_name in enumerate(lwir_frame_names):
            frameLwir = optris.load_existing_file(params_lwir_camera, lwir_frame_name)
            lwir_info.time_igni[ifile] = frameLwir.time_igni
            lwir_info.ep08[ifile] = frameLwir.corr_ref00
            lwir_info.ssim[ifile] = frameLwir.ssim
            lwir_info.filename[ifile] = lwir_frame_name
        np.save(params_rawData['root_postproc']+params_lwir_camera['dir_input']+'Frames/frame_time_info',lwir_info)
    else:
        lwir_info = np.load(params_rawData['root_postproc']+params_lwir_camera['dir_input']+'Frames/frame_time_info.npy')
        lwir_info = lwir_info.view(np.recarray)    

    #load grid from lwir
    ####################
    #arrivalTime_lwir = np.load(params_rawData['root_postproc']+params_mir_camera['file_arrivalTime_lwir'])[0] 
    #arrivalTime_lwir[np.where(mask_burnNoburn==1)] = arrivalTime_lwir[np.where(arrivalTime_lwir>=0)].min()


    #load mir time info
    ############################################
    mir_info = np.load(dir_out_mir_raw+'filename_time.npy')
    mir_info = mir_info.view(np.recarray)


    #loop over mir images: 1st warp mir on closest lwir and then orthorev mir with data from lwir.
    # if time between frame is too long we skip the mir image
    ############################################
    mir_frame_names = sorted(glob.glob(params_rawData['root_postproc'] + params_mir_camera['dir_input']+params_mir_camera['dir_img_input']+'*.MAT'))
    badId_lwir = tools.get_bad_idLwir(dir_out_georef_npy_refine_2nd) 

    params = []
    for imir, mir_frame_name in enumerate(mir_info.name):
        #if (imir < 770) | (imir >800) : continue 
        params.append( [flag_parallel, flag_restart,
                        imir, mir_frame_name, mir_info, badId_lwir, georefMode, 
                        path_, dir_out_mir_frame, dir_out_georef_npy_refine_2nd, dir_out_mir_georef_npy, dir_out_mir_georef_png,
                        params_mir_camera, plotname] )
       
    if flag_parallel:
        # set up a pool to run the parallel processing
        pool = multiprocessing.Pool(processes=tools.cpu_count())

        # then the map method of pool actually does the parallelisation  
        results = pool.map(star_stabilize_and_georef, params)
        pool.close()
        pool.join()
       
    else:
        results = []
        for param in params:
            print(os.path.basename(param[3]), end=' ') 
            results.append( star_stabilize_and_georef(param) )  
            print(results[-1])
            sys.stdout.flush()
            
