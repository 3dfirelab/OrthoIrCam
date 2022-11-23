from __future__ import division
from builtins import range
from past.utils import old_div
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import pdb 

'''
from http://www.cnblogs.com/justin_s/archive/2010/12/02/1894673.html
added mask in the ref image
'''

########################################################
def single_channel_hist( channel ):
    """ calculate cumulative histgram of single image channel
        return as a list of length 256
    """
    hist,bins = np.histogram(channel.flatten(),256,[0,256])
    cdf = hist.cumsum()
    return cdf


########################################################
def get_channels(img):
    """split jpg image file into 3 seperate channels
        return as a list of length 3
    """
    b,g,r = cv2.split(img)
    return [b,g,r]
 

########################################################
def cal_hist(channels,idx_mask=None):
    """
        cal cumulative hist for channel list
    """
    if idx_mask is None:
        idx_mask = np.where(channels[0]>=-999) # take all pixels
    
    out = []
    for channel in channels:
        #idx_mask_ = np.where(channel>0)
        hist_=single_channel_hist(channel[idx_mask])
        out.append(old_div(1.*hist_,hist_.max()))
    return out


########################################################
def cal_trans(ref,adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    i =0
    j = 0;
    table = list(range(0,256))
    for i in range( 1,256):
        for j in range(1,256):
            if (ref[i] >= adj[j-1]) & (ref[i] < adj[j]):
                table[i] = j
                break
         
    table[255] = 255
    return table


########################################################
def hist_matching(refImg,dstImg,idx_mask_ref=None,idx_mask_dst=None,flag_out_cdf=False):
    
    if len(refImg.shape)>2: 
        refChannels = get_channels(refImg)
        dstChannels = get_channels(dstImg)
        nbre_channel = refImg.shape[2]
    else:
        refChannels = [refImg]
        dstChannels = [dstImg]
        nbre_channel = 1

    hist_ref = cal_hist(refChannels,idx_mask=idx_mask_ref)
    hist_dst = cal_hist(dstChannels,idx_mask=idx_mask_dst)

    tables = [cal_trans(hist_dst[i],hist_ref[i]) for i in range(0,nbre_channel)]

    #plt.plot(np.arange(256),tables[0],c='r')
    #plt.plot(np.arange(256),tables[1],c='g')
    #plt.plot(np.arange(256),tables[2],c='b')
    #plt.show()
    #pdb.set_trace()

    for i in range(0,nbre_channel):
        for j in range(0,dstChannels[i].shape[0]):
            for k in range(0,dstChannels[i].shape[1]):
                v = dstChannels[i][j,k]
                try: 
                    dstChannels[i][j,k] = tables[i][int(v)]
                except: 
                    pdb.set_trace()

    if nbre_channel > 1:
        out = cv2.merge((dstChannels[0],dstChannels[1],dstChannels[2]))
    else:
        out = dstChannels[0]

    if flag_out_cdf:
        return out,hist_ref,hist_dst
    else:
        return out
 
 

########################################################
if __name__ == '__main__':
########################################################

    refImg = cv2.cvtColor(cv2.imread('airplane.png'),cv2.COLOR_BGR2RGB)
    dstImg_in = cv2.cvtColor(cv2.imread('lena.png'),cv2.COLOR_BGR2RGB)
     
#    mask = np.zeros(refImg.shape[:2])
#    mask[150:180,50:80] = 1 
#    idx_mask = np.where(mask == 1)

    #dstImg, hist_ref,hist_dst = hist_matching(refImg,dstImg_in,idx_mask=idx_mask,flag_out_cdf=True)
    dstImg, hist_ref,hist_dst = hist_matching(refImg,dstImg_in,flag_out_cdf=True)
    ax = plt.subplot(231)
    ax.imshow(refImg)
    #ax.imshow(np.ma.masked_where(mask==0,mask))
    ax.set_title('ref')
    ax = plt.subplot(232)
    ax.imshow(dstImg_in)
    ax.set_title('dest')
    ax = plt.subplot(233)
    ax.imshow(dstImg)
    ax.set_title('after')

    colorband = ['r','g','b']

    ax = plt.subplot(234)
    [ax.plot(np.arange(256),hist_ref[iband],c=colorband[iband]) for iband in range(3)]
    ax = plt.subplot(235)
    [ax.plot(np.arange(256),hist_dst[iband],c=colorband[iband]) for iband in range(3)]
    ax = plt.subplot(236)
    hist_dst_out = cal_hist(get_channels(dstImg))
    [ax.plot(np.arange(256),hist_dst_out[iband],c=colorband[iband]) for iband in range(3)]


    plt.show()
