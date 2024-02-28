from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import sys, os, glob, pdb
import numpy as np 
import matplotlib.pyplot as plt
import asciitable
from scipy import interpolate
import pandas as pd

######################################################################
def get_TT():
    #*****************************************************************
    # GET_TT: GET a Tabulated Temperature array in the range of 100 K
    # to 10000 K for black-body-radiance- and related calculations
    #*****************************************************************

    tt = np.zeros(551, dtype=float)
    tt[0:301] =   np.arange(301) + 100.                            #from 100 K  to  400 K - step 1 K
    tt[301:461] = np.arange(160) * 10. + 410.                    #from 410 K  to 2000 K - step 10 K
    tt[461:541] = np.arange(80) * 100. + 2100.                   #from 2100 K  to 10000 K - step 100 K
    tt[541:551] = np.arange(10) * 1000. + 10000.                 #from 11000 K  to 100000 K - step 10000 K

    return tt



#####################################################
def get_tabulated_TT_Rad(srf_file, wavelength_resolution):

    TT = get_TT()
    wavelength_SRF, SRF = read_srf(srf_file)
    
    start_band = wavelength_SRF.min() - 10 * wavelength_resolution
    width_band = wavelength_SRF.max() + 10 * wavelength_resolution - start_band

    #create a wavelength range to interpolate SRS and modtran result
    nbre_bin = int(old_div(width_band,wavelength_resolution))
    lambda_base = np.arange(nbre_bin)*wavelength_resolution + start_band
    
    #interpolate the SRF
    SRF_base   =  np.interp(lambda_base, wavelength_SRF, SRF, left=0, right=0)

    b = np.zeros(TT.shape)
    for i_tt in range(b.shape[0]):
        b[i_tt] = 0
        for i_lambda, (lambda_, SRF_) in  enumerate(zip(lambda_base, SRF_base)):
            b[i_tt] += SRF_ * planck_radiance(np.array([lambda_]),np.array([TT[i_tt]]))[0] * wavelength_resolution
    b /= (SRF_base.sum() * wavelength_resolution)


    lookUpTable_TT_Rad = np.zeros_like(b,dtype=np.dtype([('radiance',float),('temperature',float)]))
    lookUpTable_TT_Rad = lookUpTable_TT_Rad.view(np.recarray)
    lookUpTable_TT_Rad.temperature = TT
    lookUpTable_TT_Rad.radiance = b
    
    return lookUpTable_TT_Rad



######################################################
def conv_temp2Rad(temperature, srf_file, wavelength_resolution=0.01):

    wavelength_SRF, SRF = read_srf(srf_file)
   
    start_band = wavelength_SRF.min() - 10 * wavelength_resolution
    width_band = wavelength_SRF.max() + 10 * wavelength_resolution - start_band

    #create a wavelength range to interpolate SRS and modtran result
    nbre_bin = int(old_div(width_band,wavelength_resolution))
    lambda_base = np.arange(nbre_bin)*wavelength_resolution + start_band
    
    #interpolate the SRF
    SRF_base   =  np.interp(lambda_base, wavelength_SRF, SRF, left=0, right=0)

    L = np.zeros([lambda_base.shape[0],temperature.flatten().shape[0]])
    for i_lambda, (lambda_, SRF_) in  enumerate(zip(lambda_base, SRF_base)):
        lambda__ = np.zeros_like(temperature.flatten())+lambda_
        L[i_lambda,:] = (SRF_base[i_lambda] * planck_radiance(lambda__,temperature.flatten()) * wavelength_resolution)
        
    return (old_div(L.sum(axis=0),(SRF_base.sum() * wavelength_resolution)) ).reshape(temperature.shape)

######################################################
def conv_Rad2Temp(radiance, lookupTable_TTRad):

    temp = np.zeros_like(radiance.flatten())
    
    interp = interpolate.interp1d( lookupTable_TTRad.radiance, lookupTable_TTRad.temperature, kind='linear')

    idx = np.where(radiance.flatten() >= lookupTable_TTRad.radiance.min() ) 
    temp[idx] = interp(radiance.flatten()[idx] )
    
    idx = np.where(radiance.flatten() < lookupTable_TTRad.radiance.min() ) 
    temp[idx] = 0

    return temp.reshape(radiance.shape)
   
    '''
    for i, rad in enumerate(radiance.flatten()):
        if rad == 0.0: 
            temp[i] = 0.0
        else:
            try:
                idx = np.where((lookupTable_TTRad.radiance[:-1]<=rad)&(lookupTable_TTRad.radiance[1:]>rad))[0]
                temp[i] = (lookupTable_TTRad.temperature[idx+1]-lookupTable_TTRad.temperature[idx])/(lookupTable_TTRad.radiance[idx+1]-lookupTable_TTRad.radiance[idx]) * (rad-lookupTable_TTRad.radiance[idx]) \
                          + lookupTable_TTRad.temperature[idx]
            except: 
                pdb.set_trace()
    return temp.reshape(radiance.shape)
    '''

######################################################################
def planck_radiance(wavelength,BT):
    # input: wavelength is the wavelength in micron
    #        BT is the brightness Temperature
    # output: the Radiance in W/(m2.sr.um) 
    #
    #      Quantity      Sym.     Value          Unit         Relative
    #                                                      uncertainty (ppm)
    #    -------------------------------------------------------------------
    #     speed of light   c    299792458          m/s          exact
    #      in a vacuum
    #
    #        Planck        h    6.6260755(40)   1.0e-34Js       0.60
    #       constant
    #
    #       Boltzmann      k    1.380658(12)    1.0e-index      8.5
    #       constant

    c = 299792458.e0
    h = 6.6260755e-34 
    k = 1.380658e-23

    # constant
    c1 =  2*h*c*c  # [W.m2]
    c2 =  old_div(h*c,k)    # [K.m]
    np.seterr(all = "raise")
    #try:
    L = old_div(c1, ( (wavelength[:]*1.e-6)**5 * ( np.exp(old_div(c2,(wavelength[:]*1.e-6*BT[:])) ) - 1 ) )) * 1.e-6
    #except: 
    #    pdb.set_trace()
    return L


######################################################################
def planck_temperature(wavelength,radiance):
    # input: wavelength is the wavelength in micron
    #        Rad is the Radianace W/(m2.sr.um)
    # output: the Brightness Temperature
    #
    #      Quantity      Sym.     Value          Unit         Relative
    #                                                      uncertainty (ppm)
    #    -------------------------------------------------------------------
    #     speed of light   c    299792458          m/s          exact
    #      in a vacuum
    #
    #        Planck        h    6.6260755(40)   1.0e-34Js       0.60
    #       constant
    #
    #       Boltzmann      k    1.380658(12)    1.0e-23J/K      8.5
    #       constant
    
    c = 299792458.e0
    h = 6.6260755e-34 
    k = 1.380658e-23

    # constant
    c1 =  2*h*c*c  # [W.m2]
    c2 =  old_div(h*c,k)    # [K.m]
    
    return old_div(c2, ( wavelength[:]*1.e-6 * np.log( old_div(c1,( (wavelength[:]*1.e-6)**5 * radiance*1.e6 )) + 1 ) ))



######################################################
def read_srf(srf_file):
    
    df = pd.read_csv(srf_file, names=['wavelenght', 'srf'], 
                     header=None, delimiter=r"\s+" )

    ##read spectral response fct file 
    #reader = asciitable.NoHeader()
    #reader.data.splitter.delimiter = ' '
    #reader.data.splitter.process_line = None
    #reader.data.start_line = 0
    #data = reader.read(srf_file)
    
    wave_leng= np.array(df['wavelenght'])
    SRF= np.array(df['srf'])
    
    return wave_leng,SRF





######################################################
if __name__ == '__main__':
######################################################
   
    srf_file = '../data_static/Camera/optrisPI400/SpectralResponseFunction/optrisPI400.txt'
    wavelength_resolution = 0.01

    #create lookuptable between Radiance and Temperature
    lookupTable_TTRad = get_tabulated_TT_Rad(srf_file, wavelength_resolution)

    ax = plt.subplot(111)
    ax.plot(lookupTable_TTRad.temperature, lookupTable_TTRad.radiance)
    ax.set_aspect('equal')
    
    #check conversion function from T to L
    temp = np.arange(100,1000,50)
    L = conv_temp2Rad(temp, srf_file, wavelength_resolution=wavelength_resolution)
    plt.scatter(temp,L,c='k',s=100)

    temp_2 = conv_Rad2Temp(L, lookupTable_TTRad)
    plt.scatter(temp_2,L,c='r',s=20)

    plt.show()

    
