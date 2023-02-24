# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:47:14 2022

@author: Dr. Nicholas Napoli
"""
import math
from builtins import int, float, len

import numpy as np
import scipy.io as scp
import matplotlib.pyplot as plt
from scipy.fft import fft

import scipy.fftpack as scpfft
from scipy.signal import convolve as conv
#import mne






def WaveletFlatGaussian_Conv(*args):
#-----Overview-------------------------------------
#   WaveletFlatGaussian_FFT: This approach designs the wavelets filters using a flat
#   gaussian function in the frequency domain and transforms the the EEG time series to the
#   frequency domain to filter the EEG time series. A gaussian filter is
#   then implemented for smoothing. 


# ------- Variable Inputs ------------------------
    # X: EEG Time Series (N x 1) row vector
    xn = args[0]
    x = (xn.reshape(len(xn), 1)).T
    # FS: Sampling Frequency
    fs = args[1]
    # sigma:  The gaussian Mask implementation is designed to smooth the
    #            the power out. The varible is scalar value that is
    #            described by sigma. (Note: With a sampling fs=256 we have typically used a sigma around 60)
    sigma = args[2]
    # FiltCoef: This is a binary response leaving the input empty or providing new design
    #           coefficients . If you want the option to use the default filter coefficients
    #           previously designed or a different set of coefficients. The
    #           design coefficient input should be set by f x 3 dimensions,
    #           where f is the number of filters and the columns represent
    #           parameters a,b,Cf.
    if (len(args) == 4):
        plotfilters = args[3]
    # PlotFilters: A binary response, where the value of 1 would plot the
    #              designed filters and a value of 0 or leaving the input empty will not
    #              plot the filters. 
    if (len(args) == 5):
        filtcoef = args[4]
# ------- Variable Output ------------------------
    # Power: The output is the power of each designed filter. The
    #        variable dimensions are N x f. 
    
#=====================================================================================  
# NOTES  ASSUMPTIONS: The filter design was based on minimizing the
# standard deviation of the plateaut value of the filters, while
# maintaining the below EEG frequency cutoffs. The cutoff values were aimed
# to achieve a (1/e) where every other adjacent filter is orthogonal. 


#------------------------------------
# EGG Frequency Bands 
#------------------------------------
     # Frequency Bands (array index)
       # Delta 0-3.5 Hz (0)
       # Theta 4-7 Hz (1)
       # Alpha 8-12 Hz (2-3)
       # Low Beta 13-15 
       # Mid Beta 15-18 Hz (4-11)
       # High Beta 18-40
       # Gamma 32-100 Hz    

#======================================================================================================================
# Contributions: The Respiration Complexity Code was designed and written by Matthew Demas and Nicholas J. Napoli at njn5fg@virginia.edu.
#=======================================================================================================================

#==========================================
# ------Initialize Parameters for EEG---------
#==========================================
    if len(args) > 5: # Non - Default Filter Parameters

# ------ColumnVectors - ----------
        if np.shape(filtcoef, 1) > np.shape(filtcoef, 2):
            a = filtcoef[:,0]
            b = filtcoef[:,1]
            cf = filtcoef[:,2]
        else: # -------Row Vectors - -----------
            filtcoef = zip(*filtcoef)
            a = filtcoef[:, 0]
            b = filtcoef[:, 1]
            cf = filtcoef[:, 2]


    else: # Default Parameters
        # A = [.15, .003, .18, .145, .12, .02, .001, .001, .001, .001, .001, .001];
        # B = [.075, .048, .11, .15, .19, .15, .11, .075, .06, .05, .04, .03];
        # Cf = [2.25, 5.63, 8.9, 11.54, 14.09, 16.78, 19.79, 23.1, 26.7, 30.42, 34.37, 38.58];
        a = np.array([.072, .001, .101, .219, .170, .007, .001, .001, .001, .001, .003, .001]).reshape(1, 12)
        b = np.array([.095, .077, .119, .161, .180, .135, .127, .095, .090, .088, .078, .070]).reshape(1, 12)
        cf = np.array([2.349, 5.605, 8.759, 11.4, 13.859, 16.608, 19.627, 22.792, 26.094, 29.432, 32.820, 36.307]).reshape(1, 12)

#==========================================
#------------- Gaussian Mask---------------
#==========================================
    radius = math.floor(3 * sigma)
    resize = 1 + 2 * radius

    mask = np.zeros([1, resize], dtype = float)

    den = 2 * math.pi * sigma * sigma
    den = float((math.sqrt(den)))
    den = float(1 / den)

    for i in range(resize):
        z = i-radius
        mask[0, i] = den * math.exp( -0.5 * (z/sigma) * (z/sigma) )

#==============================================================
#----------Initialize Time Series EEG Variable Assignment------
#==============================================================
#==============================================
# Examine Data: Make Row Vector
#==============================================
    nm = np.zeros([1, 2], dtype = int)
    nm[0,0] = len(x)
    nm[0,1] = len(x[0])
    if (nm[0,0] > 1):
        x = np.transpose(x)
    gl_h = math.floor(len(mask[0])/2)
#===============================================
#-------- Fourier Domain to Time Domain---------
#===============================================
    n = round(len(x[0]))
    fx = fft(x)
    f = np.multiply(np.linspace(0, n-1,n),fs/n).reshape(1, n)
    j=len(cf[0])

#= == == == == == == == == == == == == == == == == == == == == == == == ==
#= == == == == == == == Design Filter Bank == == == == == == ==
#= == == == == == == == == == == == == == == == == == == == == == == == ==
    waveletsfreq = np.zeros([j, n], dtype = float)
    for i in range(j): #Something with the math.exp isnt working
        for k in range(n):
            reshape=(-a[0, i] * (f[0, k] - cf[0, i])** 2 - b[0, i] * (f[0, k] - cf[0, i])** 4)
            waveletsfreq[i,k] = math.exp(reshape)

# -------- Plot Filter Design in Frequency Domain --------
    if len(args) == 5:
        #figure
        for i in range(j):
            plt.plot(f, waveletsfreq[i,:])
            #hold on

        plt.title('Wavelet Filter Bank Design')
        plt.plot(f, sum(waveletsfreq))
        #grid on

#= == == == == == == == == == == == == == == == == == == == == == == == =
#= == =Filtering and Intensity Calculations == == == =
#= == == == == == == == == == == == == == == == == == == == == == == == =

# Retrospective Analysis(Fourier Implementation)
    #freqfiltx = np.zeros([j, n], dtype = float) 
    #y = np.zeros([j, n], dtype = float)
# Power = zeros(J, N); % Intialization
    wl_h = math.floor(len(waveletsfreq[0]) / 2) # Paddding adjust for Conv

    TempWTA = np.zeros([j, n], dtype=complex)
    Power = np.zeros([j, n-1], dtype=float)
    for i in range(j):
        TempWTA[i,:]=scpfft.fftshift(np.fft.ifft(waveletsfreq[i,:])).reshape(1, n)
        PowerTemp = conv((TempWTA[i,:]).reshape(1,n), x)
        PowerTempB = 2 * abs(PowerTemp[0, wl_h:-wl_h].reshape(1,len(PowerTemp[0])-n)) 
        #TempB = abs(ifft(FreqFiltX(j,:).*f * 2 * pi * 1i) / (2 * pi * Cf(j)) );
        PowerTempC = conv(PowerTempB, mask)
        Power[i,:]=PowerTempC[0, gl_h:-gl_h]

    return Power


def getdata(fileName):
    #Reads the data coming from the .set file and stores it into an array
    #fileObj = open(fileName, encoding="utf16")  # opens the file in read mode
    data = scp.loadmat(fileName)                                                               
    #data = fileObj.read().splitlines()  # puts the file into an array
    #fileObj.close()
    return data

#= == == == == == == == == == == == == == == == == == == == == == == == =
#= == = The Functions Below Require a Set Window Size of n  == == == =
#= == == == == == == == == == == == == == == == == == == == == == == == =

def GenWaveletFlatGaussian_Conv(n,fs,sigma):
#-----Overview-------------------------------------
#   WaveletFlatGaussian_FFT: This approach designs the wavelets filters using a flat
#   gaussian function in the frequency domain and transforms the the EEG time series to the
#   frequency domain to filter the EEG time series. A gaussian filter is
#   then implemented for smoothing. 

#=====================================================================================  
# NOTES  ASSUMPTIONS: The filter design was based on minimizing the
# standard deviation of the plateaut value of the filters, while
# maintaining the below EEG frequency cutoffs. The cutoff values were aimed
# to achieve a (1/e) where every other adjacent filter is orthogonal. 


#------------------------------------
# EGG Frequency Bands 
#------------------------------------
     # Frequency Bands (array index)
       # Delta 0-3.5 Hz (0)
       # Theta 4-7 Hz (1)
       # Alpha 8-12 Hz (2-3)
       # Low Beta 13-15 
       # Mid Beta 15-18 Hz (4-11)
       # High Beta 18-40
       # Gamma 32-100 Hz    

#======================================================================================================================
# Contributions: The Respiration Complexity Code was designed and written by Matthew Demas and Nicholas J. Napoli at njn5fg@virginia.edu.
#=======================================================================================================================

#==========================================
# ------Initialize Parameters for EEG---------
#==========================================
    a = np.array([.072, .001, .101, .219, .170, .007, .001, .001, .001, .001, .003, .001]).reshape(1, 12)
    b = np.array([.095, .077, .119, .161, .180, .135, .127, .095, .090, .088, .078, .070]).reshape(1, 12)
    cf = np.array([2.349, 5.605, 8.759, 11.4, 13.859, 16.608, 19.627, 22.792, 26.094, 29.432, 32.820, 36.307]).reshape(1, 12)

#==========================================
#------------- Gaussian Mask---------------
#==========================================
    radius = math.floor(3 * sigma)
    resize = 1 + 2 * radius

    mask = np.zeros([1, resize], dtype = float)

    den = 2 * math.pi * sigma * sigma
    den = float((math.sqrt(den)))
    den = float(1 / den)

    for i in range(resize):
        z = i-radius
        mask[0, i] = den * math.exp( -0.5 * (z/sigma) * (z/sigma) )

#==============================================================
#----------Initialize Time Series EEG Variable Assignment------
#==============================================================
#===============================================
#-------- Fourier Domain to Time Domain---------
#===============================================
    #n = round(len(x[0])) 
    #fx = fft(x)
    f = np.multiply(np.linspace(0, n-1,n),fs/n).reshape(1, n)
    j=len(cf[0])
    #print("j is",j)

#= == == == == == == == == == == == == == == == == == == == == == == == ==
#= == == == == == == == Design Filter Bank == == == == == == ==
#= == == == == == == == == == == == == == == == == == == == == == == == ==
    waveletsfreq = np.zeros([j, n], dtype = float)
    for i in range(j): #Something with the math.exp isnt working
        for k in range(n):
            reshape=(-a[0, i] * (f[0, k] - cf[0, i])** 2 - b[0, i] * (f[0, k] - cf[0, i])** 4)
            waveletsfreq[i,k] = math.exp(reshape)

    return waveletsfreq,mask

def CalcWaveletFlatGaussian_Conv(x,n,waveletsfreq,mask):
#==============================================
# Examine Data: Make Row Vector
#==============================================
    x = x.reshape(1,n)
    #print("Size of input:",np.shape(x))

    nm = np.zeros([1, 2], dtype = int)
    nm[0,0] = len(x)
    nm[0,1] = len(x[0])
    if (nm[0,0] > 1):
        x = np.transpose(x)   # maybe flip here, maybe not? 
    gl_h = math.floor(len(mask[0])/2)

#= == == == == == == == == == == == == == == == == == == == == == == == =
#= == =Filtering and Intensity Calculations == == == =
#= == == == == == == == == == == == == == == == == == == == == == == == =

# Retrospective Analysis(Fourier Implementation)
    #freqfiltx = np.zeros([j, n], dtype = float) 
    j = 12 # This is hardcoded based on how the cf filter is designed in the eariler algo
    #y = np.zeros([j, n], dtype = float)
# Power = zeros(J, N); % Intialization
    wl_h = math.floor(len(waveletsfreq[0]) / 2) # Paddding adjust for Conv

    TempWTA = np.zeros([j, n], dtype=complex)
    Power = np.zeros([j, n-1], dtype=float)
    for i in range(j):
        TempWTA[i,:]=scpfft.fftshift(np.fft.ifft(waveletsfreq[i,:])).reshape(1, n)
        # Debug insert
        #print("Temp size: ",np.shape((TempWTA[i,:]).reshape(1,n)))
        #print("x size:",np.shape(x))
        # Debug insert
        PowerTemp = conv((TempWTA[i,:]).reshape(1,n), x)
        PowerTempB = 2 * abs(PowerTemp[0, wl_h:-wl_h].reshape(1,len(PowerTemp[0])-n)) #Potential debug
        #TempB = abs(ifft(FreqFiltX(j,:).*f * 2 * pi * 1i) / (2 * pi * Cf(j)) );
        PowerTempC = conv(PowerTempB, mask)
        Power[i,:]=PowerTempC[0, gl_h:-gl_h]

    return Power
