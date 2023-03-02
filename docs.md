# Real-time MATB: Extending OpenMATB to support real-time feedback control

Developed in conjunction with the Human Informatics and Predictive Performance Optimization (HIPPO) lab at the University of Florida

# 1. Important Files

We have added additional files that work with OpenMATB via multithreading to support real-time Engagement Index (EI) calculation.

The files added are:
```
wavelet.py
WaveletFlatGaussian.py
```
 
```WaveletFlatGaussian.py``` is the filter used to calculate the EI. \

```wavelet.py``` calls filter and does windowing on LSL data to calculate EI

# 2. Additional Features

