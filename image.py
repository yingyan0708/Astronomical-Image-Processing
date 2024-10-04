from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
hdulist = fits.open('C:\\Users\\Acer\\Downloads\\Astronomical Image Processing\\Astro\\Astro\\Fits_Data\\mosaic.fits')
flatten = hdulist[0].data.flatten()
condition = (flatten < 5000)
max = np.max(flatten[condition])
mean = np.mean(flatten)
print(max)
print(mean)
print(len(flatten))
print(flatten)
plt.hist(flatten,bins=5000)
plt.xlim(3000,5000)
plt.show()