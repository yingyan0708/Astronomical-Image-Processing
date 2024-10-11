from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
from astropy.stats import sigma_clipped_stats, SigmaClip
from matplotlib.colors import LogNorm
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy import ndimage
from scipy.ndimage import label

def neighbor(i,j):
    left = [i-1,j]
    right = [i+1,j]
    above = [i,j-1]
    below = [i,j+1]
    upper_left = [i-1,j-1]
    upper_right = [i+1,j-1]
    bottom_left = [i-1,j+1]
    bottom_right = [i+1,j+1]
    neighbor_array = [[i,j],left,right,above,below,upper_left,upper_right,bottom_left,bottom_right]
    return neighbor_array


hdulist = fits.open('C:\\Users\\Acer\\Downloads\\Astronomical Image Processing\\Astro\\Astro\\Fits_Data\\mosaic.fits')
data = hdulist[0].data
flatten = data.flatten()
print(np.max(flatten))
#plt.hist(flatten,bins=len(flatten))

#plt.show()
focus_area = data[900:1000,900:1000]
plt.imshow(focus_area, cmap='viridis')
plt.show()

result = ndimage.median_filter(focus_area, size=10)
x_shape, y_shape = result.shape
print(f"x shape is {x_shape} and y shape is {y_shape}")
mean, median, std = sigma_clipped_stats(result, sigma=3.0)
print(f'mean is {mean},median is {median},std is {std}')

# for i in range(x_shape):
#     for j in range(y_shape):
#         if result[i,j] <= median + 3*std:
#             result[i,j] = 0

#max = np.max(result.flatten())
#print(max)
labeled_array, num_features = label(result > median + 3*std )
# indices = np.where(result == max)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Labeled Image')


group1 = []
max_values_descending = np.sort(result.flatten())[::-1]
print(max_values_descending)
for i in range(len(max_values_descending)):
    if max_values_descending[i] > median + 3*std:
        indices = np.where(result == max_values_descending[i])
        for j in range(len(indices[0])):
            point = [indices[1][j],indices[0][j]]
            if group1 == []:
                group1.append(neighbor(indices[1][j],indices[0][j]))
                plt.plot(point[0],point[1],'x',color='red')
            elif point in group1[0]:
                plt.plot(point[0],point[1],'x',color='red')
            elif point not in group1[0]:
                if neighbor(point[0],point[1]) in group1[0]:
                    group1[0].append(point)
                    plt.plot(point[0],point[1],'x',color='red')




# for i in range(len(indices[0])):
#     point = [indices[1][i],indices[0][i]]
#     group1.append(point)
#     result[indices[1][i],indices[0][i]] = 0
#     if point not in neighbor(point[0],point[1],result):
#         plt.plot(indices[1][i],indices[0][i],'x',color='red')

plt.imshow(result, cmap='magma')

plt.subplot(1, 2, 2)
plt.title(f'Labeled Components (Total: {num_features})')
plt.imshow(labeled_array, cmap='nipy_spectral')

plt.show()

#flatten = result.flatten()

