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
import pandas as pd
from photutils.aperture import CircularAnnulus, CircularAperture, ApertureStats, aperture_photometry


def neighborhood(y, x):
    # Define neighbors in (y, x) format
    left = [y, x-1]
    right = [y, x+1]
    above = [y-1, x]
    below = [y+1, x]
    upper_left = [y-1, x-1]
    upper_right = [y-1, x+1]
    bottom_left = [y+1, x-1]
    bottom_right = [y+1, x+1]

    # Combine into a list
    neighbor_array = [left, right, above, below, upper_left, upper_right, bottom_left, bottom_right]
    return neighbor_array


def flood_recursive(sorted_indices, threshold, result):
    group1 = []
    visited = set()

    def iterative_flood_fill(y, x, current_group, threshold):
        # Stack to hold pixels to be processed (instead of recursion)
        stack = [(y, x)]

        while stack:
            # Pop a pixel from the stack
            y, x = stack.pop()

            # If already visited, skip this pixel
            if (y, x) in visited:
                continue

            # Mark pixel as visited and add it to the current group
            visited.add((y, x))
            current_intensity = result[y, x]
            current_group.append([y, x])

            # Get neighbors of the pixel
            neighbors_array = neighborhood(y, x)

            # Process neighbors
            for ny, nx in neighbors_array:
                if 0 <= ny < result.shape[0] and 0 <= nx < result.shape[1]:  # Check bounds
                    intensity = result[ny, nx]
                    if intensity >= threshold and (ny, nx) not in visited and intensity <= current_intensity:  # Flood criteria
                        stack.append((ny, nx))  # Add neighbor to stack if it meets the criteria


    # Main loop
    for indices in sorted_indices:
        y = indices[0]
        x = indices[1]
        if (y, x) not in visited:
            new_group = []
            iterative_flood_fill(y, x, new_group,threshold)
            if new_group:
                group1.append(new_group)

    return group1

hdulist = fits.open('C:\\Users\\Acer\\Downloads\\Astronomical Image Processing\\Astro\\Astro\\Fits_Data\\mosaic.fits')
header = hdulist[0].header
mag_zpt = header.get('MAGZPT')
mag_zrr = header.get('MAGZRR')
data = hdulist[0].data
flatten = data.flatten()
print(np.max(flatten))
#plt.hist(flatten,bins=len(flatten))

#plt.show()
focus_area = data[900:1000,900:1020]
#focus_area = data[200:300,40:150]
plt.imshow(focus_area, cmap='viridis')
plt.show()

result = ndimage.median_filter(focus_area, size=10)
x_shape, y_shape = result.shape
print(f"x shape is {x_shape} and y shape is {y_shape}")
mean, median, std = sigma_clipped_stats(result, sigma=3.0)
print(f'mean is {mean},median is {median},std is {std}')


labeled_array, num_features = label(result > median + 3*std )

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Labeled Image')


threshold = median + 3 * std
mask = result > threshold
indices = np.argwhere(mask)  # Get indices where result > threshold
values_above_threshold = result[mask]
sorted_indices = indices[np.argsort(values_above_threshold)[::-1]]
group1 = flood_recursive(sorted_indices,threshold,result)

print(len(group1))
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

# for group_id, points in enumerate(group1):
#     if len(points) < 5:
#         pass
#     else:
#         group_color = colors[group_id % len(colors)]  # Assign a color to the group
#         plt.plot(points[0][1], points[0][0], 'x', color=group_color)
#         neighbors = neighborhood(points[0][1], points[0][0])
#         for neighbor in neighbors:
#             plt.plot(neighbor[0], neighbor[1], 'x', color=group_color)

catalogue = []
for group_id, points in enumerate(group1):
    if len(points) < 5:
        pass
    else:
        group_color = colors[group_id % len(colors)]  # Assign a color to the group
        x_coords = [point[1] for point in points]
        y_coords = [point[0] for point in points]

        # Calculate the centroid
        #centroid_x = np.mean(x_coords)
        #centroid_y = np.mean(y_coords)

        centroid_x = x_coords[0]
        centroid_y = y_coords[0]

        distances = np.sqrt((x_coords - centroid_x) ** 2 + (y_coords - centroid_y) ** 2)
        radius = np.mean(distances)

        # Append the centroid to the catalogue
        catalogue.append({'centroid_x': centroid_x, 'centroid_y': centroid_y, 'radius':radius})
        plt.plot(centroid_x, centroid_y, 'x', color=group_color)

plt.imshow(result, cmap='viridis', origin='lower', norm=LogNorm(), interpolation='nearest')
plt.subplot(1, 2, 2)
plt.title(f'Labeled Components (Total: {num_features})')
plt.imshow(labeled_array, cmap='nipy_spectral')
plt.show()

catalogue_df = pd.DataFrame(catalogue)
positions = np.transpose((catalogue_df['centroid_x'],catalogue_df['centroid_y']))
radii = catalogue_df['radius'].values
# Create a list of CircularAperture objects with different radii
#apertures = [CircularAperture(pos, r=r) for pos, r in zip(positions, radii)]
apertures = CircularAperture(positions, r = 6.0)
# Plot the image
plt.imshow(result, cmap='Greys', origin='lower', norm=LogNorm(), interpolation='nearest')

# Plot each aperture
#for aperture in apertures:
apertures.plot(color='blue', lw=1.5, alpha=0.5)

annulus_aperture = CircularAnnulus(positions, r_in = 10, r_out = 15)
plt.figure()
plt.imshow(result, cmap='Greys', norm=LogNorm(), origin = 'lower')
apertures.plot(color = 'blue', lw=1.5, alpha = 0.5)
annulus_aperture.plot(color = 'green', lw = 1.5, alpha = 0.5)

plt.show()

aperstats = ApertureStats(result, annulus_aperture)
bkg_mean = aperstats.mean
aperture_area = apertures.area_overlap(result)
total_bkg = bkg_mean * aperture_area
star_data = aperture_photometry(result, apertures) #function returns the sum of the (weighted) input data values within the aperture
star_data['total_bkg'] = total_bkg
star_data['subtracted_flux'] = star_data['aperture_sum'] - star_data['total_bkg']
star_data['calibrated_flux'] = mag_zpt - 2.5 * np.log10(star_data['subtracted_flux'])

for col in star_data.colnames:
    star_data[col].info.format = '%.8g'

star_data.pprint()