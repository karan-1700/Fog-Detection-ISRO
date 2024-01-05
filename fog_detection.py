#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:25:13 2022

@author: karan
"""

'''
= Permutation 3:
    1. TIR1 Dynamic Temperature Threshold
    2. VIS Dynamic Reflectance Threshold
    3. TIR1 Std Dev threshold: thresh_val=2

'''


# importing necessary libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

DATE = "01Jan2021"

INPUT_DATA_DIR = "/home/karan/Jan2021_data_merged/" + DATE
SAVE_FINAL_OUTPUT_AT = "/media/karan/Studyzz/CHANGA/SEM 8/00Jan_Final_Output/" + DATE + "_output"
SAVE_H5_AT_LOCATION = os.path.sep.join([SAVE_FINAL_OUTPUT_AT, os.path.basename(INPUT_DATA_DIR)])
SAVE_FIG_DIR = SAVE_FINAL_OUTPUT_AT + "/plots"


def addZeroPadding(input_img, kernel_shape=[3, 3]):
    # Add zero padding to the input image 
    image_padded = np.zeros((input_img.shape[0] + (kernel_shape[0]-1), 
                             input_img.shape[1] + (kernel_shape[1]-1)))   
    image_padded[(kernel_shape[0]//2):-(kernel_shape[0]//2), 
                 (kernel_shape[1]//2):-(kernel_shape[1]//2)] = input_img
    
    print("[ZeroPadding] Converted from {} to {}".format(input_img.shape, image_padded.shape))
    # It was returning data in float64 data type. I converted it to uint16 data type.
    return np.uint16(image_padded)

# get the temperature threshold (in Kelvin) value to remove the high/meduim clouds.
def get_temperature_threshold(tir1_hist_data):
    binwidth = 1
    n, bins = np.histogram(tir1_hist_data, bins=range(min(tir1_hist_data), max(tir1_hist_data)+binwidth, binwidth))
    # Apply Gaussian KDE smoothing
    import scipy.stats as stats
    density = stats.gaussian_kde(tir1_hist_data)
    density_values = density(bins)

    # Get all the local minima points of the histogram curve.
    from scipy.signal import argrelextrema
    extrema = argrelextrema(density_values, np.less)
    extrema_indexes = extrema[0]
    xx, yy = [], []
    temp_thresh = 270
    extrema_indexes = extrema[0]
    for i in extrema_indexes:
        if bins[i] >= 265 and bins[i] <= 275:
            yy.append(density_values[i])
            xx.append(bins[i])
    if len(xx) == 0:
        temp_thresh = 270
    elif len(xx) == 1:
        temp_thresh = xx[0]
    else:
        temp_thresh = np.mean(xx)
    return temp_thresh

def apply_TIR1_SD_matrix_threshold(thresh_val, tir1_data, vis_data):
    # Use sliding_window_view function to get all the (3, 3) windows from the TIR1 image.
    from numpy.lib.stride_tricks import sliding_window_view
    v = sliding_window_view(x=addZeroPadding(tir1_data, kernel_shape=[3, 3]), window_shape=(3, 3))
    # v.shape = (995, 796, 3, 3)
    
    # Create a new empty matrix of shape=v.shape[:2] for Standard Deviation.
    sd_matrix = np.zeros(shape=v.shape[:2])

    # Update the values of sd_matrix with the SD values of TIR1 image.
    for (iy, ix) in np.ndindex(v.shape[:2]):
        sd_matrix[iy, ix] = np.std(v[iy, ix, :, :])
    # sd_matrix = np.round(sd_matrix, decimals=2)

    ########################################################################
    # Keep only those VIS pixels, that have TIR1 pixel value > thresh_val=2.
    vis_data[sd_matrix >= thresh_val] = 0
    
    return vis_data

def apply_TIR1_dynamic_temp_threshold(tir1_data, vis_data):
    # get the temperature threshold (in Kelvin) value to remove the high/meduim clouds.
    temp_thresh = get_temperature_threshold(tir1_hist_data=tir1_data.ravel())

    print("[INFO] Selected Threshold of {} Kelvin".format(temp_thresh))
    
    # # Keep only those VIS pixels, in which have TIR1 pixel value > the threshold.
    vis_data[tir1_data <= temp_thresh] = 0

    return temp_thresh, vis_data

def plot_local_minima_points(hist_bins, hist_n):
    # Get all the local minima points of the histogram curve.
    from scipy.signal import argrelextrema
    local_minima_indexes = argrelextrema(hist_n, np.less)
    local_minima_indexes = local_minima_indexes[0]
    
    yy = [hist_n[i] for i in local_minima_indexes]
    xx = [hist_bins[i] for i in local_minima_indexes]
    
    plt.scatter(xx, yy, s=20, c="green", alpha=1)

    # Loop for annotation of all points and annotate them on plot
    for i in range(len(xx)):
        plt.annotate(str(int(xx[i])), (xx[i], yy[i]+ 0.005), size=10)
    
    # get all the local minima points that are b/w [30, 50).
    minima_points = [ele for ele in xx if (ele >= 30 and ele < 50)]
    return np.array(minima_points)

def plot_local_maxima_points(hist_bins, hist_n):
    # Get all the local minima points of the histogram curve.
    from scipy.signal import argrelextrema
    local_maxima_indexes = argrelextrema(hist_n, np.greater)
    local_maxima_indexes = local_maxima_indexes[0]
    
    yy = [hist_n[i] for i in local_maxima_indexes]
    xx = [hist_bins[i] for i in local_maxima_indexes]
    
    plt.scatter(xx, yy, s=20, color="red", alpha=1)

def apply_dynamic_reflectance_threshold(thresh_val, vis_data):
    vis_data[vis_data < thresh_val] = 0
    # vis_data[vis_data == 0] = 100 ## for Inv
    return vis_data

def again_apply_dynamic_reflectance_thresholding(vis_data, file_name, old_thresh_val):
    '''
    See the histogram and smooth-histogram. Plot the minima and maxima points.
    '''
    
    vis_ravel = vis_data.ravel()
    
    # Get the Histogram data
    hist_n, hist_bins = np.histogram(vis_ravel, 
                                     bins=np.arange(min(vis_ravel), max(vis_ravel)+1, 1))
    # Normalize the `hist_n` in [0,1] range.
    hist_n = (hist_n - hist_n.min()) / (hist_n.max() - hist_n.min())
    
    # Visualize the Original Histogram.
    fig = plt.figure(figsize=(12, 7), dpi=150)
    
    plt.subplot(2, 1, 1)
    plt.grid(); plt.xlim(0); plt.ylim(0, 0.050);
    plt.xticks(np.arange(0, vis_ravel.max(), 10))
    plt.title("Gray Histogram")
    plt.plot(hist_bins[:-1], hist_n, color='k')
    # Get all the local minima points of the histogram curve.
    minima_points = plot_local_minima_points(hist_bins[:-1], hist_n)
    plot_local_maxima_points(hist_bins[:-1], hist_n)
    
    #####################################################################
    # Visualize the Smoothed Histogram, Minima and Maxima points.
    
    # Apply Gaussian KDE smoothing
    import scipy.stats as stats
    kde = stats.gaussian_kde(vis_ravel)
    kde.set_bandwidth(bw_method=0.030)   #### Set Amount of Smoothing.
    kde_values = kde(hist_bins)
    
    plt.subplot(2, 1, 2)
    plt.grid(); plt.xlim(0); plt.ylim(0, 0.050);
    plt.xticks(np.arange(0, vis_ravel.max(), 10))
    plt.title("KDE Smoothed Histogram")
    plt.plot(hist_bins, kde_values, color="k")
    # Get local minima points of the histogram curve that are b/w [30, 50).
    _ = plot_local_minima_points(hist_bins[:-1], kde_values)
    plot_local_maxima_points(hist_bins[:-1], kde_values)
    
    plt.suptitle(t=file_name)
    plt.savefig(SAVE_FIG_DIR + "/" + file_name + "_8.png", bbox_inches='tight')
    # plt.show()
    plt.close()
    
    #####################################################################

    # Show the FINAL VIS output in a single plot.
    fig = plt.figure(figsize=(10, 4), dpi=150)
    
    plt.subplot(1, 2, 1)
    plt.imshow(X=vis_data, cmap='gray')
    plt.colorbar(label="Reflectance (%)")
    plt.xticks([]); plt.yticks([]);
    plt.title(label="Old Ref. Thresh=" + str(old_thresh_val))

    new_thresh_val = None
    if len(minima_points) == 0:
        new_thresh_val = old_thresh_val
    else:
        new_thresh_val = minima_points[0]
        # if (new_thresh_val < old_thresh_val-2) or (new_thresh_val > old_thresh_val+2):
        #     new_thresh_val = old_thresh_val

    vis_data = apply_dynamic_reflectance_threshold(new_thresh_val, vis_data.copy())
    print("[INFO] Again Applied Dynamic Reflectance Threshold on VIS image.")
    
    plt.subplot(1, 2, 2)
    plt.imshow(X=vis_data, cmap='gray')
    plt.colorbar(label="Reflectance (%)")
    plt.xticks([]); plt.yticks([]);
    plt.title(label="Refl. thresh=" + str(new_thresh_val))

    plt.suptitle(file_name)
    plt.savefig(SAVE_FIG_DIR + "/" + file_name + "_9.png", bbox_inches='tight')
    # plt.show()
    plt.close()

    #####################################################################

    # Show the FINAL VIS output in a single plot.
    fig = plt.figure(figsize=(7, 7), dpi=150)
    
    plt.subplot(1, 1, 1)
    plt.imshow(X=vis_data, cmap='gray')
    plt.colorbar(label="Reflectance (%)")
    # plt.xticks([]); plt.yticks([]);
    
    plt.title(label=file_name)
    plt.savefig(SAVE_FIG_DIR + "/" + file_name + "_10.png", bbox_inches='tight')
    # plt.show()
    plt.close()

    ############################################################################
    
    return vis_data

# file_paths = [
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0315_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0345_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0415_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0445_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0515_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0545_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0615_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0645_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0715_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0745_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0815_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0845_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0915_L1C_ASIA_MER.h5",
# # "/home/karan/15Jan2021/3RIMG_15JAN2021_0945_L1C_ASIA_MER.h5",
# ]


def main(file_path):
    # Load the .h5 file
    file = h5py.File(name=file_path, mode='r')
    # get the file_name from the file_path.
    file_name = os.path.basename(file_path).split(".")[0]
    
    # Generate image for IMG_TIR1 channel
    tir1_image = file["IMG_TIR1"]
    # Convert the data to a numpy matrix
    tir1_image = np.array(tir1_image)
    tir1_image = tir1_image[0, :, :]
    # Bring the TIR1 Image into TIR1_TEMP Temperature domain values.
    tir1_look_up = np.array(file["IMG_TIR1_TEMP"])
    tir1_image = tir1_look_up[tir1_image]
    tir1_image = np.uint16(tir1_image)

    # Cropping to keep only the Indian region.
    tir1_cropped_result = tir1_image[300:1192+1, 623:1391+1]

    ########################################################################

    '''Generate image for IMG_VIS channel '''

    vis_image = np.array(file["IMG_VIS"])
    vis_image = vis_image[0, :, :]
    # '''Convert VIS Image into ALBEDO domain.'''
    vis_look_up = np.array(file["IMG_VIS_ALBEDO"])
    vis_image = vis_look_up[vis_image]
    vis_image = np.uint16(vis_image)
    
    # Cropping to keep only the Indian region.
    vis_cropped_result = vis_image[300:1192+1, 623:1391+1]

    ########################################################################

    '''
    Read `Sun_Elevation` data. Derive the Solar Zenith angle.
    Perform Reflectance Normalization w.r.t Solar Zenith Angle.
    '''
    
    # read the Sun Elevation data for the database (.h5 file).
    sun_elevation = np.array(file["Sun_Elevation"])
    sun_elevation = sun_elevation[0, :, :]
    # initialize the solar_zenith_angle matrix
    solar_zenith_angle = np.zeros(sun_elevation.shape)
    # calculate Solar Zenith angle from the Sun_Elevation data
    solar_zenith_angle = 90 - (sun_elevation * 0.01)
    
    # Cropping to keep only the Indian region.
    solar_zenith_cropped = solar_zenith_angle[300:1192+1, 623:1391+1]
    # initialize the new VIS data containing the reflectance normalized data.
    vis_cropped_result = vis_cropped_result / (np.cos(solar_zenith_cropped * (np.pi / 180)))

    ########################################################################
    
    # Show the Original VIS data
    fig = plt.figure(figsize=(10, 4), dpi=150)
    
    plt.subplot(1, 2, 1)    
    plt.imshow(X=vis_cropped_result, cmap='gray')
    plt.colorbar(label="Reflectance (%)")
    plt.title(label="Original/Initial VIS data")
    plt.xticks([]); plt.yticks([]);

    #####################################################################
    #####################################################################
    
    '''
    1111111.
    Apply TIR1 Dynamic Temperature Threshold on VIS image.
    '''
    
    temp_thresh, vis_cropped_result = apply_TIR1_dynamic_temp_threshold(tir1_cropped_result.copy(), vis_cropped_result.copy())
    print("[INFO] Applied TIR1 Dynamic Temperature Threshold on VIS image.")
    
    ####################################################################
    
    plt.subplot(1, 2, 2)
    plt.imshow(X=vis_cropped_result, cmap='gray')
    plt.colorbar(label="Reflectance (%)")
    plt.title("After Dyn. Temp. thresh=" + str(temp_thresh))
    plt.xticks([]); plt.yticks([]);
    
    plt.suptitle(file_name)
    plt.savefig(SAVE_FIG_DIR + "/" + file_name + "_1.png", bbox_inches='tight')
    # plt.show()
    plt.close()

    # Save the final reflectance plot.
    fig = plt.figure()
    dpi = fig.get_dpi()
    w = vis_cropped_result.shape[1] / float(dpi)
    h = vis_cropped_result.shape[0] / float(dpi)
    fig.set_size_inches(w, h)
    # To make the content fill the whole figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # Then draw your image on it :
    ax.imshow(vis_cropped_result, cmap='gray', aspect='auto')
    fig.savefig(SAVE_FIG_DIR + "/" + file_name + "_2.png", bbox_inches='tight')
    # plt.show()
    plt.close()

    #####################################################################
        
    '''
    See the Histogram of `vis_cropped_result` to find a dynamic reflectance threshold value.
    '''

    cmap = plt.cm.gray
    vis_ravel = vis_cropped_result.ravel()
    bins = np.arange(min(vis_ravel), max(vis_ravel)+1, 1)
    
    hist_n, hist_bins = np.histogram(vis_ravel, bins=bins)
    # Normalize the `hist_n` in [0,1] range.
    hist_n = (hist_n - hist_n.min()) / (hist_n.max() - hist_n.min())

    linspace = np.linspace(start=0, stop=255, num=int(max(vis_ravel))+1, dtype=np.uint8)
    colors=cmap(linspace)
    
    # Show the Original VIS data
    fig = plt.figure(figsize=(10, 4), dpi=150)
    
    plt.subplot(1, 2, 1)
    plt.imshow(X=vis_cropped_result)
    plt.colorbar(label="Reflectance (%)")
    plt.set_cmap(cmap)    
    plt.title(label="Aft Dyn. Temp. thresh")
    plt.xticks([]); plt.yticks([]);
    
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.xlim(0); plt.ylim(0);
    plt.xticks(np.arange(0, vis_ravel.max(), 10))
    plt.title("Histogram")
    plt.plot(hist_bins[:-1], hist_n, color=(0, 0, 0))
    plt.scatter(hist_bins[:-1], [0.1]*len(hist_bins[:-1]), color=colors)
    # Plot horizontal colorbar
    plt.set_cmap(cmap)
    plt.colorbar(orientation="horizontal", pad=0.1)
    
    plt.suptitle(t=file_name)
    plt.savefig(SAVE_FIG_DIR + "/" + file_name + "_3.png", bbox_inches='tight')
    # plt.show()
    plt.close()

    #####################################################################
    
    # Set all the values that are <= 20 to 0.
    vis_cropped_result[vis_cropped_result <= 20] = 0
    
    #####################################################################
    
    '''
    See the histogram and smooth-histogram. Plot the minima and maxima points.
    '''
    
    vis_ravel = vis_cropped_result.ravel()
    
    # Get the Histogram data
    hist_n, hist_bins = np.histogram(vis_ravel, 
                                     bins=np.arange(min(vis_ravel), max(vis_ravel)+1, 1))
    # Normalize the `hist_n` in [0,1] range.
    hist_n = (hist_n - hist_n.min()) / (hist_n.max() - hist_n.min())
    
    # Visualize the Original Histogram.
    fig = plt.figure(figsize=(12, 7), dpi=150)
    
    plt.subplot(2, 1, 1)
    plt.grid(); plt.xlim(0); plt.ylim(0, 0.050);
    plt.xticks(np.arange(0, vis_ravel.max(), 10))
    plt.title("Gray Histogram")
    plt.plot(hist_bins[:-1], hist_n, color='k')
    # Get all the local minima points of the histogram curve.
    _ = plot_local_minima_points(hist_bins[:-1], hist_n)
    plot_local_maxima_points(hist_bins[:-1], hist_n)
    
    #####################################################################
    # Visualize the Smoothed Histogram, Minima and Maxima points.
    
    # Apply Gaussian KDE smoothing
    import scipy.stats as stats
    kde = stats.gaussian_kde(vis_ravel)
    kde.set_bandwidth(bw_method=0.030)   #### Set Amount of Smoothing.
    kde_values = kde(hist_bins)
    
    plt.subplot(2, 1, 2)
    plt.grid(); plt.xlim(0); plt.ylim(0, 0.050);
    plt.xticks(np.arange(0, vis_ravel.max(), 10))
    plt.title("KDE Smoothed Histogram")
    plt.plot(hist_bins, kde_values, color="k")
    # Get local minima points of the histogram curve that are b/w [30, 50).
    minima_points = plot_local_minima_points(hist_bins[:-1], kde_values)
    if len(minima_points) == 0:
        minima_points = np.append(minima_points, 37.0)
    
    plot_local_maxima_points(hist_bins[:-1], kde_values)

    plt.suptitle(t=file_name)
    plt.savefig(SAVE_FIG_DIR + "/" + file_name + "_4.png", bbox_inches='tight')
    # plt.show()
    plt.close()

    #####################################################################
    
    '''
    Perform Image threhsolding of all the local minima points that are b/w [30, 50).
    '''
    
    print("[INFO] Selected Minima points b/w [30, 50) are:", end=" ")
    print(minima_points)
    
    fig = plt.figure(figsize=(20, 5), dpi=150)
    
    i = 1; plt.subplot(1, len(minima_points)+1, i);
    plt.imshow(X=vis_cropped_result, cmap='gray')
    plt.colorbar(label="Reflectance (%)")
    plt.title(label="Gray Scale")
    plt.xticks([]); plt.yticks([]);
    
    for point in minima_points:
        i += 1; plt.subplot(1, len(minima_points)+1, i);
        plt.imshow(X=apply_dynamic_reflectance_threshold(point, vis_cropped_result.copy()), cmap='gray')
        plt.colorbar(label="Reflectance (%)")
        plt.title(label="Refl. thresh=" + str(point))
        plt.xticks([]); plt.yticks([]);

    plt.suptitle(t=file_name)
    plt.savefig(SAVE_FIG_DIR + "/" + file_name + "_5.png", bbox_inches='tight')
    # plt.show()
    plt.close()
    
    #####################################################################
    
    '''
    Select the very first local minima that is >=30.
    '''
    
    print("[INFO] Selected Minima point: ", minima_points[0])
    
    fig = plt.figure(figsize=(20, 6), dpi=150)
    
    plt.subplot(1, 3, 1)
    plt.imshow(X=vis_cropped_result, cmap='gray')
    plt.colorbar(label="Reflectance (%)")
    plt.title(label="Gray Scale")
    plt.xticks([]); plt.yticks([]);

    #####################################################################
    #####################################################################
    
    '''
    2222222.
    Apply Dynamic Reflectance Threshold.
    '''
    
    thresh_val = minima_points[0]
    old_thresh_val = thresh_val
    vis_cropped_result = apply_dynamic_reflectance_threshold(thresh_val, vis_cropped_result.copy())
    print("[INFO] Applied Dynamic Reflectance Threshold on VIS image.")

    plt.subplot(1, 3, 2);
    plt.imshow(X=vis_cropped_result, cmap='gray')
    plt.colorbar(label="Reflectance (%)")
    plt.title(label="Refl. thresh=" + str(thresh_val))
    plt.xticks([]); plt.yticks([]);

    #####################################################################
    #####################################################################
    
    '''
    3333333
    Apply TIR1 Standard Deviation Threshold on VIS image. 
    (Homogenety Test)
    Keep only those VIS pixels, that have TIR1 pixel value > the threshold=2.
    '''
    
    thresh_val = 2
    vis_cropped_result = apply_TIR1_SD_matrix_threshold(thresh_val, tir1_cropped_result.copy(), vis_cropped_result.copy())
    print("[INFO] Applied TIR1 Standard Deviation Threshold on VIS image.")
    
    ####################################################################
    
    # Visualize the Final output after removing medium/high clouds.
    # fig = plt.figure(figsize=(9, 9))
    plt.subplot(1, 3, 3)
    plt.imshow(X=vis_cropped_result, cmap='gray')
    plt.colorbar(label="Reflectance (%)")
    plt.title("Std. Dev. thresh=2")
    plt.xticks([]); plt.yticks([]);

    plt.suptitle(t=file_name)
    plt.savefig(SAVE_FIG_DIR + "/" + file_name + "_6.png", bbox_inches='tight')
    # plt.show()
    plt.close()
    
    #####################################################################

    # Show the FINAL VIS output in a single plot.
    fig = plt.figure(figsize=(7, 7), dpi=150)
    
    plt.subplot(1, 1, 1)
    plt.imshow(X=vis_cropped_result, cmap='gray')
    plt.colorbar(label="Reflectance (%)")
    # plt.xticks([]); plt.yticks([]);
    
    plt.title(label=file_name)
    plt.savefig(SAVE_FIG_DIR + "/" + file_name + "_7.png", bbox_inches='tight')
    # plt.show()
    plt.close()

    ############################################################################

    vis_cropped_result = again_apply_dynamic_reflectance_thresholding(vis_cropped_result.copy(), file_name, old_thresh_val)
    print("[Completed] ", file_name)
    
    return file_name, vis_cropped_result

############################################################################
############################################################################


'''
Main Caller Code.
Write the final output (Image with Fog patch) in an HDF5 file.
Here, I save whole day's output into one .H5 file.
'''

print("[DATE] " + DATE)

if not os.path.exists(SAVE_FINAL_OUTPUT_AT):
    os.mkdir(SAVE_FINAL_OUTPUT_AT)
if not os.path.exists(SAVE_FIG_DIR):
    os.mkdir(SAVE_FIG_DIR)

# creating a file
f = h5py.File(SAVE_H5_AT_LOCATION + ".h5", "w")

for file_path in sorted(glob.glob(INPUT_DATA_DIR + "/*.h5")):
    file_name, final_output = main(file_path=file_path)
    
    # Writing the Final Fog patch region into a HDF5 file.
    dset = f.create_dataset(file_name, data=final_output)
    print("[-Written-] ", file_name)
f.close()


##########################################################################
##########################################################################

'''
Reading the Saved Final Output .h5 file.
'''

# # importing necessary libraries
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
# import pandas as pd

# df = {"date_time" : [], "reflectance_340_440" : [], "reflectance_320_460" : []}

# INPUT_DATA_DIR = "/media/karan/Studyzz/CHANGA/SEM 8/00Jan_Final_Output/permutation_3_15Jan"

# for input_data in sorted(glob.glob(INPUT_DATA_DIR + "/15Jan2021.h5")):
#     # Load the .h5 file
#     file = h5py.File(name=input_data)
    
#     for key in file.keys():
#         # Generate image for IMG_TIR1 channel
#         data_1 = file[key]
#         # Convert the data to a numpy matrix
#         data_1 = np.array(data_1)
    
#         ############################################################################

#         fig = plt.figure(figsize=(7, 7), dpi=150)
#         plt.imshow(X=data_1, cmap='gray')
#         plt.plot(440, 340, color='red', marker='o', markersize=6)
#         plt.plot(460, 320, color='green', marker='o', markersize=6)
#         plt.colorbar(label="Reflectance (%)")
#         plt.title(label=key)
#         # plt.xticks([]); plt.yticks([]);
#         # plt.savefig("/media/karan/Studyzz/CHANGA/SEM 8/00Jan_Final_Output/old_007_tir1_image/" + key + ".png", bbox_inches='tight')
#         plt.show(); plt.close();
    
#         ############################################################################
    
#         # Append Date_time and reflectance values
#         date_time = key.split("_")[1] + "_" + key.split("_")[2]
        
#         df["date_time"].append(date_time)
#         df["reflectance_340_440"].append(data_1[340, 440])
#         df["reflectance_320_460"].append(data_1[320, 460])

#         ############################################################################

# new_date_time = ["15JAN2021_0330", 
#                  "15JAN2021_0400", "15JAN2021_0430", 
#                  "15JAN2021_0500", "15JAN2021_0530",
#                  "15JAN2021_0600", "15JAN2021_0630",
#                  "15JAN2021_0700", "15JAN2021_0730",
#                  "15JAN2021_0800", "15JAN2021_0830", 
#                  "15JAN2021_0900", "15JAN2021_0930"]
# new_reflectance = [np.nan] * len(new_date_time)

# df["date_time"].extend(new_date_time)
# df["reflectance_340_440"].extend(new_reflectance)
# df["reflectance_320_460"].extend(new_reflectance)

# df = pd.DataFrame(df)

# df["date_time"] = pd.to_datetime(df["date_time"], format="%d%b%Y_%H%M", utc=True)

# df = df.sort_values("date_time", ignore_index=True)

# # df[["reflectance_340_440", "reflectance_320_460"]] = df[["reflectance_340_440", "reflectance_320_460"]].interpolate(method ='linear', limit_direction ='forward')

# fig = plt.figure(figsize=(7, 5), dpi=150)
# plt.plot(df.dropna()["reflectance_340_440"], "ro-", label="original340_440")
# plt.plot(df["reflectance_340_440"].interpolate(method ='linear', limit_direction ='forward'), "bx", label="linearIntp340_440")

# plt.plot(df.dropna()["reflectance_320_460"], "ro-", label="original320_460")
# plt.plot(df["reflectance_320_460"].interpolate(method ='linear', limit_direction ='forward'), "bx", label="linearIntp320_460")
# plt.xticks(np.arange(0, len(df), 1))
# plt.grid()
# plt.legend()
# plt.show(); plt.close();





##########################################################################
##########################################################################
##########################################################################
##########################################################################




'''
Plot Delhi, Amritsar, Jaipur, etc cities on the map.
Reading the Saved Final Output .h5 file.
Overlap the Indian Boundary on the composite image.
'''

# # importing necessary libraries
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
# import pandas as pd

# INPUT_DATA_PATH = "/media/karan/Studyzz/CHANGA/SEM 8/00Jan_Final_Output/composite_images/15day_composite_images.h5"

# # Load the .h5 file
# file = h5py.File(name=INPUT_DATA_PATH)

# # Generate image for IMG_TIR1 channel
# data_1 = file["0715"]
# # Convert the data to a numpy matrix
# data_1 = np.array(data_1)

# def add_boundary(data_1):
#     from mpl_toolkits.basemap import Basemap
#     map = Basemap(projection='merc', 
#                 llcrnrlon=68,    # lower left corner longitude
#                 llcrnrlat=6,     # lower left corner latitude
#                 urcrnrlon=97,    # upper right corner longitude
#                 urcrnrlat=37,    # upper right corner latitude
#                 resolution='l',  # low resolution
#               )
#     map.drawmapboundary(color="gray", linewidth=0.5) # Frame 
#     map.drawcoastlines(color="gray", linewidth=0.5) # Coastline
#     map.drawcountries(color="gray", linewidth=0.5)
#     map.readshapefile(shapefile="/media/karan/Studyzz/CHANGA/SEM 8/India_Sate_Boundary_EPSG_4326/India_Sate_Boundary_EPSG_4326",
#                       name="India_Sate_Boundary_EPSG_4326",
#                       # drawbounds=True,
#                       # ax=None,
#                       # zorder=None,
#                       linewidth=0.5,
#                       color="gray"
#                       ) # Boundary around India and its states
    
#     x = np.linspace(0, map.urcrnrx, data_1.shape[1])
#     y = np.linspace(0, map.urcrnry, data_1.shape[0])
#     xx, yy = np.meshgrid(x, y)
#     temp = np.flipud(data_1)
    
#     map.pcolormesh(xx, yy, temp, cmap="gray", shading='auto')
#     plt.colorbar(label="Reflectance (%)")

#     # Add scatter points for Amritsar, New Delhi, Lucknow, Varanasi, Jaipur locations.
#     cities = ["Amritsar", "New Delhi", "Lucknow", "Varanasi", "Jaipur"]
#     lats   = [31.633980,  28.644800,   26.850000, 25.321684,  26.922070]
#     longs  = [74.872261,  77.216721,   80.949997, 82.987289,  75.778885]
#     markers= ['o', 's', '*', 'd', '^']
    
#     longs_x, lats_y = map(longs, lats)

#     map.scatter(longs_x[0], lats_y[0], s=150, marker=markers[0], color='k', zorder=2.5)
#     plt.text(longs_x[0], lats_y[0], cities[0], color="red", fontsize="x-large", fontweight="roman")

#     map.scatter(longs_x[1], lats_y[1], s=150, marker=markers[1], color='k', zorder=2.5)
#     plt.text(longs_x[1], lats_y[1], cities[1], color="red", fontsize="x-large", fontweight="roman")

#     map.scatter(longs_x[2], lats_y[2], s=250, marker=markers[2], color='k', zorder=2.5)
#     plt.text(longs_x[2], lats_y[2], cities[2], color="red", fontsize="x-large", fontweight="roman")

#     map.scatter(longs_x[3], lats_y[3], s=200, marker=markers[3], color='k', zorder=2.5)
#     plt.text(longs_x[3], lats_y[3], cities[3], color="red", fontsize="x-large", fontweight="roman")

#     map.scatter(longs_x[4], lats_y[4], s=200, marker=markers[4], color='k', zorder=2.5)
#     plt.text(longs_x[4], lats_y[4], cities[4], color="red", fontsize="x-large", fontweight="roman")
    
#     # for i in range(len(cities)):
#     #     # plt.annotate(cities[i], (longs_x[i], lats_y[i] + 0.2))
#     #     plt.text(longs_x[i], lats_y[i], cities[i], color="red", fontsize="x-large", 
#     #              fontweight="roman");


# fig = plt.figure(figsize=(9, 7), dpi=150)
# # To make the content fill the whole figure
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# # ax.set_axis_off()
# fig.add_axes(ax)

# # title = "_".join(operational_product_filename.split("_")[1:3]) + "UTC"
# # ax.text(0.5, 0.85, title, color='black', fontsize='large', fontweight="roman", 
# #         fontstretch="normal", transform=ax.transAxes)

# add_boundary(data_1)    
# # plt.imshow(X=operational_product_fog, cmap='binary')
# # plt.colorbar(label="Reflectance (%)")
# plt.title(label="15Day Composite Image: 0715")
# # plt.xticks([]); plt.yticks([]);
# # fig.savefig(SAVE_FIG_DIR + "/" + file_name + "_2.png", bbox_inches='tight')    
# # plt.savefig(OUR_PRODUCT_DIR + "/fog_validation_plots_with_india_boundary_" + DATE + "/" + operational_product_filename + ".png", bbox_inches='tight')

# plt.show()
# plt.close()



##########################################################################
##########################################################################
##########################################################################
##########################################################################







