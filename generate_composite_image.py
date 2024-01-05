#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 22:44:19 2022

@author: karan
"""

'''
Get the Composite Image using minimum Reflectance value.
'''


# importing necessary libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

INPUT_DATA_DIR = "/home/karan/Jan2021_data_merged"
SAVE_COMPOSITE_IMG_AT = "/media/karan/Studyzz/CHANGA/SEM 8/00Jan_Final_Output/composite_images"

def get_composite_image_data(time, span):
    all_data = []

    for file_path in sorted(glob.glob(INPUT_DATA_DIR + "/**/*_" + time + "_*.h5", recursive=True)):
        # Load the .h5 file
        file = h5py.File(name=file_path, mode='r')
        # get the file_name from the file_path.
        file_name = os.path.basename(file_path).split(".")[0]
    
        ########################################################################
    
        '''Generate image for IMG_VIS channel '''
        
        vis_image = np.array(file["IMG_VIS"])
        vis_image = vis_image[0, :, :]
        # '''Convert VIS Image into ALBEDO domain.'''
        vis_look_up = np.array(file["IMG_VIS_ALBEDO"])
        vis_image = vis_look_up[vis_image]
        vis_image = np.uint16(vis_image)
        # Cropping to keep only the Indian region.
        vis_image = vis_image[300:1192+1, 623:1391+1]
    
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
        vis_image = vis_image / (np.cos(solar_zenith_cropped * (np.pi / 180)))
        
        ############################################################################

        all_data.append(vis_image)
        print("[INFO] Completed:", file_name)

    # Convert list to array: Final shape becomes: (None, 893, 769)
    all_data = np.array(all_data)

    # Find the resultant minimum matrix
    vis_composite_image = np.min(all_data, axis=0)

    ############################################################################
    
    # Creating plot of VIS Composite Image
    fig = plt.figure(figsize=(7, 7), dpi=150)
    
    plt.subplot(1, 1, 1)
    plt.imshow(X=vis_composite_image, cmap='gray')
    # plt.plot(250, 450, color='red', marker='o', markersize=6)
    plt.colorbar(label="Reflectance (%)")
    plt.title(label=time + "Composite Image:" + span)
    # plt.xticks([]); plt.yticks([]);
    # plt.savefig(SAVE_COMPOSITE_IMG_AT + "/" + span + "_composite_image_" + time + ".png", bbox_inches='tight')
    # plt.show()
    plt.close()
    
    ############################################################################

    # vis_composite_image[vis_composite_image > 25] = 20
    
    ############################################################################

    # Creating plot of VIS Composite Image
    fig = plt.figure(figsize=(7, 7), dpi=150)
    
    plt.subplot(1, 1, 1)
    plt.imshow(X=vis_composite_image, cmap='gray')
    plt.plot(200, 200, color='red', marker='o', markersize=3)
    plt.text(200+1, 200+1, "(200, 200)", color='red', fontsize='medium', rotation=35)
    
    plt.plot(245, 245, color='red', marker='o', markersize=3)
    plt.text(245+1, 245+1, "(245, 245)", color='red', fontsize='medium', rotation=35)
        
    plt.plot(300, 300, color='red', marker='o', markersize=3)
    plt.text(300+1, 300+1, "(300, 300)", color='red', fontsize='medium', rotation=35)
    
    plt.plot(350, 300, color='red', marker='o', markersize=3)
    plt.text(350+1, 300+1, "(350, 300)", color='red', fontsize='medium', rotation=35)
    
    plt.plot(410, 330, color='red', marker='o', markersize=3)
    plt.text(410+1, 330+1, "(430, 330)", color='red', fontsize='medium', rotation=35)

    plt.plot(480, 340, color='red', marker='o', markersize=3)
    plt.text(480+1, 340+1, "(480, 340)", color='red', fontsize='medium', rotation=35)

    plt.colorbar(label="Reflectance (%)")
    plt.title(label=time + "Composite Image:" + span)
    # plt.grid(which='both')
    # plt.xticks([]); plt.yticks([]);
    plt.savefig(SAVE_COMPOSITE_IMG_AT + "/" + span + "_composite_image_with_points_" + time + ".png", bbox_inches='tight')
    plt.show()
    # plt.close()
    
    ############################################################################
    
    to_return = [vis_composite_image[200, 200], vis_composite_image[245, 245], 
                 vis_composite_image[300, 300], vis_composite_image[300, 350],
                 vis_composite_image[330, 410], vis_composite_image[340, 480]]
    return to_return

times = ["0" + str(i) + "15" for i in range(3, 10)]
times.extend(["0" + str(i) + "45" for i in range(3, 10)])
times = sorted(times)
# times = ["0515"]
span = "15day"

df = {"time" : [], "image_200_200" : [], "image_245_245" : [], 
                   "image_300_300" : [], "image_300_350" : [],
                   "image_330_410" : [], "image_340_480" : []}

for time in times:
    # Call the function
    a, b, c, d, e, f = get_composite_image_data(time, span)
    df["time"].append(time)
    df["image_200_200"].append(a)
    df["image_245_245"].append(b)
    df["image_300_300"].append(c)
    df["image_300_350"].append(d)
    df["image_330_410"].append(e)
    df["image_340_480"].append(f)

df = pd.DataFrame(df)
# df_sorted = df.sort_values(by=["time"], inplace=False, ignore_index=True)


# plt.xticks(np.arange(0, len(df), 1))
plt.rcParams["figure.dpi"] = 150
df.plot(x="time", y=["image_200_200", "image_245_245", "image_300_300", "image_300_350", "image_330_410", "image_340_480"], figsize=(7, 5))
plt.ylabel("Reflectance values")
plt.grid()
plt.legend()
plt.savefig(SAVE_COMPOSITE_IMG_AT + "/" + span + "_composite_image_reflectance_variation" + ".png", bbox_inches='tight')
plt.show(); plt.close();


############################################################################
############################################################################
############################################################################
############################################################################



'''
Get the Composite Image using minimum Reflectance value.
Save the composite images in a .h5 file.
'''


# # importing necessary libraries
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
# import pandas as pd

# INPUT_DATA_DIR = "/home/karan/Jan2021_data_merged"
# SAVE_COMPOSITE_IMG_AT = "/media/karan/Studyzz/CHANGA/SEM 8/00Jan_Final_Output/composite_images"

# def get_composite_image_data(time, span):
#     all_data = []

#     for file_path in sorted(glob.glob(INPUT_DATA_DIR + "/**/*_" + time + "_*.h5", recursive=True)):
#         # Load the .h5 file
#         file = h5py.File(name=file_path, mode='r')
#         # get the file_name from the file_path.
#         file_name = os.path.basename(file_path).split(".")[0]
    
#         ########################################################################
    
#         '''Generate image for IMG_VIS channel '''
        
#         vis_image = np.array(file["IMG_VIS"])
#         vis_image = vis_image[0, :, :]
#         # '''Convert VIS Image into ALBEDO domain.'''
#         vis_look_up = np.array(file["IMG_VIS_ALBEDO"])
#         vis_image = vis_look_up[vis_image]
#         vis_image = np.uint16(vis_image)
#         # Cropping to keep only the Indian region.
#         vis_image = vis_image[300:1192+1, 623:1391+1]
    
#         ########################################################################
    
#         '''
#         Read `Sun_Elevation` data. Derive the Solar Zenith angle.
#         Perform Reflectance Normalization w.r.t Solar Zenith Angle.
#         '''
        
#         # read the Sun Elevation data for the database (.h5 file).
#         sun_elevation = np.array(file["Sun_Elevation"])
#         sun_elevation = sun_elevation[0, :, :]
#         # initialize the solar_zenith_angle matrix
#         solar_zenith_angle = np.zeros(sun_elevation.shape)
#         # calculate Solar Zenith angle from the Sun_Elevation data
#         solar_zenith_angle = 90 - (sun_elevation * 0.01)
#         # Cropping to keep only the Indian region.
#         solar_zenith_cropped = solar_zenith_angle[300:1192+1, 623:1391+1]
#         # initialize the new VIS data containing the reflectance normalized data.
#         vis_image = vis_image / (np.cos(solar_zenith_cropped * (np.pi / 180)))
        
#         ############################################################################

#         all_data.append(vis_image)
#         print("[INFO] Completed:", file_name)

#     # Convert list to array: Final shape becomes: (None, 893, 769)
#     all_data = np.array(all_data)

#     # Find the resultant minimum matrix
#     vis_composite_image = np.min(all_data, axis=0)

#     ############################################################################
    
#     return vis_composite_image

# times = ["0" + str(i) + "15" for i in range(3, 10)]
# times.extend(["0" + str(i) + "45" for i in range(3, 10)])
# times = sorted(times)
# # times = ["0515"]
# span = "15day"

# # creating a file
# f = h5py.File(SAVE_COMPOSITE_IMG_AT + "/15day_composite_images.h5", "w")

# for time in times:
#     # Call the function
#     vis_composite_image = get_composite_image_data(time, span)
    
#     ############################################################################

#     # Writing the Final Fog patch region into a HDF5 file.
#     dset = f.create_dataset(time, data=vis_composite_image)

# f.close()


############################################################################
############################################################################
############################################################################
############################################################################



"""temp"""

# '''
# Get the approximate Reflectance value of Clear image i.e. land.
# '''

# # importing necessary libraries
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
# import pandas as pd

# DATE = "15Jan2021"

# INPUT_DATA_DIR = "/home/karan/Jan2021_data_merged/" + DATE

# df = {"date_time" : [], "reflectance_450_250" : []}

# for file_path in sorted(glob.glob(INPUT_DATA_DIR + "/*.h5")):
#     # Load the .h5 file
#     file = h5py.File(name=file_path, mode='r')
#     # get the file_name from the file_path.
#     file_name = os.path.basename(file_path).split(".")[0]

#     ########################################################################

#     '''Generate image for IMG_VIS channel '''
    
#     vis_image = np.array(file["IMG_VIS"])
#     vis_image = vis_image[0, :, :]
#     # '''Convert VIS Image into ALBEDO domain.'''
#     vis_look_up = np.array(file["IMG_VIS_ALBEDO"])
#     vis_image = vis_look_up[vis_image]
#     vis_image = np.uint16(vis_image)
#     # Cropping to keep only the Indian region.
#     vis_image = vis_image[300:1192+1, 623:1391+1]

#     ########################################################################

#     '''
#     Read `Sun_Elevation` data. Derive the Solar Zenith angle.
#     Perform Reflectance Normalization w.r.t Solar Zenith Angle.
#     '''
    
#     # read the Sun Elevation data for the database (.h5 file).
#     sun_elevation = np.array(file["Sun_Elevation"])
#     sun_elevation = sun_elevation[0, :, :]
#     # initialize the solar_zenith_angle matrix
#     solar_zenith_angle = np.zeros(sun_elevation.shape)
#     # calculate Solar Zenith angle from the Sun_Elevation data
#     solar_zenith_angle = 90 - (sun_elevation * 0.01)
#     # Cropping to keep only the Indian region.
#     solar_zenith_cropped = solar_zenith_angle[300:1192+1, 623:1391+1]
#     # initialize the new VIS data containing the reflectance normalized data.
#     vis_image = vis_image / (np.cos(solar_zenith_cropped * (np.pi / 180)))

#     ############################################################################
#     # Creating plot of VIS image after Reflectance Normalization
        
#     fig = plt.figure(figsize=(7, 7), dpi=150)
    
#     plt.subplot(1, 1, 1)
#     plt.imshow(X=vis_image, cmap='gray')
#     plt.plot(250, 450, color='red', marker='o', markersize=6)
#     plt.colorbar(label="Reflectance (%)")
#     plt.title(label="")
#     # plt.xticks([]); plt.yticks([]);
    
#     plt.suptitle(t=file_name)
#     # plt.savefig(OUR_PRODUCT_DIR + "/2fog_validation_plots_" + DATE + "/" + k + ".png", bbox_inches='tight')
#     # plt.show()
#     plt.close()

#     ############################################################################
#     ############################################################################

#     # Append Date_time and reflectance values
#     date_time = file_name.split("_")[1] + "_" + file_name.split("_")[2]
    
#     df["date_time"].append(date_time)
#     df["reflectance_450_250"].append(vis_image[450, 250])
    
#     ############################################################################

#     file.close()

# df = pd.DataFrame(df)







