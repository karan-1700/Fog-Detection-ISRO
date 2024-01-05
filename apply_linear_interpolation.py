#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:52:55 2022

@author: karan
"""

'''
Applying Linear Interpolation on while 2D matrix at once.
Interpolation for Data Imputation. 
Use 30-minute interval data to generate 15-minute interval data (using interpolation).
'''


# importing necessary libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

def interpolate_at_15_intervals(input_data_path):
    # Initialize an empty dictionary.
    df = {"date_time" : [], "reflectance" : []}
    
    file_name = os.path.basename(input_data_path).split(".")[0]
    
    # Load the .h5 file.
    file = h5py.File(name=input_data_path)
    
    for key in file.keys():
        # Generate image for IMG_TIR1 channel.
        data_1 = file[key]
        # Convert the data to a numpy matrix.
        data_1 = np.array(data_1)
    
        ############################################################################
    
        # Append Date_time and reflectance values.
        date_time = key.split("_")[1] + "_" + key.split("_")[2]
        df["date_time"].append(date_time)
        df["reflectance"].append(data_1)

    ############################################################################

    # Add the NEW remaining Date-Time data that are to be Interpolated.
    new_date_time = ["0330", 
                     "0400", "0430", 
                     "0500", "0530",
                     "0600", "0630",
                     "0700", "0730",
                     "0800", "0830",
                     "0900", "0930"]
    
    new_date_time = [file_name + "_" + i for i in new_date_time]
    # Initialize an Empty numpy array for the above NEW Reflectance data.
    new_reflectance = np.empty(data_1.shape)
    # Fill the empty array with NaN values.
    new_reflectance.fill(np.nan)
    new_reflectance = [new_reflectance] * len(new_date_time)
    
    # Append the NEW remaining Date-Time and Reflectance data to the original data.
    df["date_time"].extend(new_date_time)
    df["reflectance"].extend(new_reflectance)
    del new_reflectance, new_date_time
    
    ############################################################################

    # Convert the Dictionary to a Pandas DataFrame.
    df = pd.DataFrame(df)
    # Convert the "date_time" column type from String dtype to pandas.DateTime dtype.
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d%b%Y_%H%M", utc=True)
    # Sort the dataframe by the "date_time" column.
    df = df.sort_values("date_time", ignore_index=True)

    ############################################################################

    # Create a New empty dataframe
    newdf = pd.DataFrame()
    # Add the "date_time" and Reflectance-matrix data after Flattening it.
    for i in range(len(df)):
        newdf[df["date_time"][i]] = df["reflectance"][i].flatten()
    # Transpose the dataframe.
    newdf = newdf.T
    # Apply Linear Interpolation to fill the NaN values
    print("[INFO] Interpolating: ", file_name)
    newdf = newdf.interpolate(method ='linear', limit_direction ='forward')
    del df

    ############################################################################

    # Create a Final empty dataframe
    finaldf = pd.DataFrame()
    # Add the "date_time" and Reflectance-matrix data after Un-Flattening it.
    for i in range(len(newdf)):
        finaldf[newdf.index[i]] = [newdf.loc[[newdf.index[i]]].values.reshape(data_1.shape)]
    # Transpose the dataframe.
    finaldf = finaldf.T
    # Set default indexing.
    finaldf.reset_index(drop=False, inplace=True)
    # Rename the columns
    finaldf.rename(columns={"index" : "date_time", 0 : "reflectance"}, inplace=True)
    # Convert the "date_time" column type from pandas.DateTime dtype to String dtype.
    finaldf["date_time"] = finaldf["date_time"].dt.strftime("%d%b%Y_%H%M")

    ############################################################################

    """
    Writing the final INTERPOLATED data (at 15-intervals) into a new .h5 file.
    """

    # Open a .h5 file in Write mode.
    save_h5_dir = os.path.join(os.path.dirname(input_data_path), file_name+"_interpolated_15-intervals")
    if not os.path.exists(save_h5_dir):
        os.mkdir(save_h5_dir)
    save_h5_dir = os.path.join(save_h5_dir, file_name+"_interpolated_15-intervals.h5")
    
    f = h5py.File(save_h5_dir, "w")
    # Iterate through the dataframe and write the data into the .h5 file.
    for index, row in finaldf.iterrows():
        # print(row["date_time"], row["reflectance"].shape)        
        # Writing the Final Fog patch region into a HDF5 file.
        dset = f.create_dataset(row["date_time"], data=row["reflectance"])
    f.close()
    print("[INFO] Written: ", file_name)


BASE_DIR_PATH = "/media/karan/Studyzz/CHANGA/SEM 8/00Jan_Final_Output"

for i in range(21, 32):
    input_data_path = os.path.sep.join([BASE_DIR_PATH, str(i)+"Jan2021_output", str(i)+"Jan2021.h5"])
    if os.path.exists(input_data_path):
        # call the above function to interpolate at 15-min intervals
        interpolate_at_15_intervals(input_data_path)
    else:
        print("Provided path does not exists: {}".format(input_data_path))

##########################################################################
##########################################################################
##########################################################################
##########################################################################

'''
Reading the Saved Final Output .h5 file.
Saving the interpolated plots as images.
'''

# # importing necessary libraries
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
# import pandas as pd

# # df = {"date_time" : [], "reflectance_340_440" : [], "reflectance_320_460" : []}

# BASE_DIR_PATH = "/media/karan/Studyzz/CHANGA/SEM 8/00Jan_Final_Output/"

# for input_data_path in sorted(glob.glob(BASE_DIR_PATH + "/**/*_interpolated_15-intervals.h5", recursive=True)):    
#     # Load the .h5 file
#     file = h5py.File(name=input_data_path)
    
#     for key in file.keys():
#         # Generate image for IMG_TIR1 channel
#         data_1 = file[key]
#         # Convert the data to a numpy matrix
#         data_1 = np.array(data_1)
    
#         ############################################################################

#         save_fig_path = os.path.join(os.path.dirname(input_data_path), os.path.basename(input_data_path).split("_")[0] + "_interpolated_plots")
#         if not os.path.exists(save_fig_path):
#             os.mkdir(save_fig_path)
        
#         ############################################################################

#         fig = plt.figure(figsize=(7, 7), dpi=150)
#         plt.imshow(X=data_1, cmap='gray')
#         # plt.plot(440, 340, color='red', marker='o', markersize=6)
#         # plt.plot(460, 320, color='green', marker='o', markersize=6)
#         plt.colorbar(label="Reflectance (%)")
#         plt.title(label=key)
#         # plt.xticks([]); plt.yticks([]);
#         plt.savefig(save_fig_path + "/" + key + ".png", bbox_inches='tight')
#         plt.show(); plt.close();
    
#         ############################################################################
    
#         # # Append Date_time and reflectance values
#         # # date_time = key.split("_")[1] + "_" + key.split("_")[2]
        
#         # df["date_time"].append(key)
#         # df["reflectance_340_440"].append(data_1[340, 440])
#         # df["reflectance_320_460"].append(data_1[320, 460])
    
#     file.close()
#     print("[INFO] Completed: ", os.path.basename(input_data_path))


# # # df = pd.DataFrame(df)


# # # df["date_time"] = pd.to_datetime(df["date_time"], format="%d%b%Y_%H%M", utc=True)

# # # df = df.sort_values("date_time", ignore_index=True)


##########################################################################
##########################################################################
##########################################################################
##########################################################################


'''
Code to list all "interpolated_15-intervals.h5" files.
'''

# BASE_DIR_PATH = "/media/karan/Studyzz/CHANGA/SEM 8/00Jan_Final_Output"

# for file_path in sorted(glob.glob(BASE_DIR_PATH + "/**/*_interpolated_15-intervals.h5", recursive=True)):
#     # Load the .h5 file
#     # file = h5py.File(name=file_path, mode='r')
#     # get the file_name from the file_path.
#     file_name = os.path.basename(file_path)
#     print(file_name)




