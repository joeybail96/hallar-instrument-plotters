 # -*- coding: utf-8 -*-

###############################################################################
#%%# prepare workspace
###############################################################################

# import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
import datetime as dt
import tempfile




###############################################################################
#%%# define utility functions
###############################################################################


# 
def fix_x00_issue(file_path, save_dir, title):
    
    # open the csv file as a text file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # remove all the null \x00 values for all lines
    cleaned_lines = [line.replace('\x00', '') for line in lines]

    # save the cleaned content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='') as tmpfile:
        tmpfile.writelines(cleaned_lines)
        cleaned_file_path = tmpfile.name
    
    # read the cleaned file with pandas
    df = pd.read_csv(cleaned_file_path, header=None)
    
    # remove the temporary file
    os.remove(cleaned_file_path)
    
    
    raw_time = df.iloc[:,0]
    timestamps = pd.Series(raw_time)
    timestamps = pd.to_datetime(timestamps)
    dates = timestamps.dt.strftime('%Y-%m-%d')   
    date_str = dates.iloc[1]

    
    # Create the error message text file
    error_message = (
        "Error occurred in this csv file. One of the UTC time signatures contained "
        "extraneous null values of x00\\x00..."
    )
    error_filename = f"{date_str}_{title}_error.txt"
    error_file_path = os.path.join(save_dir, error_filename)
    
    with open(error_file_path, 'w') as error_file:
        error_file.write(error_message)
    
    # return the dataframe with no extraneous null \x00 values
    return df
    

# grimm_dir, grimm_efficiencies, save_dir, project_title, [2024, 2025]
def read_grimm(data_dir, eff, save_dir, title, years):
    
    
    # get a list of all subdirectories in the data_dir
    subdirectories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # initialize an empty list to store dataframes
    dfs = []

    # iterate through each subdirectory
    for subdirectory in subdirectories:
        # check if the subdirectory label contains a year from the input years
        if any(str(year) in subdirectory for year in years):
            # read every csv file in the subdirectory
            csv_files = [f for f in os.listdir(os.path.join(data_dir, subdirectory)) if f.endswith('.csv')]
            for csv_file in csv_files:
                # read the csv file, keeping the header row
                file_path = os.path.join(data_dir, subdirectory, csv_file)
                df = pd.read_csv(file_path, header=None)
                
                has_nan = df.iloc[:, 0].isna().any()
                if has_nan:    
                    df = fix_x00_issue(file_path, save_dir, title)
                
                dfs.append(df)
    
    # combine data from all csv files into one dataframe
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # return formatted file
    return format_grimm(combined_df, eff)


# format the raw grimm data
def format_grimm(df, eff):
    
    # grab time (UTC)
    utc = pd.to_datetime(df.iloc[:,0],format='mixed')

    # drop time columns from df
    df = df.iloc[:, 1:]
    
    # trim away smallest and largest bin (inaccurate measurements)
    df = df.iloc[:, 1:-1]
    
    # convert dust concentrations to n/cm3 (originally n/100ml)
    df = df / 100
            
    
    if eff != []:
        # adjust dust concentrations based on inlet efficiencies
        num_columns = df.shape[1]
        df = df / eff[0:num_columns]
    
    # return formatted grimm data
    return utc, df


# # bin aerosol data
# def bin(data, bins):
    
#     # grab indicies of neighboring elements
#     x1_indices = np.arange(0, len(bins) - 1, dtype=int)
#     x2_indices = x1_indices + 1
    
#     # calculate mean of neighboring bins
#     mean_dp = (bins[x2_indices] + bins[x1_indices]) / 2
    
#     # convert data array into dataframe
#     df = pd.DataFrame(data)
    
#     # rename columns to be mean particle diameters
#     df.columns = mean_dp

#     return df
  
    
  
def bin(df, bins):
    # Calculate the mean diameter for each bin
    bins['Dp'] = (bins['Size (µm)'] + bins['Size (µm)'].shift(-1)) / 2
    bins['dlogDp'] = np.log10(bins['Size (µm)']).shift(-1) - np.log10(bins['Size (µm)'])
    
    # Create a mapping from bin names to mean diameters and dlogDp values
    bin_mapping = {bin_num: mean_diameter for bin_num, mean_diameter in zip(bins['Bin Number'], bins['Dp'])}
    bin_dlogDp_mapping = {bin_num: dlogDp for bin_num, dlogDp in zip(bins['Bin Number'], bins['dlogDp'])}
    
    # Divide the data in each 'opc_binX' column by the corresponding 'dlogDp' value
    for column in df.columns:

        if column in bin_dlogDp_mapping:
            # Divide the data in the 'opc_bin' column by the corresponding 'dlogDp' value
            df[column] = df[column] / bin_dlogDp_mapping[column]
    
    # Rename the columns in the data DataFrame based on the bin mapping
    df = df.rename(columns=bin_mapping)
    bins = bins.iloc[:-1].reset_index(drop=True)
    

    return df, bins    
  

# normalize bin collections by log differences
def sizing(data, bins):
    
    # grab indicies of neighboring elements
    x1_indices = np.arange(0, len(bins) - 1, dtype=int)
    x2_indices = x1_indices + 1
    
    # calculate log differences
    dlogDp = np.log10(bins[x2_indices]) - np.log10(bins[x1_indices])
    
    # normalize particle #s by bin size
    result_df = data.divide(dlogDp, axis ='columns')
    
    # return the normalized data
    return result_df
 
    
# assemble time and aerosol date
def combine(date, data):
    
    # combine date with data    
    result_df = pd.concat([date, data], axis=1)
    
    # add a column title over the dates
    result_df.columns = ['Time_UTC'] + list(result_df.columns[1:])
    
    # Convert Time_UTC to timezone-aware datetime in UTC
    result_df['Time_UTC'] = pd.to_datetime(result_df['Time_UTC'], utc=True)
    
    # Add Time_MST column (UTC-7, without daylight saving)
    result_df['Time_MST'] = result_df['Time_UTC'].dt.tz_convert('Etc/GMT+7')
    
    # returned combined dataframe
    return result_df


# organize dust data by date collected
def split(df):
    
    # 
    mst = df['Time_MST']
    
    #
    dates = mst.dt.date
    
    # 
    unique_dates = dates.unique()
    
    # initialize dict for daily data collections
    data_dict = {}
    
    # iterate through days of collected data
    for date in unique_dates:
        
        # segregate data by day it was collected
        day_data = df[dates == date]
        
        # organize daily data into dictionary
        data_dict[str(date)] = day_data
    
    return data_dict
    

# iterate through each date for plotting
def process_daily_data(daily_grimm, bins, min_count, max_count, path, title):
    # Iterate through each entry in daily_grimm
    for date, day_data in daily_grimm.items():  
        plot_size_dist(day_data, date, bins, min_count, max_count, path, title)
        

# create daily contour plots
def plot_size_dist(day_data, date, bins, min_count, max_count, path, title):
    # Extract time from 'Time_MST' column
    time = pd.to_datetime(day_data['Time_MST']).dt.time

    # Use only bin columns that match the provided bins DataFrame
    bin_columns = [dp for dp in bins['Dp'] if dp in day_data.columns]
    count = day_data[bin_columns].values.T

    # Create a meshgrid for the contour plot
    X, Y = np.meshgrid(range(len(time)), bin_columns)

    # Define RGBA values for different shades of gray
    light_gray = (0.9, 0.9, 0.9, 1.0)  # RGBA values for light gray
    medium_gray = (0.7, 0.7, 0.7, 1.0)  # RGBA values for medium gray
    dark_gray = (0.5, 0.5, 0.5, 1.0)  # RGBA values for dark gray

    # Combine the colors into a list for a custom colormap
    colors = ["white", light_gray, medium_gray, dark_gray, "#0C2C84", "#225EA8", "#1D91C0", "#41B6C4",
              "#7FCDBB", "#C7E9B4", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#B10026",
              "red", "black"]
    custom_cmap = ListedColormap(colors)

    # Initialize count_range for colorbar
    count_range = []
    n = 50
    bonus = []

    # Develop the range for contour levels
    while len(count_range) < len(colors):
        count_range = np.linspace(min_count, max_count, 10)
        count_range = np.ceil(count_range / 500) * 500
        count_range = np.concatenate(([1, 2, 3, 4, 5, 10, 25, 50], count_range))

        count_range = np.concatenate((count_range, bonus))

        count_range = np.sort(count_range)


        count_range = list(np.unique(count_range.astype(int)))

        n = n + 50
        bonus.append(n)

    # Create a norm for boundary values in the colormap
    norm = BoundaryNorm(count_range, custom_cmap.N)

    # Create a contour plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    contour = ax1.contourf(X, Y, count, levels=count_range, cmap=custom_cmap, norm=norm)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax1, ticks=count_range, pad=0.1)

    # Set colorbar title above the colorbar
    cbar.ax.text(0.5, 1.05, 'dN/dlogDp', ha='center', va='center', transform=cbar.ax.transAxes, weight='bold')

    # Set labels and title
    ax1.set_ylabel('Diameter Midpoint (μm)', fontsize=14)

    # Set the title with formatted date
    ax1.set_title(f'{title} Aerosol Distributions (MST): {date}', fontsize=14, weight='bold')

    # Set the x-axis ticks and labels at 1-hour intervals from the minimum to maximum time
    min_hr = min(time).hour
    max_hr = max(time).hour

    # Generate evenly spaced ticks
    num_ticks = max_hr - min_hr + 2  # adjust as needed
    tick_positions = np.linspace(0, len(time) - 1, num_ticks, dtype=int)

    # Convert tick positions to time series and round to the nearest hour
    try:
        time_series = pd.to_datetime(time.astype(str).iloc[tick_positions], format='%H:%M:%S.%f')
    except ValueError:
        time_series = pd.to_datetime(time.astype(str).iloc[tick_positions], format='%H:%M:%S')

    rounded_times = time_series.dt.round('h')

    # Convert rounded times to string format
    rounded_times_str = rounded_times.dt.strftime('%H:%M')  # change the format here

    # Set x-axis ticks and labels
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(rounded_times_str, rotation=45, ha='right')

    # Set y-axis to log scale
    ax1.set_yscale('log')

    # Specify sensible tick labels
    y_ticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks, fontsize=8)
    

    # Save plot
    year = pd.to_datetime(day_data['Time_MST']).dt.year.iloc[0]
    year_directory = os.path.join(path, str(year))
    os.makedirs(year_directory, exist_ok=True)
    filepath = os.path.join(year_directory, f"{date}_{title}.png")
    plt.savefig(filepath, dpi=400, bbox_inches='tight')
    plt.close()




###############################################################################
#%%# process aerosol # concentrations for GRIMM
###############################################################################




# define directories to instrument directory
# # each directory must contain a "collection" folder and "plots" folder
grimm_dir = '/uufs/chpc.utah.edu/common/home/hallar-group2/data/site/wbb'
save_dir = '/uufs/chpc.utah.edu/common/home/hallar-group2/plots/site/wbb/GRIMM/'

project_title = 'GRIMM_(WBB)'


# # specify inlet efficiencies for GRIMM @ WBB
grimm_efficiencies = [0.9984480257, 0.9981661124, 0.9979658865, 0.9974219954, 0.9968158814,
                0.9961473774, 0.9954165081, 0.9941177784, 0.992851444, 0.991873087,
                0.9897330241, 0.9847273933, 0.9754386725, 0.9640684329, 0.9457855443,
                0.9181626064, 0.8856016482, 0.8485313359, 0.8074187886, 0.7150943161,
                0.5595642092, 0.451205686, 0.3445254765, 0.1974961192, 0.0234574173,
                0, 0, 0, 0, 0, 0, 0]

# read grimm csv data for current day
grimm_utc, grimm = read_grimm(grimm_dir, grimm_efficiencies, save_dir, project_title, [2024, 2025])

# bin grimm data
# grimm_bins = np.array([0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.58, 0.65,
#               0.70, 0.80, 1.00, 1.30, 1.60, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00,
#               6.50, 7.50, 8.50, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 32.0])
# grimm = bin(grimm, grimm_bins)



# grimm_bins = pd.DataFrame({
#     'Bin Number': [f'bin{i}' for i in range(31)],
#     'Size (µm)': [0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.58, 0.65,
#                   0.70, 0.80, 1.00, 1.30, 1.60, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00,
#                   6.50, 7.50, 8.50, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 32.0]
# })

grimm_bins = pd.DataFrame({
    'Bin Number': list(range(2, 32)) + ['XX'],
    'Size (µm)': [0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.58, 0.65,
                  0.70, 0.80, 1.00, 1.30, 1.60, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00,
                  6.50, 7.50, 8.50, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 32.0]
})

grimm, grimm_bins = bin(grimm, grimm_bins)


# normalize bins by particle size and bin size
#grimm = sizing(grimm, grimm_bins)

# combine all csv files into one
grimm = combine(grimm_utc, grimm)

daily_grimm = split(grimm)


###############################################################################
#%%# plot aerosol # concentrations for GRIMM & QUANT
###############################################################################

# process and plot dust concentrations
#plot_contour(grimm, 0, 60000, save_dir, project_title)
process_daily_data(daily_grimm, grimm_bins, 0, 60000, save_dir, project_title)











