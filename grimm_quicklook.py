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



# define directories to instrument directory
# # each directory must contain a "collection" folder and "plots" folder
grimm_dir = '/uufs/chpc.utah.edu/common/home/hallar-group2/data'
save_dir = '/uufs/chpc.utah.edu/common/home/hallar-group2/plots/site/wbb/QUANT'


###############################################################################
#%%# define utility functions
###############################################################################


# read quant csv file(s)
def read_quant(data_dir, file, eff, years):
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
                df = pd.read_csv(os.path.join(data_dir, subdirectory, csv_file))
                dfs.append(df)
    
    # combine data from all csv files into one dataframe
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # return all formatted and combined data
    return format_quant(combined_df, eff)



# format the raw quant data
def format_quant(df, eff):
    
    # check if efficiency was specified for quant
    if eff == []:
         
        # Drop unnecessary columns
        df.drop(columns=['Unnamed: 0', 'timestamp_local'], inplace=True)

        # Rename 'timestamp' to 'Time_UTC'
        df.rename(columns={'timestamp': 'Time_UTC'}, inplace=True)

        # Convert 'Time_UTC' to datetime and create 'Time_MST'
        df['Time_UTC'] = pd.to_datetime(df['Time_UTC'])
        df['Time_MST'] = df['Time_UTC'].dt.tz_localize('UTC').dt.tz_convert('MST').dt.tz_localize(None)

        # Define OPC columns and reorder
        opc_columns = [
            'Time_UTC', 'Time_MST',  # Time-related columns
            'opc_bin0', 'opc_bin1', 'opc_bin2', 'opc_bin3', 'opc_bin4', 'opc_bin5', 'opc_bin6',
            'opc_bin7', 'opc_bin8', 'opc_bin9', 'opc_bin10', 'opc_bin11', 'opc_bin12', 'opc_bin13',
            'opc_bin14', 'opc_bin15', 'opc_bin16', 'opc_bin17', 'opc_bin18', 'opc_bin19', 'opc_bin20',
            'opc_bin21', 'opc_bin22', 'opc_bin23', 
            'opc_pm1', 'opc_pm25', 'opc_pm10'
        ]

        # Define NEPH columns and reorder
        neph_columns = [
            'Time_UTC', 'Time_MST',  # Time-related columns
            'neph_bin0', 'neph_bin1', 'neph_bin2', 'neph_bin3', 'neph_bin4', 'neph_bin5',
            'neph_pm1', 'neph_pm25', 'neph_pm10'
        ]

        # Extract OPC and NEPH DataFrames
        opc_df = df[opc_columns].copy()
        neph_df = df[neph_columns].copy()
        
        # Sort both DataFrames by 'Time_MST' in chronological order
        opc_df.sort_values(by='Time_MST', inplace=True)
        neph_df.sort_values(by='Time_MST', inplace=True)
        
      
        # return formatted quant data
        return opc_df, neph_df
    
    else:
        print("Need to define efficiency for quant")
        return


# bin aerosol data
def bin(df, bins, instrument):
    # Calculate the mean diameter for each bin
    bins['Dp'] = (bins['Size (µm)'] + bins['Size (µm)'].shift(-1)) / 2
    bins['dlogDp'] = np.log10(bins['Size (µm)']).shift(-1) - np.log10(bins['Size (µm)'])
    
    # Create a mapping from bin names to mean diameters and dlogDp values
    bin_mapping = {f'{instrument}_{bin_num}': mean_diameter for bin_num, mean_diameter in zip(bins['Bin Number'], bins['Dp'])}
    bin_dlogDp_mapping = {f'{instrument}_{bin_num}': dlogDp for bin_num, dlogDp in zip(bins['Bin Number'], bins['dlogDp'])}
    
    # Divide the data in each 'opc_binX' column by the corresponding 'dlogDp' value
    for column in df.columns:
        if column.startswith(f'{instrument}_bin'):
            bin_number = column.split('_')[1]  # Extract bin number (e.g., 'opc_bin0' -> '0')
            
            if f'{instrument}_{bin_number}' in bin_dlogDp_mapping:
                # Divide the data in the 'opc_bin' column by the corresponding 'dlogDp' value
                df[column] = df[column] / bin_dlogDp_mapping[f'{instrument}_{bin_number}']
    
    # Rename the columns in the data DataFrame based on the bin mapping
    df = df.rename(columns=bin_mapping)
    bins = bins.iloc[:-1].reset_index(drop=True)
    

    return df, bins


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
def process_daily_data(daily_quant, bins, min_count, max_count, path, title):
    # Iterate through each entry in daily_grimm
    for date, day_data in daily_quant.items():     
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
    
    
    # Identify PM10 column by searching for 'pm10' in the column name
    day_data.set_index('Time_MST', inplace=True)
    pm10_column = next(col for col in day_data.columns if '_pm10' in str(col))
    pm10_data = day_data[pm10_column]
    day_data.reset_index(inplace=True)


    # Plot PM10 on the second y-axis
    # ax2 = ax1.twinx()
    # ax2.plot(range(len(time)), pm10_data, color='tab:red', label='PM10 1min Avg', linestyle='-', marker='o', markersize=4, alpha=1)
    # ax2.set_ylabel('PM10 (µg/m³)', color='tab:red')

    
    # ax2.tick_params(axis='y', labelcolor='tab:red')
    # ax2.set_ylim(-500, 500)  # Starting halfway up
    # ax2.set_yticks([0, 100, 200, 300, 400, 500])  # PM10 ticks

    # Save plot
    year = pd.to_datetime(day_data['Time_MST']).dt.year.iloc[0]
    year_directory = os.path.join(path, str(year))
    os.makedirs(year_directory, exist_ok=True)
    filepath = os.path.join(year_directory, f"{date}_{title}.png")
    plt.savefig(filepath, dpi=400, bbox_inches='tight')
    plt.close()
    



###############################################################################
#%%# process aerosol # concentrations for QUANT
###############################################################################

# specify inlet efficiencies for QUANT at WBB
quant_efficiencies = []

# read all csv data in filepath
# # can also specify specific, individual files in 2nd input (e.g, "2023-11-24.csv")
quant_opc, quant_neph = read_quant(quant_dir, "", quant_efficiencies, [2023])

# bin quant data
opc_bins = pd.DataFrame({
    'Bin Number': ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 
                   'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15', 'bin16', 'bin17', 
                   'bin18', 'bin19', 'bin20', 'bin21', 'bin22', 'bin23', 'binXX'],
    'Size (µm)': [0.35, 0.46, 0.66, 1.00, 1.30, 1.70, 2.30, 3.00, 4.00, 
                  5.20, 6.50, 8.00, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 
                  22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0]
})
quant_opc, opc_bins = bin(quant_opc, opc_bins, 'opc')


neph_bins = pd.DataFrame({
    'Bin Number': ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'binXX'],
    'Size (µm)': [0.35, 0.46, 0.66, 1.00, 1.30, 1.70, 2.30]
})
quant_neph, neph_bins = bin(quant_neph, neph_bins, 'neph')



# split the combine data set according to day dust was collected
daily_opc = split(quant_opc)

daily_neph = split(quant_neph)




###############################################################################
#%%# plot aerosol # concentrations for QUANT
###############################################################################


# # process and plot dust concentrations
process_daily_data(daily_opc, opc_bins, 0, 10000, save_dir, "QUANT_OPC_WBB")


process_daily_data(daily_neph, neph_bins, 0, 10000, save_dir, "QUANT_NEPH_WBB")





