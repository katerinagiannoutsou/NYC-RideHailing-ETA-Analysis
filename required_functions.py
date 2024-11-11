import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import dataframe_image as dfi  # Required for saving DataFrame as an image

import numpy as np
from scipy import stats
import seaborn as sns
import os
import concurrent.futures
import random
import requests
import osmnx as ox
from itertools import combinations

import networkx as nx
from libpysal.weights import Queen
from libpysal.weights import Rook

import dask.dataframe as dd
import dask.array as da
from dask_ml.linear_model import LinearRegression
from dask.array import concatenate
from dask import delayed, compute

import datashader as ds
import datashader.transfer_functions as tf
from datashader.mpl_ext import dsshow
#from datashader.glyphs import Point

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, rand_score, mutual_info_score
from sklearn.metrics import homogeneity_completeness_v_measure, fowlkes_mallows_score, jaccard_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


import statsmodels.api as sm

#this file containts ALL the functions required for the several 'main' scripts

#### ========= 1. Data Cleaning  ========= ####

def optimize_dataframe(df):
    """
    Reduce DataFrame memory footprint by downcasting numerical columns
    and converting objects to category type where applicable.
    This function is adapted for pandas.
    """
    # Identify columns by data type
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    object_cols = df.select_dtypes(include=['object']).columns

    # Downcast numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Convert object columns to category based on a heuristic
    for col in object_cols:
        distinct_count = df[col].nunique()
        total_count = len(df[col])
        if distinct_count / total_count < 0.5:
            df[col] = df[col].astype('category')

    return df


def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # filter the column to remove outliers
    filtered_column = column[(column >= lower_bound) & (column <= upper_bound)]
    return filtered_column

def clean_data(df, taxi_zones, company):
    """
    cleans the provided ride-share company DataFrame by setting up taxi zones, filtering by company,
    and optimizing data types for efficiency.
    """
    # setup taxi zones optimally
    taxi_zones = taxi_zones.drop_duplicates(subset=['LocationID']).set_index('LocationID')
    taxi_zones['borough'] = taxi_zones['borough'].astype('category')
    print("Taxi Zones Set Up Successfully!")

    # ensure IDs in df are of type int for efficient merging
    df['PULocationID'] = df['PULocationID'].astype(int)
    df['DOLocationID'] = df['DOLocationID'].astype(int)
    print(df['PULocationID'].dtype, df['DOLocationID'].dtype)
    print("Data types optimized successfully!")

    # replace license numbers with company names using a mapping
    license_mapping = {
        'HV0002': 'Juno',
        'HV0003': 'Uber',
        'HV0004': 'Via',
        'HV0005': 'Lyft'
    }
    df['hvfhs_license_num'] = df['hvfhs_license_num'].map(license_mapping)
    print("License numbers mapped successfully!")

    # filter dataframe for a specific company
    df = df[df['hvfhs_license_num'] == company]
    print(f"Data filtered for company: {company}")

    # merge operations
    df = df.merge(taxi_zones[['borough']].rename(columns={'borough': 'PUBorough'}),
                  left_on='PULocationID', right_index=True, how='left')
    df = df.merge(taxi_zones[['borough']].rename(columns={'borough': 'DOBorough'}),
                  left_on='DOLocationID', right_index=True, how='left', suffixes=('', '_DO'))
    print("Data merged with pickup and drop-off locations successfully!")

    # filter out unwanted values
    df = df[(df['PUBorough'] != 'EWR') & (df['PUBorough'] != 'Unknown') &
            (df['DOBorough'] != 'EWR') & (df['DOBorough'] != 'Unknown')]
    print("Data filtered successfully!")

    # define the airport location IDs to filter out 132=JFK, 138=LaGuardia
    airport_values = [132, 138]
    count_unwanted = df[(df['DOLocationID'].isin(airport_values) | df['PULocationID'].isin(airport_values))].shape[0]
    print(f"Number of rows with 'DOLocationID' or 'PULocationID' equal to 132 or 138 before operation: {count_unwanted}")

    # filter out rows where 'DOLocationID' or 'PULocationID' is either 132 or 138
    df = df[~(df['DOLocationID'].isin(airport_values) | df['PULocationID'].isin(airport_values))]

    print("Airport locations filtered successfully!")
    # count rows where 'DOLocationID' or 'PULocationID' is either 132 or 138
    count_unwanted = df[(df['DOLocationID'].isin(airport_values) | df['PULocationID'].isin(airport_values))].shape[0]

    print(f"Number of rows with 'DOLocationID' or 'PULocationID' equal to 132 or 138 after operation: {count_unwanted}")

    # drop rows with NaN in specific columns
    df = df.dropna(subset=['PUBorough', 'DOBorough'])
    print("Data operations completed successfully!")

    # date and time conversion
    #request
    df['request_datetime'] = pd.to_datetime(df['request_datetime'], errors='coerce')
    df['request_date'] = df['request_datetime'].dt.date
    #pickup
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df['pickup_date'] = df['pickup_datetime'].dt.date
    #dropoff
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
    df['dropoff_date'] = df['dropoff_datetime'].dt.date
   
    print("Dates extracted successfully!")

    df = df.dropna(subset=['request_datetime', 'pickup_datetime', 'dropoff_datetime'])
    print("Data filtered for date and time successfully!")

    # apply filtering to remove incorrect rows
    df = df[
        (df['dropoff_datetime'] >= df['request_datetime']) &
        (df['dropoff_datetime'] >= df['pickup_datetime']) &
        (df['pickup_datetime'] >= df['request_datetime'])
    ]
    print("Data filtered for date and time consistency successfully!")

    # apply borough abbreviations
    borough_abbreviations = {
        'Manhattan': 'M',
        'Brooklyn': 'BKL',
        'Bronx': 'BNX',
        'Queens': 'Q',
        'Staten Island': 'SI'
    }
    df['PUBorough'] = df['PUBorough'].map(borough_abbreviations)
    df['DOBorough'] = df['DOBorough'].map(borough_abbreviations)
    print("Borough abbreviations added successfully!")

    #outliers removal
    df['trip_miles'] = remove_outliers(df['trip_miles'])
    df['trip_time'] = remove_outliers(df['trip_time'])
    print("Outliers removed successfully!")

    return df

def clean_data_taxi(df, taxi_zones):
    """
    cleans the provided Taxi DataFrame by setting up taxi zones, filtering by company,
    and optimizing data types for efficiency.
    """
    # setup taxi zones optimally
    taxi_zones = taxi_zones.drop_duplicates(subset=['LocationID']).set_index('LocationID')
    taxi_zones['borough'] = taxi_zones['borough'].astype('category')
    print("Taxi Zones Set Up Successfully!")

    # ensure IDs in df are of type int for efficient merging
    df['PULocationID'] = df['PULocationID'].astype(int)
    df['DOLocationID'] = df['DOLocationID'].astype(int)
    print(df['PULocationID'].dtype, df['DOLocationID'].dtype)
    print("Data types optimized successfully!")

    # merge operations
    df = df.merge(taxi_zones[['borough']].rename(columns={'borough': 'PUBorough'}),
                  left_on='PULocationID', right_index=True, how='left')
    df = df.merge(taxi_zones[['borough']].rename(columns={'borough': 'DOBorough'}),
                  left_on='DOLocationID', right_index=True, how='left', suffixes=('', '_DO'))
    print("Data merged with pickup and drop-off locations successfully!")

    # filter out unwanted values
    df = df[(df['PUBorough'] != 'EWR') & (df['PUBorough'] != 'Unknown') &
            (df['DOBorough'] != 'EWR') & (df['DOBorough'] != 'Unknown')]
    print("Data filtered successfully!")

    # define the airport location IDs to filter out 132=JFK, 138=LaGuardia
    airport_values = [132, 138]
    count_unwanted = df[(df['DOLocationID'].isin(airport_values) | df['PULocationID'].isin(airport_values))].shape[0]
    print(f"Number of rows with 'DOLocationID' or 'PULocationID' equal to 132 or 138 before operation: {count_unwanted}")

    # filter out rows where 'DOLocationID' or 'PULocationID' is either 132 or 138
    df = df[~(df['DOLocationID'].isin(airport_values) | df['PULocationID'].isin(airport_values))]

    print("Airport locations filtered successfully!")
    # count rows where 'DOLocationID' or 'PULocationID' is either 132 or 138
    count_unwanted = df[(df['DOLocationID'].isin(airport_values) | df['PULocationID'].isin(airport_values))].shape[0]

    print(f"Number of rows with 'DOLocationID' or 'PULocationID' equal to 132 or 138 after operation: {count_unwanted}")

    # drop rows with NaN in specific columns
    df = df.dropna(subset=['PUBorough', 'DOBorough'])
    print("Data operations completed successfully!")

    # date and time conversion 
    #rename the taxi pickup and dropoff columns to match uber and lyft
    #pickup
    #df.rename(columns={'tpep_pickup_datetime': 'pickup_datetime'}, inplace=True)
    df.rename(columns={'lpep_pickup_datetime': 'pickup_datetime'}, inplace=True)

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df['pickup_date'] = df['pickup_datetime'].dt.date
    #dropoff
    #df.rename(columns={'tpep_dropoff_datetime': 'dropoff_datetime'}, inplace=True)
    df.rename(columns={'lpep_dropoff_datetime': 'dropoff_datetime'}, inplace=True)
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
    df['dropoff_date'] = df['dropoff_datetime'].dt.date
    print("Dates extracted successfully!")

    df = df.dropna(subset=['pickup_datetime', 'dropoff_datetime'])
    print("Data filtered for date and time successfully!")

    # apply filtering to remove incorrect rows
    df = df[(df['dropoff_datetime'] >= df['pickup_datetime'])]
    print("Data filtered for date and time consistency successfully!")

    # apply borough abbreviations
    borough_abbreviations = {
        'Manhattan': 'M',
        'Brooklyn': 'BKL',
        'Bronx': 'BNX',
        'Queens': 'Q',
        'Staten Island': 'SI'
    }
    df['PUBorough'] = df['PUBorough'].map(borough_abbreviations)
    df['DOBorough'] = df['DOBorough'].map(borough_abbreviations)
    print("Borough abbreviations added successfully!")

    #outliers removal
    #df['Trip_distance'] = remove_outliers(df['Trip_distance'])
    #create trip_time column based on pickup and dropoff (like uber and lyft) and then remove outliers
    # calculate the trip time in seconds
    df['trip_time'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()
    df['trip_time'] = remove_outliers(df['trip_time'])
    print("Outliers removed successfully!")

    return df


def calculate_mean_distances(df):
    """
    calculate mean distances for each PUBorough and DOBorough pair.

    inputs:
        df (pandas.DataFrame): DataFrame containing 'PUBorough', 'DOBorough', and 'trip_miles'.

    outputs:
        pandas.DataFrame: DataFrame with the mean trip miles for each borough pair.
    """
    # calculate mean distance and count of trips for each PUBorough and DOBorough pair
    mean_distances = df.groupby(['PUBorough', 'DOBorough']).agg(
        MeanTripMiles=('trip_miles', 'mean'),
        count=('trip_miles', 'count')
    ).reset_index()

    # create a new column for the bar labels
    mean_distances['BoroughPair'] = mean_distances['PUBorough'] + "-" + mean_distances['DOBorough']


    return mean_distances


def prepare_ride_counts(locations):
    """
    aggregates the count of rides for each drop-off location ID.

    inputs:
        df (pandas.DataFrame): DataFrame containing the ride data with 'DOLocationID'.

    outputs:
        pandas.DataFrame: Aggregated DataFrame with columns ['LocationID', 'RideCount'].
    """
    # count the number of rides for each location ID
    ride_counts = locations.value_counts().reset_index(name='RideCount')
    ride_counts.columns = ['LocationID', 'RideCount']


    return ride_counts


def calculate_daily_averages(dropoff_date, PUBorough, DOBorough):
    # combine the necessary columns into a single DataFrame
    df = pd.DataFrame({
        'dropoff_date': dropoff_date,
        'PUBorough': PUBorough,
        'DOBorough': DOBorough
    })
    
    # group by drop-off date and borough pair, then count the number of rides
    daily_counts = df.groupby(['dropoff_date', 'PUBorough', 'DOBorough']).size().reset_index(name='DailyRideCount')
    
    # calculate the average daily rides for each borough pair over the month
    average_daily_rides = daily_counts.groupby(['PUBorough', 'DOBorough']).agg(
        AvgDailyRides=('DailyRideCount', 'mean')
    ).reset_index()

    # create a new column for the bar labels
    average_daily_rides['BoroughPair'] = average_daily_rides['PUBorough'] + "-" + average_daily_rides['DOBorough']
    
    return average_daily_rides

#### ========= 2. Max Flow Simulation  ========= ####


def calculate_requests(df, method='current', time_window_minutes=5):
    # Calculate time windows for requests
    #df['Request Time Window'] = df['request_datetime'].dt.floor(f'{time_window_minutes}min').astype('datetime64[ns]')
    print("Time windows calculated successfully!")

    # Calculate request counts based on Request Time Window and PULocationID
    request_counts_loc =  df.groupby(['Request Time Window', 'PULocationID']).size().reset_index().rename(columns={0: 'requests'})
    print("Request counts calculated successfully!")
    print('The columns in request_counts_loc are:', request_counts_loc.columns)

    # Adjust Request Time Window based on the method
    if method == 'shifted':
        request_counts_loc['Request Time Window'] = request_counts_loc['Request Time Window'] + pd.Timedelta(minutes=time_window_minutes)
        print("Request time window adjusted successfully!")
    elif method == 'extended':
        # Create an additional column for the previous time window
        request_counts_loc['Next Time Window'] = request_counts_loc['Request Time Window'] + pd.Timedelta(minutes=time_window_minutes)
        print("Request time window adjusted successfully!")

    # Merge based on the method
    if method == 'extended':
        # Merging on current and previous time windows
        df = df.merge(request_counts_loc[['Request Time Window', 'PULocationID', 'requests']],
                how='left',
                left_on=['Request Time Window', 'PULocationID'],
                right_on=['Request Time Window', 'PULocationID'])
    
        # Fill NaN values for 'completed_rides'
        df['requests'] = df['requests'].fillna(0)

        df = df.merge(request_counts_loc[['Next Time Window', 'PULocationID', 'requests']],
                how='left',
                left_on=['Request Time Window', 'PULocationID'],
                right_on=['Next Time Window', 'PULocationID'],
                suffixes=('', '_previous'))
        print("Data merged successfully!")
        # Summing requests from current and previous time windows
        df['requests'] = df['requests'] + df['requests_previous']
        print("requests calculated successfully!")
    else:
        df = df.merge(request_counts_loc, how='left', left_on=['Request Time Window', 'PULocationID'], right_on=['Request Time Window', 'PULocationID'])
        print("Data merged successfully!")
        print("requests calculated successfully!")
    
    df['requests'] = df['requests'].fillna(0)  # Replace NaNs with 0
    #df['requests'] = df['requests'].fillna(0).compute()

    #print('NAs produced are ', df['requests'].isna().sum())
    # Return the new DataFrame
    return df


def calculate_dropoffs(df, method, time_window_minutes):
    # Calculate the dropoff time window
    #print("Sample dropoff_datetime values:", df['dropoff_datetime'].head())

    #df['Dropoff Time Window'] = df['dropoff_datetime'].dt.floor(f'{time_window_minutes}min').astype('datetime64[ns]')
    #print("Dropoff Time Windows calculated successfully!")
    print("The columns in the dataframe are:", df.columns)
    print(df['Dropoff Time Window'].head())

    # Group by Dropoff Time Window and DOLocationID to calculate completed rides
    dropoff_counts_loc = df.groupby(['Dropoff Time Window', 'DOLocationID']).size().reset_index().rename(columns={0: 'dropoffs'})
    print("Dropoff counts calculated successfully!")
    print(dropoff_counts_loc.head())

    # Adjust Dropoff Time Window based on the method
    if method == 'shifted':
        dropoff_counts_loc['Dropoff Time Window'] = dropoff_counts_loc['Dropoff Time Window'] + pd.Timedelta(minutes=time_window_minutes)
    elif method == 'extended':
        dropoff_counts_loc['Next Time Window'] = dropoff_counts_loc['Dropoff Time Window'] + pd.Timedelta(minutes=time_window_minutes)

    # Merge the data based on the method
    if method == 'extended':
        df = df.merge(dropoff_counts_loc[['Dropoff Time Window', 'DOLocationID', 'dropoffs']],
                      how='left',
                      left_on=['Request Time Window', 'PULocationID'],
                      right_on=['Dropoff Time Window', 'DOLocationID'])

        df['dropoffs'] = df['dropoffs'].fillna(0)
        
        df = df.merge(dropoff_counts_loc[['Next Time Window', 'DOLocationID', 'dropoffs']],
                      how='left',
                      left_on=['Request Time Window', 'PULocationID'],
                      right_on=['Next Time Window', 'DOLocationID'],
                      suffixes=('', '_previous'))

        df['dropoffs'] = df['dropoffs'] + df['dropoffs_previous'].fillna(0)
        df = df.drop(columns=['dropoffs_previous', 'Next Time Window'])
    else:
        df = df.merge(dropoff_counts_loc, 
                      how='left', 
                      left_on=['Request Time Window', 'PULocationID'], 
                      right_on=['Dropoff Time Window', 'DOLocationID'])

    df['dropoffs'] = df['dropoffs'].fillna(0)  # Replace NaNs with 0
    
    # Debug check
    print(f"Number of NaNs in 'dropoffs': {df['dropoffs'].isna().sum()}")

    # Debug print statements to verify results
    print("After merge, dropoffs column:")
    print(df[['Request Time Window', 'PULocationID', 'dropoffs']].head())

    print("description of dropoffs in df after merge: ")
    print(df['dropoffs'].describe())
    return df    

def eta_create(df, upper_bound, eta_calc):
    # Calculate ETA in minutes
    print("At calculating ETA step, the columns of the df are:", df.columns)
    if eta_calc == 'on_scene':
        print("Calculating revised ETA: on scene datetime")
        df['ETA'] = (df['on_scene_datetime'] - df['request_datetime']).dt.total_seconds() / 60.0
    else:
        print("Calculating original ETA: pickup datetime")
        df['ETA'] = (df['pickup_datetime'] - df['request_datetime']).dt.total_seconds() / 60.0
    print("ETA calculated successfully!")

    # Define bounds for ETA
    lower_bound = 0
    upper_bound = 50
    print(f"NAs in ETA are: {df['ETA'].isna().sum().compute()}")

    # Filter out ETAs not within the desired range and ensure open_drivers > 0
    #filtered_df = df[(df['ETA'] > lower_bound) & (df['ETA'] < upper_bound) & (df['completed_rides'] > 0)]
    filtered_df = df[(df['ETA'] > lower_bound) & (df['ETA'] < upper_bound)]
    print("Data filtered for ETA and open drivers successfully!")
    
    # Return only the 'ETA' column
    return filtered_df['ETA']

# function to classify the time of day
def classify_time_of_day(dt):
    if dt.hour >= 6 and dt.hour < 10:
        return 'morning_rush'
    elif dt.hour >= 10 and dt.hour < 16:
        return 'midday'
    elif dt.hour >= 16 and dt.hour < 20:
        return 'evening_rush'
    elif dt.hour >= 20 and dt.hour < 24:
        return 'night'
    else:
        return 'late_night'
    
#account for missing on_scene_datetime values
def fill_na(df):
    df['on_scene_datetime'] = df['on_scene_datetime'].fillna(df['pickup_datetime'])
    return df
    
def process_and_append_data_sim(clean_data_path, company, year, demand_method, code, segment, eta_upper_bound, time_window_minutes, fraction, eta_calc):
    # initialize an empty Dask DataFrame to accumulate results
    big_dataframe = None
  
    for number in range(1, 13):  # from 01 to 02 $CHANGE BACK TO 13
        file_number = f'{number:02}'  # format number to two digits
        file_path = f'{clean_data_path}cleaned_{company}_tripdata_{year}_{file_number}.parquet'
        print(f"Processing file: {file_number}")

        # load the parquet file
        df = dd.read_parquet(file_path)
        #df = df.compute()
        print(f"File: {file_number}, loaded successfully!")
        print(f'The columns in the dataframe are:', df.columns)
        print(f"The number of rows in the dataframe is: {df.shape[0].compute()}")
        
        #keep omly the columns we need
        df = df[['dropoff_datetime', 'request_datetime', 'pickup_datetime', 'on_scene_datetime','PULocationID', 'DOLocationID', 'PUBorough', 'DOBorough', 'shared_request_flag', 'trip_miles', 'trip_time']]
        
        # filter data for the specified borough code
        df = df[(df['PUBorough'] == code) & (df['DOBorough'] == code)]
        print(f"Data filtered for borough {code} successfully!")

       #sample data if necessary
        if fraction < 1:
            df = df.sample(frac=fraction, random_state=12)
            print("Sampled Data Succesfully")

        #filter data for no shared rides
        df = df[df['shared_request_flag'] == 'N']
        print(f"Data filtered for no shared rides successfully!")

        df = df[['dropoff_datetime', 'request_datetime', 'pickup_datetime', 'on_scene_datetime', 'PULocationID', 'DOLocationID', 'PUBorough', 'DOBorough', 'trip_miles', 'trip_time']]

        df['dropoff_datetime'] = dd.to_datetime(df['dropoff_datetime'])
        df['request_datetime'] = dd.to_datetime(df['request_datetime'])
        df['on_scene_datetime'] = dd.to_datetime(df['on_scene_datetime'])
        df['pickup_datetime'] = dd.to_datetime(df['pickup_datetime'])
        print(f"Columns filtered successfully!")

        print(f"NAs in on_scene_datetime are: {df['on_scene_datetime'].isna().sum().compute()}")
        print(f"NAs in pickup_datetime are: {df['pickup_datetime'].isna().sum().compute()}")
       
        df = df.map_partitions(fill_na)
        
        print(f"NAs in on_scene_datetime after accounted for by pickuptime are: {df['on_scene_datetime'].isna().sum().compute()}")
        print(f"NAs in pickup_datetime are: {df['pickup_datetime'].isna().sum().compute()}")
       
        #create the timw windows
        df['Request Time Window'] = df['request_datetime'].dt.floor(f'{time_window_minutes}min').astype('datetime64[ns]')
        print(df['Request Time Window'].head())
        #df['Dropoff Time Window'] = df['dropoff_datetime'].dt.floor(f'{time_window_minutes}min').astype('datetime64[ns]')
        print(f"Time windows created successfully!")
        
        df = calculate_requests(df, demand_method, time_window_minutes)
        print(f"Data processed for demand successfully!")

        #ETA
        df['ETA'] = eta_create(df, eta_upper_bound, eta_calc)
        print("ETA calculated successfully!")
        
        #print('The mean ETA is:', df['ETA'].mean().compute())
        #zero_eta_count = (df['ETA'] == 0).sum().compute()
        #print(f"Number of rows with ETA equal to 0: {zero_eta_count}")
        # ensure no zero or negative values for log transformation
        df = df[df['ETA'] > 0]
        #df = df[df['completed_rides'] > 0]
        print("Data filtered for log transformation successfully!")

        #calculating open drivers -- earlier attempts
        #df = calculate_open_drivers(df, gamma, demand_adj, Dmax=None)

        # filter out weekends
        df['day_of_week'] = df['Request Time Window'].dt.dayofweek
        df = df[df['day_of_week'] < 5]
        print(f"Dataframe filtered for weekdays.")

        # classify time of day (using map_partitions to maintain Dask efficiency)
        df['Time of Day'] = df['Request Time Window'].map_partitions(lambda x: x.apply(classify_time_of_day))
        print(f"Time of day classified.")

        if segment != 'all':
            #filter for given time (of day) segment
            df = df[df['Time of Day'] == segment]
            print(f"Filtered for time of day: {segment}")

        # append the processed DataFrame to the big dataframe
        if big_dataframe is None:
            big_dataframe = df
        else:
            big_dataframe = dd.concat([big_dataframe, df])
            print(f"Data appended successfully!")

    return big_dataframe


def calculate_adjacency_matrix(taxi_zone_datasets_path, method='queen'):
    """
    Calculates the adjacency matrix for taxi zones using either the Queen or Rook contiguity method.
    Cleans the DataFrame by removing duplicate or conflicting LocationIDs before setting the index.
    
    inputs:
    - taxi_zone_datasets_path (str): Path to the directory containing the taxi zones shapefile.
    - method (str): Adjacency method to use ('queen' or 'rook'). Default is 'queen'.
    
    outputs:
    - pd.DataFrame: A DataFrame representing the adjacency matrix with LocationID as both index and columns.
    """
    # load the taxi zones shapefile
    taxi_zones = gpd.read_file(f"{taxi_zone_datasets_path}/taxi_zones.shp")
    
    # clean the DataFrame by removing duplicates/conflicting LocationIDs    
        # for 'LocationID' 103, keep only the first occurrence and drop the rest
    taxi_zones = taxi_zones.drop_duplicates(subset=['LocationID'], keep='first')  # Automatically handles 103, 56
    
    # remove rows where 'LocationID' is 1, 132, or 138
    taxi_zones = taxi_zones[~taxi_zones['LocationID'].isin([1, 132, 138])]

    # verify the changes
    print(taxi_zones[taxi_zones['LocationID'].isin([1, 132, 138])])  # This should print an empty DataFrame
    
    #remove airport values

    # ensure 'LocationID' is set as index for adjacency calculations
    # note: This is only for adjacency calculation, index will be reset later to keep 'LocationID' as a column
    taxi_zones.set_index('LocationID', inplace=True)
    
    # determine the contiguity method
    if method == 'queen':
        w = Queen.from_dataframe(taxi_zones)
    elif method == 'rook':
        w = Rook.from_dataframe(taxi_zones)
    else:
        raise ValueError("Method must be either 'queen' or 'rook'.")

    # generate the full adjacency matrix
    adjacency_matrix = w.full()[0]  #extract the adjacency matrix
    ids = taxi_zones.index.tolist()  #set the index
    
    # set the diagonal values to 1 to ensure self-adjacency
    np.fill_diagonal(adjacency_matrix, 1)

    # convert back to a DataFrame to ensure correct structure and indexing
    matrix_df = pd.DataFrame(adjacency_matrix, index=ids, columns=ids)
    
    # reset index to keep 'LocationID' as a column for merging purposes
    matrix_df.reset_index(inplace=True)
    matrix_df.rename(columns={'index': 'LocationID'}, inplace=True)

    return matrix_df

def map_adjacencies_to_data(big_df, adjacency_matrix):
    # correctly create the adjacency dictionary using pandas boolean indexing
    adjacency_dict = {}
    for loc in adjacency_matrix['LocationID']:
        # filter correctly to avoid misalignments
        try:
            # ensure we're working with the row correctly by filtering with .iloc
            adjacency_row = adjacency_matrix.loc[adjacency_matrix['LocationID'] == loc].iloc[0]  # get the full row as a Series
            adjacent_columns = adjacency_row[adjacency_row == 1].index.tolist()  # filter to get indices with value 1
            adjacency_dict[loc] = [int(x) for x in adjacent_columns if x != 'LocationID']  # ensure correct types and avoid errors
        except Exception as e:
            print(f"Error occurred at LocationID {loc}: {e}")
    
    big_df['adjacent_locations'] = big_df['PULocationID'].map(adjacency_dict)

    return big_df

def prepare_df_for_max_flow_simulation(big_df, time_window_minutes, time_range, zones, supply_method):
    if isinstance(big_df, dd.DataFrame):
        big_df = big_df.compute()
        print('big_df computed')
    else:
        print('big_df is already a pandas DataFrame')

    # create a DataFrame with all possible combinations of time windows and zones
    full_time_df = pd.DataFrame([(zone, time) for zone in zones for time in time_range],
                                columns=['PULocationID', 'Request Time Window'])
    print(f"Rows in full_time_df: {len(full_time_df)}")
    #full_time_df['PU_Time_Index'] = full_time_df['PULocationID'].astype(str) + '_' + full_time_df['Request Time Window'].astype(str)

    # remove rows where 'PULocationID' or 'DOLocationID' equals 1 (EWR)
    big_df = big_df[~big_df['PULocationID'].isin([1]) & ~big_df['DOLocationID'].isin([1])]

    # create Dropoff Time Window
    big_df['Dropoff Time Window'] = big_df['dropoff_datetime'].dt.floor(f'{time_window_minutes}min')
    print('Dropoff Time Window added to big_df')
    print(big_df['Dropoff Time Window'].head())

    # merge the full time range DataFrame with big_df on 'PULocationID' and 'Request Time Window'
    #merged_df = full_time_df.merge(big_df, on=['PU_Time_Index', 'PULocationID', 'Request Time Window'], how='left')

    merged_df = full_time_df.merge(big_df, on=['PULocationID', 'Request Time Window'], how='left')

    print('Merge completed between full_time_df and big_df')

    # fill missing 'requests' with zeroes
    merged_df['requests'] = merged_df['requests'].fillna(0)
    print(merged_df[(merged_df['Request Time Window'] == '2021-01-01 00:10:00') & (merged_df['PULocationID'] == 4)]['requests'])
    print(merged_df[(merged_df['Request Time Window'] == '2021-01-01 01:10:00') & (merged_df['PULocationID'] == 113)]['requests'])

    # calculate dropoffs (implement the correct logic for dropoffs calculation)
    merged_df = calculate_dropoffs(merged_df, supply_method, time_window_minutes)

    # fill missing dropoff values with zero
    merged_df['dropoffs'] = merged_df['dropoffs'].fillna(0)
    print(f"Number of NaNs in 'dropoffs': {merged_df['dropoffs'].isna().sum()}")
    print(merged_df[(merged_df['Request Time Window'] == '2021-01-01 01:10:00') & (merged_df['PULocationID'] == 113)]['dropoffs'])

    unique_puloc_df = big_df.drop_duplicates(subset=['PULocationID'])

    # merge the unique 'PULocationID' DataFrame with 'adjacent_locations' data
    adjacent_locs_df = unique_puloc_df[['PULocationID', 'adjacent_locations']]
    print('Adjacency information extracted and duplicates removed based on PULocationID')

    # aggregate 'requests' and 'dropoffs' based on 'PULocationID' and 'Request Time Window'
    aggregated_df = merged_df.groupby(['PULocationID', 'Request Time Window']).agg({
        'requests': 'first',            # using first because values are identical
        'dropoffs': 'first'             # using first because values are identical
    }).reset_index()
    print('Requests and dropoffs aggregated')

    # merge aggregated 'adjacent_locations' with the main aggregated DataFrame
    aggregated_df = aggregated_df.merge(adjacent_locs_df, on='PULocationID', how='left')
    print('Merged adjacency information into aggregated_df')

    print(aggregated_df[(aggregated_df['Request Time Window'] == '2021-01-01 00:10:00') & (aggregated_df['PULocationID'] == 4)]['requests'])
    print(aggregated_df[(aggregated_df['Request Time Window'] == '2021-01-01 01:10:00') & (aggregated_df['PULocationID'] == 113)]['requests'])

    # sort the DataFrame by 'Request Time Window' and 'PULocationID'
    aggregated_df.sort_values(['Request Time Window', 'PULocationID'], inplace=True)
    print('sorted')

    print(aggregated_df[(aggregated_df['Request Time Window'] == '2021-01-01 00:10:00') & (aggregated_df['PULocationID'] == 4)]['requests'])
    print(aggregated_df[(aggregated_df['Request Time Window'] == '2021-01-01 01:10:00') & (aggregated_df['PULocationID'] == 113)]['requests'])

    # initialize the simulation variables (initial drivers for all zones)
    #initial_dr = {zone: {initial_driver_count} for zone in zones}

    # set MultiIndex for easy access
    aggregated_df.set_index(['PULocationID', 'Request Time Window'], inplace=True)
    print('Set MultiIndex on aggregated_df')

    # add columns for the simulation
    aggregated_df['driver_avail_start'] = 0
    aggregated_df['driver_avail_end'] = 0
    aggregated_df['unmatched'] = 0
    aggregated_df['driver_abandon'] = 0
    aggregated_df['added_drivers'] = 0
    aggregated_df['driver_avail_avge'] = 0
    aggregated_df['f_d'] = 0
    aggregated_df['f_s'] = 0

    return aggregated_df


##pre-built the graph
def initialize_graph(aggregated_df):
    G = nx.DiGraph()
    source = 'source'
    sink = 'sink'
    G.add_node(source)
    G.add_node(sink)

    # Build graph structure without capacities
    for location in aggregated_df.index.get_level_values('PULocationID').unique():
        x_node = f'X{location}'
        y_node = f'Y{location}'
        G.add_node(x_node)
        G.add_node(y_node)
        
        # Add edges without setting capacities
        G.add_edge(source, x_node, capacity=0)  # Placeholder capacity
        G.add_edge(y_node, sink, capacity=0)   # Placeholder capacity

        # Add edges for adjacent locations with infinite capacity
        for adj_loc in aggregated_df.loc[(location, slice(None)), 'adjacent_locations'].iloc[0]:
            adj_y_node = f'Y{adj_loc}'
            if adj_y_node in G:
                G.add_edge(x_node, adj_y_node, capacity=float('inf'))  # Static capacity

    return G


def compute_max_flow(current_time, df, G):
    # Define the source and sink explicitly
    source = 'source'
    sink = 'sink'
    
    # Update capacities based on the current time window
    idx_current_time = df.index.get_level_values('Request Time Window') == current_time

    for location in df[idx_current_time].index.get_level_values('PULocationID'):
        x_node = f'X{location}'
        y_node = f'Y{location}'

        try:
            start_avail = df.loc[(location, current_time), 'driver_avail_start']
        except KeyError:
            print(f"KeyError: Unable to find ({location}, {current_time}) in index. Skipping this location.")
            continue

        # Update the capacities from source to X node
        G[source][x_node]['capacity'] = start_avail

        # Update the capacities from Y node to sink
        request = df.loc[(location, current_time), 'requests']
        G[y_node][sink]['capacity'] = request

    # Compute the maximum flow
    flow_value, flow_dict = nx.maximum_flow(G, 'source', 'sink', flow_func=nx.algorithms.flow.edmonds_karp)
    print(f"Max flow value at time {current_time}: {flow_value}.")
    
    # Extract flow values for drivers matched (f_s) and requests fulfilled (f_d)
    f_s = [
        flow_dict['source'].get(f'X{location}', 0) for location in df[idx_current_time].index.get_level_values('PULocationID')
    ]
    f_d = [
        flow_dict.get(f'Y{location}', {}).get('sink', 0) for location in df[idx_current_time].index.get_level_values('PULocationID')
    ]

    return f_s, f_d


def run_max_flow_simulation(big_df, aggregated_df, initial_dr, time_range, p, results_path, sim_num):
    """
    Run the simulation for a given simulation ID.

    inputs:
        simulation_id (int): Unique identifier for the simulation run.
        big_df (pd.DataFrame): The original big DataFrame before simulation.
        aggregated_df (pd.DataFrame): The prepared aggregated DataFrame.
        initial_dr (dict): Initial driver count by zone.
        time_range (pd.DatetimeIndex): The time range for the simulation.
        zones (list): List of zones (PULocationIDs).
        p (float): Probability of drivers leaving.
        results_path (str): Path to save the results.

    """
    # ensure the results directory exists
    os.makedirs(results_path, exist_ok=True)
    # copy the DataFrame to avoid modifying the original
    df = aggregated_df
    initial_dr = 0
    print(df.index.names)
    G = initialize_graph(df)
    print("Graph initialized")

    for t in range(len(time_range)):
        current_time = time_range[t]
        print(f"\nCurrent Time Window: {current_time}")

        #start checks
        sliced_df = df.loc[df.index.get_level_values('Request Time Window') == current_time]
        # check for NaNs in critical columns after slicing or updates
        nan_check = df.loc[df.index.get_level_values('Request Time Window') == current_time, 'driver_avail_start'].isna().sum()
        print(f"NaNs in 'driver_avail_start' for {current_time}: {nan_check}")
        #end checks

        # check the unique values of the sliced index
        print(sliced_df.index.get_level_values('Request Time Window').unique())
        if t == 0:
            # for the first time window, initialize driver_avail_start with the initial driver count
            df.loc[df.index.get_level_values('Request Time Window') == current_time, 'driver_avail_start'] = initial_dr
            print(f"Initialization for Time Window: {current_time}")
        else:
            # for subsequent time windows, update driver_avail_start from the previous end availability
            previous_time = time_range[t - 1]
            print(f"Previous Time Window: {previous_time}")

            # extract driver_avail_end from the previous time window and ensure alignment
            previous_end_avail = df.loc[
                df.index.get_level_values('Request Time Window') == previous_time, 'driver_avail_end'
            ].fillna(0)  # Fill NaNs with zero to prevent propagation of NaNs

            # update current driver_avail_start using previous driver_avail_end
            df.loc[
                df.index.get_level_values('Request Time Window') == current_time, 'driver_avail_start'
            ] = previous_end_avail.values
            print(f"Updated driver_avail_start for Time Window: {current_time}")

        # debugging: Verify the updated 'driver_avail_start' before proceeding
        #driver_avail_start_values = df.loc[
         #   df.index.get_level_values('Request Time Window') == current_time, 'driver_avail_start'
        #].tolist()
        #print("driver_avail_start, current:", driver_avail_start_values)

        # call the max flow function
        f_s, f_d = compute_max_flow(current_time, df, G)

        # update the DataFrame with the results from the max flow function
        idx_current_time = df.index.get_level_values('Request Time Window') == current_time
        df.loc[idx_current_time, 'f_s'] = f_s
        df.loc[idx_current_time, 'f_d'] = f_d

        # calculate added drivers (a_{j,t}) and unmatched drivers (u_{j,t})
        df.loc[idx_current_time, 'added_drivers'] = df.loc[idx_current_time, 'requests'] - df.loc[idx_current_time, 'f_d']
        df.loc[idx_current_time, 'unmatched'] = df.loc[idx_current_time, 'driver_avail_start'] - df.loc[idx_current_time, 'f_s']

        # abandoning drivers (b_{j,t}) based on binomial distribution
        df.loc[idx_current_time, 'driver_abandon'] = np.random.binomial(df.loc[idx_current_time, 'unmatched'], p)

        # update the end availability (s^{end}_{j,t})
        df.loc[idx_current_time, 'driver_avail_end'] = (
            df.loc[idx_current_time, 'driver_avail_start'] 
            - df.loc[idx_current_time, 'f_s'] 
            - df.loc[idx_current_time, 'driver_abandon'] 
            + df.loc[idx_current_time, 'dropoffs']
        )

        #debugging: Confirm updates at the end of the current iteration
        #updated_driver_avail_end = df.loc[idx_current_time, 'driver_avail_end'].tolist()
        #print(f"Updated driver_avail_end for {current_time}:", updated_driver_avail_end)
        print("-" * 50) 

    df['driver_avail_avge'] = (df['driver_avail_start'] + df['driver_avail_end']) / 2

    print("PU_Time_Index in big_df columns:", 'PU_Time_Index' in big_df.columns)
    print("PU_Time_Index in df columns:", 'PU_Time_Index' in df.columns)
    print(f"Index levels in df before reset: {df.index.names}")

    print("The columns in df are:", df.columns)
    # reset the index of df to turn 'PULocationID' and 'Request Time Window' back into columns
    df = df.reset_index() ## this is the df that will be saved

    # merge the results back into big_df
    merged_df = big_df.merge(df, on=['PULocationID', 'Request Time Window'], how='left')

    # keep the correct 'requests' and 'adjacent_locations' columns from big_df (_x columns)
    merged_df['requests'] = merged_df['requests_x']
    merged_df['adjacent_locations'] = merged_df['adjacent_locations_x']

    # drop the now unnecessary '_x' and '_y' columns
    merged_df.drop(columns=['requests_x', 'requests_y', 'adjacent_locations_x', 'adjacent_locations_y'], inplace=True)
    print("The columns in merged_df are:", merged_df.columns)

    #check
    print(merged_df[(merged_df['Request Time Window'] == '2021-01-01 00:10:00') & (merged_df['PULocationID'] == 4)]['requests'])
    print(merged_df[(merged_df['Request Time Window'] == '2021-01-01 01:10:00') & (merged_df['PULocationID'] == 113)]['requests'])
    print(merged_df[(merged_df['Request Time Window'] == '2021-01-01 00:10:00') & (merged_df['PULocationID'] == 4)]['adjacent_locations'])
    print(merged_df[(merged_df['Request Time Window'] == '2021-01-01 01:10:00') & (merged_df['PULocationID'] == 113)]['adjacent_locations'])
    
    # generate file paths
    aggregate_output_filename = os.path.join(results_path, f'aggregated_simulation_results_{sim_num}.parquet')
    dataset_output_filename = os.path.join(results_path, f'dataset_with_results_{sim_num}.parquet')

    # save the results
    df.to_parquet(aggregate_output_filename)
    merged_df.to_parquet(dataset_output_filename)
    print(f'Simulation completed and saved.')


def plot_eta_vs_average_num_avail(plots_general_path, df, p, year):
    plots_path = f'{plots_general_path}/eta_vs_drivers'
    os.makedirs(plots_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
    
     # define the start time and the end time for the first three weeks
    t_min = pd.Timestamp(f"{year}-01-01 00:00:00")
    #threshold = t_min + pd.Timedelta(days=25)
    threshold = t_min + pd.Timedelta(days=25)
    #threshold = t_min + pd.Timedelta(weeks=3)

    #df.rename(columns={'Request Time Window_x': 'Request Time Window', 'PULocationID_x': 'PULocationID'}, inplace=True)

    # Filter out the observations that fall within the first three weeks to avoid transient states
    df = df[df['Request Time Window'] > threshold]
    
    borough_codes = ['M', 'BNX', 'BKL', 'Q', 'SI']

    segments = ['morning_rush', 'midday', 'evening_rush', 'night', 'late_night']
    for borough in borough_codes:
        for segment in segments:
            # filter the DataFrame for the specific borough and segment
            filtered_df = df[(df['PUBorough'] == borough) & (df['Time of Day'] == segment)]
            
            # check if the filtered DataFrame is not empty
            if not filtered_df.empty:
                # create the scatter plot
                plt.figure(figsize=(10, 6))
            # calculating and plotting the mean ETA for each avge_number_avail value
                mean_ETA = filtered_df.groupby('driver_avail_avge')['ETA'].mean().reset_index()
                plt.scatter(mean_ETA['driver_avail_avge'], mean_ETA['ETA'], alpha=0.6)

                plt.title(f'Mean ETA vs Average # Available for {borough} - {segment}, leave probability: {p}')
                plt.xlabel('Average Number Available')
                plt.ylabel('ETA')
                plt.legend()

                # save the plot
                plot_filename = f"{plots_path}/mean_eta_vs_average_driver_avail_{borough}_{segment}.png"
                plt.savefig(plot_filename, bbox_inches='tight')
                plt.close()
            else:
                print(f'No data for {borough} - {segment}')


def plot_sim_drivers_overtime(plots_general_path, aggegate_results, selected_zones, p, year):
    #aggegate_results['PULocationID'] = aggegate_results['PU_Time_Index'].apply(lambda x: int(x.split('_')[0]))
    #aggegate_results['Request Time Window'] = pd.to_datetime(aggegate_results['PU_Time_Index'].apply(lambda x: x.split('_')[1]))

    aggegate_results['Time of Day'] = aggegate_results['Request Time Window'].apply(classify_time_of_day)
    print(type(selected_zones), selected_zones)  # Debugging line
    
    # define the start time and the end time for the first three weeks
    t_min = pd.Timestamp(f"{year}-01-01 00:00:00")
    threshold = t_min + pd.Timedelta(days=25)
    #threshold = t_min + pd.Timedelta(weeks=3)

    # filter out the observations that fall within the first three weeks to avoid transient states
    aggegate_results = aggegate_results[
        aggegate_results['Request Time Window'] > threshold
    ]
    
    segments = ['morning_rush', 'midday', 'evening_rush', 'night', 'late_night']
    columns_to_plot = ['driver_avail_start', 'unmatched', 
                   'driver_abandon', 'added_drivers', 'requests']
    
    plots_path = f'{plots_general_path}/drivers_overtime'
    os.makedirs(plots_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
    
   # selected_zones = [113, 114, 7, 12, 18, 37, 45, 66, 82, 90, 161, 162, 163, 164, 187, 195, 217, 221, 238, 261]

    for zone in selected_zones:
        # filter the DataFrame for the specific borough and segment
        filtered_df = aggegate_results[(aggegate_results['PULocationID'] == zone)]
        
        # check if the filtered DataFrame is not empty
        if not filtered_df.empty:
            plt.figure(figsize=(12, 8))
            
            # plot each column with a different color
            for column in columns_to_plot:
                plt.plot(filtered_df['Request Time Window'], filtered_df[column], label=column)
            
            # set plot title and labels
            plt.title(f'Driver Metrics over Time for Zone {zone},  leave probability: {p}')
            plt.xlabel('Request Time Window')
            plt.ylabel('Number of Drivers')
            
            # show legend
            plt.legend(loc='best')
            
                # save the plot
            plt.grid(True)
            plot_filename = f"{plots_path}/drivers_overtime_{zone}.png"
            plt.savefig(plot_filename, bbox_inches='tight')
            plt.close()

        else:
            print(f'No data for Zone {zone}')


def filter_drivers_from_time_window_samples(df):
    # calculate the total number of unique 'Request Time Window' values
    total_unique_time_windows = df['Request Time Window'].nunique()
    print(f"Total unique 'Request Time Window' values: {total_unique_time_windows}")
    
    # determine the threshold as 1% of the total unique 'Request Time Window' values
    threshold = max(1, int(total_unique_time_windows * 0.005))  # Ensure at least 1 if the percentage is less than 1
    print(f"Threshold for filtering: {threshold}")
    
    # count unique 'Request Time Window' values for each 'driver_avail_avge'
    counts = df.groupby('driver_avail_avge')['Request Time Window'].nunique()
    print("Counts of unique 'Request Time Window' per 'driver_avail_avge':")
    print(counts)
    print(f"Number of unique 'driver_avail_avge': {counts.shape[0]}")
    
    # filter to get 'driver_avail_avge' values with at least the threshold number of unique 'Request Time Window' values
    valid_driver_avail = counts[counts >= threshold].index
    print("Valid 'driver_avail_avge' values meeting the threshold:")
    print(valid_driver_avail)
    
    # filter the original DataFrame to keep only rows with valid 'driver_avail_avge' values
    filtered_df = df[df['driver_avail_avge'].isin(valid_driver_avail)]
    return filtered_df


def aggregate_mean_eta_per_location(results_path, location_id, time_segment, num_simulations, output_path, year, x_column):
    # ensure x_column is one of the expected values
    if x_column not in ['driver_avail_start', 'driver_avail_end', 'driver_avail_avge']:
        raise ValueError("x_column must be one of 'driver_avail_start', 'driver_avail_end', or 'driver_avail_avge'")
    
    # define the start time and the threshold for filtering (first 25 days)
    t_min = pd.Timestamp(f"{year}-01-01 00:00:00")
    threshold = t_min + pd.Timedelta(days=25)
    
    # initialize cumulative sum DataFrame for mean ETA calculations
    cumulative_mean_eta_df = None
    
    for simulation_id in range(1, num_simulations + 1):
        dataset_path = f'{results_path}/dataset_with_results_{simulation_id}.parquet'
        print(f"Reading dataset: {dataset_path}")
        df = dd.read_parquet(dataset_path)

        # filter out the first 25 days
        df = df[df['Request Time Window'] > threshold]

        # filter for the given location_id
        df_location = df[(df['PULocationID'] == location_id) & (df['Time of Day'] == time_segment)]
        print("Data filtered for location and time of day successfully!")
        
        df_location = df_location.compute()  # Convert to pandas DataFrame for further processing
        
        # depending on the value of x_column, set driver_avail_avge
        if x_column == 'driver_avail_start':
            df_location['driver_avail_avge'] = df_location['driver_avail_start']
        elif x_column == 'driver_avail_end':
            df_location['driver_avail_avge'] = df_location['driver_avail_end']
        # if x_column is 'driver_avail_avge', no change is needed
        
        # further filtering with external function
        df_location = filter_drivers_from_time_window_samples(df_location)

        # group by driver_avail_avge and calculate mean ETA
        mean_ETA = df_location.groupby('driver_avail_avge')['ETA'].mean().reset_index().rename(columns={'ETA': 'mean_ETA'})

        # checking for zero or negative values before applying the logarithm
        print("Checking for zero or negative values before taking the logarithm...")
        print(f"Number of zero or negative values in mean_ETA: {(mean_ETA['mean_ETA'] <= 0).sum()}")
        print(f"Number of zero or negative values in driver_avail_avge: {(mean_ETA['driver_avail_avge'] <= 0).sum()}")

        mean_ETA = mean_ETA[(mean_ETA['mean_ETA'] > 0) & (mean_ETA['driver_avail_avge'] > 0)].dropna()

        # add current mean_ETA to cumulative sum DataFrame (pandas DataFrame now)
        if cumulative_mean_eta_df is None:
            cumulative_mean_eta_df = mean_ETA
        else:
            cumulative_mean_eta_df = cumulative_mean_eta_df.merge(mean_ETA, on='driver_avail_avge', how='outer', suffixes=('', f'_sim{simulation_id}'))

    # calculate the average mean_ETA across all simulations
    mean_eta_cols = [col for col in cumulative_mean_eta_df.columns if 'mean_ETA' in col]
    cumulative_mean_eta_df['mean_ETA_avg'] = cumulative_mean_eta_df[mean_eta_cols].mean(axis=1)

    # save the resulting DataFrame for the specific location
    final_df = cumulative_mean_eta_df[['driver_avail_avge', 'mean_ETA_avg']]

    output_file = f'{output_path}/mean_eta_aggregated_location_{location_id}_{time_segment}_{x_column}.parquet'
    final_df.to_parquet(output_file)
    print(f"Aggregated results saved to {output_file}")

#### ========= 3. Regression  ========= ####
#regression post simulation using open driver proxy from simulation, after conditional averaging performed

def load_taxi_zones(taxi_zones_path):
    # load the taxi zones shapefile
    taxi_zones = gpd.read_file(f"{taxi_zones_path}/taxi_zones.shp")
    
    # clean the DataFrame by removing duplicates/conflicting LocationIDs    
    # for 'LocationID' 103, keep only the first occurrence and drop the rest
    taxi_zones = taxi_zones.drop_duplicates(subset=['LocationID'], keep='first')  # Automatically handles 103, 56
    
    # remove rows where 'LocationID' is 1, 132, or 138
    taxi_zones = taxi_zones[~taxi_zones['LocationID'].isin([1, 132, 138])]

    # verify the changes
    print(taxi_zones[taxi_zones['LocationID'].isin([1, 132, 138])])  # This should print an empty DataFrame

    return taxi_zones



def logarithmic_regression(X, y, original_y):
    # ensure X is 2D (reshape directly without flatten)
    if X.ndim != 2:
        X = X.reshape(-1, 1)  # Reshape X to (-1, 1) if not already 2D
    
    # ensure y is 1D
    if y.ndim != 1:
        y = y.flatten()  # Flatten y to ensure it is 1D
    print("Data reshaped successfully!")

    # use sklearn's LinearRegression for the regression
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    print("Regression model fitted successfully!")

    # calculate predictions and compute residuals
    predicted_log_ETA = model.predict(X)
    residuals = y - predicted_log_ETA
    print("Predictions and residuals computed successfully!")

    # estimate E[e^epsilon] using the sample average of e^residuals
    residuals_exp = np.exp(residuals)
    E_e_epsilon = np.mean(residuals_exp)
    print("E[e^epsilon] estimated successfully!")

    # correct the tau estimate
    ln_tau_corrected = np.exp(model.intercept_) * E_e_epsilon
    print("Corrected ln(tau) calculated successfully!")

    # sum of squares of residuals
    SSR = np.sum(residuals ** 2)
    n, p = X.shape

    # R-squared calculation
    SS_total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - SSR / SS_total

    # Root Mean Squared Error (RMSE)
    RMSE = np.sqrt(SSR / n)

    # Residual Standard Error (RSE)
    RSE = np.sqrt(SSR / (n - p - 1))

    # calculate MSE on the original scale
    predicted_mean_ETA = np.exp(predicted_log_ETA)  # Convert back from log to original scale
    MSE_original = np.mean((original_y - predicted_mean_ETA) ** 2)  # MSE on original scale

    # compute standard error and p-values (as before)
    MSE = SSR / (n - p - 1)
    X_with_intercept = np.hstack([np.ones((n, 1)), X])
    XTX_inv = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept))
    standard_errors = np.sqrt(np.diag(XTX_inv) * MSE)

    t_intercept = model.intercept_ / standard_errors[0]
    t_coefficient = model.coef_[0] / standard_errors[1]

    p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), df=n - p - 1))
    p_coefficient = 2 * (1 - stats.t.cdf(np.abs(t_coefficient), df=n - p - 1))

    print(f"ln(tau) = {model.intercept_}")
    print(f"alpha = {model.coef_[0]}")
    print(f"Corrected ln(tau) = {ln_tau_corrected}")
    print(f"Standard error (intercept) = {standard_errors[0]}")
    print(f"Standard error (alpha) = {standard_errors[1]}")
    print(f"p-value (intercept) = {p_intercept}")
    print(f"p-value (alpha) = {p_coefficient}")
    print(f"R-squared = {r_squared}")
    print(f"RMSE = {RMSE}")
    print(f"RSE = {RSE}")
    print(f"MSE (Original scale) = {MSE_original}")

    return {
        'ln(tau)': model.intercept_,
        'alpha': model.coef_[0],
        'Corrected ln(tau)': ln_tau_corrected,
        'SE(intercept)': standard_errors[0],
        'SE(alpha)': standard_errors[1],
        'p-value(intercept)': p_intercept,
        'p-value(alpha)': p_coefficient,
        'R-squared': r_squared,
        'RMSE': RMSE,
        'RSE': RSE,
        'MSE (Original scale)': MSE_original
    }

def perform_refined_conditional_regression_zone_level_post_sim_existing_filtered_df(location_id, eta_subtract, time_segment, x_column, cond_sim_results_path):
    print(f"Location ID: {location_id}")
    
    # initialize the results dictionary with default values
    results = {
        'PULocationID': location_id,
        'driver_upper_bound': None,
        'location_dataset_size': None,
        'ln(tau)': None,
        'alpha': None,
        'Corrected ln(tau)': None,
        'SE(intercept)': None,
        'SE(alpha)': None,
        'p-value(intercept)': None,
        'p-value(alpha)': None,
        'R-squared': None,
        'RMSE': None,
        'RSE': None,
        'MSE (Original scale)': None,
        'error': None
    }

    # path to the dataset
    file_path = f'{cond_sim_results_path}/mean_eta_aggregated_location_{location_id}_{time_segment}_{x_column}.parquet'
    
    # check if the file exists
    if not os.path.exists(file_path):
        results['error'] = 'Averaged dataset for this location not found at time of reg'
        print(f"Dataset for Location ID {location_id} not found.")
        return results

    # if the file exists, try to load it
    try:
        mean_ETA = pd.read_parquet(file_path)
    except Exception as e:
        results['error'] = f'Error loading dataset: {str(e)}'
        print(f"Error loading dataset for Location ID {location_id}: {str(e)}")
        return results

    if mean_ETA.empty:
        print("DataFrame is empty")
    if 'driver_avail_avge' not in mean_ETA.columns:
        print("Column 'driver_avail_avge' does not exist")

    # this dataset is already filtered for location, threshold, and outliers, and at the desired form
    mean_ETA['mean_ETA'] = mean_ETA['mean_ETA_avg']

    # account for ETA -- matching time 
    mean_ETA['adjusted_mean_ETA'] = mean_ETA['mean_ETA'] - eta_subtract
    mean_ETA = mean_ETA[(mean_ETA['adjusted_mean_ETA'] > 0)]

    # take the logarithm of the ADJUSTED mean ETA and completed rides
    mean_ETA['ln_mean_ETA'] = np.log(mean_ETA['adjusted_mean_ETA'])
    mean_ETA['ln_driver_avail_avge'] = np.log(mean_ETA['driver_avail_avge'])
    print("Logarithm of mean ETA, and driver_avail_avge calculated successfully!")

    # check for NaNs or Infs
    print("Checking for NaNs or Infs after taking the logarithm...")
    print(f"NaNs in ln_mean_ETA: {mean_ETA['ln_mean_ETA'].isna().sum()}")
    print(f"Infs in ln_mean_ETA: {np.isinf(mean_ETA['ln_mean_ETA']).sum()}")
    print(f"NaNs in ln_driver_avail_avge: {mean_ETA['ln_driver_avail_avge'].isna().sum()}")
    print(f"Infs in ln_driver_avail_avge: {np.isinf(mean_ETA['ln_driver_avail_avge']).sum()}")

    # filter out NaN values
    mean_ETA = mean_ETA.dropna()
    print("Data filtered for NaN values successfully!")

    # prepare the data for regression
    X = mean_ETA[['ln_driver_avail_avge']].values  # 2D numpy array
    y = mean_ETA['ln_mean_ETA'].values  # 1D numpy array
    original_y = mean_ETA['adjusted_mean_ETA'].values  # 1D numpy array
    print("Data prepared for regression successfully!")

    unique_values = mean_ETA['driver_avail_avge'].unique()

    # if no unique values are found, return an error
    if unique_values.size == 0:
        results['error'] = "No unique values in 'driver_avail_avge', regression not performed."
        print(f"No unique values in 'driver_avail_avge' for Location ID: {location_id}")
        return results
    
    # find the maximum value among the unique values
    driver_max = unique_values.max()

    # update the results dictionary with data size and upper bound
    results['driver_upper_bound'] = driver_max
    results['location_dataset_size'] = mean_ETA.shape[0]

    # check if X or y is empty
    if X.size == 0 or y.size == 0:
        results['error'] = 'Empty data, regression not performed.'
        print(f"Empty data for location ID: {location_id}")
        return results

    # check if all values in X are constant
    if np.all(X == X[0, :], axis=0):
        results['error'] = 'Constant columns detected, regression not performed.'
        print(f"Constant columns detected for location ID: {location_id}")
        return results

    print("Univariate Regression")
    
    # perform the regression
    regression_results = logarithmic_regression(X, y, original_y)
    # update the results dictionary with regression results
    results.update(regression_results)

    return results


def plot_reg_mean_eta_vs_drivers_loc_refined_conditional(df, reg_results, time_segment, location_id, year, plots_path, eta_subtract, avge_filter):
    # define the start time and the end time for the first three weeks
    t_min = pd.Timestamp(f"{year}-01-01 00:00:00")
    threshold = t_min + pd.Timedelta(days=25)

    # filter out the observations that fall within the first three weeks to avoid transient states
    df = df[df['Request Time Window'] > threshold]

    # filter dataframe for the given PULocationID
    df_location = df[df['PULocationID'] == location_id]
    print("Data filtered for location successfully!")

    # filter out drivers with few corresponding time window samples
    df_location = filter_drivers_from_time_window_samples(df_location)

    # group by number of drivers and calculate the mean ETA
    mean_ETA = df_location.groupby('driver_avail_avge')['ETA'].mean().reset_index().rename(columns={'ETA': 'mean_ETA'})

    # check and filter for valid values
    print("Checking for zero or negative values before taking the logarithm...")
    print(f"Number of zero or negative values in mean_ETA: {(mean_ETA['mean_ETA'] <= 0).sum()}")
    print(f"Number of zero or negative values in driver_avail_avge: {(mean_ETA['driver_avail_avge'] <= 0).sum()}")
    mean_ETA = mean_ETA[
        (mean_ETA['mean_ETA'] > 0) & 
        (mean_ETA['driver_avail_avge'] > 0)
    ]
    mean_ETA = mean_ETA.dropna()

    # account for ETA -- matching time 
    mean_ETA['adjusted_mean_ETA'] = mean_ETA['mean_ETA'] - eta_subtract

    # extract regression results for the given location_id
    reg_row = reg_results[reg_results['PULocationID'] == location_id].iloc[0]
    intercept = reg_row['Corrected ln(tau)']
    alpha = reg_row['alpha']
    r_squared = reg_row['R-squared']

    # calculate fitted values using the regression parameters
    #mean_ETA['fitted_mean_ETA'] = np.exp(intercept) * np.exp(alpha * np.log(mean_ETA['driver_avail_avge'])) this is equivalent to the one below:
    mean_ETA['fitted_mean_ETA'] = intercept * mean_ETA['driver_avail_avge'] ** alpha

    # plotting mean ETA vs. driver_avail_avge
    plt.figure(figsize=(10, 6))
    
    # scatter plot for mean_ETA
    plt.scatter(mean_ETA['driver_avail_avge'], mean_ETA['mean_ETA'], alpha=0.3, label='Mean ETA', color='blue')
    
    # scatter plot for adjusted_mean_ETA
    plt.scatter(mean_ETA['driver_avail_avge'], mean_ETA['adjusted_mean_ETA'], alpha=0.3, label='Adjusted Mean ETA', color='orange')
    
    # plot fitted regression line
    plt.plot(mean_ETA['driver_avail_avge'], mean_ETA['fitted_mean_ETA'], label='Fitted Regression Line', color='red', linestyle='--')

    # setting labels and title with R-squared
    plt.xlabel('Average Number of Available Drivers')
    plt.ylabel('Mean ETA')
    plt.title(f'Mean ETA vs Driver Availability - Conditional\nR-squared: {r_squared:.3f}, Alpha: {alpha:.3f}')
    plt.grid(True)

    # adding a legend
    plt.legend()

    # save the plot
    plt.savefig(f"{plots_path}/mean_eta_vs_drivers_{location_id}_{time_segment}_conditional.png")
    plt.close()


#### ========= 4. Density - alpha & Taxi - Uber Comparisons  ========= ####

def plot_taxi_vs_uber_rankings(rankings, plots_path):
    # create the scatter plot without color coding based on boroughs
    plt.figure(figsize=(10, 6))
    
    # get the min and max values for the axes
    min_val = min(rankings[['uber_rank', 'taxi_rank']].min())
    max_val = max(rankings[['uber_rank', 'taxi_rank']].max())

    # plot the points without borough color coding
    plt.scatter(rankings['uber_rank'], rankings['taxi_rank'], color='black', alpha=0.5)

    # add the 45-degree line
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label='45-degree line')

    # add labels and title
    plt.xlabel('Uber Rankings')
    plt.ylabel('Taxi Rankings')
    plt.title('Uber vs Taxi Ride Rankings by Location')
    
    # save the plot to the specified path
    plt.savefig(f"{plots_path}/uber_vs_taxi_ride_rankings_no_color.png")
    plt.close()


def load_and_filter_regression_results(results_directory, year, company, supply_method, demand_method, 
                                       time_window_minutes, adj_method, initial_driver_count, p, 
                                       grouping, eta_subtract, avge_filter, time_segment, x_column, codes, threshold_r_squared=0.3):
    """
    Loads and filters the regression results for multiple boroughs and combines them into one DataFrame.

    outputs:
    pd.DataFrame: Combined and filtered DataFrame of regression results across all boroughs.
    """
    # list of borough codes
    #codes = ['M', 'BNX', 'BKL', 'Q']

    # initialize an empty list to store DataFrames for this eta_subtract value
    dfs = []

    # loop over each borough code
    for code in codes:
        # construct the path for the regression results file based on eta_subtract and other parameters
        reg_results_path = (f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_'
                            f'{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/'
                            f'refined_{grouping}_results_sub_{eta_subtract}_sim_avge_{avge_filter}')
        
        # file path depends on avge_filter condition
        if avge_filter == 'loc_cond_mean':
            reg_filename = f"{reg_results_path}/reg_coefficients_{time_segment}_{code}_{x_column}.csv"
        else:
            reg_filename = f"{reg_results_path}/reg_coefficients_{time_segment}_{code}.csv"
        
        # check if the file exists before trying to load it
        if os.path.exists(reg_filename):
            # load the CSV file into a DataFrame
            df = pd.read_csv(reg_filename)
            # add the borough code to a new column
            df['borough_code'] = code
            # append the DataFrame to the list
            dfs.append(df)
        else:
            print(f"File {reg_filename} not found!")
    
    # combine all DataFrames for this eta_subtract value into one DataFrame
    if dfs:  # check if dfs list is not empty
        reg_results = pd.concat(dfs, ignore_index=True)
        
        # filter rows based on 'driver_upper_bound' and 'R-squared'
        reg_results = reg_results[reg_results['driver_upper_bound'] >= 5]
        reg_results = reg_results[reg_results['R-squared'] >= threshold_r_squared]
        reg_results = reg_results[reg_results['alpha']  < 0]

        return reg_results
    else:
        print("No data loaded.")
        return pd.DataFrame()  # return an empty DataFrame if no files were loaded


def calculate_and_merge_ride_volume(dataset_path, updated_reg_results, time_segment):
    # load the original dataset
    original_df = pd.read_parquet(dataset_path)
    
    # filter the dataset to only include rows matching the given time segment
    filtered_df = original_df[original_df['Time of Day'] == time_segment]
    
    # group by PULocationID and count the number of rides for each location
    ride_volume_df = filtered_df.groupby('PULocationID').size().reset_index(name='Ride Volume')
    
    # merge the ride volume into updated_reg_results based on PULocationID
    merged_df = pd.merge(updated_reg_results, ride_volume_df, on='PULocationID', how='left')
    
    return merged_df

def plot_alpha_vs_column(reg_results, column_name, plots_path, plot_filename):
    """
    Creates and saves a plot of 'alpha' versus the specified column from reg_results.

    inputs:
    reg_results (pd.DataFrame): The DataFrame containing the regression results, including 'alpha' and other columns.
    column_name (str): The name of the column to plot on the y-axis against 'alpha'.
    plots_path (str): The directory where the plot will be saved.
    plot_filename (str): The filename to use for saving the plot.

    outputs:
    None
    """
    # create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(reg_results['alpha'], reg_results[column_name], alpha=0.7)
    
    # add labels and title
    plt.xlabel('Alpha')
    plt.ylabel(column_name)
    plt.title(f'Alpha vs {column_name}')
    
    # save the plot
    plt.tight_layout()
    plot_path = f"{plots_path}/{plot_filename}"
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved at {plot_path}")

## LAND USE

def map_to_location_id(input_gdf, taxi_zones):
    """
    Function to perform spatial join and assign 'LocationID' from taxi_zones to the input GeoDataFrame.
    
    inputs:
    - input_gdf: The input GeoDataFrame (e.g., subway_data) to which 'LocationID' should be assigned.
    - taxi_zones: The taxi_zones GeoDataFrame that contains 'LocationID' and 'geometry'.
    
    outputs:
    - input_gdf_with_location_id: The GeoDataFrame with 'LocationID' assigned to each observation.
    """
    # ensure both GeoDataFrames have the same CRS (Coordinate Reference System)
    input_gdf = input_gdf.to_crs(taxi_zones.crs)

    # perform a spatial join to classify input_gdf into taxi zones
    input_gdf_with_location_id = gpd.sjoin(input_gdf, taxi_zones[['LocationID', 'geometry']], how="left", predicate="within")

    return input_gdf_with_location_id

def calculate_shannon_diversity(land_use_with_location_id):
    # group by LocationID
    location_groups = land_use_with_location_id.groupby('LocationID')
    
    # dictionary to store Shannon Diversity Index for each LocationID
    shannon_diversity_dict = {}

    # loop through each LocationID group
    for location_id, group in location_groups:
        # count the occurrences of each land use category
        land_use_counts = group['LandUse'].value_counts()
        
        # calculate the proportions (p_i)
        total_count = land_use_counts.sum()
        proportions = land_use_counts / total_count
        
        # calculate Shannon Diversity Index
        shannon_index = -np.sum(proportions * np.log(proportions))
        
        # store the result for this LocationID
        shannon_diversity_dict[location_id] = shannon_index
    
    # convert the dictionary to a DataFrame for easier merging
    shannon_diversity_df = pd.DataFrame(list(shannon_diversity_dict.items()), columns=['LocationID', 'shannon_index'])
    
    return shannon_diversity_df


def plot_shannon_diversity_map(taxi_zones_with_diversity, plots_path, filename='shannon_diversity_map.png'):
    """
    Function to create and save a map plot of the Shannon Diversity Index for each taxi zone.

    inputs:
    - taxi_zones_with_diversity: GeoDataFrame containing 'LocationID' and 'shannon_index'.
    - plots_path: Directory path where the plot will be saved.
    - filename: Name of the file to save the plot (default: 'shannon_diversity_map.png').
    
    outpus:
    - None. The plot is saved to the specified path.
    """
    
    # ensure the plots directory exists
    os.makedirs(plots_path, exist_ok=True)

    # plotting the Shannon Diversity Index on the map
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # define a color map (e.g., 'YlGn')
    cmap = plt.get_cmap('YlGn')

    # normalize the color scale based on the Shannon Diversity Index values
    norm = mcolors.Normalize(vmin=taxi_zones_with_diversity['shannon_index'].min(), 
                             vmax=taxi_zones_with_diversity['shannon_index'].max())

    # plot the taxi zones, coloring them by the Shannon Diversity Index
    taxi_zones_with_diversity.plot(column='shannon_index', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='black', norm=norm)

    # add a color legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Shannon Diversity Index')

    # add a title
    ax.set_title('Shannon Diversity Index per NYC Taxi Zone', fontsize=16)

    # remove axis for a cleaner look
    ax.set_axis_off()

    # save the plot to the specified path
    save_path = os.path.join(plots_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # show a message indicating where the plot has been saved
    print(f"Plot saved to: {save_path}")

def calculate_simpson_index(land_use_with_location_id):
    # group by LocationID
    location_groups = land_use_with_location_id.groupby('LocationID')
    
    # dictionary to store Simpson Index for each LocationID
    simpson_index_dict = {}

    # loop through each LocationID group
    for location_id, group in location_groups:
        # count the occurrences of each land use category
        land_use_counts = group['LandUse'].value_counts()
        
        # calculate the proportions (p_i)
        total_count = land_use_counts.sum()
        proportions = land_use_counts / total_count
        
        # calculate Simpson Index (1 - sum(p_i^2))
        simpson_index = 1 - np.sum(proportions ** 2)
        
        # store the result for this LocationID
        simpson_index_dict[location_id] = simpson_index
    
    # convert the dictionary to a DataFrame for easier merging
    simpson_index_df = pd.DataFrame(list(simpson_index_dict.items()), columns=['LocationID', 'simpson_index'])
    
    return simpson_index_df

def plot_simpson_index_map(taxi_zones_with_simpson, plots_path, filename='simpson_index_map.png'):
    """
    Function to create and save a map plot of the Simpson Index for each taxi zone.

    inputs:
    - taxi_zones_with_simpson: GeoDataFrame containing 'LocationID' and 'simpson_index'.
    - plots_path: Directory path where the plot will be saved.
    - filename: Name of the file to save the plot (default: 'simpson_index_map.png').
    
    outputs:
    - None. The plot is saved to the specified path.
    """
    
    # ensure the plots directory exists
    os.makedirs(plots_path, exist_ok=True)

    # plotting the Simpson Index on the map
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # define a color map (e.g., 'YlOrBr')
    cmap = plt.get_cmap('YlOrBr')

    # normalize the color scale based on the Simpson Index values
    norm = mcolors.Normalize(vmin=taxi_zones_with_simpson['simpson_index'].min(), 
                             vmax=taxi_zones_with_simpson['simpson_index'].max())
    
    # plot the taxi zones, coloring them by the Simpson Index
    taxi_zones_with_simpson.plot(column='simpson_index', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='black', norm=norm)

    # add a color legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Simpson Diversity Index')

    # add a title
    ax.set_title('Simpson Diversity Index per NYC Taxi Zone', fontsize=16)

    # remove axis for a cleaner look
    ax.set_axis_off()

    # save the plot to the specified path
    save_path = os.path.join(plots_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # show a message indicating where the plot has been saved
    print(f"Plot saved to: {save_path}")

def load_land_use_and_merge_div_indices_with_reg_results(density_datasets_path, taxi_zones, reg_results, plots_path):
    """
    Loads land use data, calculates Shannon and Simpson Diversity Indices, and merges these indices into reg_results.
    Also adds adjacency-based diversity indices by considering adjacent zones' land use data.

    inputs:
    density_datasets_path (str): Path to the land use dataset.
    taxi_zones (gpd.GeoDataFrame): GeoDataFrame with the taxi zones.
    reg_results (pd.DataFrame): DataFrame containing regression results with 'PULocationID' and 'adjacent_locations'.
    plots_path (str): Path where the index maps will be saved.

    outputs:
    pd.DataFrame: Updated reg_results DataFrame with Shannon and Simpson indices and their adjacency-based versions.
    """
    
    # load the land use data and filter out Staten Island and empty land use values
    land_use = gpd.read_file(f"{density_datasets_path}/land_use/nyc_mappluto_24v3_shp/MapPLUTO.shp")
    land_use = land_use[land_use['Borough'] != 'SI']
    land_use = land_use[land_use['LandUse'] != 'None']
    
    # map land use data to taxi zones and create LocationID
    land_use_with_location_id = map_to_location_id(land_use, taxi_zones)
    
    # calculate Shannon Diversity Index
    shannon_diversity_df = calculate_shannon_diversity(land_use_with_location_id)
    
    # merge Shannon Diversity Index into taxi_zones
    taxi_zones_with_diversity = taxi_zones.merge(shannon_diversity_df, on='LocationID', how='left')
    taxi_zones_with_diversity['shannon_index'] = taxi_zones_with_diversity['shannon_index'].fillna(0)
    
    # plot the Shannon Diversity Index map
    plot_shannon_diversity_map(taxi_zones_with_diversity, plots_path, filename='shannon_index_map.png')
    
    # calculate Simpson Index
    simpson_index_df = calculate_simpson_index(land_use_with_location_id)
    
    # merge Simpson Index into taxi_zones
    taxi_zones_with_simpson = taxi_zones.merge(simpson_index_df, on='LocationID', how='left')
    taxi_zones_with_simpson['simpson_index'] = taxi_zones_with_simpson['simpson_index'].fillna(0)
    
    # plot the Simpson Index map
    plot_simpson_index_map(taxi_zones_with_simpson, plots_path, filename='simpson_index_map.png')
    
    # calculate adjacency-based Shannon and Simpson indices
    taxi_zones_with_diversity_adj = calculate_adjacent_diversity_indices(land_use_with_location_id, reg_results, taxi_zones, 'shannon_index')
    taxi_zones_with_simpson_adj = calculate_adjacent_diversity_indices(land_use_with_location_id, reg_results, taxi_zones, 'simpson_index')
    
    # merge the original and adjacency-based indices back into reg_results
    reg_results = reg_results.merge(taxi_zones_with_diversity[['LocationID', 'shannon_index']], 
                                    left_on='PULocationID', right_on='LocationID', how='left')
    if 'LocationID' in reg_results.columns:
        reg_results = reg_results.drop(columns=['LocationID'])

    reg_results = reg_results.merge(taxi_zones_with_simpson[['LocationID', 'simpson_index']], 
                                    left_on='PULocationID', right_on='LocationID', how='left')
    if 'LocationID' in reg_results.columns:
        reg_results = reg_results.drop(columns=['LocationID'])

    reg_results = reg_results.merge(taxi_zones_with_diversity_adj[['LocationID', 'shannon_index_adj']], 
                                    left_on='PULocationID', right_on='LocationID', how='left')
    if 'LocationID' in reg_results.columns:
        reg_results = reg_results.drop(columns=['LocationID'])

    reg_results = reg_results.merge(taxi_zones_with_simpson_adj[['LocationID', 'simpson_index_adj']], 
                                    left_on='PULocationID', right_on='LocationID', how='left')
    
    # drop the 'LocationID' column from the regression results if it was added during the merge
    reg_results.drop(columns=['LocationID'], inplace=True)
    
    return reg_results

def calculate_adjacent_diversity_indices(land_use_with_location_id, reg_results, taxi_zones, index_column):
    """
    Calculates adjacency-based diversity indices (Shannon or Simpson) by considering adjacent zones' land use data.

    inputs:
    land_use_with_location_id (gpd.GeoDataFrame): GeoDataFrame with land use data and LocationID.
    reg_results (pd.DataFrame): DataFrame with adjacent locations in the 'adjacent_locations' column.
    taxi_zones (gpd.GeoDataFrame): GeoDataFrame containing taxi zones.
    index_column (str): Column name for the diversity index ('shannon_index' or 'simpson_index').

    outputs:
    gpd.GeoDataFrame: GeoDataFrame with the adjacency-based diversity indices.
    """
    adj_diversity = {}

    # loop through each zone and calculate the index based on the zone and its adjacent zones
    for location_id, row in reg_results.iterrows():
        adjacent_locations = row['adjacent_locations']  # list of adjacent locations including the zone itself
        adjacent_land_use = land_use_with_location_id[land_use_with_location_id['LocationID'].isin(adjacent_locations)]
        
        # calculate the diversity index for the adjacent zones (can use Shannon or Simpson)
        if index_column == 'shannon_index':
            adj_diversity_value = calculate_shannon_diversity(adjacent_land_use)['shannon_index'].mean()
        elif index_column == 'simpson_index':
            adj_diversity_value = calculate_simpson_index(adjacent_land_use)['simpson_index'].mean()
        else:
            raise ValueError("Invalid index_column parameter. Use 'shannon_index' or 'simpson_index'.")

        adj_diversity[location_id] = adj_diversity_value

    # convert the adjacency-based diversity indices to a DataFrame
    adj_diversity_df = pd.DataFrame(list(adj_diversity.items()), columns=['LocationID', f'{index_column}_adj'])
    
    # merge with taxi_zones
    taxi_zones_with_adj_diversity = taxi_zones.merge(adj_diversity_df, on='LocationID', how='left')
    
    # fill NaN values with 0 where no data is available
    taxi_zones_with_adj_diversity[f'{index_column}_adj'] = taxi_zones_with_adj_diversity[f'{index_column}_adj'].fillna(0)
    
    return taxi_zones_with_adj_diversity


## BUILDINGS

def calculate_building_height_stats(buildings_with_location_id, taxi_zones, stat, stat_column_name):
    """
    Function to calculate building height statistics (max, min, median, quantiles) for each LocationID.
    
    inputs:
    - buildings_with_location_id: GeoDataFrame containing building information with 'heightroof' and 'LocationID' columns.
    - taxi_zones: GeoDataFrame containing taxi zones with 'LocationID' and 'geometry' columns.
    - stat: The statistic to compute ('max', 'min', 'median', 'upper_quantile', 'lower_quantile').
    - stat_column_name: The name to assign to the resulting column in taxi_zones_with_stat.

    outputs:
    - taxi_zones_with_stat: GeoDataFrame with the calculated statistic for each zone.
    """
    
    # check for the correct height column in the building data
    if 'heightroof' not in buildings_with_location_id.columns:
        raise ValueError("Please make sure the 'heightroof' column exists in the buildings_with_location_id GeoDataFrame.")
    
    # calculate statistics based on the 'stat' parameter
    if stat == 'max':
        building_stat = buildings_with_location_id.groupby('LocationID')['heightroof'].max().reset_index()
    elif stat == 'min':
        building_stat = buildings_with_location_id.groupby('LocationID')['heightroof'].min().reset_index()
    elif stat == 'median':
        building_stat = buildings_with_location_id.groupby('LocationID')['heightroof'].median().reset_index()
    elif stat == 'upper_quantile':
        building_stat = buildings_with_location_id.groupby('LocationID')['heightroof'].quantile(0.75).reset_index()
    elif stat == 'lower_quantile':
        building_stat = buildings_with_location_id.groupby('LocationID')['heightroof'].quantile(0.25).reset_index()
    else:
        raise ValueError("Invalid stat parameter. Choose from 'max', 'min', 'median', 'upper_quantile', 'lower_quantile'.")
    
    # rename the calculated column to the specified stat_column_name
    building_stat.rename(columns={'heightroof': stat_column_name}, inplace=True)

    # merge the calculated statistic with the taxi zones GeoDataFrame
    taxi_zones_with_stat = taxi_zones.merge(building_stat, on='LocationID', how='left')
    
    # fill NaN values with 0 where there are no buildings in the taxi zone
    taxi_zones_with_stat[stat_column_name] = taxi_zones_with_stat[stat_column_name].fillna(0)
    
    return taxi_zones_with_stat

def calculate_building_height_stats_with_adj(buildings_with_location_id, taxi_zones, reg_results, stat, stat_column_name):
    """
    Function to calculate building height statistics (max, min, median, quantiles) for each LocationID
    and its adjacent locations.
    
    inputs:
    - buildings_with_location_id: GeoDataFrame containing building information with 'heightroof' and 'LocationID' columns.
    - taxi_zones: GeoDataFrame containing taxi zones with 'LocationID' and 'geometry' columns.
    - reg_results: DataFrame containing the adjacent locations for each zone in the 'adjacent_locations' column.
    - stat: The statistic to compute ('max', 'min', 'median', 'upper_quantile', 'lower_quantile').
    - stat_column_name: The name to assign to the resulting column in taxi_zones_with_stat.

    outputs:
    - taxi_zones_with_stat: GeoDataFrame with the calculated statistic for each zone, including its adjacent zones.
    """
    
    # check for the correct height column in the building data
    if 'heightroof' not in buildings_with_location_id.columns:
        raise ValueError("Please make sure the 'heightroof' column exists in the buildings_with_location_id GeoDataFrame.")
    
    # prepare a dictionary to store the computed statistics for each LocationID
    adj_building_stats = {}

    # loop through each zone and its adjacent zones
    for location_id, row in reg_results.iterrows():
        adjacent_locations = row['adjacent_locations']  # List of adjacent locations, including the zone itself
        buildings_in_zone_and_adj = buildings_with_location_id[buildings_with_location_id['LocationID'].isin(adjacent_locations)]
        
        # calculate the required statistic
        if stat == 'max':
            stat_value = buildings_in_zone_and_adj['heightroof'].max()
        elif stat == 'min':
            stat_value = buildings_in_zone_and_adj['heightroof'].min()
        elif stat == 'median':
            stat_value = buildings_in_zone_and_adj['heightroof'].median()
        elif stat == 'upper_quantile':
            stat_value = buildings_in_zone_and_adj['heightroof'].quantile(0.75)
        elif stat == 'lower_quantile':
            stat_value = buildings_in_zone_and_adj['heightroof'].quantile(0.25)
        else:
            raise ValueError("Invalid stat parameter. Choose from 'max', 'min', 'median', 'upper_quantile', 'lower_quantile'.")

        adj_building_stats[location_id] = stat_value

    # convert the stats dictionary to a DataFrame and merge with taxi zones
    adj_building_stats_df = pd.DataFrame(list(adj_building_stats.items()), columns=['LocationID', stat_column_name])
    
    # merge the calculated statistic with the taxi zones GeoDataFrame
    taxi_zones_with_stat = taxi_zones.merge(adj_building_stats_df, on='LocationID', how='left')
    
    # fill NaN values with 0 where there are no buildings in the taxi zone
    taxi_zones_with_stat[stat_column_name] = taxi_zones_with_stat[stat_column_name].fillna(0)
    
    return taxi_zones_with_stat

def plot_building_height_map(taxi_zones_with_stat, stat_label, plots_path, filename='building_height_map.png'):
    """
    Function to plot and save a building height map based on the given statistic.
    
    inputs:
    - taxi_zones_with_stat: GeoDataFrame containing the building height statistic for each zone.
    - stat_label: Label for the statistic being plotted (e.g., 'max_height', 'min_height').
    - plots_path: Directory where the plot will be saved.
    - filename: Name of the file to save the plot (default: 'building_height_map.png').
    
    outputs:
    - None. The plot is saved to the specified path.
    """
    
    # plotting the building height statistic on the map
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # define a color map
    cmap = plt.get_cmap('YlGnBu')

    # normalize the color scale based on the height statistic
    norm = mcolors.Normalize(vmin=taxi_zones_with_stat['heightroof'].min(), vmax=taxi_zones_with_stat['heightroof'].max())

    # plot the taxi zones, coloring them by the height statistic
    taxi_zones_with_stat.plot(column='heightroof', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='black', norm=norm)

    # add a color legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(stat_label)

    # add a title
    ax.set_title(f'{stat_label} per NYC Taxi Zone', fontsize=16)

    # remove axis for a cleaner look
    ax.set_axis_off()

    # save the plot to the specified path
    save_path = os.path.join(plots_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # show a message indicating where the plot has been saved
    print(f"Plot saved to: {save_path}")    


def add_building_stats_to_reg_results(density_datasets_path, taxi_zones, reg_results, plots_path):
    """
    Loads buildings dataset, calculates building stats, and adds them as new columns to reg_results
    by mapping LocationID to PULocationID. Also adds normalized and adjacency-based versions of stats (by zone's area).

    inputs:
    buildings_dataset_path (str): Path to the buildings dataset.
    taxi_zones (gpd.GeoDataFrame): GeoDataFrame with the taxi zones.
    reg_results (pd.DataFrame): DataFrame containing regression results with 'PULocationID' and 'adjacent_locations'.
    plots_path (str): Path where building height plots will be saved.

    outputs:
    pd.DataFrame: Updated reg_results DataFrame with building height statistics columns.
    """
    # load the buildings dataset
    buildings = gpd.read_file(f"{density_datasets_path}/building_footprints/building_p/geo_export_000a8f4e-8313-4a31-b2c2-25751ae34bff.shp")

    # map the buildings dataset to the taxi zones to assign LocationID
    buildings_with_location_id = map_to_location_id(buildings, taxi_zones)

    # Define the building stats to calculate
    building_stats = {
        'max': 'max_height',
        'median': 'median_height',  # New median stat added
        'min': 'min_height',
        'upper_quantile': 'upper_quantile_height',
        'lower_quantile': 'lower_quantile_height'
    }

    # add area in square kilometers to the taxi zones
    taxi_zones['area_km2'] = taxi_zones.geometry.area / 1e6  # Convert from m² to km²

    for stat_type, stat_name in building_stats.items():
        # calculate the building height stats for the specified type (max, min, etc.)
        taxi_zones_with_stat = calculate_building_height_stats(buildings_with_location_id, taxi_zones, stat_type, stat_name)
        
        # normalize the stat by the zone's area (stat / area)
        taxi_zones_with_stat[f'{stat_name}_norm'] = taxi_zones_with_stat[stat_name] / taxi_zones_with_stat['area_km2']

        # calculate stats including adjacency
        taxi_zones_with_adj_stat = calculate_building_height_stats_with_adj(buildings_with_location_id, taxi_zones, reg_results, stat_type, f'{stat_name}_adj')

        # normalize the adj stat by the combined area of adjacent zones
        taxi_zones_with_adj_stat['combined_area_km2'] = reg_results['adjacent_locations'].apply(
            lambda adj_locs: taxi_zones[taxi_zones['LocationID'].isin(adj_locs)]['area_km2'].sum()
        )
        taxi_zones_with_adj_stat[f'{stat_name}_adj_norm'] = taxi_zones_with_adj_stat[f'{stat_name}_adj'] / taxi_zones_with_adj_stat['combined_area_km2']

        # merge all stats (raw, adj, and normalized) into reg_results
        reg_results = reg_results.merge(
            taxi_zones_with_stat[['LocationID', stat_name, f'{stat_name}_norm']],
            left_on='PULocationID', right_on='LocationID', how='left'
        )
        reg_results.drop(columns=['LocationID'], inplace=True)
        if 'LocationID' in reg_results.columns:
            reg_results = reg_results.drop(columns=['LocationID'])

        reg_results = reg_results.merge(
            taxi_zones_with_adj_stat[['LocationID', f'{stat_name}_adj', f'{stat_name}_adj_norm']],
            left_on='PULocationID', right_on='LocationID', how='left'
        )
        reg_results.drop(columns=['LocationID'], inplace=True)

    return reg_results

def plot_building_stats(reg_results, plots_path, building_stat_columns):
    """
    Plots the raw, normalized, adjacency-based, and adjacency-normalized versions of building statistics.

    inputs:
    reg_results (pd.DataFrame): DataFrame containing regression results with building stats.
    plots_path (str): Path where the plots will be saved.
    building_stat_columns (list): List of base building statistics to plot (e.g., 'max_height', 'median_height').

    """
    # define the filenames for the plots
    plot_filenames = {
        'max_height': 'alpha_vs_max_building_height.png',
        'max_height_norm': 'alpha_vs_max_building_height_norm.png',
        'max_height_adj': 'alpha_vs_max_building_height_adj.png',
        'max_height_adj_norm': 'alpha_vs_max_building_height_adj_norm.png',
        
        'median_height': 'alpha_vs_median_building_height.png',
        'median_height_norm': 'alpha_vs_median_building_height_norm.png',
        'median_height_adj': 'alpha_vs_median_building_height_adj.png',
        'median_height_adj_norm': 'alpha_vs_median_building_height_adj_norm.png',

        'min_height': 'alpha_vs_min_building_height.png',
        'min_height_norm': 'alpha_vs_min_building_height_norm.png',
        'min_height_adj': 'alpha_vs_min_building_height_adj.png',
        'min_height_adj_norm': 'alpha_vs_min_building_height_adj_norm.png',

        'upper_quantile_height': 'alpha_vs_upper_quantile_building_height.png',
        'upper_quantile_height_norm': 'alpha_vs_upper_quantile_building_height_norm.png',
        'upper_quantile_height_adj': 'alpha_vs_upper_quantile_building_height_adj.png',
        'upper_quantile_height_adj_norm': 'alpha_vs_upper_quantile_building_height_adj_norm.png',

        'lower_quantile_height': 'alpha_vs_lower_quantile_building_height.png',
        'lower_quantile_height_norm': 'alpha_vs_lower_quantile_building_height_norm.png',
        'lower_quantile_height_adj': 'alpha_vs_lower_quantile_building_height_adj.png',
        'lower_quantile_height_adj_norm': 'alpha_vs_lower_quantile_building_height_adj_norm.png'
    }

    # loop through each stat and plot raw, normalized, adjacency-based, and adjacency-normalized versions
    for stat_name in building_stat_columns:
        # plot the raw version
        plot_alpha_vs_column(reg_results, stat_name, plots_path, plot_filenames[stat_name])
        
        # plot the normalized version
        norm_stat_name = f'{stat_name}_norm'
        plot_alpha_vs_column(reg_results, norm_stat_name, plots_path, plot_filenames[norm_stat_name])
        
        # plot the adjacency-based version
        adj_stat_name = f'{stat_name}_adj'
        plot_alpha_vs_column(reg_results, adj_stat_name, plots_path, plot_filenames[adj_stat_name])
        
        # plot the adjacency-normalized version
        adj_norm_stat_name = f'{stat_name}_adj_norm'
        plot_alpha_vs_column(reg_results, adj_norm_stat_name, plots_path, plot_filenames[adj_norm_stat_name])

## PEDESTRIAN COUNTS
def add_ped_count_stats_to_reg_results(density_datasets_path, taxi_zones, reg_results, plots_path, time_segment):
    """
    Loads the pedestrian count dataset, calculates pedestrian count statistics (AM or PM based on time_segment),
    and adds them as new columns to reg_results by mapping LocationID to PULocationID.
    Also adds normalized versions of pedestrian count statistics by the zone's area and adjacency-based versions.

    inputs:
    density_datasets_path (str): Path to the pedestrian counts dataset.
    taxi_zones (gpd.GeoDataFrame): GeoDataFrame with the taxi zones.
    reg_results (pd.DataFrame): DataFrame containing regression results with 'PULocationID' and 'adjacent_locations'.
    plots_path (str): Path where any related plots will be saved (not used in this case, but included for consistency).
    time_segment (str): Time segment for which to calculate pedestrian counts ('morning_rush' or 'evening_rush').

    outputs:
    pd.DataFrame: Updated reg_results DataFrame with pedestrian count statistics columns.
    """
    # load the pedestrian count dataset
    peds = gpd.read_file(f"{density_datasets_path}/pedestrian_counts/Shapefile/PedCounts_Fall2023.shp")

    # map the pedestrian dataset to the taxi zones to assign LocationID
    peds_with_location_id = map_to_location_id(peds, taxi_zones)

    # adjust columns based on the time_segment argument
    if time_segment == 'morning_rush':
        # use AM counts for morning time segment
        peds_time_2021 = peds_with_location_id[['LocationID', 'May21_AM', 'Oct21_AM']]
        count_column_name = 'avg_am_count_2021'
    elif time_segment == 'evening_rush':
        # use PM counts for evening time segment
        peds_time_2021 = peds_with_location_id[['LocationID', 'May21_PM', 'Oct21_PM']]
        count_column_name = 'avg_pm_count_2021'
    else:
        raise ValueError("Invalid time_segment. Choose 'morning_rush' or 'evening_rush'.")

    # calculate the average pedestrian count for May and October for each location
    peds_time_2021[count_column_name] = peds_time_2021.mean(axis=1)

    # group by LocationID and calculate the average pedestrian count per taxi zone
    avg_count_per_zone = peds_time_2021.groupby('LocationID')[count_column_name].mean().reset_index()

    # merge the pedestrian counts with the taxi_zones GeoDataFrame
    taxi_zones_with_counts = taxi_zones.merge(avg_count_per_zone, on='LocationID', how='left')

    # fill NaN values with 0 where there are no pedestrian counts for that zone
    taxi_zones_with_counts[count_column_name] = taxi_zones_with_counts[count_column_name].fillna(0)

    # calculate additional pedestrian count statistics
    count_stations = peds_with_location_id.groupby('LocationID').size().reset_index(name='number_of_count_stations')

    # add zone area in square kilometers to taxi_zones
    taxi_zones_with_counts['area_km2'] = taxi_zones_with_counts.geometry.area / 1e6  # Convert from m² to km²

    # normalize the pedestrian count stats by area
    taxi_zones_with_counts[f'{count_column_name}_norm'] = taxi_zones_with_counts[count_column_name] / taxi_zones_with_counts['area_km2']
    count_stations['number_of_count_stations_norm'] = count_stations['number_of_count_stations'] / taxi_zones_with_counts['area_km2']

    # calculate adjacency-based pedestrian counts and count stations
    taxi_zones_with_counts_adj = calculate_adjacent_pedestrian_stats(taxi_zones, reg_results, taxi_zones_with_counts, count_stations, count_column_name)

    # merge the pedestrian stats into reg_results
    reg_results = reg_results.merge(taxi_zones_with_counts[['LocationID', count_column_name, f'{count_column_name}_norm']], 
                                    left_on='PULocationID', right_on='LocationID', how='left')
    reg_results[count_column_name] = reg_results[count_column_name].fillna(0)
    reg_results[f'{count_column_name}_norm'] = reg_results[f'{count_column_name}_norm'].fillna(0)
    if 'LocationID' in reg_results.columns:
        reg_results = reg_results.drop(columns=['LocationID'])

    reg_results = reg_results.merge(count_stations[['LocationID', 'number_of_count_stations', 'number_of_count_stations_norm']], 
                                    left_on='PULocationID', right_on='LocationID', how='left')
    reg_results['number_of_count_stations'] = reg_results['number_of_count_stations'].fillna(0)
    reg_results['number_of_count_stations_norm'] = reg_results['number_of_count_stations_norm'].fillna(0)
    if 'LocationID' in reg_results.columns:
        reg_results = reg_results.drop(columns=['LocationID'])

    # merge the adjacency-based pedestrian counts and count stations into reg_results
    reg_results = reg_results.merge(taxi_zones_with_counts_adj[['LocationID', f'{count_column_name}_adj', f'{count_column_name}_adj_norm', 
                                                               'number_of_count_stations_adj', 'number_of_count_stations_adj_norm']], 
                                    left_on='PULocationID', right_on='LocationID', how='left')
    if 'LocationID' in reg_results.columns:
        reg_results = reg_results.drop(columns=['LocationID'])

    # has_count_stations: Whether there is at least one count station in each zone (1 if count_stations > 0, else 0)
    peds_with_location_id['has_count_stations'] = peds_with_location_id['LocationID'].map(
        lambda loc_id: 1 if loc_id in count_stations['LocationID'].values else 0
    ).reset_index(drop=True)

    if 'LocationID' in reg_results.columns:
        reg_results = reg_results.drop(columns=['LocationID'])

    reg_results = reg_results.merge(peds_with_location_id[['LocationID', 'has_count_stations']].drop_duplicates(), 
                                    left_on='PULocationID', right_on='LocationID', how='left')
    reg_results['has_count_stations'] = reg_results['has_count_stations'].fillna(0)

    return reg_results

def calculate_adjacent_pedestrian_stats(taxi_zones, reg_results, taxi_zones_with_counts, count_stations, count_column_name):
    """
    Calculate adjacency-based pedestrian stats (counts and stations) by considering adjacent zones.

    inputs:
    taxi_zones (gpd.GeoDataFrame): GeoDataFrame containing taxi zones.
    reg_results (pd.DataFrame): DataFrame with adjacent locations in the 'adjacent_locations' column.
    taxi_zones_with_counts (gpd.GeoDataFrame): GeoDataFrame with pedestrian counts and area for each zone.
    count_stations (pd.DataFrame): DataFrame with count stations for each zone.
    count_column_name (str): The column name for pedestrian count statistics (e.g., 'avg_am_count_2021').

    outputs:
    gpd.GeoDataFrame: GeoDataFrame with adjacency-based pedestrian stats.
    """
    adj_stats = []

    # loop through each zone and calculate the adjacency-based stats
    for location_id, row in reg_results.iterrows():
        adjacent_locations = row['adjacent_locations']  # list of adjacent locations including the zone itself
        adjacent_counts = taxi_zones_with_counts[taxi_zones_with_counts['LocationID'].isin(adjacent_locations)]
        adjacent_stations = count_stations[count_stations['LocationID'].isin(adjacent_locations)]

        # calculate the sum of counts and the number of stations for adjacent zones
        count_adj = adjacent_counts[count_column_name].sum()
        count_adj_norm = count_adj / adjacent_counts['area_km2'].sum()  # normalize by the combined area of adjacent zones
        stations_adj = adjacent_stations['number_of_count_stations'].sum()
        stations_adj_norm = stations_adj / adjacent_counts['area_km2'].sum()  # normalize by the combined area of adjacent zones

        adj_stats.append({
            'LocationID': location_id,
            f'{count_column_name}_adj': count_adj,
            f'{count_column_name}_adj_norm': count_adj_norm,
            'number_of_count_stations_adj': stations_adj,
            'number_of_count_stations_adj_norm': stations_adj_norm
        })

    # convert the adjacency-based stats to a DataFrame
    adj_stats_df = pd.DataFrame(adj_stats)

    # merge with taxi zones to preserve GeoDataFrame structure
    taxi_zones_with_adj_stats = taxi_zones.merge(adj_stats_df, on='LocationID', how='left')
    
    return taxi_zones_with_adj_stats

def plot_ped_stats(reg_results, plots_path, time_segment):
    """
    Plots pedestrian count statistics and their normalized versions, including adjacency-based stats.

    inputs:
    reg_results (pd.DataFrame): DataFrame containing regression results with pedestrian count stats.
    plots_path (str): Path where the plots will be saved.
    time_segment (str): Time segment for which to plot pedestrian counts ('morning_rush' or 'evening_rush').

    outputs:
    None
    """
    # plot for 'has_count_stations'
    plot_alpha_vs_column(reg_results, 'has_count_stations', plots_path, 'alpha_vs_count_station_presence.png')

    # plot for 'number_of_count_stations' and its normalized version
    plot_alpha_vs_column(reg_results, 'number_of_count_stations', plots_path, 'alpha_vs_count_station_number.png')
    plot_alpha_vs_column(reg_results, 'number_of_count_stations_norm', plots_path, 'alpha_vs_count_station_number_norm.png')

    # determine the appropriate count column name based on the time segment
    if time_segment == 'morning_rush':
        count_column_name = 'avg_am_count_2021'
    elif time_segment == 'evening_rush':
        count_column_name = 'avg_pm_count_2021'
    else:
        raise ValueError("Invalid time_segment. Choose 'morning_rush' or 'evening_rush'.")

    # plot for the chosen count column (AM or PM) and its normalized version
    plot_alpha_vs_column(reg_results, count_column_name, plots_path, f'alpha_vs_{count_column_name}.png')
    plot_alpha_vs_column(reg_results, f'{count_column_name}_norm', plots_path, f'alpha_vs_{count_column_name}_norm.png')

    # plot for adjacency-based pedestrian stats
    plot_alpha_vs_column(reg_results, f'{count_column_name}_adj', plots_path, f'alpha_vs_{count_column_name}_adj.png')
    plot_alpha_vs_column(reg_results, f'{count_column_name}_adj_norm', plots_path, f'alpha_vs_{count_column_name}_adj_norm.png')

    # plot for adjacency-based count stations and normalized versions
    plot_alpha_vs_column(reg_results, 'number_of_count_stations_adj', plots_path, 'alpha_vs_count_station_number_adj.png')
    plot_alpha_vs_column(reg_results, 'number_of_count_stations_adj_norm', plots_path, 'alpha_vs_count_station_number_adj_norm.png')


## WIFI HOTSPOTS
def add_wifi_stats_to_reg_results(density_datasets_path, taxi_zones, reg_results):
    """
    Loads the Wi-Fi dataset, calculates Wi-Fi statistics, and adds them as new columns to reg_results
    by mapping LocationID to PULocationID. Also adds normalized and adjacency-based versions of 
    'number_of_wifi_hotspots' by the taxi zone's area in km².

    inputs:
    density_datasets_path (str): Path to the Wi-Fi hotspots dataset (CSV).
    taxi_zones (gpd.GeoDataFrame): GeoDataFrame with the taxi zones.
    reg_results (pd.DataFrame): DataFrame containing regression results with 'PULocationID' and 'adjacent_locations'.

    outputs:
    pd.DataFrame: Updated reg_results DataFrame with Wi-Fi statistics columns, including adjacency-based versions.
    """
    # load the Wi-Fi dataset from the CSV file
    wifi = pd.read_csv(f"{density_datasets_path}/wifi_hotspots/NYC_Wi-Fi_Hotspot_Locations_20240930.csv")

    # convert the Wi-Fi dataset to a GeoDataFrame by creating Point geometries based on Latitude and Longitude
    wifi['geometry'] = wifi.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    wifi_gdf = gpd.GeoDataFrame(wifi, geometry='geometry')

    # set the CRS of the Wi-Fi GeoDataFrame to match that of the taxi zones (EPSG:4326 for latitude/longitude)
    wifi_gdf = wifi_gdf.set_crs('EPSG:4326')  # Assuming latitude/longitude is in EPSG:4326
    wifi_gdf = wifi_gdf.to_crs(taxi_zones.crs)  # Reproject to the CRS of taxi_zones

    # perform a spatial join to map Wi-Fi hotspots to taxi zones based on their geometry
    wifi_with_location_id = gpd.sjoin(wifi_gdf, taxi_zones, how='left', predicate='within')

    # calculate the number of Wi-Fi hotspots per LocationID
    wifi_hotspot_count = wifi_with_location_id.groupby('LocationID').size().reset_index(name='number_of_wifi_hotspots')

    # add zone area in square kilometers to taxi_zones
    taxi_zones['area_km2'] = taxi_zones.geometry.area / 1e6  # Convert from m² to km²

    # normalize 'number_of_wifi_hotspots' by the zone's area in km²
    wifi_hotspot_count = wifi_hotspot_count.merge(taxi_zones[['LocationID', 'area_km2']], on='LocationID', how='left')
    wifi_hotspot_count['number_of_wifi_hotspots_norm'] = wifi_hotspot_count['number_of_wifi_hotspots'] / wifi_hotspot_count['area_km2']

    # create a binary column 'has_wifi_hotspot' indicating if a taxi zone has at least one hotspot
    wifi_hotspot_zones = set(wifi_hotspot_count['LocationID'].values)
    wifi_with_location_id['has_wifi_hotspot'] = wifi_with_location_id['LocationID'].apply(
        lambda loc_id: 1 if loc_id in wifi_hotspot_zones else 0
    )

    # calculate adjacency-based Wi-Fi stats, including 'has_wifi_hotspot_adj'
    taxi_zones_with_wifi_adj = calculate_adjacent_wifi_stats(taxi_zones, reg_results, wifi_hotspot_count)

    # merge the Wi-Fi stats into reg_results
    if 'LocationID' in reg_results.columns:
        reg_results = reg_results.drop(columns=['LocationID'])
    
    reg_results = reg_results.merge(
        wifi_hotspot_count[['LocationID', 'number_of_wifi_hotspots', 'number_of_wifi_hotspots_norm']], 
        left_on='PULocationID', right_on='LocationID', how='left'
    ).drop(columns=['LocationID'])

    reg_results['number_of_wifi_hotspots'] = reg_results['number_of_wifi_hotspots'].fillna(0)
    reg_results['number_of_wifi_hotspots_norm'] = reg_results['number_of_wifi_hotspots_norm'].fillna(0)

    # merge the adjacency-based stats, including 'has_wifi_hotspot_adj'
    if 'LocationID' in reg_results.columns:
        reg_results = reg_results.drop(columns=['LocationID'])
    
    reg_results = reg_results.merge(taxi_zones_with_wifi_adj[['LocationID', 'number_of_wifi_hotspots_adj', 'number_of_wifi_hotspots_adj_norm', 'has_wifi_hotspot_adj']], 
                                    left_on='PULocationID', right_on='LocationID', how='left').drop(columns=['LocationID'])

    # merge the 'has_wifi_hotspot' column
    if 'LocationID' in reg_results.columns:
        reg_results = reg_results.drop(columns=['LocationID'])
    
    reg_results = reg_results.merge(
        wifi_with_location_id[['LocationID', 'has_wifi_hotspot']].drop_duplicates(), 
        left_on='PULocationID', right_on='LocationID', how='left'
    ).drop(columns=['LocationID'])

    reg_results['has_wifi_hotspot'] = reg_results['has_wifi_hotspot'].fillna(0)

    return reg_results



def calculate_adjacent_wifi_stats(taxi_zones, reg_results, wifi_hotspot_count):
    """
    Calculates adjacency-based Wi-Fi stats by considering adjacent zones.

    inputs:
    taxi_zones (gpd.GeoDataFrame): GeoDataFrame containing taxi zones.
    reg_results (pd.DataFrame): DataFrame with adjacent locations in the 'adjacent_locations' column.
    wifi_hotspot_count (pd.DataFrame): DataFrame with Wi-Fi hotspots count and area for each zone.

    outputs:
    gpd.GeoDataFrame: GeoDataFrame with adjacency-based Wi-Fi stats.
    """
    adj_stats = []

    # loop through each zone and calculate adjacency-based stats
    for location_id, row in reg_results.iterrows():
        adjacent_locations = row['adjacent_locations']  # List of adjacent locations including the zone itself
        adjacent_wifi_counts = wifi_hotspot_count[wifi_hotspot_count['LocationID'].isin(adjacent_locations)]

        # calculate the sum of Wi-Fi hotspots for adjacent zones
        wifi_hotspots_adj = adjacent_wifi_counts['number_of_wifi_hotspots'].sum()

        # calculate the combined area of the adjacent zones
        total_area_adj = adjacent_wifi_counts['area_km2'].sum()

        # normalize the adjacency-based Wi-Fi hotspots count by the total area of adjacent zones
        wifi_hotspots_adj_norm = wifi_hotspots_adj / total_area_adj if total_area_adj > 0 else 0

        # check if there is any Wi-Fi hotspot in the adjacent zones
        has_wifi_hotspot_adj = 1 if wifi_hotspots_adj > 0 else 0

        adj_stats.append({
            'LocationID': location_id,
            'number_of_wifi_hotspots_adj': wifi_hotspots_adj,
            'number_of_wifi_hotspots_adj_norm': wifi_hotspots_adj_norm,
            'has_wifi_hotspot_adj': has_wifi_hotspot_adj
        })

    # convert adjacency-based stats to a DataFrame
    adj_stats_df = pd.DataFrame(adj_stats)

    # merge with taxi zones to preserve GeoDataFrame structure
    taxi_zones_with_adj_stats = taxi_zones.merge(adj_stats_df, on='LocationID', how='left')

    return taxi_zones_with_adj_stats

## ROAD NETWORK

def calculate_road_metrics(taxi_zones, combined_road_network):
    """
    Function to calculate total road length, intersection count, road density, and intersection density for each taxi zone.
    
    inputs:
    - taxi_zones: GeoDataFrame containing taxi zones with 'LocationID' and 'geometry' columns.
    - combined_road_network: GeoDataFrame of the road network for the boroughs (from OSMnx).
    
    outputs:
    - taxi_zones_with_metrics: GeoDataFrame with calculated total road length, intersection count, road density, and intersection density.
    """
    # ensure both GeoDataFrames are projected to UTM zone 18N (EPSG:32618)
    taxi_zones_meters = taxi_zones.to_crs(epsg=32618)
    combined_road_network_meters = combined_road_network.to_crs(epsg=32618)
    
    # calculate the total road length per LocationID
    combined_road_network_meters['road_length'] = combined_road_network_meters.geometry.length
    road_length_per_zone = combined_road_network_meters.groupby('LocationID')['road_length'].sum().reset_index()
    
    # reproject taxi zones to EPSG:4326 for bbox-based network download
    taxi_zones_4326 = taxi_zones.to_crs(epsg=4326)
    bbox = taxi_zones_4326.total_bounds  # minx, miny, maxx, maxy
    
    # download the road network graph using the bounding box in EPSG:4326
    G = ox.graph_from_bbox(north=bbox[3], south=bbox[1], east=bbox[2], west=bbox[0], network_type='drive')

    # extract nodes (intersections) and edges from the road network
    nodes, _ = ox.graph_to_gdfs(G)

    # reproject nodes to UTM zone 18N (EPSG:32618)
    nodes = nodes.to_crs(epsg=32618)

    # perform a spatial join to map intersections (nodes) to taxi zones
    nodes_with_location_id = gpd.sjoin(nodes, taxi_zones_meters[['LocationID', 'geometry']], how="left", predicate="within")
    
    # count the intersections per LocationID
    intersections_per_zone = nodes_with_location_id.groupby('LocationID').size().reset_index(name='intersection_count')

    # merge road length and intersection count with taxi zones
    taxi_zones_with_metrics = taxi_zones_meters.merge(road_length_per_zone, on='LocationID', how='left')
    taxi_zones_with_metrics = taxi_zones_with_metrics.merge(intersections_per_zone, on='LocationID', how='left')

    # calculate the area of each taxi zone in square kilometers
    taxi_zones_with_metrics['area_km2'] = taxi_zones_with_metrics.geometry.area / 1e6  # Convert from m² to km²
    
    # calculate road density (road length per square kilometer)
    taxi_zones_with_metrics['road_density_km'] = taxi_zones_with_metrics['road_length'] / taxi_zones_with_metrics['area_km2']

    # calculate intersection density (number of intersections per square kilometer)
    taxi_zones_with_metrics['intersection_density'] = taxi_zones_with_metrics['intersection_count'] / taxi_zones_with_metrics['area_km2']
    
    # fill NaN values with 0 where there are no roads or intersections
    taxi_zones_with_metrics['road_length'] = taxi_zones_with_metrics['road_length'].fillna(0)
    taxi_zones_with_metrics['intersection_count'] = taxi_zones_with_metrics['intersection_count'].fillna(0)
    taxi_zones_with_metrics['road_density_km'] = taxi_zones_with_metrics['road_density_km'].fillna(0)
    taxi_zones_with_metrics['intersection_density'] = taxi_zones_with_metrics['intersection_density'].fillna(0)
    
    return taxi_zones_with_metrics


def add_road_metrics_to_reg_results(road_network_path, reg_results, taxi_zones):
    """
    Loads road network data, calculates road metrics, and maps the metrics to reg_results 
    by matching LocationID to PULocationID. Also adds adjacency-based versions of the metrics.

    inputs:
    road_network_path (str): Path to the road network GeoJSON file.
    reg_results (pd.DataFrame): DataFrame containing regression results with 'PULocationID' and 'adjacent_locations'.
    taxi_zones (gpd.GeoDataFrame): GeoDataFrame containing taxi zones with 'LocationID'.

    outputs:
    pd.DataFrame: Updated reg_results DataFrame with road network metrics, including adjacency-based metrics.
    """
    # load the road network data with LocationID
    roads_with_location_id = gpd.read_file(f"{road_network_path}/roads_with_location_id.geojson")

    # ensure both road and taxi_zones data are projected to UTM (EPSG:32618) for distance/area calculations
    if taxi_zones.crs != 'EPSG:32618':
        taxi_zones = taxi_zones.to_crs('EPSG:32618')
    roads_with_location_id = roads_with_location_id.to_crs('EPSG:32618')

    # calculate road metrics using the adjusted calculate_road_metrics function
    taxi_zones_with_metrics = calculate_road_metrics(taxi_zones, roads_with_location_id)

    # calculate adjacency-based road metrics
    taxi_zones_with_metrics_adj = calculate_adjacent_road_metrics(taxi_zones, reg_results, taxi_zones_with_metrics)

    # define the road network metrics to add
    road_metrics = ['road_length', 'intersection_count', 'road_density_km', 'intersection_density']

    # merge each road metric into reg_results based on PULocationID and LocationID
    for metric in road_metrics:
        # Merge the calculated road metrics
        reg_results = reg_results.merge(
            taxi_zones_with_metrics[['LocationID', metric]],
            left_on='PULocationID',
            right_on='LocationID',
            how='left'
        ).drop(columns=['LocationID'])

        # merge the adjacency-based road metrics
        reg_results = reg_results.merge(
            taxi_zones_with_metrics_adj[['LocationID', f'{metric}_adj']],
            left_on='PULocationID',
            right_on='LocationID',
            how='left'
        ).drop(columns=['LocationID'])

    return reg_results

def calculate_adjacent_road_metrics(taxi_zones, reg_results, taxi_zones_with_metrics):
    """
    Calculates adjacency-based road metrics by considering adjacent zones.

    inputs:
    taxi_zones (gpd.GeoDataFrame): GeoDataFrame containing taxi zones.
    reg_results (pd.DataFrame): DataFrame with adjacent locations in the 'adjacent_locations' column.
    taxi_zones_with_metrics (gpd.GeoDataFrame): GeoDataFrame with road metrics for each zone.

    outputs:
    gpd.GeoDataFrame: GeoDataFrame with adjacency-based road metrics.
    """
    adj_metrics = []

    # loop through each zone and calculate adjacency-based metrics
    for location_id, row in reg_results.iterrows():
        adjacent_locations = row['adjacent_locations']  # List of adjacent locations including the zone itself
        adjacent_metrics = taxi_zones_with_metrics[taxi_zones_with_metrics['LocationID'].isin(adjacent_locations)]

        # calculate the sum of road metrics for adjacent zones
        road_length_adj = adjacent_metrics['road_length'].sum()
        intersection_count_adj = adjacent_metrics['intersection_count'].sum()

        # calculate the combined area of the adjacent zones
        total_area_adj = adjacent_metrics['area_km2'].sum()

        # normalize the adjacency-based road metrics by the total area of adjacent zones
        road_density_adj = road_length_adj / total_area_adj if total_area_adj > 0 else 0
        intersection_density_adj = intersection_count_adj / total_area_adj if total_area_adj > 0 else 0

        adj_metrics.append({
            'LocationID': location_id,
            'road_length_adj': road_length_adj,
            'intersection_count_adj': intersection_count_adj,
            'road_density_km_adj': road_density_adj,
            'intersection_density_adj': intersection_density_adj
        })

    # convert adjacency-based metrics to a DataFrame
    adj_metrics_df = pd.DataFrame(adj_metrics)

    # merge with taxi zones to preserve GeoDataFrame structure
    taxi_zones_with_adj_metrics = taxi_zones.merge(adj_metrics_df, on='LocationID', how='left')

    return taxi_zones_with_adj_metrics


def normalize_and_save_density_metrics_and_cdf(reg_results, density_metrics, reg_results_path):
    """
    Normalizes density metrics in the reg_results dataframe in two different ways: 
    1) by dividing by the maximum value, and 
    2) by calculating the empirical CDF.
    
    Changes the sign of the 'alpha' column (from negative to positive),
    and leaves binary metrics ('has_count_stations' and 'has_wifi_hotspot') unchanged.
    
    Both resulting dataframes only keep the density metrics and the 'alpha' column.
    The resulting dataframes are saved to the provided directory.

    inputs:
    reg_results (pd.DataFrame): The dataframe containing regression results and density metrics.
    density_metrics (list): List of column names corresponding to density metrics.
    reg_results_path (str): Directory to save the dataframes with normalized values and empirical CDFs.

    outputs:
    tuple: The dataframe with normalized density metrics and the dataframe with empirical CDF values.
    """
    # create a filtered DataFrame that only keeps the density metrics and the 'alpha' column
    filtered_reg_results = reg_results[density_metrics + ['alpha']].copy()

    # specify the binary metrics that should remain unchanged
    binary_metrics = ['has_count_stations', 'has_wifi_hotspot']
    
    # ensure the binary metrics are not normalized and remain unchanged
    binary_metrics_included = [col for col in binary_metrics if col in filtered_reg_results.columns]

    # normalize the density metrics (0-1 scaling based on max value), excluding binary metrics
    normalized_data_values = filtered_reg_results.copy()
    non_binary_metrics = [col for col in density_metrics if col not in binary_metrics_included]
    normalized_data_values[non_binary_metrics] = normalized_data_values[non_binary_metrics].div(
        normalized_data_values[non_binary_metrics].max(axis=0), axis=1
    )

    # change the sign of 'alpha' and ensure it's unchanged for other columns
    normalized_data_values['alpha'] = -filtered_reg_results['alpha']
    
    # create the empirical CDF dataframe, keeping binary metrics unchanged
    empirical_cdf = filtered_reg_results.copy()
    for metric in non_binary_metrics:
        empirical_cdf[metric] = empirical_cdf[metric].rank(method='max', pct=True)

    # change the sign of 'alpha' for empirical_cdf as well
    empirical_cdf['alpha'] = -filtered_reg_results['alpha']
    
    # save both dataframes to the provided directory
    normalized_data_values.to_csv(f'{reg_results_path}/normalized_data_values.csv', index=False)
    empirical_cdf.to_csv(f'{reg_results_path}/empirical_cdf.csv', index=False)
    
    return normalized_data_values, empirical_cdf


def combine_normalized_and_cdf_density_metrics(reg_results, continuous_metrics, discrete_metrics, results_path):
    """
    Combines the normalized discrete metrics and the empirical CDF of the continuous metrics.
    
    1) The continuous metrics are processed using the empirical CDF.
    2) The discrete metrics are normalized using (0-1 scaling based on max value).
    3) The function leaves binary metrics unchanged.
    
    changes the sign of the 'alpha' column (from negative to positive).
    
    The resulting combined dataframe is saved to the specified directory.

    inputs:
    reg_results (pd.DataFrame): The dataframe containing regression results and density metrics.
    continuous_metrics (list): List of column names corresponding to continuous density metrics.
    discrete_metrics (list): List of column names corresponding to discrete density metrics.
    reg_results_path (str): Directory to save the resulting combined dataframe.

    outputs:
    pd.DataFrame: The combined dataframe with empirical CDF for continuous metrics and normalized values for discrete metrics.
    """
    # ensure 'PULocationID' is included in the filtered dataframe
    # check if 'alpha' column exists
    if 'alpha' in reg_results.columns:
        columns_to_keep = ['PULocationID', 'borough_code'] + continuous_metrics + discrete_metrics + ['alpha']
    else:
        columns_to_keep = ['PULocationID', 'borough_code'] + continuous_metrics + discrete_metrics

    # create a filtered DataFrame that only keeps the necessary columns
    filtered_reg_results = reg_results[columns_to_keep].copy()

    # specify binary metrics that should remain unchanged (assuming they are part of discrete metrics)
    binary_metrics = ['has_count_stations', 'has_wifi_hotspot']
    binary_metrics_included = [col for col in binary_metrics if col in filtered_reg_results.columns]

    # continuous Metrics: Apply empirical CDF processing (rank with pct)
    combined_df = filtered_reg_results.copy()
    for metric in continuous_metrics:
        combined_df[metric] = filtered_reg_results[metric].rank(method='max', pct=True)

    # discrete Metrics: Apply normalization (0-1 scaling based on max value), excluding binary metrics
    non_binary_discrete_metrics = [col for col in discrete_metrics if col not in binary_metrics_included]
    combined_df[non_binary_discrete_metrics] = filtered_reg_results[non_binary_discrete_metrics].div(
        filtered_reg_results[non_binary_discrete_metrics].max(axis=0), axis=1
    )

    # binary metrics remain unchanged

    # if 'alpha' exists, change the sign
    if 'alpha' in combined_df.columns:
        combined_df['alpha'] = -filtered_reg_results['alpha']
    
    # save the combined dataframe to the specified path
    combined_df.to_csv(f'{results_path}/combined_norm_density_metrics.csv', index=False)
    
    return combined_df



def create_and_save_correlation_matrix(normalized_df, density_metrics, plots_path):
    """
    Creates a correlation matrix for the density metrics in normalized_df and saves the correlation matrix 
    and the heatmap plot to the specified directory.

    inputs:
    normalized_df (pd.DataFrame): The dataframe containing normalized density metrics.
    density_metrics (list): List of column names corresponding to density metrics.
    plots_path (str): Directory to save the correlation matrix and the heatmap plot.

    outputs:
    pd.DataFrame: The correlation matrix dataframe.
    """
    # calculate the correlation matrix
    corr_matrix = normalized_df[density_metrics].corr()

    # save the correlation matrix as a CSV
    corr_matrix.to_csv(f'{plots_path}/correlation_matrix_norm_density_metrics.csv')

    # plot the heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)

    # adjust the x-axis labels: make them smaller and rotate them
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.title("Correlation Matrix of Density Metrics")

    # save the plot
    plt.tight_layout()  # Ensures labels are not cut off
    plt.savefig(f'{plots_path}/correlation_matrix_norm_density_metrics_heatmap.png')

    plt.close()

    return corr_matrix


def perform_pca_and_save_results(df, continuous_metrics, plots_path):
    """
    Performs PCA on the given continuous density metrics and saves the explained variance, 
    component loadings, and relevant plots to the specified directory. The function also
    includes the metrics used in the file names and plot titles for easier comparison.

    inputs:
    df (pd.DataFrame): The dataframe containing the data (e.g., normalized_df).
    continuous_metrics (list): List of continuous metrics for PCA (e.g., ['shannon_index', 'simpson_index', ...]).
    plots_path (str): Directory to save the PCA results and plots.

    outputs:
    pca (PCA object): Fitted PCA model.
    """
    # create a string of the metrics used for PCA for file names and titles
    metrics_str = "_".join(continuous_metrics)

    # extract the data for the continuous metrics
    X = df[continuous_metrics].copy()

    # perform PCA
    pca = PCA()
    pca.fit(X)

    # explained variance by each component
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # save explained variance and cumulative variance to a CSV
    variance_df = pd.DataFrame({
        'Principal Component': np.arange(1, len(explained_variance) + 1),
        'Explained Variance': explained_variance,
        'Cumulative Variance': cumulative_variance
    })
    variance_df.to_csv(f'{plots_path}/pca_explained_variance_{metrics_str}.csv', index=False)

    # plot the explained variance (Scree Plot)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
    plt.step(range(1, len(explained_variance) + 1), cumulative_variance, where='mid', color='red', label='Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'PCA - Explained Variance by Principal Component ({metrics_str})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plots_path}/pca_explained_variance_plot_{metrics_str}.png')
    plt.show()

    # save component loadings (how much each metric contributes to each principal component)
    loadings_df = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(len(explained_variance))], 
        index=continuous_metrics
    )
    loadings_df.to_csv(f'{plots_path}/pca_component_loadings_{metrics_str}.csv')

    # visualize the loadings as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(pca.components_, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(continuous_metrics)), [f'PC{i+1}' for i in range(len(explained_variance))], rotation=45)
    plt.yticks(range(len(continuous_metrics)), continuous_metrics)
    plt.title(f'PCA Component Loadings (Heatmap) ({metrics_str})')
    plt.tight_layout()
    plt.savefig(f'{plots_path}/pca_loadings_heatmap_{metrics_str}.png')
    plt.show()

    return pca


def run_pca_for_combinations(df, plots_path):
    combinations = {
        'not_including_intersection': ['shannon_index', 'simpson_index', 'median_height', 'avg_am_count_2021_norm', 'road_density_km'],
        'not_including_shannon_simpson': ['median_height', 'avg_am_count_2021_norm', 'road_density_km', 'intersection_density'],
        'not_including_intersection_shannon_simpson': ['median_height', 'avg_am_count_2021_norm', 'road_density_km']
    }

    for key, metrics in combinations.items():
        print(f"Running PCA for: {key} -> Metrics: {metrics}")
        perform_pca_and_save_results(df, metrics, plots_path)



def process_and_append_data_for_comparison(clean_data_path, company, year, demand_method, code, segment, eta_upper_bound, time_window_minutes, fraction, eta_calc):
    # initialize an empty Dask DataFrame to accumulate results
    big_dataframe = None
  
    for number in range(1, 13):  # from 01 to 02 $CHANGE BACK TO 13
        file_number = f'{number:02}'  # format number to two digits
       
        file_path = f'{clean_data_path}cleaned_{company}_tripdata_{year}_{file_number}.parquet'
        print(f"Processing file: {file_number}, {company}")

        # load the parquet file
        df = dd.read_parquet(file_path)
        #df = df.compute()
        print(f"File: {file_number}, loaded successfully!")
        print(f'The columns in the {company} dataframe are:', df.columns)
        print(f"The number of rows in the {company} dataframe is: {df.shape[0].compute()}")
        
        #keep omly the columns we need
        if company == 'Uber':
            df = df[['dropoff_datetime', 'request_datetime', 'pickup_datetime', 'on_scene_datetime','PULocationID', 'DOLocationID', 'PUBorough', 'DOBorough', 'shared_request_flag', 'trip_miles', 'trip_time', 'base_passenger_fare', 'congestion_surcharge']]
        else:    
            df = df[['dropoff_datetime', 'pickup_datetime', 'PULocationID', 'DOLocationID', 'PUBorough', 'DOBorough', 'trip_distance', 'fare_amount']]

        #df = df[['dropoff_datetime', 'request_datetime', 'pickup_datetime', 'on_scene_datetime','PULocationID', 'DOLocationID', 'PUBorough', 'DOBorough', 'shared_request_flag', 'trip_miles', 'trip_time', 'base_passenger_fare', 'congestion_surcharge']]

        # filter data for the specified borough code
        df = df[(df['PUBorough'] == code) & (df['DOBorough'] == code)]
        print(f"Data filtered for borough {code} successfully!")

       #sample data if necessary
        if fraction < 1:
            df = df.sample(frac=fraction, random_state=12)
            print("Sampled Data Succesfully")

        if company == 'Uber':
            #filter data for no shared rides
            df = df[df['shared_request_flag'] == 'N']
            print(f"Data filtered for no shared rides successfully!")

            df = df[['dropoff_datetime', 'request_datetime', 'pickup_datetime', 'on_scene_datetime', 'PULocationID', 'DOLocationID', 'PUBorough', 'DOBorough', 'trip_miles', 'trip_time', 'base_passenger_fare', 'congestion_surcharge']]

        df['dropoff_datetime'] = dd.to_datetime(df['dropoff_datetime'])
        df['pickup_datetime'] = dd.to_datetime(df['pickup_datetime'])
        print(f"Columns filtered successfully!")

        print(f"NAs in pickup_datetime are: {df['pickup_datetime'].isna().sum().compute()}")
       
        

        #create the timw windows
        if company == 'Uber':
            df = df.map_partitions(fill_na)
        
            print(f"NAs in pickup_datetime are: {df['pickup_datetime'].isna().sum().compute()}")
            df['request_datetime'] = dd.to_datetime(df['request_datetime'])
            df['on_scene_datetime'] = dd.to_datetime(df['on_scene_datetime'])
            print(f"NAs in on_scene_datetime are: {df['on_scene_datetime'].isna().sum().compute()}")
            print(f"NAs in on_scene_datetime after accounted for by pickuptime are: {df['on_scene_datetime'].isna().sum().compute()}")
            print(f"NAs in request_datetime are: {df['request_datetime'].isna().sum().compute()}")

            df['Request Time Window'] = df['request_datetime'].dt.floor(f'{time_window_minutes}min').astype('datetime64[ns]')
            print(df['Request Time Window'].head())
            #df['Dropoff Time Window'] = df['dropoff_datetime'].dt.floor(f'{time_window_minutes}min').astype('datetime64[ns]')
            print(f"Time windows created successfully!")
        
            df = calculate_requests(df, demand_method, time_window_minutes)
            print(f"Data processed for demand successfully!")

            #ETA calculation
            df['ETA'] = eta_create(df, eta_upper_bound, eta_calc)
            print("ETA calculated successfully!")
            
            #print('The mean ETA is:', df['ETA'].mean().compute())
            #zero_eta_count = (df['ETA'] == 0).sum().compute()
            #print(f"Number of rows with ETA equal to 0: {zero_eta_count}")
            # Ensure no zero or negative values for log transformation
            df = df[df['ETA'] > 0]
            #df = df[df['completed_rides'] > 0]
            print("Data filtered for log transformation successfully!")
        else:
            df['Request Time Window'] = df['pickup_datetime'].dt.floor(f'{time_window_minutes}min').astype('datetime64[ns]')
            print(df['Request Time Window'].head())

        #calculating open drivers
        #df = calculate_open_drivers(df, gamma, demand_adj, Dmax=None)

        # filter out weekends
        df['day_of_week'] = df['Request Time Window'].dt.dayofweek
        df = df[df['day_of_week'] < 5]
        print(f"Dataframe filtered for weekdays.")

        # classify time of day (using map_partitions to maintain Dask efficiency)
        df['Time of Day'] = df['Request Time Window'].map_partitions(lambda x: x.apply(classify_time_of_day))
        print(f"Time of day classified.")

        if segment != 'all':
            #filter for given time (of day) segment
            df = df[df['Time of Day'] == segment]
            print(f"Filtered for time of day: {segment}")

        # append the processed DataFrame to the big dataframe
        if big_dataframe is None:
            big_dataframe = df
        else:
            big_dataframe = dd.concat([big_dataframe, df])
            print(f"Data appended successfully!")

    return big_dataframe


def save_borough_observation_counts(df, company, plots_path):
    """
    Groups the DataFrame by 'PUBorough', counts the number of observations for each value,
    and saves the result as a CSV file.

    inputs:
    - df: The DataFrame containing the ride data.
    - company: A string ('Uber' or 'Taxi') to indicate which company the data is for.
    - plots_path: Path where the CSV file will be saved.
    """
    # ensure the output directory exists
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # group by 'PUBorough' and count the number of observations for each unique value
    borough_counts = df['PUBorough'].value_counts().reset_index()

    # rename the columns for clarity
    borough_counts.columns = ['PUBorough', 'Observation Count']

    # save the table to a CSV file with the company name in the filename
    output_file = f'{plots_path}/{company}_borough_observation_counts.csv'
    borough_counts.to_csv(output_file, index=False)

    print(f"Table saved as '{output_file}'.")



def plot_ride_data_on_map(taxi_zones_gdf, df, company, plots_path, data_type):
    """
    Plots ride volume or revenue data on a map for a specified company and saves both the raw
    and proportion plots.

    inputs:
    - taxi_zones_gdf: GeoDataFrame containing taxi zones with 'LocationID' and 'geometry'.
    - df: DataFrame containing rides data for the specified company.
    - company: 'Uber' or 'Taxi' indicating which company's data is provided.
    - plots_path: Path to save the plot.
    - data_type: 'ride_volume' or 'revenue' indicating which data to plot.
    """
    # validate input parameters
    if company not in ['Uber', 'Taxi']:
        raise ValueError("Invalid company. Expected 'Uber' or 'Taxi'.")
    if data_type not in ['ride_volume', 'revenue']:
        raise ValueError("Invalid data_type. Expected 'ride_volume' or 'revenue'.")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    
    # set company_name for labeling
    company_name = company  # Since company is already 'Uber' or 'Taxi'
    
    # ensure 'PULocationID' is present
    if 'PULocationID' not in df.columns:
        raise KeyError("'PULocationID' not found in the DataFrame.")
    
    # determine the appropriate revenue column for Uber or Taxi
    if data_type == 'revenue':
        if company == 'Uber':
            revenue_column = 'base_passenger_fare'
            if revenue_column not in df.columns:
                raise KeyError(f"'{revenue_column}' not found in Uber DataFrame.")
        elif company == 'Taxi':
            revenue_column = 'fare_amount'
            if revenue_column not in df.columns:
                raise KeyError(f"'{revenue_column}' not found in Taxi DataFrame.")
    
    # calculate raw ride volume or revenue per location
    if data_type == 'ride_volume':
        ride_data = df.groupby('PULocationID').size().reset_index(name=f'{company_name} Ride Volume')
        data_column_raw = f'{company_name} Ride Volume'
    elif data_type == 'revenue':
        # calculate total revenue using the appropriate column
        ride_data = df.groupby('PULocationID')[revenue_column].sum().reset_index(name=f'{company_name} Total Revenue')
        data_column_raw = f'{company_name} Total Revenue'
    
    # merge with taxi_zones_gdf
    merged_gdf = taxi_zones_gdf.merge(ride_data, left_on='LocationID', right_on='PULocationID', how='left')
    merged_gdf[data_column_raw] = merged_gdf[data_column_raw].fillna(0)
    
    # --------- Raw Plot ---------
    # plotting the raw data
    plt.figure(figsize=(12, 10))
    merged_gdf.plot(column=data_column_raw, cmap='OrRd', linewidth=0.8, edgecolor='0.8', legend=True)
    plt.title(f'{company_name} {data_type.replace("_", " ").title()} by Taxi Zone (Raw)', fontsize=15)
    plt.axis('off')

    # save the raw plot
    raw_filename = f'raw_{company_name}_{data_type}.png'
    raw_filepath = os.path.join(plots_path, raw_filename)
    plt.savefig(raw_filepath, bbox_inches='tight')
    plt.close()
    
    print(f"Raw plot saved to {raw_filepath}")

    # --------- Proportion Plot ---------
    # calculate total ride volume or revenue for the company
    total_metric = merged_gdf[data_column_raw].sum()
    
    # avoid division by zero
    if total_metric > 0:
        merged_gdf[f'{company_name} Proportion'] = merged_gdf[data_column_raw] / total_metric
    else:
        merged_gdf[f'{company_name} Proportion'] = 0
    
    # plotting the proportional data
    plt.figure(figsize=(12, 10))
    merged_gdf.plot(column=f'{company_name} Proportion', cmap='OrRd', linewidth=0.8, edgecolor='0.8', legend=True)
    plt.title(f'{company_name} {data_type.replace("_", " ").title()} Proportion by Taxi Zone', fontsize=15)
    plt.axis('off')

    # save the proportion plot
    proportion_filename = f'proportion_{company_name}_{data_type}.png'
    proportion_filepath = os.path.join(plots_path, proportion_filename)
    plt.savefig(proportion_filepath, bbox_inches='tight')
    plt.close()

    print(f"Proportion plot saved to {proportion_filepath}")


def plot_uber_taxi_comparison_map(taxi_zones_gdf, uber_df, taxi_df, plots_path, data_type):
    """
    Plots a map showing the proportion of Uber to total (Uber + Taxi) metric for each location.

    inputs:
    - taxi_zones_gdf: GeoDataFrame containing taxi zones with 'LocationID' and 'geometry'.
    - uber_df: DataFrame containing Uber rides data.
    - taxi_df: DataFrame containing Taxi rides data.
    - plots_path: Path to save the plot.
    - data_type: 'ride_volume' or 'revenue' indicating which metric to compare.
    """
    # validate input parameters
    data_type = data_type.lower()
    if data_type not in ['ride_volume', 'revenue']:
        raise ValueError("Invalid data_type. Expected 'ride_volume' or 'revenue'.")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # ensure 'PULocationID' is present
    if 'PULocationID' not in uber_df.columns or 'PULocationID' not in taxi_df.columns:
        raise KeyError("'PULocationID' not found in one of the DataFrames.")

    # ensure 'base_passenger_fare' and 'fare_amount' is present for revenue calculation
    if data_type == 'revenue':
        if 'base_passenger_fare' not in uber_df.columns or 'fare_amount' not in taxi_df.columns:
            raise KeyError("'base_passenger_fare' or 'fare_amount not found in one of the DataFrames.")

    # calculate the metric per location for Uber and Taxi
    if data_type == 'ride_volume':
        # Uber ride volume per location
        uber_metric = uber_df.groupby('PULocationID').size().reset_index(name='Uber Metric')
        # Taxi ride volume per location
        taxi_metric = taxi_df.groupby('PULocationID').size().reset_index(name='Taxi Metric')
    elif data_type == 'revenue':
        # Uber revenue per location
        uber_metric = uber_df.groupby('PULocationID')['base_passenger_fare'].sum().reset_index(name='Uber Metric')
        # Taxi revenue per location
        taxi_metric = taxi_df.groupby('PULocationID')['fare_amount'].sum().reset_index(name='Taxi Metric')

    # merge Uber and Taxi metrics
    merged_metrics = pd.merge(uber_metric, taxi_metric, on='PULocationID', how='outer').fillna(0)

    # calculate the Uber proportion
    merged_metrics['Total Metric'] = merged_metrics['Uber Metric'] + merged_metrics['Taxi Metric']
    # avoid division by zero
    merged_metrics = merged_metrics[merged_metrics['Total Metric'] > 0]
    merged_metrics['Uber Proportion'] = merged_metrics['Uber Metric'] / merged_metrics['Total Metric']

    # merge with taxi_zones_gdf
    merged_gdf = taxi_zones_gdf.merge(merged_metrics, left_on='LocationID', right_on='PULocationID', how='left')
    merged_gdf['Uber Proportion'] = merged_gdf['Uber Proportion'].fillna(0)

     # plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    merged_gdf.plot(
        column='Uber Proportion',
        cmap='coolwarm',
        linewidth=0.8,
        edgecolor='0.8',
        legend=False,
        ax=ax
    )
    ax.set_title(f'Uber Proportion of Total {data_type.replace("_", " ").title()} by Taxi Zone', fontsize=15)
    ax.axis('off')

    # create colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Uber Proportion', rotation=270, labelpad=15)

    # save the plot
    filename = f'Uber_Taxi_{data_type}_comparison.png'
    filepath = os.path.join(plots_path, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()



def create_ride_rankings(big_uber_df, big_taxi_df, plots_path):
    # group by PULocationID and count the number of rides for each location for Uber and Taxi
    uber_rides_by_location = big_uber_df.groupby(['PULocationID', 'PUBorough']).size().reset_index(name='ride_count')
    print(f"The unique values in PULocationID for Uber are: {big_uber_df['PULocationID'].nunique()}")
    taxi_rides_by_location = big_taxi_df.groupby(['PULocationID', 'PUBorough']).size().reset_index(name='ride_count')
    print(f"The unique values in PULocationID for Taxi are: {big_taxi_df['PULocationID'].nunique()}")
    
    # rank locations by number of rides (descending order) separately for Uber and Taxi
    uber_rides_by_location['uber_rank'] = uber_rides_by_location['ride_count'].rank(ascending=False, method='min')
    taxi_rides_by_location['taxi_rank'] = taxi_rides_by_location['ride_count'].rank(ascending=False, method='min')
    
    # merge the Uber and Taxi rankings on PULocationID
    rankings = pd.merge(uber_rides_by_location[['PULocationID', 'PUBorough', 'uber_rank']],
                        taxi_rides_by_location[['PULocationID', 'taxi_rank']],
                        on='PULocationID', how='outer')
    
    # drop rows without a taxi or Uber rank 
    rankings.dropna(subset=['uber_rank', 'taxi_rank'], inplace=True)

    # define a color palette for boroughs
    borough_palette = sns.color_palette("hsv", len(rankings['PUBorough'].unique()))
    borough_colors = {borough: borough_palette[i] for i, borough in enumerate(rankings['PUBorough'].unique())}
    
    # create the scatter plot with color coding by PUBorough
    plt.figure(figsize=(10, 6))
    
     # get the min and max values for the axes
    min_val = min(rankings[['uber_rank', 'taxi_rank']].min())
    max_val = max(rankings[['uber_rank', 'taxi_rank']].max())

    # plot each borough with its assigned color
    for borough in rankings['PUBorough'].unique():
        subset = rankings[rankings['PUBorough'] == borough]
        plt.scatter(subset['uber_rank'], subset['taxi_rank'], label=borough, color=borough_colors[borough], alpha=0.5)

    # add the 45-degree line
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label='45-degree line')

    # add labels and title
    plt.xlabel('Uber Rankings')
    plt.ylabel('Taxi Rankings')
    plt.title('Uber vs Taxi Ride Rankings by Location (Color-Coded by Borough)')
    
    # add a legend explaining the color coding
    plt.legend(title='Borough', loc='best')

    # save the plot to the specified path
    plt.savefig(f"{plots_path}/uber_vs_taxi_ride_rankings_colored.png")
    plt.close()

    # return the dataframes of rankings for further analysis if needed
    return rankings


def combine_normalized_and_cdf_density_metrics(reg_results, continuous_metrics, discrete_metrics, results_path):
    """
    Combines the normalized discrete metrics and the empirical CDF of the continuous metrics.
    
    1) The continuous metrics are processed using the empirical CDF.
    2) The discrete metrics are normalized using (0-1 scaling based on max value).
    3) The function leaves binary metrics unchanged.
    
    Additionally, changes the sign of the 'alpha' column (from negative to positive).
    
    The resulting combined dataframe is saved to the specified directory.

    inputs:
    reg_results (pd.DataFrame): The dataframe containing regression results and density metrics.
    continuous_metrics (list): List of column names corresponding to continuous density metrics.
    discrete_metrics (list): List of column names corresponding to discrete density metrics.
    reg_results_path (str): Directory to save the resulting combined dataframe.

    outputs:
    pd.DataFrame: The combined dataframe with empirical CDF for continuous metrics and normalized values for discrete metrics.
    """
    # ensure 'PULocationID' is included in the filtered dataframe
    # check if 'alpha' column exists
    if 'alpha' in reg_results.columns:
        columns_to_keep = ['PULocationID', 'borough_code'] + continuous_metrics + discrete_metrics + ['alpha']
    else:
        columns_to_keep = ['PULocationID', 'borough_code'] + continuous_metrics + discrete_metrics

    # create a filtered DataFrame that only keeps the necessary columns
    filtered_reg_results = reg_results[columns_to_keep].copy()

    # specify binary metrics that should remain unchanged (assuming they are part of discrete metrics)
    binary_metrics = ['has_count_stations', 'has_wifi_hotspot']
    binary_metrics_included = [col for col in binary_metrics if col in filtered_reg_results.columns]

    # continuous Metrics: Apply empirical CDF processing (rank with pct)
    combined_df = filtered_reg_results.copy()
    for metric in continuous_metrics:
        combined_df[metric] = filtered_reg_results[metric].rank(method='max', pct=True)

    # discrete Metrics: Apply normalization (0-1 scaling based on max value), excluding binary metrics
    non_binary_discrete_metrics = [col for col in discrete_metrics if col not in binary_metrics_included]
    combined_df[non_binary_discrete_metrics] = filtered_reg_results[non_binary_discrete_metrics].div(
        filtered_reg_results[non_binary_discrete_metrics].max(axis=0), axis=1
    )

    # binary metrics remain unchanged

    # if 'alpha' exists, change the sign
    if 'alpha' in combined_df.columns:
        combined_df['alpha'] = -filtered_reg_results['alpha']
    
    # save the combined dataframe to the specified path
    combined_df.to_csv(f'{results_path}/combined_norm_density_metrics.csv', index=False)
    
    return combined_df


def create_borough_summary_table(combined_df, density_metrics, plots_path):
    """
    Creates a summary table that reports the average alpha and average density score for each borough.
    
    inputs:
    combined_df (pd.DataFrame): DataFrame containing combined density metrics and alpha values.
    density_metrics (list): List of density metrics used for the average density score calculation.
    plots_path (str): Path to save the resulting summary table.
    
    outputs:
    pd.DataFrame: A summary DataFrame with average alpha and average density score per borough.
    """
    # calculate the average alpha for each borough
    avg_alpha_by_borough = combined_df.groupby('borough_code')['alpha'].mean().reset_index(name='average_alpha')

    # calculate the average density score for each location (PULocationID)
    combined_df['average_density_score'] = combined_df[density_metrics].mean(axis=1)

    # calculate the average of the average density scores for each borough
    avg_density_score_by_borough = combined_df.groupby('borough_code')['average_density_score'].mean().reset_index(name='average_density_score')

    # merge the results into a summary table
    borough_summary = pd.merge(avg_alpha_by_borough, avg_density_score_by_borough, on='borough_code')

    # save the summary table to the specified path
    summary_table_path = f"{plots_path}/borough_summary_table.csv"
    borough_summary.to_csv(summary_table_path, index=False)

    print(f"Summary table saved to {summary_table_path}")
    
    return borough_summary

def plot_ride_volume_vs_density(big_uber_df, big_taxi_df, combined_df, density_metrics, plots_path):
    """
    Plots the relationship between ride volume and density metrics for Uber and taxis by borough.
    """
    # group by PULocationID and count rides for Uber and Taxi
    uber_ride_counts = big_uber_df.groupby(['PULocationID', 'PUBorough']).size().reset_index(name='ride_count')
    taxi_ride_counts = big_taxi_df.groupby(['PULocationID', 'PUBorough']).size().reset_index(name='ride_count')

    # add service type to both Uber and Taxi DataFrames
    uber_ride_counts['service_type'] = 'Uber'
    taxi_ride_counts['service_type'] = 'Taxi'

    # merge Uber and Taxi ride counts with density scores
    uber_merged_df = pd.merge(uber_ride_counts, combined_df[['PULocationID', 'borough_code'] + density_metrics], on='PULocationID', how='left')
    taxi_merged_df = pd.merge(taxi_ride_counts, combined_df[['PULocationID', 'borough_code'] + density_metrics], on='PULocationID', how='left')

    # combine Uber and Taxi DataFrames
    final_merged_df = pd.concat([uber_merged_df, taxi_merged_df], ignore_index=True)

    # calculate average density score
    final_merged_df['average_density_score'] = final_merged_df[density_metrics].mean(axis=1)
    
    # plot for Uber and Taxi
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=final_merged_df, x='average_density_score', y='ride_count', hue='PUBorough', style='service_type')
    
    # add labels and title
    plt.title('Ride Volume vs Average Density Score (Uber vs Taxi)')
    plt.ylabel('Ride Volume')
    plt.xlabel('Average Density Score')
    
    # save and show the plot
    plt.tight_layout()
    plt.savefig(f'{plots_path}/ride_volume_vs_density.png')
    plt.show()


def plot_alpha_vs_density(combined_df, density_metrics, plots_path):
    """
    Plots the relationship between alpha coefficients and average density score.
    
    inputs:
    combined_df (pd.DataFrame): DataFrame containing alpha values, density metrics, and other relevant information.
    density_metrics (list): List of density metrics to be averaged.
    plots_path (str): Path to save the resulting plot.
    """
    # calculate average density score for each location
    combined_df['average_density_score'] = combined_df[density_metrics].mean(axis=1)

    # plot the relationship between alpha and average density score
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=combined_df, x='average_density_score', y='alpha', hue='borough_code')

    # add labels and title
    plt.title('Alpha Coefficients vs Average Density Score')
    plt.ylabel('Alpha Coefficient (ETA Elasticity)')
    plt.xlabel('Average Density Score')

    # save and show the plot
    plt.tight_layout()
    plt.savefig(f'{plots_path}/alpha_vs_density.png')
    plt.show()


def classify_average_based(df, density_metrics, plots_path, table_used, threshold):
    """
    Classifies each row as 'dense' or 'not_dense' based on the average of the density metrics and a given threshold.

    inputs:
    df (pd.DataFrame): The dataframe containing density metrics.
    density_metrics (list): List of column names corresponding to density metrics.
    threshold (float): The threshold for the average value to classify as 'dense'.
    column_prefix (str): Prefix for the classification column name (default: 'average_classification').

    outputs:
    pd.DataFrame: The original dataframe with a new classification column.
    """
    # calculate the average of the density metrics for each row
    avg_density = df[density_metrics].mean(axis=1)

    # create a dynamic column name for the classification based on the threshold
    column_name = f'{threshold}_average_classification'

    # classify as 'dense' if the average of the metrics is greater than or equal to the threshold
    df[column_name] = avg_density.apply(lambda x: 'more dense' if x >= threshold else 'less dense')

    #plot_alpha_vs_column(df, column_name, plots_path, f'alpha_vs_{column_name}_{table_used}.png')

    return df

def plot_uber_vs_taxi_rankings_with_density(combined_df, rankings, density_metrics, plots_path, ranking_type='ride'):
    """
    Plots Uber vs Taxi rankings color-coded by density classification based on a set of density metrics.
    
    inputs:
    combined_df (pd.DataFrame): DataFrame containing combined density metrics and regression results.
    rankings (pd.DataFrame): DataFrame containing either ride or revenue rankings for Uber and Taxi.
    density_metrics (list): List of density metrics used for classification.
    plots_path (str): Path to save the resulting plots.
    ranking_type (str): Type of ranking ('ride' or 'revenue'). Used for plot titles and file names.
    """
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        # classify zones as dense or not dense based on the current threshold
        classified_df = classify_average_based(combined_df, density_metrics, plots_path, table_used='combined', threshold=threshold)

        # merge classification results with rankings
        merged_df = pd.merge(rankings, classified_df[['PULocationID', f'{threshold}_average_classification']], 
                             on='PULocationID', how='left')

        # create marker styles for dense vs not dense classification
        classification_styles = {
            'more dense': {'marker': 'o', 'facecolors': 'blue', 'edgecolors': 'blue'},  # Solid blue circle
            'less dense': {'marker': 'o', 'facecolors': 'none', 'edgecolors': 'blue'}  # Open blue circle
        }
    
        # create the scatter plot for Uber vs Taxi rankings color-coded by density classification
        plt.figure(figsize=(10, 6))
        for classification, style in classification_styles.items():
            subset = merged_df[merged_df[f'{threshold}_average_classification'] == classification]
            plt.scatter(subset['uber_rank'], subset['taxi_rank'], 
                        label=classification, **style, alpha=0.5)

        # add a 45-degree dashed line
        max_rank = max(merged_df['uber_rank'].max(), merged_df['taxi_rank'].max())
        plt.plot([0, max_rank], [0, max_rank], linestyle='--', color='gray')

        # add labels and title based on ranking type
        if ranking_type == 'ride':
            plot_title = f'Uber vs Taxi Ride Rankings by Location (Classified by Density, Threshold = {threshold})'
            file_name = f"{plots_path}/uber_vs_taxi_ride_rankings_density_classification_threshold_{threshold}.png"
        elif ranking_type == 'revenue':
            plot_title = f'Uber vs Taxi Revenue Rankings by Location (Classified by Density, Threshold = {threshold})'
            file_name = f"{plots_path}/uber_vs_taxi_revenue_rankings_density_classification_threshold_{threshold}.png"
        else:
            raise ValueError("Invalid ranking_type. Choose 'ride' or 'revenue'.")

        # add labels and title
        plt.xlabel('Uber Rankings')
        plt.ylabel('Taxi Rankings')
        plt.title(plot_title)
        
        # add a legend explaining the color coding
        plt.legend(title=f'{threshold} Density Classification', loc='best')

        # save the plot to the specified path
        plt.savefig(file_name)
        plt.close()


def classify_alpha_based_on_percentile(df, threshold):
    """
    Classifies each zone as 'dense' or 'not_dense' based on the alpha percentile rank.
    
    inputs:
    df (pd.DataFrame): DataFrame containing the alpha values.
    threshold (float): The percentile threshold to classify zones as dense or not dense.

    outputs:
    pd.DataFrame: The DataFrame with a new column 'alpha_threshold_classification'.
    """
    # create a dynamic column name for the classification based on the threshold
    column_name = f'alpha_percentile_threshold_{threshold}_classification'

    # calculate the percentile rank of each alpha value
    df['alpha_percentile'] = df['alpha'].rank(pct=True)
    
    # classify as 'dense' if the alpha percentile rank is greater than or equal to the threshold
    df[column_name] = df['alpha_percentile'].apply(lambda x: 'more dense' if x >= threshold else 'less dense')
    
    return df

def plot_uber_vs_taxi_rankings_with_alpha_threshold_classification(rankings, reg_results, thresholds, plots_path, ranking_type='ride'):
    """
    Plots Uber vs Taxi rankings color-coded by alpha classification based on thresholds (with positive alphas).
    
    inputs:
    rankings (pd.DataFrame): DataFrame containing the Uber and taxi rankings (ride or revenue).
    reg_results (pd.DataFrame): DataFrame containing alpha values and classifications.
    thresholds (list of float): List of thresholds to classify alpha values.
    plots_path (str): Path to save the resulting plot.
    ranking_type (str): Type of rankings, either 'ride' or 'revenue'. Used in plot title and file name.
    """
    # flip the alpha values to positive
    reg_results['alpha'] = -reg_results['alpha']  # Flip the sign of alpha

    for threshold in thresholds:
        # classify zones as dense or not dense based on the alpha percentile threshold
        classified_df = classify_alpha_based_on_percentile(reg_results, threshold)

        # merge classification results with rankings
        column_name = f'alpha_percentile_threshold_{threshold}_classification'
        merged_df = pd.merge(rankings, classified_df[['PULocationID', column_name]], on='PULocationID', how='left')
        
        # create color map for alpha classification
        classification_colors = {'more dense': 'blue', 'less dense': 'red'}
        
        # create the scatter plot for Uber vs Taxi rankings color-coded by alpha classification
        plt.figure(figsize=(10, 6))
        for classification in ['more dense', 'less dense']:
            subset = merged_df[merged_df[column_name] == classification]
            plt.scatter(subset['uber_rank'], subset['taxi_rank'], 
                        label=classification, color=classification_colors[classification], alpha=0.5)

        # add labels and title
        plt.xlabel('Uber Rankings')
        plt.ylabel('Taxi Rankings')
        plt.title(f'Uber vs Taxi {ranking_type.capitalize()} Rankings by Location (Alpha Percentile Threshold = {threshold})')
        
        # add a legend explaining the color coding
        plt.legend(title=f'Alpha Classification (Threshold = {threshold})', loc='best')

        # save the plot to the specified path, with filename indicating the ranking type
        plt.savefig(f"{plots_path}/uber_vs_taxi_{ranking_type}_rankings_alpha_classification_threshold_{threshold}_percentile.png")
        plt.close()

def plot_combined_density_and_alpha_classification(
        combined_df, reg_results, rankings, density_metrics, 
        density_thresholds, alpha_thresholds, plots_path, ranking_type='ride'):
    """
    Plots Uber vs Taxi rankings color-coded by both density metrics classification and alpha threshold classification.
    Points with agreement (classified as dense or not dense by both methods) will be shown as squares.
    
    inputs:
    combined_df (pd.DataFrame): DataFrame containing combined density metrics and regression results.
    reg_results (pd.DataFrame): DataFrame containing alpha values and classifications.
    rankings (pd.DataFrame): DataFrame containing either ride or revenue rankings for Uber and Taxi.
    density_metrics (list): List of density metrics used for classification.
    density_thresholds (list of float): List of density thresholds to classify density values.
    alpha_thresholds (list of float): List of thresholds to classify alpha values.
    plots_path (str): Path to save the resulting plots.
    ranking_type (str): Type of ranking ('ride' or 'revenue'). Used for plot titles and file names.
    """
    
    # loop over density and alpha thresholds
    for density_threshold in density_thresholds:
        for alpha_threshold in alpha_thresholds:
            
            # classify zones based on density metrics
            classified_density_df = classify_average_based(
                combined_df, density_metrics, plots_path, 
                table_used='combined', threshold=density_threshold
            )
            
            # classify zones based on alpha thresholds
            classified_alpha_df = classify_alpha_based_on_percentile(
                reg_results, alpha_threshold
            )

            # merge density and alpha classifications with rankings
            merged_df = pd.merge(
                rankings, 
                classified_density_df[['PULocationID', f'{density_threshold}_average_classification']], 
                on='PULocationID', how='left'
            )
            merged_df = pd.merge(
                merged_df, 
                classified_alpha_df[['PULocationID', f'alpha_percentile_threshold_{alpha_threshold}_classification']], 
                on='PULocationID', how='left'
            )
            
            # create marker styles for density classification
            classification_styles = {
                'more dense': {'marker': 'o', 'facecolors': 'blue', 'edgecolors': 'blue'},  # Solid blue circle
                'less dense': {'marker': 'o', 'facecolors': 'none', 'edgecolors': 'blue'}  # Open blue circle
            }
            
            # define marker styles for agreement cases
            agreement_styles = {
                'more dense': {'marker': 's', 'facecolors': 'blue', 'edgecolors': 'blue'},  # Solid blue square
                'less dense': {'marker': 's', 'facecolors': 'none', 'edgecolors': 'blue'}  # Open blue square
            }

            # clear previous figure and legend for a fresh start
            plt.figure(figsize=(10, 6))

            # loop through the classifications and plot the points
            for classification, style in classification_styles.items():
                subset = merged_df[merged_df[f'{density_threshold}_average_classification'] == classification]
                
                # identify points that are classified the same under both methods
                agreement = (subset[f'alpha_percentile_threshold_{alpha_threshold}_classification'] == classification)
                
                # define custom labels
                if classification == 'more dense':
                    label_agree = 'high density - high alpha'
                    label_disagree = 'high density - low alpha'
                else:
                    label_agree = 'low density - low alpha'
                    label_disagree = 'low density - high alpha'
                
                # plot points with agreement (squares)
                plt.scatter(
                    subset.loc[agreement, 'uber_rank'], 
                    subset.loc[agreement, 'taxi_rank'], 
                    label=label_agree, 
                    **agreement_styles[classification], alpha=0.8
                )
                
                # plot regular points without agreement (circles)
                plt.scatter(
                    subset.loc[~agreement, 'uber_rank'], 
                    subset.loc[~agreement, 'taxi_rank'], 
                    label=label_disagree, 
                    **style, alpha=0.5
                )
            
            # add 45-degree line
            min_val = min(merged_df[['uber_rank', 'taxi_rank']].min())
            max_val = max(merged_df[['uber_rank', 'taxi_rank']].max())
            plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label='45-degree line')

            # add labels and title
            plt.xlabel('Uber Rankings')
            plt.ylabel('Taxi Rankings')
            plot_title = (
                f'Uber vs Taxi {ranking_type.capitalize()} Rankings (Density Threshold = {density_threshold}, '
                f'Alpha Threshold = {alpha_threshold})'
            )
            plt.title(plot_title)

            # clear previous legends to avoid residual labels
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles, labels, title='Classification', loc='best')
            
            # save the plot to the specified path
            file_name = f"{plots_path}/uber_vs_taxi_{ranking_type}_rankings_density_alpha_classification_" \
                        f"density_{density_threshold}_alpha_{alpha_threshold}.png"
            plt.savefig(file_name)
            plt.close()


def plot_combined_density_and_alpha_classification_with_ids(
        combined_df, reg_results, rankings, density_metrics, 
        density_thresholds, alpha_thresholds, plots_path, ranking_type='ride'):
    """
    Plots Uber vs Taxi rankings color-coded by both density metrics classification and alpha threshold classification.
    Points with agreement (classified as dense or not dense by both methods) will be shown as squares.
    Also retrieves the PULocationIDs classified as 'more dense' in both classifications and their rankings.
    
    inputs:
    combined_df (pd.DataFrame): DataFrame containing combined density metrics and regression results.
    reg_results (pd.DataFrame): DataFrame containing alpha values and classifications.
    rankings (pd.DataFrame): DataFrame containing either ride or revenue rankings for Uber and Taxi.
    density_metrics (list): List of density metrics used for classification.
    density_thresholds (list of float): List of density thresholds to classify density values.
    alpha_thresholds (list of float): List of thresholds to classify alpha values.
    plots_path (str): Path to save the resulting plots.
    ranking_type (str): Type of ranking ('ride' or 'revenue'). Used for plot titles and file names.
    
    outputs:
    pd.DataFrame: DataFrame with PULocationIDs classified as 'more dense' under both methods, along with their rankings.
    """
    # initialize list to capture agreement PULocationIDs for specified thresholds
    agreement_dense_ids = []

    # loop over density and alpha thresholds
    for density_threshold in density_thresholds:
        for alpha_threshold in alpha_thresholds:
            
            # classify zones based on density metrics
            classified_density_df = classify_average_based(
                combined_df, density_metrics, plots_path, 
                table_used='combined', threshold=density_threshold
            )
            
            # classify zones based on alpha thresholds
            classified_alpha_df = classify_alpha_based_on_percentile(
                reg_results, alpha_threshold
            )

            # merge density and alpha classifications with rankings
            merged_df = pd.merge(
                rankings, 
                classified_density_df[['PULocationID', f'{density_threshold}_average_classification']], 
                on='PULocationID', how='left'
            )
            merged_df = pd.merge(
                merged_df, 
                classified_alpha_df[['PULocationID', f'alpha_percentile_threshold_{alpha_threshold}_classification']], 
                on='PULocationID', how='left'
            )
            
            # extract PULocationIDs classified as 'more dense' under both methods for specific thresholds
            if density_threshold == 0.6 and alpha_threshold == 0.5:
                agreement = merged_df[
                    (merged_df[f'{density_threshold}_average_classification'] == 'more dense') & 
                    (merged_df[f'alpha_percentile_threshold_{alpha_threshold}_classification'] == 'more dense')
                ]
                agreement_dense_ids = agreement['PULocationID'].tolist()

            # create marker styles for density classification
            classification_styles = {
                'more dense': {'marker': 'o', 'facecolors': 'blue', 'edgecolors': 'blue'},  # Solid blue circle
                'less dense': {'marker': 'o', 'facecolors': 'none', 'edgecolors': 'blue'}  # Open blue circle
            }
            
            # define marker styles for agreement cases
            agreement_styles = {
                'more dense': {'marker': 's', 'facecolors': 'blue', 'edgecolors': 'blue'},  # Solid blue square
                'less dense': {'marker': 's', 'facecolors': 'none', 'edgecolors': 'blue'}  # Open blue square
            }

            plt.figure(figsize=(10, 6))

            # loop through the classifications and plot the points
            for classification, style in classification_styles.items():
                subset = merged_df[merged_df[f'{density_threshold}_average_classification'] == classification]
                
                # identify points that are classified the same under both methods
                agreement = (subset[f'alpha_percentile_threshold_{alpha_threshold}_classification'] == classification)
                
                # plot points with agreement (squares)
                plt.scatter(
                    subset.loc[agreement, 'uber_rank'], 
                    subset.loc[agreement, 'taxi_rank'], 
                    label=f'{classification} (agree)', 
                    **agreement_styles[classification], alpha=0.8
                )
                
                # plot regular points without agreement (circles)
                plt.scatter(
                    subset.loc[~agreement, 'uber_rank'], 
                    subset.loc[~agreement, 'taxi_rank'], 
                    label=f'{classification}', 
                    **style, alpha=0.5
                )
            
            # add 45-degree line
            min_val = min(merged_df[['uber_rank', 'taxi_rank']].min())
            max_val = max(merged_df[['uber_rank', 'taxi_rank']].max())
            plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label='45-degree line')

            # add labels and title
            plt.xlabel('Uber Rankings')
            plt.ylabel('Taxi Rankings')
            plot_title = (
                f'Uber vs Taxi {ranking_type.capitalize()} Rankings (Density Threshold = {density_threshold}, '
                f'Alpha Threshold = {alpha_threshold})'
            )
            plt.title(plot_title)

            # add legend
            plt.legend(title='Classification', loc='best')
            
            # save the plot to the specified path
            file_name = f"{plots_path}/uber_vs_taxi_{ranking_type}_rankings_density_alpha_classification_" \
                        f"density_{density_threshold}_alpha_{alpha_threshold}_extra.png"
            plt.savefig(file_name)
            plt.close()
    
    # filter the rankings DataFrame for the agreement IDs and sort by Uber and Taxi ranks
    agreement_dense_rankings = rankings[rankings['PULocationID'].isin(agreement_dense_ids)].copy()
    agreement_dense_rankings = agreement_dense_rankings[['PULocationID', 'uber_rank', 'taxi_rank']]
    
    return agreement_dense_rankings