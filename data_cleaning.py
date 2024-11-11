#!/usr/bin/env python3

#general
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
from math import sqrt 
import numpy as np 
from datetime import datetime, timedelta
import sys
import pandas as pd

import io
from dask.distributed import Client, LocalCluster

import dask.dataframe as dd

#geospatial 
import geopandas as gpd

# Import cleaning functions
from required_functions import optimize_dataframe, clean_data, clean_data_taxi, calculate_mean_distances, prepare_ride_counts, calculate_daily_averages

def main():
    #fetch working directory
    current_directory = os.getcwd()

    # fetch SLURM environment variables
    ntasks = int(os.getenv('SLURM_NTASKS', '1'))  # Default to 1 if not set
    cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK', '1'))  # Default to 1 if not set

    # configure the Dask Client
    cluster = LocalCluster(n_workers=ntasks, threads_per_worker=cpus_per_task)
    client = Client(cluster)

    # read the paths from command-line arguments
    month = sys.argv[1]
    year = sys.argv[2]
    company = sys.argv[3]
    taxi_zones_path = sys.argv[4]
    #plots_path = sys.argv[4]


    # ensure the directories exist
    #os.makedirs(plots_path, exist_ok=True)
    os.makedirs(taxi_zones_path, exist_ok=True)

    # load Taxi Zones GeoDataFrame
    taxi_zones = gpd.read_file(f"{taxi_zones_path}/taxi_zones.shp") 
    print("Taxi Zones Loaded Successfully!")

    files_directory =  f'{current_directory}/data_{year}_{company}'
    os.makedirs(files_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists

    # path where the parquet file will be stored
    clean_data_directory = f'{files_directory}/cleaned_rides_data_{year}_{company}'
    os.makedirs(clean_data_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists

    aggregate_data_directory = f'{files_directory}/aggregate_data_{year}_{company}'
    os.makedirs(aggregate_data_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    # clean data
    if company == 'Taxi':
        #define the file path
        path_to_file = f'{current_directory}/{year}_taxi/yellow_tripdata_{year}-{month}.parquet'

        # read the Parquet file using pandas
        df = pd.read_parquet(path_to_file)

        # optimize dataframe
        df = optimize_dataframe(df)
        print("DataFrame info:")
        print(df.dtypes)

        df_cleaned = clean_data_taxi(df, taxi_zones)
    elif company == 'Green_Taxi':
        #define the file path
        path_to_file = f'{current_directory}/{year}_greentaxi/green_tripdata_{year}-{month}.parquet'

        # read the Parquet file using pandas
        df = pd.read_parquet(path_to_file)

        # optimize dataframe
        df = optimize_dataframe(df)
        print("DataFrame info:")
        print(df.dtypes)
        
        df_cleaned = clean_data_taxi(df, taxi_zones)

    else:
         #define the file path
        path_to_file = f'{current_directory}/{year}_fhvhv/fhvhv_tripdata_{year}-{month}.parquet'
        
        # read the Parquet file using pandas
        df = pd.read_parquet(path_to_file)

        # optimize dataframe
        df = optimize_dataframe(df)
        print("DataFrame info:")
        print(df.dtypes)
        df_cleaned = clean_data(df, taxi_zones, company)

    # export the cleaned data to a new parquet file
    cleaned_data_path = f'{clean_data_directory}/cleaned_{company}_tripdata_{year}_{month}.parquet'  
    df_cleaned.to_parquet(cleaned_data_path)
    print(f'Month {month}, {company} cleaned data saved successfully!')
    print(f"The column names of the cleaned data are: {df_cleaned.columns}")

    #mean distances
    '''
    mean_distances = calculate_mean_distances(df_cleaned[['PUBorough', 'DOBorough', 'trip_miles']])

    mean_distances_directory = f'{aggregate_data_directory}/mean_distances_{year}_{company}'
    os.makedirs(mean_distances_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    mean_distances_path =  f'{mean_distances_directory}/mean_distances_{company}_{year}_{month}.parquet' 
    # save the monthly summary to a Parquet file for later aggregation
    mean_distances.to_parquet(mean_distances_path)
    print(f'Month {month}, {company} distance summary saved successfully!')

    #Daily Averages
    daily_averages = calculate_daily_averages(df_cleaned['dropoff_date'], df_cleaned['PUBorough'], df_cleaned['DOBorough'])

    daily_averages_directory = f'{aggregate_data_directory}/daily_averages_{year}_{company}'
    os.makedirs(daily_averages_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    daily_averages_path =  f'{daily_averages_directory}/daily_averages_{company}_{year}_{month}.parquet'  
    daily_averages.to_parquet(daily_averages_path)
    print(f'Month {month}, {company} daily averages saved successfully!')

    #ride_Counts 
    ride_counts_directory = f'{aggregate_data_directory}/ride_counts_{year}_{company}'
    os.makedirs(ride_counts_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    #dropoff ride_Counts
    ride_counts_path_dropoff =  f'{ride_counts_directory}/dropoff_ride_counts_df_{company}_{year}_{month}.parquet'  
    ride_counts_df = prepare_ride_counts(df_cleaned['DOLocationID'])
    ride_counts_df.to_parquet(ride_counts_path_dropoff)
    print(f'Month {month}, {company} dropoff ride counts summary saved successfully!')

     #pickup ride_Counts 
    ride_counts_path_pickup =  f'{ride_counts_directory}/pickup_ride_counts_df_{company}_{year}_{month}.parquet'  
    ride_counts_df = prepare_ride_counts(df_cleaned['PULocationID'])
    ride_counts_df.to_parquet(ride_counts_path_pickup)
    print(f'Month {month}, {company} pickup ride counts summary saved successfully!')
    '''

if __name__ == '__main__':
    main()