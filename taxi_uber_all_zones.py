import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
from math import sqrt 
import numpy as np 
from datetime import datetime, timedelta
import sys
import pandas as pd
import dask.dataframe as dd
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
import dask.array as da
#import logging
import concurrent.futures

import geopandas as gpd
import concurrent.futures
from scipy.spatial.distance import pdist, squareform
import osmnx as ox

from required_functions import load_taxi_zones, plot_taxi_vs_uber_rankings

def main():

     #fetch working directory
    current_directory = os.getcwd()

    # fetch SLURM environment variables
    ntasks = int(os.getenv('SLURM_NTASKS', '1'))  # Default to 1 if not set
    cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK', '1'))  # Default to 1 if not set

   # set up the Dask LocalCluster based on SLURM environment
    cluster = LocalCluster(n_workers=ntasks, threads_per_worker=cpus_per_task)
    client = Client(cluster)

    #logging.basicConfig(level=logging.INFO)
    #logger = logging.getLogger(__name__)
    
    #fetch variables
    year = sys.argv[1]
    print(f"Year: {year}")
    company = sys.argv[2]
    print(f"Company: {company}")
    supply_method = sys.argv[3]
    print(f"Supply Method: {supply_method}")
    demand_method = sys.argv[4]
    print(f"Demand Method: {demand_method}")
    demand_adj = sys.argv[5]
    print(f"Demand Adjustment: {demand_adj}")
    time_segment = sys.argv[6]
    print(f"Time Segment: {time_segment}")
    time_window_minutes = float(sys.argv[7])
    print(f"Time Window: {time_window_minutes}")
    initial_driver_count = int(sys.argv[8])
    print(f"Initial Drivers: {initial_driver_count}")
    p = float(sys.argv[9])
    print(f"Leave Probability: {p}")
    #code = sys.argv[10]
    #print(f"Code: {code}")
    #num_simulations = int(sys.argv[11])
    #print(f"Number of Simulations: {num_simulations}")
    adj_method = sys.argv[10]
    print(f"Adjacency Method: {adj_method}")
    eta_calc = sys.argv[11]
    print(f"ETA Calculation: {eta_calc}")
    eta_subtract = float(sys.argv[12])
    print(f"ETA Subtract: {eta_subtract}")
    grouping = sys.argv[13]
    print(f"Grouping: {grouping}")
    avge_filter = sys.argv[14]
    print(f"Average Filter: {avge_filter}")
    x_column = sys.argv[15]
    print(f"X Column: {x_column}")
    max_k = int(sys.argv[16])
    print(f"Max K: {max_k}")
    n_clusters = sys.argv[17]
    print(f"N Clusters: {n_clusters}")

    eta_upper_bound = 50
    fraction = 1
    # path where the results will be stored
    results_directory = f'{current_directory}/max_flow_open_driver_simulation_{eta_calc}_eta' 
    os.makedirs(results_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
    
    reg_results_path = f'{results_directory}/full_zones_density_results'
    os.makedirs(reg_results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
    
    clean_uber_data_path =f'{current_directory}/data_{year}_{company}/cleaned_rides_data_{year}_{company}/'

    clean_yellow_taxi_data_path =f'{current_directory}/data_{year}_Taxi/cleaned_rides_data_{year}_Taxi/'

    clean_green_taxi_data_path =f'{current_directory}/data_{year}_Green_Taxi/cleaned_rides_data_{year}_Green_Taxi/'

    taxi_zones_path = f'{current_directory}/taxi_zones'

    density_datasets_path = f'{current_directory}/density_datasets'

    road_network_path = f'{current_directory}/road_network'

    plots_path = f'{reg_results_path}/density_plots_norm_{time_segment}'
    os.makedirs(plots_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    taxi_zones = load_taxi_zones(taxi_zones_path)
    taxi_zones = taxi_zones[taxi_zones['borough'] != 'Staten Island'] #remove SI
    
    ''' 
    ## UBER & TAXI - DATA LOADING - ##
    '''

    #load the big_uber_df and big_taxi_df if already created
    big_uber_df =  pd.read_csv(f'{clean_uber_data_path}big_uber_{time_segment}_df.csv') #pd.read_parquet(dataset_path)

    yellow_taxi_df = pd.read_csv(f'{clean_yellow_taxi_data_path}yellow_taxi_{time_segment}_df.csv')
   
    green_taxi_df = pd.read_csv(f'{clean_green_taxi_data_path}green_taxi_{time_segment}_df.csv')

    save_borough_observation_counts(big_uber_df, 'Uber', plots_path)
    save_borough_observation_counts(yellow_taxi_df, 'Yellow_Taxi', plots_path) 
    save_borough_observation_counts(green_taxi_df, 'Green_Taxi', plots_path) 

    # add a 'TaxiType' column to each DataFrame
    yellow_taxi_df['TaxiType'] = 'Yellow'
    green_taxi_df['TaxiType'] = 'Green'

    # combine the two DataFrames vertically
    big_taxi_df = pd.concat([yellow_taxi_df, green_taxi_df], ignore_index=True)

    ''' 
    ## UBER & TAXI - EXPLORATORY PLOTS  ##
    '''
    
    # call the function for normalized ride volume comparison over total
    ride_rankings = create_ride_rankings(big_uber_df, big_taxi_df, plots_path)

    plot_taxi_vs_uber_rankings(ride_rankings, plots_path)


if __name__ == "__main__":
    main()