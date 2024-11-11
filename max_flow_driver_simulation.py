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

from required_functions import process_and_append_data_sim, calculate_adjacency_matrix, map_adjacencies_to_data, prepare_df_for_max_flow_simulation, run_max_flow_simulation

def main():

     #fetch working directory
    current_directory = os.getcwd()

    # fetch SLURM environment variables
    ntasks = int(os.getenv('SLURM_NTASKS', '1'))  # Default to 1 if not set
    cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK', '1'))  # Default to 1 if not set

    # define the number of partitions based on SLURM tasks
    #total_cpus = ntasks * cpus_per_task

    # configure the Dask Client
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
    sim_num = int(sys.argv[10])
    print(f"Simulation bash job Number: {sim_num}")
    adj_method = sys.argv[11]
    print(f"Adjacency Method: {adj_method}")
    eta_calc = sys.argv[12]
    print(f"ETA: {eta_calc}")

    #variables we do not longer pass on but fix them from here because they are unlikely to change
    eta_upper_bound = 50
    print(f"ETA Upper Bound: {eta_upper_bound}")
    fraction = 1
    print(f"Fraction of data: {fraction}")

    # path where the results will be stored
    results_directory = f'{current_directory}/max_flow_open_driver_simulation_{eta_calc}_eta' 
    os.makedirs(results_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}'
    os.makedirs(results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    taxi_zones_path = f'{current_directory}/taxi_zones'
 
    files_directory =  f'{current_directory}/data_{year}_{company}'

    clean_data_path =f'{files_directory}/cleaned_rides_data_{year}_{company}/'
    
     # loop through borough codes and load the data, append to a common dataframe
    borough_codes = ['M', 'BNX', 'BKL', 'Q', 'SI'] #    borough_codes = ['M', 'BNX', 'BKL', 'Q', 'SI']
    #borough_codes = ['M'] #    borough_codes = ['M', 'BNX', 'BKL', 'Q', 'SI']

    # loading and concatenating data across boroughs
    big_df = None
    for code in borough_codes:
        df = process_and_append_data_sim(clean_data_path, company, year, demand_method, code, time_segment, eta_upper_bound, time_window_minutes, fraction, eta_calc)
        if big_df is None:
            big_df = df
        else:
            big_df = dd.concat([big_df, df])

    print(f"Data loaded for all boroughs and appended.")
    #print("Columns in big_df:", big_df.columns)
    
    big_df = big_df.compute()
    print("Dataset computed")
    print("Columns in big_df:", big_df.columns)
    '''
    #some checks
    print(big_df[(big_df['Request Time Window'] == '2021-01-01 00:10:00') & (big_df['PULocationID'] == 4)]['requests'])
    print(big_df[(big_df['Request Time Window'] == '2021-01-01 01:10:00') & (big_df['PULocationID'] == 113)]['requests'])
    print(big_df[(big_df['Request Time Window'] == '2021-03-01 01:10:00') & (big_df['PULocationID'] == 66)]['requests'])
    '''
    adjacency_matrix = calculate_adjacency_matrix(taxi_zones_path, method='queen')

    big_df = map_adjacencies_to_data(big_df, adjacency_matrix)
    #check
    # filter rows in big_df where 'PULocationID' is 113
    location_rows = big_df[big_df['PULocationID'] == 113]

    # Print the 'adjacent_locations' for these rows
    print("Adjacent locations for PULocationID 113:")
    print(location_rows['adjacent_locations'])

    ## time range and zones set up
    t_min = pd.Timestamp(f'{year}-01-01 00:00:00')
    print(f"First Request Time Window: {t_min}")

    #TODO: Make sure to adjust for whole year when not running test version
    time_range = pd.date_range(start=t_min, end=t_min + pd.Timedelta(days=365), freq=f'{time_window_minutes}min')
    #time_range = pd.date_range(start=t_min, end=t_min + pd.Timedelta(days=5), freq=f'{time_window_minutes}min')

    print(f"Time Range: {time_range}")
    print(f"Time Range Length: {len(time_range)}")

    # zones set-up --> get unique location IDs from the big_df
    #zones = list(range(1, 263))
    unique_pulocation_ids = big_df['PULocationID'].unique()
    # convert the unique values to a list
    zones = list(unique_pulocation_ids)
    # print the list
    print(zones)

    # prepare the dataframe for the max flow simulation
    aggregated_df = prepare_df_for_max_flow_simulation(big_df, time_window_minutes, time_range, zones, supply_method)
    
    print(aggregated_df.index.names)
    print(aggregated_df.index.values)
    '''
    #checks 
    print(aggregated_df.loc[(4, '2021-01-01 00:10:00'), 'requests'])
    print(aggregated_df.loc[(113, '2021-01-01 01:10:00'), 'requests'])
    # check for None or NaN values in 'adjacent_locations'
    missing_adjacent_locs = aggregated_df[aggregated_df['adjacent_locations'].isna()]

    # print the PULocationIDs and corresponding Request Time Windows with missing 'adjacent_locations'
    if not missing_adjacent_locs.empty:
        print("PULocationIDs with None or NaN values in 'adjacent_locations':")
        print(missing_adjacent_locs.index.get_level_values('PULocationID').unique())
        print("Details of rows with missing 'adjacent_locations':")
        print(missing_adjacent_locs[['adjacent_locations']])
    else:
        print("No None or NaN values found in 'adjacent_locations'.")
    '''

    final_results_path = f'{results_path}/simulation_datasets'
    os.makedirs(final_results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
  
    # run the max flow simulation
    run_max_flow_simulation(big_df, aggregated_df, initial_driver_count, time_range, p, final_results_path, sim_num)



    
if __name__ == "__main__":
    #if len(sys.argv) != 10:
     #   print("Usage: zone_level_eta_regression.py <CODE> <YEAR> <COMPANY> <SUPPLY_METHOD> <DEMAND_METHOD> <DEMAND_ADJ> <TIME_SEGMENT> <ETA_UPPER_BOUND> <TIME_WINDOW_MINUTES>")
      #  sys.exit(1)
    main()
