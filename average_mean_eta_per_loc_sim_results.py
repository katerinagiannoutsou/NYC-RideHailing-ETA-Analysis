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

from required_functions import aggregate_mean_eta_per_location

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
    code = sys.argv[10]
    print(f"Code: {code}")
    num_simulations = int(sys.argv[11])
    print(f"Number of Simulations: {num_simulations}")
    adj_method = sys.argv[12]
    print(f"Adjacency Method: {adj_method}")
    eta_calc = sys.argv[13]
    print(f"ETA: {eta_calc}")
    avge_filter = sys.argv[14]
    print(f"Average Filter: {avge_filter}")
    x_column = sys.argv[15]
    print(f"X Column: {x_column}")

    #variables we do not longer pass on but fix them from here because they are unlikely to change
    eta_upper_bound = 50
    print(f"ETA Upper Bound: {eta_upper_bound}")
    fraction = 1
    print(f"Fraction of data: {fraction}")
    #regression_type = 'per_time_window'
    #print(f"Regression Type: {regression_type}")

    # path where the results will be stored
    results_directory = f'{current_directory}/max_flow_open_driver_simulation_{eta_calc}_eta' 
    os.makedirs(results_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/simulation_datasets'
    os.makedirs(results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
    
    output_path = f'{results_path}/sim_avge_{avge_filter}_{x_column}'
    os.makedirs(output_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists

    #read based df to get location id 
    dataset_path = f'{results_path}/dataset_with_results_{num_simulations}.parquet'
    df = pd.read_parquet(dataset_path)
    print(f'The columns in the dataset are: {df.columns}')

    #filter for borough code and time segment
    df = df[(df['PUBorough'] == code) & (df['Time of Day'] == time_segment)] 
    print('Data filtered for borough code and time segment')
    
    '''
    unique_location_ids = [113, 166, 88] #TODO: Change back to above when done testing

    tasks = [delayed(aggregate_mean_eta_per_location)(results_path, location_id, time_segment, num_simulations, output_path, year, x_column) for location_id in unique_location_ids]
    compute(*tasks)

    '''
     # Get unique PULocationIDs
    unique_location_ids = df['PULocationID'].unique() #.compute()
    print('Unique location IDs fetched')
    # Convert to a list
    unique_location_ids = unique_location_ids.tolist()

    split_size = len(unique_location_ids) // 3
    first_part = unique_location_ids[:split_size]
    second_part = unique_location_ids[split_size:2*split_size]
    third_part = unique_location_ids[2*split_size:]
    print(f"First part: {first_part}")
    print(f"Second part: {second_part}")
    print(f"Third part: {third_part}")
    first_part_tasks = [delayed(aggregate_mean_eta_per_location)(results_path, location_id, time_segment, num_simulations, output_path, year, x_column) for location_id in first_part]
    second_part_tasks = [delayed(aggregate_mean_eta_per_location)(results_path, location_id, time_segment, num_simulations, output_path, year, x_column) for location_id in second_part]
    third_part_tasks = [delayed(aggregate_mean_eta_per_location)(results_path, location_id, time_segment, num_simulations, output_path, year, x_column) for location_id in third_part]

    compute(*first_part_tasks)
    compute(*second_part_tasks)
    compute(*third_part_tasks)
    
    
if __name__ == "__main__":
    #if len(sys.argv) != 10:
     #   print("Usage: zone_level_eta_regression.py <CODE> <YEAR> <COMPANY> <SUPPLY_METHOD> <DEMAND_METHOD> <DEMAND_ADJ> <TIME_SEGMENT> <ETA_UPPER_BOUND> <TIME_WINDOW_MINUTES>")
      #  sys.exit(1)
    main()