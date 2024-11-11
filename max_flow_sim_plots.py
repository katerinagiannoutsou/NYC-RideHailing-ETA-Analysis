import pandas as pd
import matplotlib.pyplot as plt
#import plotly.express as px
import os
from math import sqrt 
#import numpy as np 
from datetime import datetime, timedelta
import sys
import pandas as pd
import dask.dataframe as dd
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
import dask.array as da
#import logging

import geopandas as gpd

from required_functions import plot_eta_vs_average_num_avail, plot_sim_drivers_overtime

def main():

     #fetch working directory
    current_directory = os.getcwd()

    # fetch SLURM environment variables
    ntasks = int(os.getenv('SLURM_NTASKS', '1'))  # Default to 1 if not set
    cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK', '1'))  # Default to 1 if not set

    # define the number of partitions based on SLURM tasks
    total_cpus = ntasks * cpus_per_task

    # configure the Dask Client
    #cluster = LocalCluster(n_workers=ntasks, threads_per_worker=cpus_per_task)
    #client = Client(cluster)

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
    print(f"Simulation Number: {sim_num}")
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
    #results_directory = f'{current_directory}/max_flow_open_driver_simulation_eta_rev' 

    os.makedirs(results_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    #results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}_test'
    results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}'
    os.makedirs(results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    plots_path = f'{results_path}/simulation_plots'
    os.makedirs(plots_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
    
    # Specify the filenames
    aggregate_filename = f'{results_path}/simulation_datasets/aggregated_simulation_results_{sim_num}.parquet'
    dataset_filename = f'{results_path}/simulation_datasets/dataset_with_results_{sim_num}.parquet'
    #aggregate_filename = f'{results_path}/aggregated_simulation_results_1.parquet'
    #dataset_filename = f'{results_path}/dataset_with_results_1.parquet'

    # load the DataFrames with the results
    aggegate_results = dd.read_parquet(aggregate_filename)
    df = dd.read_parquet(dataset_filename)

    aggegate_results = aggegate_results.compute()
    df = df.compute()
    print("Data loaded and computed")

    print(f"the columns in df are: {df.columns}")
    print(f"Number of rows in df: {df.shape[0]}")

    df.rename(columns={'Request Time Window_x': 'Request Time Window', 'PULocationID_x': 'PULocationID', 'requests_x': 'requests'}, inplace=True)

    print(f"the columns in aggegate_results are: {aggegate_results.columns}")
    print(f"Number of rows in aggegate_results: {aggegate_results.shape[0]}")
    
    plot_eta_vs_average_num_avail(plots_path, df, p, year)

    selected_zones = [113, 114, 7, 12, 18, 37, 45, 66, 82, 90, 161, 162, 163, 164, 187, 195, 217, 221, 238, 261]

    #plot_sim_drivers_overtime(plots_path, df, selected_zones, p)
    plot_sim_drivers_overtime(plots_path, aggegate_results, selected_zones, p, year)

 
if __name__ == "__main__":

    main()
