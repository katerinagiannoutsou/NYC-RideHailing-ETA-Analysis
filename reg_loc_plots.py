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

from required_functions import load_taxi_zones, plot_reg_mean_eta_vs_drivers_loc_refined_conditional, plot_reg_mean_eta_vs_drivers_loc_refined_conditional_adjusted_shape_area

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
    print(f"ETA Calculation: {eta_calc}")
    eta_subtract = float(sys.argv[14])
    print(f"ETA Subtract: {eta_subtract}")
    grouping = sys.argv[15]
    print(f"Grouping: {grouping}")
    avge_filter = sys.argv[16]
    print(f"Average Filter: {avge_filter}")
    x_column = sys.argv[17]
    print(f"X Column: {x_column}")

    #grouping = sys.argv[15]
    #print(f"Grouping: {grouping}")
    
    # path where the results will be stored
    results_directory = f'{current_directory}/max_flow_open_driver_simulation_{eta_calc}_eta' 
    os.makedirs(results_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    sim_results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/simulation_datasets'
    os.makedirs(sim_results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    #plots_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/location_plots_unfiltered_avge_sim_rounded/loc_plots_{code}_{time_segment}'
    plots_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/refined_reg_location_plots_eta_sub_{eta_subtract}_sim_avge_{avge_filter}/reg_loc_plots_{code}_{time_segment}_{x_column}_area_adjusted'
    os.makedirs(plots_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
    
    reg_results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/refined_{grouping}_results_sub_{eta_subtract}_sim_avge_{avge_filter}_area_adjusted'
    os.makedirs(reg_results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists

    taxi_zones_path = f'{current_directory}/taxi_zones'

       # read the dataset into a DataFrame
    #when num_simulations referes to average of that many simulations   
    dataset_path = f'{sim_results_path}/average_simulation_results_over_{num_simulations}.parquet'
    #when only one simulation, num_simulations refers to the simulation id
    #dataset_path = f'{sim_results_path}/dataset_with_results_{num_simulations}.parquet'

    df = pd.read_parquet(dataset_path)
    print(f'The columns in the dataset are: {df.columns}')

    #filter for borough code and time segment
    df = df[(df['PUBorough'] == code) & (df['Time of Day'] == time_segment)]
    print('Data filtered for borough code and time segment')
    print(f"Unique values in Time of Day: {df['Time of Day'].unique()}")
    print(f"Unique values in PUBorough: {df['PUBorough'].unique()}")


    #taxi zones
    taxi_zones = load_taxi_zones(taxi_zones_path)
    # recalculate the area in square feet (the default for EPSG:2263)
    taxi_zones['Shape_Area_sqft'] = taxi_zones.geometry.area
    # convert the area from square feet to square kilometers
    taxi_zones['Shape_Area_km2'] = taxi_zones['Shape_Area_sqft'] * 9.2903e-8

    #read regression results
    if avge_filter == 'loc_cond_mean':
        reg_filename = f"{reg_results_path}/reg_coefficients_{time_segment}_{code}_{x_column}.csv"
    else:
        reg_filename = f"{reg_results_path}/reg_coefficients_{time_segment}_{code}.csv"
    
    reg_results = pd.read_csv(reg_filename)

    # get unique PULocationIDs
    unique_location_ids = df['PULocationID'].unique() #.compute()
    print('Unique location IDs fetched')
    # convert to a list
    unique_location_ids = unique_location_ids.tolist()

    #unique_location_ids = [113, 166, 88, 164, 244, 41] #TODO: Change back to above when done testing

    for location_id in unique_location_ids:
        plot_reg_mean_eta_vs_drivers_loc_refined_conditional(df, reg_results, time_segment, location_id, year, plots_path, eta_subtract, avge_filter)
        #plot_reg_mean_eta_vs_drivers_loc_refined_conditional_adjusted_shape_area(df, reg_results, time_segment, location_id, year, plots_path, eta_subtract, avge_filter, taxi_zones)

    
    
if __name__ == "__main__":
    main()
