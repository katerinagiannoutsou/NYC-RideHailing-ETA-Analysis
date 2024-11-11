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

from required_functions import load_taxi_zones, perform_refined_conditional_regression_zone_level_post_sim_existing_filtered_df, perform_refined_conditional_regression_zone_level_adjusted_shape_area

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

    # path where the results will be stored
    results_directory = f'{current_directory}/max_flow_open_driver_simulation_{eta_calc}_eta' 
    os.makedirs(results_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    sim_results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/simulation_datasets'
    os.makedirs(sim_results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    reg_results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/refined_{grouping}_results_sub_{eta_subtract}_sim_avge_{avge_filter}'
    os.makedirs(reg_results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists

    taxi_zones_path = f'{current_directory}/taxi_zones'

    
    cond_sim_results_path = f'{sim_results_path}/sim_avge_{avge_filter}_{x_column}'
    
       # read the dataset into a DataFrame
    #when num_simulations referes to average of that many simulations   
    base_dataset_path = f'{sim_results_path}/average_simulation_results_over_{num_simulations}.parquet'
    #when only one simulation, num_simulations refers to the simulation id
    #base_dataset_path = f'{sim_results_path}/dataset_with_results_{num_simulations}.parquet'

    base_df = pd.read_parquet(base_dataset_path)
    print(f'The columns in the dataset are: {base_df.columns}')

    #filter for borough code and time segment
    base_df = base_df[(base_df['PUBorough'] == code) & (base_df['Time of Day'] == time_segment)]
    print('Data filtered for borough code and time segment')
    print(f"Unique values in Time of Day: {base_df['Time of Day'].unique()}")
    print(f"Unique values in PUBorough: {base_df['PUBorough'].unique()}")

    # get unique PULocationIDs
    unique_location_ids = base_df['PULocationID'].unique() #.compute()
    print('Unique location IDs fetched')
    # convert to a list
    unique_location_ids = unique_location_ids.tolist()

    #taxi zones
    taxi_zones = load_taxi_zones(taxi_zones_path)
    # recalculate the area in square feet (the default for EPSG:2263)
    taxi_zones['Shape_Area_sqft'] = taxi_zones.geometry.area
    # convert the area from square feet to square kilometers
    taxi_zones['Shape_Area_km2'] = taxi_zones['Shape_Area_sqft'] * 9.2903e-8

    '''
    unique_location_ids = [113, 166, 88] #TODO: Change back to above when done testing
    # Compute each part sequentially
    tasks = [delayed(perform_refined_conditional_regression_zone_level_post_sim_existing_filtered_df)(location_id, eta_subtract, time_segment, x_column, cond_sim_results_path) for location_id in unique_location_ids]
    
    results = compute(*tasks)
    borough_results_df = pd.DataFrame(results)

    '''
    # check the value of the code parameter
    if code in ['M', 'Q']:
        # split the unique_location_ids into three parts if code is 'M' or 'Q'
        split_size = len(unique_location_ids) // 3
        first_part = unique_location_ids[:split_size]
        second_part = unique_location_ids[split_size:2*split_size]
        third_part = unique_location_ids[2*split_size:]
        print(f"First part: {first_part}")
        print(f"Second part: {second_part}")
        print(f"Third part: {third_part}")
        
        
        # create delayed tasks for each part
        #first_part_tasks = [delayed(perform_refined_conditional_regression_zone_level_adjusted_shape_area)(location_id, eta_subtract, time_segment, x_column, cond_sim_results_path, taxi_zones) for location_id in first_part]

        first_part_tasks = [delayed(perform_refined_conditional_regression_zone_level_post_sim_existing_filtered_df)(location_id, eta_subtract, time_segment, x_column, cond_sim_results_path) for location_id in first_part]
        second_part_tasks = [delayed(perform_refined_conditional_regression_zone_level_post_sim_existing_filtered_df)(location_id, eta_subtract, time_segment, x_column, cond_sim_results_path) for location_id in second_part]
        third_part_tasks = [delayed(perform_refined_conditional_regression_zone_level_post_sim_existing_filtered_df)(location_id, eta_subtract, time_segment, x_column, cond_sim_results_path) for location_id in third_part]

        # compute each part sequentially
        first_part_results = compute(*first_part_tasks)
        first_part_results_df = pd.DataFrame(first_part_results)

        second_part_results = compute(*second_part_tasks)
        second_part_results_df = pd.DataFrame(second_part_results)

        third_part_results = compute(*third_part_tasks)
        third_part_results_df = pd.DataFrame(third_part_results)

        # combine all results
        borough_results_df = pd.concat([first_part_results_df, second_part_results_df, third_part_results_df], ignore_index=True)

    else:
        # default splitting into two parts if code is not 'M' or 'Q'
        mid_index = len(unique_location_ids) // 2
        first_half = unique_location_ids[:mid_index]
        second_half = unique_location_ids[mid_index:]

        # create delayed tasks for each half
        first_half_tasks = [delayed(perform_refined_conditional_regression_zone_level_post_sim_existing_filtered_df)(location_id, eta_subtract, time_segment, x_column, cond_sim_results_path) for location_id in first_half]
        second_half_tasks = [delayed(perform_refined_conditional_regression_zone_level_post_sim_existing_filtered_df)(location_id, eta_subtract, time_segment, x_column, cond_sim_results_path) for location_id in second_half]

        # compute each half sequentially
        first_half_results = compute(*first_half_tasks)
        first_half_results_df = pd.DataFrame(first_half_results)

        second_half_results = compute(*second_half_tasks)
        second_half_results_df = pd.DataFrame(second_half_results)

        # combine the results
        borough_results_df = pd.concat([first_half_results_df, second_half_results_df], ignore_index=True)
    
    # append additional information to the combined results DataFrame
    borough_results_df['company'] = company
    borough_results_df['year'] = year
    borough_results_df['supply_method'] = supply_method
    borough_results_df['demand_method'] = demand_method
    borough_results_df['demand_adj'] = demand_adj
    borough_results_df['time_segment'] = time_segment
    borough_results_df['adjacency_method'] = adj_method
    borough_results_df['eta_subtract'] = eta_subtract

    # save the results to a CSV file
    filename = f"{reg_results_path}/reg_coefficients_{time_segment}_{code}_{x_column}.csv"
    borough_results_df.to_csv(filename, index=False)
        
        
    
if __name__ == "__main__":
    main()
