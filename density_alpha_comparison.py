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

from required_functions import load_and_filter_regression_results, calculate_and_merge_ride_volume, load_taxi_zones, calculate_adjacency_matrix, map_adjacencies_to_data, load_land_use_and_merge_div_indices_with_reg_results, plot_alpha_vs_column, add_building_stats_to_reg_results, add_ped_count_stats_to_reg_results, add_wifi_stats_to_reg_results, add_road_metrics_to_reg_results, plot_building_stats, plot_ped_stats, normalize_and_save_density_metrics_and_cdf, combine_normalized_and_cdf_density_metrics, classify_top_percentile_based, run_percentile_classification_loops, run_average_classification_loops, create_and_save_correlation_matrix, perform_pca_and_save_results, run_pca_for_combinations

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

    # path where the results will be stored
    results_directory = f'{current_directory}/max_flow_open_driver_simulation_{eta_calc}_eta' 
    os.makedirs(results_directory, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    sim_results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/simulation_datasets'
    os.makedirs(sim_results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    reg_results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/refined_{grouping}_results_sub_{eta_subtract}_sim_avge_{avge_filter}'
    #reg_results_path = f'{results_directory}/full_zones_density_results'
    os.makedirs(reg_results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
    
    dataset_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/simulation_datasets/dataset_with_results_1.parquet'

    taxi_zones_path = f'{current_directory}/taxi_zones'

    density_datasets_path = f'{current_directory}/density_datasets'

    road_network_path = f'{current_directory}/road_network'

    plots_path = f'{reg_results_path}/density_plots_{time_segment}_{x_column}_max_k_{max_k}_{n_clusters}_norm_tables'
    #plots_path = f'{reg_results_path}/density_plots_raw'
    os.makedirs(plots_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    taxi_zones = load_taxi_zones(taxi_zones_path)
    taxi_zones = taxi_zones[taxi_zones['borough'] != 'Staten Island'] #remove SI
    
    '''
    ## REGRESSION RESULTS  ##
    '''
    
    #load the regression results for each borough
    codes = ['M', 'BNX', 'BKL', 'Q']
    
    #load regression results
    reg_results = load_and_filter_regression_results(results_directory, year, company, supply_method, demand_method, 
                                       time_window_minutes, adj_method, initial_driver_count, p, 
                                       grouping, eta_subtract, avge_filter, time_segment, x_column, codes, threshold_r_squared=0.3)
    reg_results['alpha'] = - reg_results['alpha']
    #add ride volume
    reg_results = calculate_and_merge_ride_volume(dataset_path, reg_results, time_segment)

    # perform clustering and update the 'reg_results' dataframe

    '''
    WHEN DOING FOR FULL ZONES 
    #creating full zones dataset to add the density metrics
   # extract the unique 'LocationID' and corresponding 'borough' from taxi_zones
    unique_locations_borough = taxi_zones[['LocationID', 'borough']].drop_duplicates()

    # rename the columns to 'PULocationID' and 'borough_code'
    reg_results = unique_locations_borough.rename(columns={'LocationID': 'PULocationID', 'borough': 'borough_code'})

    print(f"The number of rows in the 'reg_results' DataFrame is: {len(reg_results)}")
    '''

    '''
    ## DENSITY METRICS  ##
    '''

    adjacency_matrix = calculate_adjacency_matrix(taxi_zones_path, method='queen')

    reg_results = map_adjacencies_to_data(reg_results, adjacency_matrix)
    #check
    # Filter rows in reg_results where 'PULocationID' is 113
    #location_rows = reg_results[reg_results['PULocationID'] == 113]

    # Print the 'adjacent_locations' for these rows
    #print("Adjacent locations for PULocationID 113:")
    #print(location_rows['adjacent_locations'])

    
    # LAND USE -- Shannon & Simpson Index
    reg_results = load_land_use_and_merge_div_indices_with_reg_results(density_datasets_path, taxi_zones, reg_results, plots_path)
    #plots
    plot_alpha_vs_column(reg_results, 'shannon_index', plots_path, 'alpha_vs_shannon_index.png')
    plot_alpha_vs_column(reg_results, 'simpson_index', plots_path, 'alpha_vs_simpson_index.png')
    #plot_alpha_vs_column(reg_results, 'shannon_index_adj', plots_path, 'alpha_vs_shannon_index_adj.png')
    #plot_alpha_vs_column(reg_results, 'simpson_index_adj', plots_path, 'alpha_vs_simpson_index_adj.png')
    
    # BUILDINGS -- Max Height
    reg_results = add_building_stats_to_reg_results(density_datasets_path, taxi_zones, reg_results, plots_path)
    # define the list of statistics to plot (both raw and normalized versions)
    building_stat_columns = ['max_height', 'median_height', 'min_height', 'upper_quantile_height', 'lower_quantile_height']
    plot_building_stats(reg_results, plots_path, building_stat_columns)
    
    # PEDESTRIAN COUNTS
    reg_results = add_ped_count_stats_to_reg_results(density_datasets_path, taxi_zones, reg_results, plots_path, time_segment)
    plot_ped_stats(reg_results, plots_path, time_segment)
    
    #reg_results = add_ped_count_stats_to_reg_results(density_datasets_path, taxi_zones, reg_results, plots_path, 'morning_rush')
    #reg_results = add_ped_count_stats_to_reg_results(density_datasets_path, taxi_zones, reg_results, plots_path, 'evening_rush')


    # WIFI HOTSPOTS
    reg_results = add_wifi_stats_to_reg_results(density_datasets_path, taxi_zones, reg_results)
    plot_alpha_vs_column(reg_results, 'has_wifi_hotspot', plots_path, 'alpha_vs_wifi_presence.png')
    plot_alpha_vs_column(reg_results, 'number_of_wifi_hotspots', plots_path, 'alpha_vs_wifi_number.png')
    plot_alpha_vs_column(reg_results, 'number_of_wifi_hotspots_norm', plots_path, 'alpha_vs_wifi_number_norm.png')
    #plot_alpha_vs_column(reg_results, 'has_wifi_hotspot_adj', plots_path, 'alpha_vs_wifi_presence_adj.png')
    #plot_alpha_vs_column(reg_results, 'number_of_wifi_hotspots_adj', plots_path, 'alpha_vs_wifi_number_adj.png')
    #plot_alpha_vs_column(reg_results, 'number_of_wifi_hotspots_adj_norm', plots_path, 'alpha_vs_wifi_number_norm_adj.png')


    # ROAD NETWORK 
    reg_results = add_road_metrics_to_reg_results(road_network_path, reg_results, taxi_zones)
    plot_alpha_vs_column(reg_results, 'road_density_km', plots_path, 'alpha_vs_road_density_km.png')
    plot_alpha_vs_column(reg_results, 'intersection_density', plots_path, 'alpha_vs_intersection_density.png')
    #plot_alpha_vs_column(reg_results, 'road_density_km_adj', plots_path, 'alpha_vs_road_density_km_adj.png')
    #plot_alpha_vs_column(reg_results, 'intersection_density_adj', plots_path, 'alpha_vs_intersection_density_adj.png')

    # save the updated DataFrame with the cluster column to a CSV file
    reg_results.to_csv(f'{reg_results_path}/reg_results_with_density_metrics_{time_segment}_{x_column}.csv', index=False)

    
    '''
    ## DENSITY - ALPHA COMPARISON  ##
        ## TABLES ## 
    '''

    
    reg_results = pd.read_csv(f'{reg_results_path}/reg_results_with_density_metrics_{time_segment}_{x_column}.csv')

    if time_segment == 'morning_rush':
        count_column_name = 'avg_am_count_2021_norm'
    elif time_segment == 'evening_rush':
        count_column_name = 'avg_pm_count_2021_norm'
    else:
        raise ValueError("Invalid time_segment. Choose 'morning_rush' or 'evening_rush'.")

    density_metrics = ['shannon_index', 'simpson_index', 'median_height', 'has_count_stations', 'number_of_count_stations_norm', count_column_name, 'has_wifi_hotspot','number_of_wifi_hotspots_norm', 'road_density_km', 'intersection_density']

    tables_path = f'{reg_results_path}/density_norm_tables_{time_segment}_{x_column}'
    os.makedirs(tables_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists

    normalized_df, empirical_cdf_df = normalize_and_save_density_metrics_and_cdf(reg_results, density_metrics, tables_path)

    continuous_metrics = ['shannon_index', 'simpson_index', 'median_height',  'road_density_km', 'intersection_density']
    discrete_metrics = ['has_count_stations', 'number_of_count_stations_norm', 'has_wifi_hotspot', 'number_of_wifi_hotspots_norm', count_column_name]

    combined_df = combine_normalized_and_cdf_density_metrics(reg_results, continuous_metrics, discrete_metrics, tables_path)

    #CORRELATION MATRIX
    cor_path = f'{tables_path}/density_correlations'
    os.makedirs(cor_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists

    create_and_save_correlation_matrix(normalized_df, density_metrics, cor_path)

    #PCA
    pca_path = f'{tables_path}/pca'
    os.makedirs(pca_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists

    #continuous_metrics = ['median_height', count_column_name,  'road_density_km', 'intersection_density']

    #pca_model = perform_pca_and_save_results(normalized_df, continuous_metrics, pca_path)
    run_pca_for_combinations(normalized_df, pca_path)

    
    # perecntile classification
    #normalized_df = classify_top_percentile_based(normalized_df, density_metrics, tables_path, threshold_percentile=0.2, min_metrics_above_threshold=5)
    #empirical_cdf_df = classify_top_percentile_based(empirical_cdf_df, density_metrics, tables_path, threshold_percentile=0.2, min_metrics_above_threshold=5)

    '''
    percentile_plots_path = f'{tables_path}/percentile_class_plots'
    os.makedirs(percentile_plots_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
    average_plots_path = f'{tables_path}/average_class_plots'
    os.makedirs(average_plots_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists

    normalized_df, empirical_cdf_df = run_percentile_classification_loops(normalized_df, empirical_cdf_df, density_metrics, percentile_plots_path)

    normalized_df, empirical_cdf_df = run_average_classification_loops(normalized_df, empirical_cdf_df, density_metrics, average_plots_path)

    normalized_df.to_csv(f'{tables_path}/normalized_data_values_class.csv', index=False)
    empirical_cdf_df.to_csv(f'{tables_path}/empirical_cdf_class.csv', index=False)
    '''
   


if __name__ == "__main__":
    main()