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

from required_functions import load_taxi_zones,  process_and_append_data_for_comparison, plot_ride_data_on_map, plot_uber_taxi_comparison_map, save_borough_observation_counts, create_ride_rankings,  combine_normalized_and_cdf_density_metrics, plot_uber_vs_taxi_rankings_with_density, classify_zones_with_gmm, plot_alpha_gmm_distribution, plot_uber_vs_taxi_rankings_with_gmm_classification, plot_uber_vs_taxi_rankings_with_alpha_threshold_classification, create_borough_summary_table, plot_ride_volume_vs_density, plot_alpha_vs_density, plot_combined_density_and_alpha_classification, plot_rankings_with_density_and_alpha_grading, plot_combined_density_and_alpha_classification_with_ids

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
   
    sim_results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/simulation_datasets'
    os.makedirs(sim_results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    reg_results_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/refined_{grouping}_results_sub_{eta_subtract}_sim_avge_{avge_filter}'
    os.makedirs(reg_results_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
    
    dataset_path = f'{results_directory}/results_{year}_{company}_supply_{supply_method}_demand_{demand_method}_{time_window_minutes}_loc_adj_{adj_method}_initial_drivers_{initial_driver_count}_leave_prob_{p}/simulation_datasets/dataset_with_results_1.parquet'

    clean_uber_data_path =f'{current_directory}/data_{year}_{company}/cleaned_rides_data_{year}_{company}/'

    clean_yellow_taxi_data_path =f'{current_directory}/data_{year}_Taxi/cleaned_rides_data_{year}_Taxi/'

    clean_green_taxi_data_path =f'{current_directory}/data_{year}_Green_Taxi/cleaned_rides_data_{year}_Green_Taxi/'

    taxi_zones_path = f'{current_directory}/taxi_zones'

    density_datasets_path = f'{current_directory}/density_datasets'

    road_network_path = f'{current_directory}/road_network'

    plots_path = f'{reg_results_path}/taxi_vs_uber_{time_segment}_{x_column}_borough_density_paper_plots_presentation'
    os.makedirs(plots_path, exist_ok=True)  # 'exist_ok=True' prevents throwing an error if the directory already exists
   
    taxi_zones = load_taxi_zones(taxi_zones_path)
    taxi_zones = taxi_zones[taxi_zones['borough'] != 'Staten Island'] #remove SI
    
    ''' 
    ## UBER & TAXI - DATA LOADING - ##
    '''

    
    codes = ['M', 'BNX', 'BKL', 'Q']
    
    
    ############################# UBER DATA LOADING ################################
    ## original df
    # loading and concatenating data across boroughs
    big_uber_df = None
    for code in codes:
        df = process_and_append_data_for_comparison(clean_uber_data_path, company, year, demand_method, code, time_segment, eta_upper_bound, time_window_minutes, fraction, eta_calc)
        if big_uber_df is None:
            big_uber_df = df
        else:
            big_uber_df = dd.concat([big_uber_df, df])

    print(f"Uber Data loaded for all boroughs and appended.")
    print("Columns in big_uber_df:", big_uber_df.columns)
    
    #compute df
    big_uber_df = big_uber_df.compute()

    #save the big_uber_df as csv in the clean_uber_data_path
    big_uber_df.to_csv(f'{clean_uber_data_path}big_uber_{time_segment}_df.csv', index=False) #TODO: UNCOMMENT
    
    
    ############################# YELLOW TAXI DATA LOADING ################################
    # loading and concatenating data across boroughs
    yellow_taxi_df = None
    for code in codes:
        df = process_and_append_data_for_comparison(clean_yellow_taxi_data_path, 'Taxi', year, demand_method, code, time_segment, eta_upper_bound, time_window_minutes, fraction, eta_calc)
        if yellow_taxi_df is None:
            yellow_taxi_df = df
        else:
            yellow_taxi_df = dd.concat([yellow_taxi_df, df])

    print(f"Taxi Data loaded for all boroughs and appended.")
    print("Columns in big_taxi_df:", yellow_taxi_df.columns)
    
    print("The values in the 'PUBorough' column are:", yellow_taxi_df['PUBorough'].value_counts().compute())
    print("The values in the 'PULocationID' column are:", yellow_taxi_df['PULocationID'].value_counts().compute())

        #compute df
    yellow_taxi_df = yellow_taxi_df.compute()

    #save the big_taxi_df as csv in the clean_taxi_data_path
    yellow_taxi_df.to_csv(f'{clean_yellow_taxi_data_path}yellow_taxi_{time_segment}_df.csv', index=False) #TODO: UNCOMMENT
    
    ############################# GREEN TAXI DATA LOADING ################################

    # loading and concatenating data across boroughs
    green_taxi_df = None
    for code in codes:
        df = process_and_append_data_for_comparison(clean_green_taxi_data_path, 'Green_Taxi', year, demand_method, code, time_segment, eta_upper_bound, time_window_minutes, fraction, eta_calc)
        if green_taxi_df is None:
            green_taxi_df = df
        else:
            green_taxi_df = dd.concat([green_taxi_df, df])

    print(f"Taxi Data loaded for all boroughs and appended.")
    print("Columns in big_taxi_df:", green_taxi_df.columns)
    
    print("The values in the 'PUBorough' column are:", green_taxi_df['PUBorough'].value_counts().compute())
    print("The values in the 'PULocationID' column are:", green_taxi_df['PULocationID'].value_counts().compute())

        #compute df
    green_taxi_df = green_taxi_df.compute()

    #save the big_taxi_df as csv in the clean_taxi_data_path
    green_taxi_df.to_csv(f'{clean_green_taxi_data_path}green_taxi_{time_segment}_df.csv', index=False) #TODO: UNCOMMENT
    
    
    ####
    
    
    #load the big_uber_df and big_taxi_df if already created
    '''
    big_uber_df =  pd.read_csv(f'{clean_uber_data_path}big_uber_{time_segment}_df.csv') #pd.read_parquet(dataset_path)

    yellow_taxi_df = pd.read_csv(f'{clean_yellow_taxi_data_path}yellow_taxi_{time_segment}_df.csv')
   
    green_taxi_df = pd.read_csv(f'{clean_green_taxi_data_path}green_taxi_{time_segment}_df.csv')
    '''
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
    
    # List of companies and data types
    
    companies = ['Uber', 'Taxi']
    data_types = ['ride_volume', 'revenue']
    # Loop over companies and data types
    for company in companies:
        # Select the appropriate DataFrame based on the company
        if company == 'Uber':
            df = big_uber_df
        elif company == 'Taxi':
            df = big_taxi_df
        else:
            continue  # Skip if company is not recognized
        
        for data_type in data_types:
            # Call the plotting function
            plot_ride_data_on_map(taxi_zones, df, company, plots_path, data_type)
    



    # call the function for ride volume comparison over total 
    #plot_uber_taxi_comparison_map(taxi_zones, big_uber_df, big_taxi_df, plots_path, data_type='ride_volume')

   
    # call the function for normalized ride volume comparison over total
    ride_rankings = create_ride_rankings(big_uber_df, big_taxi_df, plots_path)
    #revenue_rankings = create_weighted_revenue_rankings(big_uber_df, big_taxi_df, plots_path)    #create_revenue_rankings_with_normalization(big_uber_df, big_taxi_df, plots_path)

    '''
    ## REGRESSION RESULTS -- ALPHA  ##
    '''  
    #load the regression results for each borough
    codes = ['M', 'BNX', 'BKL', 'Q']
    
    #load regression results
    reg_results = pd.read_csv(f'{reg_results_path}/reg_results_with_density_metrics_{time_segment}_{x_column}.csv')

    if time_segment == 'morning_rush':
        count_column_name = 'avg_am_count_2021_norm'
    elif time_segment == 'evening_rush':
        count_column_name = 'avg_pm_count_2021_norm'
    else:
        raise ValueError("Invalid time_segment. Choose 'morning_rush' or 'evening_rush'.")

    #use this subset of metrics to compare to the regression results
    density_metrics = ['median_height', 'road_density_km', 'intersection_density', 'has_count_stations', 'has_wifi_hotspot', 'number_of_wifi_hotspots_norm', count_column_name]
    continuous_metrics = ['median_height',  'road_density_km', 'intersection_density'] #shannon_index
    discrete_metrics = ['has_count_stations', 'has_wifi_hotspot', 'number_of_wifi_hotspots_norm', count_column_name]

    combined_df = combine_normalized_and_cdf_density_metrics(reg_results, continuous_metrics, discrete_metrics, plots_path)

    summary_df = create_borough_summary_table(combined_df, density_metrics, plots_path)

    '''
    plot_ride_volume_vs_density(big_uber_df, big_taxi_df, combined_df, density_metrics, plots_path)
    plot_alpha_vs_density(combined_df, density_metrics, plots_path)
    '''
    
    
    # plot Uber vs Taxi rankings based on ride and revenue with density comparison
    
    plot_uber_vs_taxi_rankings_with_density(combined_df, ride_rankings, density_metrics, plots_path, ranking_type='ride')

    # apply thresholds for alpha values and plot for both ride and revenue rankings
    alpha_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    plot_uber_vs_taxi_rankings_with_alpha_threshold_classification(ride_rankings, reg_results, alpha_thresholds, plots_path, ranking_type='ride')
    
    
    density_thresholds = [0.5, 0.6, 0.7]
    plot_combined_density_and_alpha_classification(
        combined_df, reg_results, ride_rankings, density_metrics, 
        density_thresholds, alpha_thresholds, plots_path, ranking_type='ride')
    
    '''
    # classify zones with GMM and plot for ride rankings
    n_classes = max_k
    reg_results = classify_zones_with_gmm(reg_results, alpha_column='alpha', n_classes=n_classes)
    plot_alpha_gmm_distribution(reg_results, plots_path, alpha_column='alpha', n_classes=n_classes)

    # plot Uber vs Taxi rankings based on ride and revenue with GMM classification
    plot_uber_vs_taxi_rankings_with_gmm_classification(ride_rankings, reg_results, n_classes, plots_path, ranking_type='ride')
    plot_uber_vs_taxi_rankings_with_gmm_classification(revenue_rankings, reg_results, n_classes, plots_path, ranking_type='revenue')
    '''

    agreement_dense_df = plot_combined_density_and_alpha_classification_with_ids(combined_df, reg_results, ride_rankings, density_metrics, density_thresholds=[0.6], alpha_thresholds=[0.5], plots_path=plots_path)

    # convert to a DataFrame and save to CSV
    #agreement_dense_df = pd.DataFrame(agreement_dense_ids, columns=['PULocationID'])
    agreement_dense_df.to_csv(f"{plots_path}/agreement_dense_ids_0_6_0_5.csv", index=False)

    
if __name__ == "__main__":
    main()