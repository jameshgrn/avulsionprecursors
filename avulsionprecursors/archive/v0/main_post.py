import os
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
from db_ops import create_db_engine, get_dataframe
from based_river_inference import load_data, load_model_and_params, process_river_data
from icesat_ops import process_icesat_data

load_dotenv()

# Define the PostgreSQL connection parameters
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

name = "V7_2"

def main():
    # Load the labeled data from the CSV file
    input_filename = f"data/{name}_output.csv"
    discharge_filename = "data/river_discharge_data.csv"
    model_filename = "data/based_us_sans_trampush_early_stopping_combat_overfitting.ubj"
    params_filename = "data/inverted_discharge_params.pickle"
    
    # Load data
    river_data = load_data(input_filename, discharge_filename)
    
    # Load model and parameters
    xgb_reg, params = load_model_and_params(model_filename, params_filename)
    
    # Process river data using BASED inference
    based_results = process_river_data(river_data, xgb_reg, params)
    
    # Load the original cross-section data
    all_elevations_gdf = gpd.read_parquet(f'data/all_elevations_gdf_{name}.parquet')
    
    # Process ICESat data
    icesat_results = process_icesat_data(based_results, all_elevations_gdf)
    
    # Combine results and save
    final_results = pd.merge(based_results, icesat_results, on=['reach_id', 'node_id'])
    final_results.to_csv(f"data/{name}_final_results.csv", index=False)
    
    print(f"Final results saved to data/{name}_final_results.csv")

if __name__ == "__main__":
    main()