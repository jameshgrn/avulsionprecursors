import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
import os

def inverse_power_law(y, a, b):
    return (y / a) ** (1 / b)

def process_river_data(input_filename, discharge_filename, model_filename, params_filename, output_filename=None):
    if output_filename is None:
        output_filename = input_filename.replace('.csv', '_based.csv')

    # Load river data
    river_data = pd.read_csv(input_filename)
    print(f"River data shape: {river_data.shape}")

    # Extract river name from the filename or river_data
    river_name = os.path.basename(input_filename).split('_output')[0]
    print(f"Processing river: {river_name}")

    # Load discharge data
    discharge_data = pd.read_csv(discharge_filename)
    print(f"Full discharge data shape: {discharge_data.shape}")
    print("Unique river names in discharge data:")
    print(discharge_data['river_name'].unique())

    # Filter discharge data for the specific river
    river_discharge = discharge_data[discharge_data['river_name'] == river_name]
    print(f"Filtered discharge data shape: {river_discharge.shape}")
    
    if river_discharge.empty:
        print(f"WARNING: No discharge data found for river {river_name}")
    else:
        print("Sample of filtered discharge data:")
        print(river_discharge.head())

    # Merge river data with filtered discharge data
    river_data = pd.merge(river_data, river_discharge[['reach_id', 'discharge_value']], on='reach_id', how='left')
    print(f"Merged river data shape: {river_data.shape}")
    print("Sample of merged data:")
    print(river_data[['reach_id', 'discharge_value']].head())

    # Check for any reach_ids that didn't get a discharge value
    missing_discharge = river_data[river_data['discharge_value'].isna()]
    if not missing_discharge.empty:
        print(f"Number of reach_ids without discharge data: {len(missing_discharge)}")
        print("Sample of reach_ids without discharge data:")
        print(missing_discharge['reach_id'].head())

    # Load model and parameters
    xgb_reg = xgb.XGBRegressor()
    xgb_reg.load_model(model_filename)

    with open(params_filename, 'rb') as f:
        params = pickle.load(f)

    # Correct discharge
    river_data['corrected_discharge'] = inverse_power_law(river_data['discharge_value'], *params)
    print("Sample of corrected discharge:")
    print(river_data[['discharge_value', 'corrected_discharge']].head())
    river_data['slope'] = river_data['slope'] / 1000

    epsilon = 1e-10
    min_superelevation = 0.01  # Changed from 0 to 0.01
    min_slope = 1e-5  # Minimum threshold for slope

    # Apply minimum threshold to slope
    river_data['slope'] = np.maximum(river_data['slope'], min_slope)

    # Debugging: Check for NaNs in slope
    nan_slope_count = river_data['slope'].isna().sum()
    print(f"Number of NaN slope values: {nan_slope_count}")

    # Prepare data for XGB prediction
    guesswork = river_data[['width', 'slope', 'corrected_discharge']].astype(float)
    guesswork.columns = ['width', 'slope', 'discharge']

    # Predict depth
    river_data['XGB_depth'] = xgb_reg.predict(guesswork)
    river_data['XGB_depth'] = river_data['XGB_depth'].clip(lower=0)

    # Calculate floodplain distances to river center
    river_data['floodplain1_dist_to_river_center'] = abs(river_data['floodplain1_dist_along'] - river_data['channel_dist_along'])
    river_data['floodplain2_dist_to_river_center'] = abs(river_data['floodplain2_dist_along'] - river_data['channel_dist_along'])

    # Calculate slopes with safeguards
    river_data['ridge1_slope'] = (river_data['ridge1_elevation'] - river_data['floodplain1_elevation']) / (abs(river_data['ridge1_dist_along'] - river_data['floodplain1_dist_along']) + epsilon)
    river_data['ridge2_slope'] = (river_data['ridge2_elevation'] - river_data['floodplain2_elevation']) / (abs(river_data['ridge2_dist_along'] - river_data['floodplain2_dist_along']) + epsilon)

    # Handle NaNs and infinities in slope calculations
    river_data['ridge1_slope'] = river_data['ridge1_slope'].replace([np.inf, -np.inf], np.nan)
    river_data['ridge2_slope'] = river_data['ridge2_slope'].replace([np.inf, -np.inf], np.nan)

    # Calculate gamma values
    river_data['gamma1'] = np.abs(river_data['ridge1_slope']) / river_data['slope']
    river_data['gamma2'] = np.abs(river_data['ridge2_slope']) / river_data['slope']

    # Handle single-side measurements
    river_data['gamma_mean'] = np.where(
        river_data['gamma2'].isna(),
        river_data['gamma1'],
        river_data[['gamma1', 'gamma2']].mean(axis=1, skipna=True)
    )

    # Calculate a_b and corrected_denominator
    river_data['a_b_1'] = (river_data['ridge1_elevation'] - river_data['channel_elevation']) / (river_data['XGB_depth'])
    river_data['a_b_2'] = (river_data['ridge2_elevation'] - river_data['channel_elevation']) / (river_data['XGB_depth'])
    river_data['a_b'] = river_data[['a_b_1', 'a_b_2']].mean(axis=1, skipna=True)

    conditions = [river_data['a_b'] <= 1.25, river_data['a_b'] > 1.25]
    choices = [
        river_data['XGB_depth'],
        river_data[['ridge1_elevation', 'ridge2_elevation']].mean(axis=1, skipna=True) - river_data['channel_elevation']
    ]
    river_data['corrected_denominator'] = np.select(conditions, choices)

    # Calculate superelevation with a minimum threshold
    river_data['superelevation1'] = np.maximum((river_data['ridge1_elevation'] - river_data['floodplain1_elevation']) / river_data['corrected_denominator'], min_superelevation)
    river_data['superelevation2'] = np.maximum((river_data['ridge2_elevation'] - river_data['floodplain2_elevation']) / river_data['corrected_denominator'], min_superelevation)
    
    # Handle single-side measurements for superelevation
    river_data['superelevation_mean'] = np.where(
        river_data['superelevation2'].isna(),
        river_data['superelevation1'],
        river_data[['superelevation1', 'superelevation2']].mean(axis=1, skipna=True)
    )

    # Calculate lambda
    river_data['lambda'] = river_data['gamma_mean'] * river_data['superelevation_mean']

    # Add flags for potentially problematic measurements
    river_data['flag_high_gamma'] = river_data['gamma_mean'] > 100
    river_data['flag_low_superelevation'] = river_data['superelevation_mean'] < 0.05
    river_data['flag_single_side'] = river_data['gamma2'].isna()

    # Diagnostic information
    print("\nDetailed diagnostics:")
    print(f"Total number of measurements: {len(river_data)}")
    print(f"Number of single-side measurements: {river_data['gamma2'].isna().sum()}")
    print(f"Number of high gamma values (>100): {river_data['flag_high_gamma'].sum()}")
    print(f"Number of low superelevation values (<0.05): {river_data['flag_low_superelevation'].sum()}")
    
    print("\nGamma statistics:")
    print(river_data['gamma_mean'].describe())
    
    print("\nSuperelevation statistics:")
    print(river_data['superelevation_mean'].describe())
    
    print("\nLambda statistics:")
    print(river_data['lambda'].describe())

    # Display extreme cases
    print("\nTop 5 highest gamma values:")
    print(river_data.nlargest(5, 'gamma_mean')[['gamma_mean', 'superelevation_mean', 'lambda', 'ridge1_slope', 'ridge2_slope', 'slope']])

    print("\nTop 5 lowest superelevation values:")
    print(river_data.nsmallest(5, 'superelevation_mean')[['gamma_mean', 'superelevation_mean', 'lambda', 'ridge1_elevation', 'ridge2_elevation', 'floodplain1_elevation', 'floodplain2_elevation', 'channel_elevation']])

    # Add new diagnostic information
    print("\nPercentage of problematic measurements:")
    print(f"High gamma (>100): {river_data['flag_high_gamma'].mean()*100:.2f}%")
    print(f"Low superelevation (<0.05): {river_data['flag_low_superelevation'].mean()*100:.2f}%")
    print(f"Single-side measurements: {river_data['flag_single_side'].mean()*100:.2f}%")

    # Save the processed DataFrame to a CSV file
    river_data.to_csv(output_filename, index=False)
    print(f"Processed data saved to {output_filename}")

def main():
    # Set your input parameters here
    river_name = 'ANJOBONY'
    input_filename = f"src/data_handling/data/{river_name}_output.csv"
    discharge_filename = "src/data_handling/data/river_discharge_data.csv"
    model_filename = "src/data_handling/data/based_us_sans_trampush_early_stopping_combat_overfitting.ubj"
    params_filename = "src/data_handling/data/inverted_discharge_params.pickle"
    output_filename = f"src/data_handling/data/{river_name}_output_based.csv"

    process_river_data(input_filename, discharge_filename, model_filename, params_filename, output_filename)

if __name__ == "__main__":
    main()