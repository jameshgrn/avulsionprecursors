import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from custom_variogram import Variogram
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
import logging
import csv
from adjustText import adjust_text  # Import adjustText for annotation adjustment


#make vector pdf for illustrator
plt.rcParams['pdf.fonttype'] = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detrend_data(data: pd.DataFrame, variable: str) -> np.ndarray:
    X = data['dist_out'].values
    y = data[variable].values

    # apply LOWESS smoothing
    frac = 0.5  # adjust this fraction as needed
    lowess_result = lowess(y, X, frac=frac, it=10)

    # extract the trend
    trend = lowess_result[:, 1]

    # detrend the data
    detrended = y - trend

    # Shift detrended data to ensure all values are positive
    min_detrended = np.min(detrended)
    if min_detrended < 0:
        detrended += abs(min_detrended)

    return detrended

def update_supplementary_table(river_name: str, range_value: float, range_error: float, 
                             avg_width: float, dominant_wavelength: float, 
                             dominant_wavelength_error: float, half_wavelength: float,
                             median_meander_length: float):
    original_file_path = 'data/manuscript_data/supplementary_table.csv'
    output_file = 'data/manuscript_data/supplementary_table.csv'

    # Convert range and error to kilometers
    range_km = range_value / 1000
    range_error_km = range_error / 1000
    median_meander_length_km = median_meander_length / 1000  # Convert to km

    rows = []
    found = False
    fieldnames = [
        'river_name', 'variogram_range', 'variogram_range_error',
        'avg_width', 'dominant_wavelength', 'dominant_wavelength_error',
        'half_wavelength', 'median_meander_length'  # Added field
    ]
    
    try:
        with open(original_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            original_fieldnames = reader.fieldnames
            
            for field in original_fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)

            for row in reader:
                if row['river_name'] == river_name:
                    row['variogram_range'] = f"{range_km:.2f}"
                    row['variogram_range_error'] = f"{range_error_km:.2f}"
                    row['avg_width'] = f"{avg_width:.2f}"
                    row['dominant_wavelength'] = f"{dominant_wavelength:.2f}"
                    row['dominant_wavelength_error'] = f"{dominant_wavelength_error:.2f}"
                    row['half_wavelength'] = f"{half_wavelength:.2f}"
                    row['median_meander_length'] = f"{median_meander_length_km:.2f}"  # Added field
                    found = True
                rows.append(row)

    except FileNotFoundError:
        logger.error(f"File not found: {original_file_path}")
        found = False

    if not found:
        new_row = {
            'river_name': river_name,
            'variogram_range': f"{range_km:.2f}",
            'variogram_range_error': f"{range_error_km:.2f}",
            'avg_width': f"{avg_width:.2f}",
            'dominant_wavelength': f"{dominant_wavelength:.2f}",
            'dominant_wavelength_error': f"{dominant_wavelength_error:.2f}",
            'half_wavelength': f"{half_wavelength:.2f}",
            'median_meander_length': f"{median_meander_length_km:.2f}"  # Added field
        }
        for field in fieldnames:
            if field not in new_row:
                new_row[field] = ''
        rows.append(new_row)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def process_river(river):
    try:
        # Read and prepare data
        df = pd.read_csv(f'data/{river}_recalculated_edited.csv')
        df = df.sort_values('dist_out').reset_index(drop=True)
        df['log_lambda'] = np.log1p(df['lambda'])

        # Calculate average meander length
        median_meander_length = df['meand_len'].median()  # Added line

        # Detrend log-transformed lambda
        detrended_log_lambda = detrend_data(df[['dist_out', 'log_lambda']], 'log_lambda')
        df['detrended_log_lambda'] = detrended_log_lambda

        # Calculate lag distances and empirical semivariance
        coords = df['dist_out'].values
        values = df['detrended_log_lambda'].values
        
        # Calculate median spacing between points
        median_spacing = np.median(np.abs(np.diff(coords)))
        
        # Calculate max distance (half the river length)
        max_distance = (coords.max() - coords.min()) * .5
        
        # Use median spacing as step size
        lag_step = median_spacing
        
        # Create lags up to max_distance
        lags = np.arange(1, max_distance, lag_step)
        
        # If we have too few lags, use half the median spacing
        if len(lags) < 10:
            lag_step = median_spacing
            lags = np.arange(lag_step, max_distance, lag_step)

            
        logger.info(f"{river} - points: {len(values)}, max_distance: {max_distance:.2f}, " 
                   f"median_spacing: {median_spacing:.2f}, step: {lag_step:.2f}, n_lags: {len(lags)}")
        
        # Calculate empirical semivariance using vectorized operations
        semivariance = []
        valid_lags = []
        
        # Create distance matrix once
        i, j = np.triu_indices(len(coords), k=1)
        distances = np.abs(coords[j] - coords[i])
        value_diffs = values[j] - values[i]
        
        for h in lags:
            # Find pairs within the current lag bin
            mask = np.abs(distances - h) <= lag_step/2
            if np.any(mask):
                gamma = np.mean(value_diffs[mask]**2) / 2
                semivariance.append(gamma)
                valid_lags.append(h)

        if not semivariance:
            raise ValueError(f"No valid semivariance values calculated for {river}")
            
        semivariance = np.array(semivariance)
        lags = np.array(valid_lags)

        # Check sampling regularity
        spacings = np.abs(np.diff(coords))
        
        logger.info(f"{river} sampling stats:")
        logger.info(f"  Mean spacing: {np.mean(spacings):.2f}")
        logger.info(f"  Median spacing: {np.median(spacings):.2f}")
        logger.info(f"  Std spacing: {np.std(spacings):.2f}")
        logger.info(f"  CV spacing: {np.std(spacings)/np.mean(spacings):.3f}")  # Coefficient of variation
        
        # Plot spacing histogram
        plt.figure(figsize=(6, 2))
        plt.rcParams.update({
            'font.size': 13,
            'axes.labelsize': 13,
            'axes.titlesize': 13,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 8
        })

        
        # Create and fit the custom Variogram
        variogram = Variogram(df) 
        fitted_params = variogram.fit(semivariance, lags)
        
        # Extract parameters
        theoretical_range = fitted_params['a']
        effective_range = 3 * theoretical_range  # convert to effective range for reporting
        
        # Get wavelengths and convert to km
        dominant_wavelength = variogram.get_dominant_wavelength() / 1000
        second_harmonic = variogram.get_second_harmonic() / 1000 if variogram.get_second_harmonic() is not None else None
        
        # Calculate average width
        avg_width = df['width'].mean()

        # Get effective range and its error
        effective_range, effective_range_error = variogram.get_effective_range_with_error()
        
        # Get wavelength and its error
        component_info = variogram.get_periodic_component_info()
        dominant_wavelength = component_info[0]['wavelength'] / 1000  # convert to km
        wavelength_error = component_info[0]['wavelength_error'] / 1000  # convert to km
        
        #Update supplementary table with both errors and meander length
        update_supplementary_table(
            river_name=river,
            range_value=effective_range,
            range_error=effective_range_error,
            avg_width=avg_width,
            dominant_wavelength=dominant_wavelength,
            dominant_wavelength_error=wavelength_error,
            half_wavelength=dominant_wavelength/2,
            median_meander_length=median_meander_length  # Added parameter
        )

        # Determine if the second harmonic should be kept
        keep_second_harmonic, reason = variogram.should_keep_second_harmonic()
        logger.info(f"{river} - Second harmonic decision: {reason}")

        # Plot experimental variogram and fitted model
        plt.figure(figsize=(6, 3))
        plt.rcParams.update({
            'font.size': 13,
            'axes.labelsize': 13,
            'axes.titlesize': 13,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 8
        })
        
        # Plot empirical semivariogram as black dots
        scatter = plt.plot(lags / 1000, semivariance, 'ko', markersize=4)[0]
        
        # Generate fitted model values and plot the red fitted-model line with direct labeling
        h_smooth = np.linspace(np.min(lags), np.max(lags), 200)
        fitted_values = variogram.get_model_values(h_smooth)
        line1 = plt.plot(
            h_smooth / 1000,
            fitted_values,
            'r-',
            linewidth=2,
            label=f"Fitted Model ($L_C$ = {effective_range/1000:.1f} km, $L_\lambda$ = {dominant_wavelength:.1f} km)"
        )[0]

        # Plot vertical lines indicating Lc and Lλ/2
        plt.axvline(
            x=effective_range / 1000,
            color='darkred',
            linestyle='--',
            linewidth=2.5,
            alpha=0.8
        )
        
        # Add vertical line for Lλ/2
        plt.axvline(
            x=dominant_wavelength/2,
            color='darkblue',
            linestyle='--',
            linewidth=2.5,
            alpha=0.8
        )

        # Set axis labels (the x-axis label remains simple; no additional or "Ac" label)
        plt.xlabel('Lag Distance (km)')
        plt.ylabel('Semivariance')
        plt.xlim(0, np.max(lags) / 1000)
        plt.ylim(np.min(semivariance) * 0.9, np.max(semivariance) * 1.1)

        # Use a legend that now only includes the directly labeled red fitted model line
        plt.legend(loc='upper right', fontsize=9, frameon=False)
        
        plt.tight_layout()

        plt.close()
        
       
        # Get all relevant parameters from the variogram
        c0, c, a = fitted_params['c0'], fitted_params['c'], fitted_params['a']
        
        # Calculate statistics
        regularity = variogram.calculate_regularity()
        total_wave_height = variogram.calculate_total_wave_height()
        
        # Create comprehensive results dictionary
        return {
            'River': river,
            'Nugget (c0)': c0,
            'Partial_Sill (c)': c,
            'Total_Sill': c0 + c,
            'Theoretical_Range': theoretical_range,
            'Effective_Range': effective_range,
            'Dominant_Wavelength': dominant_wavelength,
            'Half_Wavelength': dominant_wavelength/2,
            'Average_Width': avg_width,
            'Regularity': regularity,
            'Total_Wave_Height': total_wave_height,
            # Add periodic component parameters
            **{f'Periodic_Amplitude_{i+1}': fitted_params[f'p{3*i}'] for i in range(variogram.n_periodic_components)},
            **{f'Periodic_Phase_{i+1}': fitted_params[f'p{3*i+1}'] for i in range(variogram.n_periodic_components)},
            **{f'Periodic_Wavelength_{i+1}': fitted_params[f'p{3*i+2}'] for i in range(variogram.n_periodic_components)},
            'Second_Harmonic': second_harmonic,
            'Keep_Second_Harmonic': keep_second_harmonic,
        }

    except Exception as e:
        logger.error(f"Error processing {river}: {str(e)}")
        return None

if __name__ == '__main__':
    # List of river names
    river_names = ["MUSA", "B14_2", "ARG_LAKE_2", "COLOMBIA_2011_2", "VENEZ_2022_N", "VENEZ_2023", "ANJOBONY",
                   "RUVU_2", "V7_2", "V11_2", "LILONGWE", "TURKWEL", "MANGOKY", "BEMARIVO"]

    # Create a directory for plots if it doesn't exist

    # Process all rivers
    results = []
    for river in river_names:
        result = process_river(river)
        if result:
            results.append(result)
        else:
            logger.warning(f"Skipping {river} due to processing error")

    # Save all parameters to a separate CSV file
    if results:
        range_df = pd.DataFrame(results)
        range_df.to_csv('best_parameters.csv', index=False)
        logger.info("All parameters saved to best_parameters.csv")
        logger.info("Supplementary table updated with effective range, average width, dominant wavelength, and half wavelength at: /Users/jakegearon/staging/Revision 1/supplementary_table.csv")
    else:
        logger.error("No rivers were successfully processed")
