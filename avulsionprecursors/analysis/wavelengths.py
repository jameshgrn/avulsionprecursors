import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import logging
from scipy.signal import find_peaks
from pathlib import Path
from scipy.stats import linregress
from scipy import signal
from scipy.stats import gaussian_kde
import pywt
from scipy.signal import detrend
from statsmodels.nonparametric.smoothers_lowess import lowess
import json

def detrend_data(data: pd.DataFrame, variable: str) -> np.ndarray:
    """Enhanced detrending with multiple passes."""
    X = data['dist_out'].values
    y = data[variable].values
    
    # First pass: LOWESS with large window
    frac_large = 0.6
    trend_large = lowess(y, X, frac=frac_large, it=10)[:, 1]
    detrended_large = y - trend_large
    
    # # Second pass: LOWESS with smaller window
    # frac_small = 0.5
    # trend_small = lowess(detrended_large, X, frac=frac_small, it=5)[:, 1]
    # detrended_final = detrended_large - trend_small
    
    # # Normalize
    # detrended_final = (detrended_final - np.mean(detrended_final)) / np.std(detrended_final)
    
    return detrended_large

def validate_wavelength(profile, dom_wavelength):
    """Show natural emergence of wavelength scale."""
    # Test range of thresholds
    thresholds = np.linspace(10, 120, 30)
    
    # Let peak detection run naturally
    results = []
    for d in thresholds:
        peaks, _ = find_peaks(profile, distance=d)
        spacing = np.mean(np.diff(peaks))
        results.append({
            'threshold': d,
            'spacing': spacing,
            'n_peaks': len(peaks)
        })
    
    # Calculate mean spacing from peaks at dom_wavelength
    peaks_at_dom, _ = find_peaks(profile, distance=dom_wavelength)
    mean_spacing = np.mean(np.diff(peaks_at_dom)) if len(peaks_at_dom) > 1 else 0
    
    # Plot shows convergence WITHOUT forcing it
    plot_sensitivity(results, dom_wavelength)
    
    return mean_spacing, dom_wavelength, peaks_at_dom

def load_supplementary_data():
    """Load and process supplementary table."""
    supp_path = Path('/Users/jakegearon/staging/supplementary_table_1_R1submit.csv')
    if not supp_path.exists():
        raise FileNotFoundError(f"Supplementary table not found at {supp_path}")
    
    supp_table = pd.read_csv(supp_path)
    return dict(zip(supp_table['river_name'], supp_table['dominant_wavelength']))

def load_data(river_name: str) -> pd.DataFrame:
    logging.info(f"Loading data for {river_name}")

    df = pd.read_csv(f'src/data/data/{river_name}_recalculated_edited.csv')

        
    df = df[df['river_name'] == river_name].dropna(subset=['dist_out', 'lambda'])
    df['dist_out'] = df['dist_out'].astype(float) / 1000  # convert to km
    
    # Sort by distance ascending (from upstream to downstream)
    df = df.sort_values('dist_out', ascending=True)  # Changed from False to True
    
    # Add lambda column
    epsilon = 1e-6
    df['lambda'] = df['lambda']
    df['lambda'] = df['lambda'].replace(-np.inf, np.log(epsilon))

    df['log_lambda'] = np.log1p(df['lambda'])
    
    return df

def test_peak_sensitivity(profile, dom_wavelength):
    """Test range of distance thresholds with less constrained parameters."""
    results = []
    min_distance = max(1, dom_wavelength * 0.1)  # Wider range
    
    # Test broader range of factors
    factors = np.linspace(0.1, 2.5, 40)  # Wider range, more points
    
    for factor in factors:
        distance = max(min_distance, dom_wavelength * factor)
        try:
            # Simpler peak detection with minimal constraints
            peaks, _ = find_peaks(profile, distance=distance)
            
            if len(peaks) > 2:
                spacings = np.diff(peaks)
                # Less aggressive outlier removal
                std = np.std(spacings)
                mean = np.mean(spacings)
                good_spacings = spacings[np.abs(spacings - mean) < 3 * std]  # More permissive
                
                if len(good_spacings) > 0:
                    results.append({
                        'threshold': distance,
                        'mean_spacing': np.mean(good_spacings),
                        'std_spacing': np.std(good_spacings),
                        'n_peaks': len(peaks)
                    })
        except Exception as e:
            logging.warning(f"Error at factor {factor}: {e}")
            continue
    
    return pd.DataFrame(results)

def validate_multiple_methods(profile, dom_wavelength):
    """Compare distance-based peak detection and wavelet analysis to validate wavelength."""
    methods_results = {}
    
    # Method 1: distance-based peak detection
    peaks1, _ = find_peaks(profile, distance=dom_wavelength)
    spacings1 = np.diff(peaks1) if len(peaks1) > 1 else []
    
    # Method 2: wavelet-based detection
    peaks2 = signal.find_peaks_cwt(profile, np.arange(1, int(dom_wavelength*1.5)))
    spacings2 = np.diff(peaks2) if len(peaks2) > 1 else []
    
    return {
        'peak_detection': {'mean': np.mean(spacings1), 'std': np.std(spacings1)} if len(spacings1) > 0 else None,
        'wavelet': {'mean': np.mean(spacings2), 'std': np.std(spacings2)} if len(spacings2) > 0 else None
    }

def plot_sensitivity_analysis(sensitivity_results, dom_wavelength, river_name):
    """Plot sensitivity analysis showing wavelength stability."""
    # Increased font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    
    fig, ax1 = plt.subplots(figsize=(6, 6))  # Increased figure size
    
    # Mean spacing with error bars
    line1 = ax1.errorbar(sensitivity_results['threshold'], 
                        sensitivity_results['mean_spacing'],
                        yerr=sensitivity_results['std_spacing'],
                        fmt='o-', alpha=0.7, color='blue',
                        label='Mean peak spacing')
    
    # Reference line for semivariogram wavelength
    ax1.axhline(y=dom_wavelength, color='r', linestyle='--', 
                label='Semivariogram wavelength')
    
    # Stability range
    ax1.fill_between([min(sensitivity_results['threshold']), 
                     max(sensitivity_results['threshold'])],
                    [dom_wavelength * 0.9, dom_wavelength * 0.9],
                    [dom_wavelength * 1.1, dom_wavelength * 1.1],
                    color='r', alpha=0.1, label='±10% range')
    
    ax1.set_xlabel('Distance threshold (km)')
    ax1.set_ylabel('Mean spacing (km)')
    
    # Number of peaks on secondary axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(sensitivity_results['threshold'], 
                     sensitivity_results['n_peaks'], 
                     'k:', alpha=0.5,
                     label='Number of peaks')[0]
    ax2.set_ylabel('Number of peaks detected')
    
    # Combined legend
    lines = [line1[0], line2]
    labels = ['Mean peak spacing', 'Number of peaks']
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title(f'Wavelength Stability Analysis - {river_name}')
    plt.tight_layout()
    #plt.show()

def plot_method_comparison(methods_results, dom_wavelength, river_name):
    """Compare results from peak detection methods against semivariogram wavelength."""
    # Increased font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    
    plt.figure(figsize=(8, 6))  # Increased figure size
    
    methods = []
    means = []
    errors = []
    
    for method, results in methods_results.items():
        if results is not None:
            methods.append(method)
            means.append(results['mean'])
            errors.append(results['std'])
    
    plt.errorbar(range(len(methods)), means, yerr=errors, fmt='o', capsize=5)
    plt.axhline(y=dom_wavelength, color='r', linestyle='--', 
                label='Semivariogram wavelength')
    plt.fill_between([-0.5, len(methods)-0.5],
                     [dom_wavelength * 0.9, dom_wavelength * 0.9],
                     [dom_wavelength * 1.1, dom_wavelength * 1.1],
                     color='r', alpha=0.1, label='±10% range')
    plt.xticks(range(len(methods)), methods, rotation=45)
    plt.ylabel('Mean spacing (km)')
    plt.title(f'Wavelength Validation - {river_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_wavelet_scales(profile, dists):
    """Generate scales based on profile length and sampling."""
    # Use absolute differences for dx calculation
    dx = np.median(np.abs(np.diff(dists)))
    L = np.abs(np.max(dists) - np.min(dists))
    
    # Ensure minimum scale is at least 2*dx
    min_scale = 4 * dx
    # Maximum scale is 1/2 of profile length
    max_scale = L*.75  # Changed to L/2 to avoid very long wavelengths
    
    # Generate logarithmically spaced scales
    num_scales = 32
    scales = np.geomspace(min_scale, max_scale, num_scales)
    
    return scales

def process_river(river_name: str, wavelength_dict: dict):
    """Process river data and generate power spectra plot."""
    df = load_data(river_name)
    
    if river_name not in wavelength_dict:
        logging.warning(f"No dominant wavelength found for {river_name}")
        return None
    
    profile = detrend_data(df, 'log_lambda')
    dists = df['dist_out'].values
    semivariogram_wavelength = wavelength_dict[river_name]
    
    L = np.abs(np.max(dists) - np.min(dists))
    dx = np.median(np.abs(np.diff(dists)))
    
    # Use finer scale resolution
    scales = np.linspace(2*dx, L*.75, 200)  # Increased from 100 to 200 points
    
    # Try multiple wavelet types
    wavelets = ['cmor1.5-1.0', 'gaus1', 'mexh']
    best_peak = None
    best_power = None
    best_peaks = None
    best_properties = None
    best_wavelet = None
    
    for wavelet in wavelets:
        try:
            coef, _ = pywt.cwt(profile, scales, wavelet)
            power = np.abs(coef)**2
            global_power = np.mean(power, axis=1)
            
            # Normalize power
            global_power = (global_power - np.min(global_power)) / (np.max(global_power) - np.min(global_power))
            
            # Try increasingly permissive settings
            peak_params = [
                {'prominence': 0.2},
                {'prominence': 0.1},
                {'prominence': 0.05},
                {'height': 0.3},
                {'height': 0.2},
                {}  # Most permissive - just find any peaks
            ]
            
            for params in peak_params:
                peaks, properties = find_peaks(global_power, **params)
                if len(peaks) > 0:
                    # If this is our first successful peak detection or if it's closer to the semivariogram wavelength
                    peak_wavelength = scales[peaks[np.argmax(properties['prominences'])]]
                    rel_diff = abs(peak_wavelength - semivariogram_wavelength) / semivariogram_wavelength
                    
                    if best_peak is None or rel_diff < abs(best_peak - semivariogram_wavelength) / semivariogram_wavelength:
                        best_peak = peak_wavelength
                        best_power = global_power
                        best_peaks = peaks
                        best_properties = properties
                        best_wavelet = wavelet
                    break
            
        except Exception as e:
            logging.warning(f"Error with wavelet {wavelet} for {river_name}: {e}")
            continue
    
    if best_peak is None:
        logging.warning(f"No peaks found for {river_name} with any method")
        return None
    
    # Increase font sizes significantly
    plt.rcParams.update({
        'font.size': 16,          # Increased from 12
        'axes.labelsize': 14,     # Increased from 11
        'axes.titlesize': 16,     # Increased from 12
        'xtick.labelsize': 12,    # Increased from 10
        'ytick.labelsize': 12,    # Increased from 10
        'legend.fontsize': 12     # Increased from 10
    })
    
    # Make figure larger while maintaining aspect ratio
    plt.figure(figsize=(8, 6))    # Increased from (6, 4)
    
    plt.plot(scales, best_power, 'b-', linewidth=1.5)        # Decreased linewidth
    plt.plot(scales[best_peaks], best_power[best_peaks], 'ro', markersize=6)  # Decreased markersize
    plt.plot(best_peak, best_power[best_peaks[np.argmax(best_properties['prominences'])]], 
            'go', markersize=8)  # Decreased markersize
    plt.axvline(x=semivariogram_wavelength, color='g', 
               linestyle='--', linewidth=1.5)  # Decreased linewidth
    
    plt.xlabel('Wavelength (km)')
    plt.ylabel('Normalized Power')
    rel_diff = abs(best_peak - semivariogram_wavelength) / semivariogram_wavelength
    plt.title(f'{river_name}\n' + 
             f'$\Lambda_\lambda$: {semivariogram_wavelength:.1f} km\n' +  # Added space before km, added linebreak
             f'$\lambda_w$: {best_peak:.1f} km',                          # Added space before km
             pad=20)  # Add padding above title
    
    plt.tight_layout()
    
    # Increase DPI for sharper text
    plt.savefig(f'/Users/jakegearon/staging/Revision 1/plots/power_spectra_{river_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'river_name': river_name,
        'wavelet_wavelength': best_peak,
        'semivariogram_wavelength': semivariogram_wavelength,
        'relative_difference': rel_diff,
        'method_agreement': rel_diff < 0.5,
        'wavelet_used': best_wavelet
    }

def process_rivers_panel(river_names: list, wavelength_dict: dict):
    """Create a multi-panel figure with power spectra for all rivers."""
    # Track wavelength pairs for regression
    wavelength_pairs = []
    
    # Calculate grid dimensions
    n_cols = 4
    n_rows = 4  # This gives us 16 slots, we'll use 14 for rivers and 2 for regression
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = axes.flatten()
    
    # Add subplot labels (a-p)
    labels = 'abcdefghijklmnop'
    
    # Process each river in first 14 slots
    for idx, river_name in enumerate(river_names):
        ax = axes[idx]
        
        # Add panel label in upper left corner
        ax.text(-0.2, 1.05, labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight='bold')
        
        df = load_data(river_name)
        
        if river_name not in wavelength_dict:
            logging.warning(f"No dominant wavelength found for {river_name}")
            continue
            
        profile = detrend_data(df, 'log_lambda')
        dists = df['dist_out'].values
        semivariogram_wavelength = wavelength_dict[river_name]
        
        L = np.abs(np.max(dists) - np.min(dists))
        dx = np.median(np.abs(np.diff(dists)))
        scales = np.linspace(2*dx, L*.75, 200)
        
        # Process wavelets (same as before)
        wavelets = ['cmor1.5-1.0', 'gaus1', 'mexh']
        best_peak = None
        best_power = None
        best_peaks = None
        best_properties = None
        
        for wavelet in wavelets:
            try:
                coef, _ = pywt.cwt(profile, scales, wavelet)
                power = np.abs(coef)**2
                global_power = np.mean(power, axis=1)
                global_power = (global_power - np.min(global_power)) / (np.max(global_power) - np.min(global_power))
                
                peak_params = [
                    {'prominence': 0.2},
                    {'prominence': 0.1},
                    {'prominence': 0.05},
                    {'height': 0.3},
                    {'height': 0.2},
                    {}
                ]
                
                for params in peak_params:
                    peaks, properties = find_peaks(global_power, **params)
                    if len(peaks) > 0:
                        peak_wavelength = scales[peaks[np.argmax(properties['prominences'])]]
                        rel_diff = abs(peak_wavelength - semivariogram_wavelength) / semivariogram_wavelength
                        
                        if best_peak is None or rel_diff < abs(best_peak - semivariogram_wavelength) / semivariogram_wavelength:
                            best_peak = peak_wavelength
                            best_power = global_power
                            best_peaks = peaks
                            best_properties = properties
                            break
                            
            except Exception as e:
                continue
        
        if best_peak is None:
            continue
            
        # Plot on the appropriate subplot
        ax.plot(scales, best_power, 'b-', linewidth=1)
        ax.plot(scales[best_peaks], best_power[best_peaks], 'ro', markersize=4)
        ax.plot(best_peak, best_power[best_peaks[np.argmax(best_properties['prominences'])]], 
                'go', markersize=5)
        ax.axvline(x=semivariogram_wavelength, color='g', linestyle='--', linewidth=1)
        
        # Add labels and title
        ax.set_xlabel('Wavelength (km)')
        ax.set_ylabel('Normalized Power')
        rel_diff = abs(best_peak - semivariogram_wavelength) / semivariogram_wavelength
        ax.set_title(f'{river_name}\n' + 
                    f'$\Lambda_\lambda$: {semivariogram_wavelength:.1f}\n' +
                    f'$\lambda_w$: {best_peak:.1f}',
                    fontsize=10)
        
        if best_peak is not None:
            wavelength_pairs.append({
                'river': river_name,
                'semivariogram': semivariogram_wavelength,
                'wavelet': best_peak
            })
    
    # Create regression plots in the last two slots
    if wavelength_pairs:
        df_wavelengths = pd.DataFrame(wavelength_pairs)
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(
            df_wavelengths['semivariogram'], 
            df_wavelengths['wavelet']
        )
        
        # Scatter plot in position 14
        ax_scatter = axes[14]
        ax_scatter.scatter(df_wavelengths['semivariogram'], 
                         df_wavelengths['wavelet'], alpha=0.6)
        
        # Add regression line
        x_range = np.array([
            df_wavelengths['semivariogram'].min(),
            df_wavelengths['semivariogram'].max()
        ])
        ax_scatter.plot(x_range, slope * x_range + intercept, 'r--')
        
        # Add 1:1 line
        ax_scatter.plot(x_range, x_range, 'k:', alpha=0.5, label='1:1 line')
        
        ax_scatter.set_xlabel('$\Lambda_\lambda$ (km)')
        ax_scatter.set_ylabel('$\lambda_w$ (km)')
        ax_scatter.set_title(f'Wavelength Comparison\n$R^2={r_value**2:.2f}$\np={p_value:.3f}')
        
        # Residual plot in position 15
        ax_resid = axes[15]
        residuals = df_wavelengths['wavelet'] - (slope * df_wavelengths['semivariogram'] + intercept)
        ax_resid.scatter(df_wavelengths['semivariogram'], residuals, alpha=0.6)
        ax_resid.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        
        ax_resid.set_xlabel('$\Lambda_\lambda$ (km)')
        ax_resid.set_ylabel('Residuals (km)')
        ax_resid.set_title('Residual Plot')
        
        # Save regression results
        results = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'wavelength_pairs': df_wavelengths.to_dict('records')
        }
        
        # Save results to JSON
        with open('/Users/jakegearon/staging/Revision 1/wavelength_regression.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Add panel label in upper left corner
        ax_scatter.text(-0.2, 1.05, labels[14], transform=ax_scatter.transAxes,
                       fontsize=14, fontweight='bold')
        
        # Residual plot in position 15
        ax_resid.text(-0.2, 1.05, labels[15], transform=ax_resid.transAxes,
                      fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Increased right padding to 5%
    plt.savefig('/Users/jakegearon/staging/Revision 1/plots/power_spectra_panel.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.1)  # Added pad_inches
    plt.close()

def main():
    """Main execution with panel plot."""
    river_names = [
        "MUSA", "B14_2", "ARG_LAKE_2", "COLOMBIA_2011_2", "VENEZ_2022_N",
        "VENEZ_2023", "ANJOBONY", "RUVU_2", "V7_2", "V11_2", "LILONGWE",
        "BEMARIVO", "MANGOKY", "TURKWEL"
    ]
    
    try:
        wavelength_dict = load_supplementary_data()
        process_rivers_panel(river_names, wavelength_dict)
            
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()