"""BASED (Bayesian Analysis of Superelevation and Elevation Differences) implementation."""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from ..sword.base import SwordReach, SwordNode
import pickle
import geopandas as gpd
import joblib

class BASEDAnalyzer:
    """Analyzes river morphology using BASED methodology."""
    
    def __init__(
        self,
        model_path: str,
        params_path: str,
        min_superelevation: float = 0.01,
        min_slope: float = 1e-5
    ):
        self.min_superelevation = min_superelevation
        self.min_slope = min_slope
        self.model = self._load_model(model_path)
        self.params = self._load_params(params_path)
    
    def analyze_reach(self, reach: SwordReach) -> pd.DataFrame:
        """
        Analyze a reach using BASED methodology.
        
        Args:
            reach: SwordReach object with elevation data
            
        Returns:
            DataFrame with BASED analysis results
        """
        # Create initial dataframe from nodes
        df = self._nodes_to_dataframe(reach)
        
        # Calculate basic parameters
        df['slope'] = df['slope'].clip(lower=self.min_slope)
        df = self._calculate_discharge(df)
        df = self._predict_depth(df)
        
        # Calculate ridge and floodplain parameters
        df = self._calculate_ridge_parameters(df)
        df = self._calculate_gamma(df)
        df = self._calculate_superelevation(df)
        
        # Calculate final metrics
        df['lambda'] = df['gamma_mean'] * df['superelevation_mean']
        
        # Add quality flags
        df = self._add_quality_flags(df)
        
        return df
    
    def _nodes_to_dataframe(self, reach: SwordReach) -> pd.DataFrame:
        """Convert reach nodes to DataFrame."""
        records = []
        for node in reach.nodes:
            if not node.cross_section:
                continue
                
            record = {
                'node_id': node.node_id,
                'reach_id': reach.reach_id,
                'dist_out': node.dist_out,
                'width': node.width,
                'slope': node.slope or self.min_slope,
                'elevation': node.elevation
            }
            records.append(record)
        
        return pd.DataFrame(records) 

    def _load_model(self, model_path: str) -> xgb.XGBRegressor:
        """Load XGBoost model from file."""
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model

    def _load_params(self, params_path: str) -> Dict[str, float]:
        """Load discharge parameters from pickle file."""
        with open(params_path, 'rb') as f:
            return pickle.load(f)

    def _calculate_discharge(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate corrected discharge using power law."""
        def inverse_power_law(y: float, a: float, b: float) -> float:
            return (y / a) ** (1 / b)
        
        if 'discharge_value' not in df.columns:
            raise ValueError("Discharge value missing from data")
        
        df['corrected_discharge'] = df['discharge_value'].apply(
            lambda x: inverse_power_law(x, *self.params)
        )
        return df

    def _predict_depth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict depth using XGBoost model."""
        features = df[['width', 'slope', 'corrected_discharge']].copy()
        features.columns = ['width', 'slope', 'discharge']
        
        df['XGB_depth'] = self.model.predict(features)
        df['XGB_depth'] = df['XGB_depth'].clip(lower=0)
        return df

    def _calculate_ridge_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ridge and floodplain parameters."""
        # Calculate distances to river center
        df['floodplain1_dist_to_river_center'] = abs(
            df['floodplain1_dist_along'] - df['channel_dist_along']
        )
        df['floodplain2_dist_to_river_center'] = abs(
            df['floodplain2_dist_along'] - df['channel_dist_along']
        )
        
        # Calculate ridge slopes with safeguards
        epsilon = 1e-10
        df['ridge1_slope'] = (
            (df['ridge1_elevation'] - df['floodplain1_elevation']) /
            (abs(df['ridge1_dist_along'] - df['floodplain1_dist_along']) + epsilon)
        )
        df['ridge2_slope'] = (
            (df['ridge2_elevation'] - df['floodplain2_elevation']) /
            (abs(df['ridge2_dist_along'] - df['floodplain2_dist_along']) + epsilon)
        )
        
        # Handle infinities
        df['ridge1_slope'] = df['ridge1_slope'].replace([np.inf, -np.inf], np.nan)
        df['ridge2_slope'] = df['ridge2_slope'].replace([np.inf, -np.inf], np.nan)
        
        return df

    def _calculate_gamma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gamma values."""
        df['gamma1'] = np.abs(df['ridge1_slope']) / df['slope']
        df['gamma2'] = np.abs(df['ridge2_slope']) / df['slope']
        
        # Handle single-side measurements
        df['gamma_mean'] = np.where(
            df['gamma2'].isna(),
            df['gamma1'],
            df[['gamma1', 'gamma2']].mean(axis=1, skipna=True)
        )
        return df

    def _calculate_superelevation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate superelevation metrics."""
        from .metrics import calculate_ridge_metrics
        
        # Calculate a_b ratios
        df['a_b_1'] = (df['ridge1_elevation'] - df['channel_elevation']) / df['XGB_depth']
        df['a_b_2'] = (df['ridge2_elevation'] - df['channel_elevation']) / df['XGB_depth']
        df['a_b'] = df[['a_b_1', 'a_b_2']].mean(axis=1, skipna=True)
        
        # Determine denominator based on a_b ratio
        conditions = [df['a_b'] <= 1.25, df['a_b'] > 1.25]
        choices = [
            df['XGB_depth'],
            df[['ridge1_elevation', 'ridge2_elevation']].mean(axis=1, skipna=True) - 
            df['channel_elevation']
        ]
        df['corrected_denominator'] = np.select(conditions, choices)
        
        # Calculate superelevation
        df['superelevation1'] = np.maximum(
            (df['ridge1_elevation'] - df['floodplain1_elevation']) / 
            df['corrected_denominator'],
            self.min_superelevation
        )
        df['superelevation2'] = np.maximum(
            (df['ridge2_elevation'] - df['floodplain2_elevation']) / 
            df['corrected_denominator'],
            self.min_superelevation
        )
        
        # Handle single-side measurements
        df['superelevation_mean'] = np.where(
            df['superelevation2'].isna(),
            df['superelevation1'],
            df[['superelevation1', 'superelevation2']].mean(axis=1, skipna=True)
        )
        
        return df

    def _add_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality control flags to the data."""
        df['flag_high_gamma'] = df['gamma_mean'] > 100
        df['flag_low_superelevation'] = df['superelevation_mean'] < 0.05
        df['flag_single_side'] = df['gamma2'].isna()
        
        # Add confidence score
        df['confidence_score'] = 1.0
        df.loc[df['flag_high_gamma'], 'confidence_score'] *= 0.5
        df.loc[df['flag_low_superelevation'], 'confidence_score'] *= 0.7
        df.loc[df['flag_single_side'], 'confidence_score'] *= 0.8
        
        return df

def inverse_power_law(y: float, a: float, b: float) -> float:
    """
    Compute the inverse power law.

    Given an equation of the form Q = a * d^b, this returns d = (Q/a)^(1/b).
    """
    return (y / a) ** (1 / b)

def run_based_model(discharge: float, wse_profile: gpd.GeoDataFrame) -> dict:
    """
    Run the BASED model on uncorrected discharge and water surface elevation (WSE) profile data.
    First, correct the discharge using the inverse power law (with parameters loaded from a pickle file),
    then predict channel depth using a pre-trained XGBoost model from a .ubj file.

    Args:
        discharge (float): The uncorrected discharge value for the reach.
        wse_profile (gpd.GeoDataFrame): A GeoDataFrame containing the processed ICESat-2 WSE profile;
            it must include an 'orthometric_height' column representing the water surface elevation.

    Returns:
        dict: A dictionary with BASED model results including:
            - 'discharge': the corrected discharge,
            - 'predicted_depth': the channel depth predicted by the BASED model,
            - 'median_wse': the median water surface elevation from the profile, and
            - 'channel_elevation': an estimate of channel elevation computed as (median_wse - predicted_depth).
    """
    # Load inverse power law parameters
    try:
        with open("inverse_power_law_params.pickle", "rb") as f:
            loaded_params = pickle.load(f)
        # Debug: print the loaded_params to inspect its structure
        print("Loaded inverse power law params:", loaded_params)
        if isinstance(loaded_params, dict):
            a = loaded_params["a"]
            b = loaded_params["b"]
        elif isinstance(loaded_params, (tuple, list)) and len(loaded_params) >= 2:
            a, b = loaded_params[0], loaded_params[1]
        elif hasattr(loaded_params, "shape") and loaded_params.shape[0] >= 2:
            # Handle numpy arrays
            a, b = loaded_params[0], loaded_params[1]
        else:
            raise ValueError("inverse_power_law_params.pickle is in an unrecognized format")
    except Exception as e:
        raise Exception(f"Error loading inverse power law parameters: {e}")
    
    # Correct the discharge using the inverse power law
    corrected_discharge = inverse_power_law(discharge, a, b)
    
    # Load the pre-trained BASED model from the correct .ubj file
    model = joblib.load("basic_model_20250202_164836.joblib")

    # Create a default feature vector for prediction.
    # The current model expects features: 'width', 'slope', 'discharge'
    default_width = 50.0
    default_slope = 0.001
    features = pd.DataFrame({
        'width': [default_width],
        'slope': [default_slope],
        'discharge': [corrected_discharge]
    })
    predicted_depth = model.predict(features)[0]
    predicted_depth = max(predicted_depth, 0)
    
    # Determine the median water surface elevation from the WSE profile.
    if 'orthometric_height' in wse_profile.columns and not wse_profile.empty:
        median_wse = wse_profile['orthometric_height'].median()
    else:
        median_wse = np.nan
    
    # Estimate channel elevation as the median WSE minus predicted depth.
    channel_elevation = median_wse - predicted_depth if not np.isnan(median_wse) else np.nan
    
    results = {
        "discharge": corrected_discharge,
        "predicted_depth": predicted_depth,
        "median_wse": median_wse,
        "channel_elevation": channel_elevation
    }
    return results