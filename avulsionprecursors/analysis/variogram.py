import numpy as np
from scipy.optimize import differential_evolution, minimize
import warnings

class Variogram:
    def __init__(self, data):
        self.data = data
        self.params = None
        self.param_errors = None
        self.n_periodic_components = 2

    def exponential_component(self, h, c0, c, a):
        return c0 + c * (1 - np.exp(-h / a))

    def periodic_component(self, h, *args):
        result = 0
        for i in range(0, len(args), 3):
            a, b, l = args[i:i+3]
            result += a * np.cos(2 * np.pi * h / l) + b * np.sin(2 * np.pi * h / l)
        return result

    def combined_model(self, h, params):
        c0, c, a = params[:3]
        periodic_params = params[3:]
        return self.exponential_component(h, c0, c, a) + self.periodic_component(h, *periodic_params)

    def objective_function(self, params, h, empirical_values, exponential_only=False, exp_weight=1):
        if exponential_only:
            modeled_values = self.exponential_component(h, *params)
        else:
            modeled_values = self.combined_model(h, params)
        
        exp_component = self.exponential_component(h, *params[:3])
        
        # Modified weighting scheme
        # Give less weight to points that are close together (likely correlated)
        # and points that are far apart (less reliable)
        weights = 1 / (h + h[1]) # h[1] is the lag spacing
        weights *= np.exp(-h / (np.max(h)))  # Downweight distant points
        weights /= np.sum(weights)  # Normalize weights
        
        # Add uncertainty scaling based on the empirical values
        uncertainty = np.maximum(0.1 * np.abs(empirical_values), 0.01)  # At least 10% uncertainty
        
        # Calculate weighted MSE with uncertainty scaling
        residuals = (empirical_values - modeled_values) / uncertainty
        mse = np.sum(weights * residuals ** 2)
        
        if not exponential_only:
            exp_residuals = (empirical_values - exp_component) / uncertainty
            exp_mse = np.sum(weights * exp_residuals ** 2)
            mse += exp_weight * exp_mse
        
        return mse

    def fit(self, empirical_values, h):
        # Single-stage fitting with reasonable bounds
        bounds = [
            (0, np.max(empirical_values)),  # c0 (nugget)
            (0, np.max(empirical_values)),  # c (sill)
            (0, np.max(h)),                 # a (range)
        ]
        
        max_amp = np.max(empirical_values) * 0.6
        max_wavelength = np.max(h) * 0.8  # Restrict to half the maximum lag distance
        
        for _ in range(self.n_periodic_components):
            bounds.extend([
                (-max_amp, max_amp),        # a (amplitude)
                (-max_amp, max_amp),        # b (amplitude)
                (h[1], max_wavelength)      # l (wavelength) - more restricted upper bound
            ])
        
        result = differential_evolution(
            lambda params: self.objective_function(params, h, empirical_values),
            bounds,
            maxiter=1000,
            seed=42
        )
        
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        
        # Second stage: Use minimize with BFGS method
        result_refined = minimize(
            lambda params: self.objective_function(params, h, empirical_values),
            result.x,
            method='BFGS',  # Changed back to BFGS which provides Hessian information
            options={'maxiter': 1000}
        )
        
        if not result_refined.success:
            warnings.warn("Refinement optimization failed, parameter errors may be unreliable")
            
        # Calculate parameter uncertainties
        try:
            # Get the number of data points and parameters
            n_points = len(h)
            n_params = len(result.x)
            
            # Calculate residuals and MSE with weights
            residuals = empirical_values - self.combined_model(h, result_refined.x)
            
            # Calculate weights as before
            weights = 1 / (h + h[1]) 
            weights *= np.exp(-h / (np.max(h) / 3))
            weights /= np.sum(weights)
            
            # Add intrinsic uncertainty of semivariance
            # For a variogram, the variance of γ(h) is approximately γ(h)²/N(h)
            # where N(h) is effectively the number of pairs at each lag
            # We'll estimate N(h) based on sampling density
            N_h = n_points / len(h)  # approximate number of pairs per lag
            intrinsic_variance = (empirical_values**2) / N_h
            
            # Combined uncertainty
            total_variance = intrinsic_variance + residuals**2
            mse = np.mean(total_variance * weights)
            
            print(f"DEBUG: MSE with intrinsic variance = {mse}")
            
            # Get Hessian inverse from BFGS
            hessian_inv = result_refined.hess_inv
            hessian_inv = np.array(hessian_inv, dtype=float)
            
            # Scale the covariance matrix with total uncertainty
            scaling_factor = 2 * mse / (n_points - n_params)
            print(f"DEBUG: New scaling factor = {scaling_factor}")
            
            covariance = scaling_factor * hessian_inv
            
            # Calculate parameter errors
            self.param_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
            
            # Set minimum error threshold based on parameter values and intrinsic uncertainty
            min_errors = np.maximum(
                0.05 * np.abs(result_refined.x),  # at least 5% of parameter value
                np.sqrt(scaling_factor)  # or based on overall fit uncertainty
            )
            
            self.param_errors = np.maximum(self.param_errors, min_errors)
            
        except Exception as e:
            warnings.warn(f"Failed to calculate parameter uncertainties: {str(e)}")
            self.param_errors = np.full_like(result_refined.x, np.nan)
        
        self.params = result_refined.x
        return dict(zip(['c0', 'c', 'a'] + 
                       [f'p{i}' for i in range(len(self.params) - 3)], 
                       self.params))

    def get_model_values(self, h):
        return self.combined_model(h, self.params)

    def get_exponential_component(self, h):
        c0, c, a = self.params[:3]
        return self.exponential_component(h, c0, c, a)

    def get_periodic_component(self, h):
        periodic_params = self.params[3:]
        return self.periodic_component(h, *periodic_params)

    def calculate_regularity(self):
        _, c, a = self.params[:3]
        return np.sqrt(c) / a

    def calculate_total_wave_height(self):
        _, c, _ = self.params[:3]
        periodic_amplitudes = self.params[3::3]  # Get all 'a' parameters from periodic components
        return 2 * (np.sum(np.abs(periodic_amplitudes)) + np.sqrt(c))

    def get_dominant_wavelength(self):
        periodic_params = self.params[3:]
        amplitudes = periodic_params[::3]
        wavelengths = periodic_params[2::3]
        return wavelengths[np.argmax(np.abs(amplitudes))]

    def get_second_harmonic(self):
        periodic_params = self.params[3:]
        amplitudes = periodic_params[::3]
        wavelengths = periodic_params[2::3]
        
        # Find indices of the two largest amplitudes
        sorted_indices = np.argsort(np.abs(amplitudes))
        if len(sorted_indices) < 2:
            warnings.warn("Less than two periodic components found.")
            return None
        
        # Get the second largest amplitude's wavelength
        second_largest_index = sorted_indices[-2]
        return wavelengths[second_largest_index]

    def variance_explained_by_harmonics(self):
        if self.params is None:
            raise ValueError("Model parameters are not set. Fit the model first.")

        # Get parameters
        c0, c, a = self.params[:3]  # nugget, partial sill, range
        periodic_params = self.params[3:]
        
        # For periodic components, we need to consider both sine and cosine terms together
        periodic_variances = []
        for i in range(self.n_periodic_components):
            # Get a and b coefficients for this component
            a_i = periodic_params[i*3]
            b_i = periodic_params[i*3 + 1]
            # Variance contribution is (a² + b²)/2 for each periodic component
            periodic_variances.append((a_i**2 + b_i**2)/2)

        # Total variance is sum of all components (including exponential)
        total_variance = c + sum(periodic_variances)

        # Calculate proportions
        exp_proportion = c / total_variance if total_variance > 0 else 0
        periodic_proportions = [v/total_variance if total_variance > 0 else 0 for v in periodic_variances]

        # Return both exponential and periodic variances as proportions
        return np.array([exp_proportion] + periodic_proportions)

    def should_keep_second_harmonic(self, threshold=0.05):
        variance_explained = self.variance_explained_by_harmonics()
        if len(variance_explained) < 3:  # 1 for exponential + at least 2 harmonics
            return False, "Less than two harmonics available."

        # Skip exponential variance (first element)
        periodic_variance = variance_explained[1:]
        second_harmonic_variance = periodic_variance[np.argsort(np.abs(periodic_variance))[-2]]
        return second_harmonic_variance > threshold, f"Second harmonic explains {second_harmonic_variance:.2%} of variance."

    def get_individual_periodic_components(self, h):
        """Return each periodic component separately"""
        periodic_params = self.params[3:]
        components = []
        
        for i in range(self.n_periodic_components):
            # Get parameters for this component
            a, b, l = periodic_params[i*3:(i+1)*3]
            component = a * np.cos(2 * np.pi * h / l) + b * np.sin(2 * np.pi * h / l)
            components.append(component)
            
        return components

    def get_periodic_component_info(self):
        """Get information about each periodic component including errors"""
        if self.params is None or self.param_errors is None:
            raise ValueError("Model must be fit first")
            
        periodic_params = self.params[3:]
        periodic_errors = self.param_errors[3:]
        variance_explained = self.variance_explained_by_harmonics()
        
        # Skip the first element (exponential variance)
        periodic_variance = variance_explained[1:]
        
        component_info = []
        for i in range(self.n_periodic_components):
            a, b, l = periodic_params[i*3:(i+1)*3]
            a_err, b_err, l_err = periodic_errors[i*3:(i+1)*3]
            
            amplitude = np.sqrt(a**2 + b**2)
            # Error propagation for amplitude
            amplitude_err = np.sqrt((a*a_err)**2 + (b*b_err)**2) / amplitude if amplitude > 0 else 0
            
            info = {
                'wavelength': l,
                'wavelength_error': l_err,
                'amplitude': amplitude,
                'amplitude_error': amplitude_err,
                'variance_explained': periodic_variance[i]
            }
            component_info.append(info)
            
        return component_info

    def get_effective_range_with_error(self):
        """
        Calculate the effective range (3*a for exponential model) and its uncertainty
        Returns:
            tuple: (effective_range, error)
        """
        if self.params is None or self.param_errors is None:
            raise ValueError("Model must be fit first")
            
        # For exponential model, effective range is 3*a
        a = self.params[2]  # Range parameter
        a_error = self.param_errors[2]  # Error in range parameter
        
        effective_range = 3 * a
        # Error propagation: if y = 3x, then σy = 3σx
        effective_range_error = 3 * a_error
        
        return effective_range, effective_range_error
