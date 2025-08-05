"""
Main FWI interface with multi-frequency support.
Integrates Devito wave propagation with pySolver optimization.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union

# Import from GenericSolver with correct paths
from GenericSolver.pyVector import vectorIC, superVector
from GenericSolver.pyNonLinearSolver import NLCGsolver, LBFGSsolver
from GenericSolver.pyStopper import BasicStopper
from GenericSolver.pyStepper import ParabolicStep, StrongWolfe

# Import the problem classes correctly
import GenericSolver.pyProblem as pyProblem
from GenericSolver.pyProblem import Problem

from pydeviseis.inversion.problems import FWIProblem, BoundedFWIProblem
from pydeviseis.core.vector_adapters import MultiShotVectorAdapter


class DevitoFWI:
    """
    Main FWI class with comprehensive multi-frequency inversion capability.
    
    This provides complete control over all FWI parameters and supports
    sophisticated multi-frequency continuation strategies.
    """
    
    def __init__(self, wave_equation, 
                 observed_data: Union[vectorIC, superVector], 
                 geometry_params: Dict,
                 # Optimization parameters
                 solver_type: str = 'nlcg', stepper_type: str = 'parabolic',
                 # Default iteration parameters  
                 max_iterations: int = 20, grad_tolerance: float = 1e-6,
                 obj_tolerance: float = 1e-10,
                 # Multi-frequency parameters
                 frequency_bands: Optional[List[Tuple[float, float]]] = None,
                 iterations_per_band: Optional[List[int]] = None,
                 grad_tolerance_per_band: Optional[List[float]] = None,
                 # Model constraints
                 vmin: float = 1400.0, vmax: float = 4000.0,
                 epsilon: float = 0.01,
                 # Advanced options
                 stepper_params: Optional[Dict] = None,
                 solver_params: Optional[Dict] = None):
        """
        Initialize comprehensive FWI with full parameter control.
        
        Args:
            wave_equation: AcousticWaveEquation instance
            observed_data: Observed seismic data
            geometry_params: Acquisition geometry parameters
            
            # Optimization Control
            solver_type: Optimization algorithm ('nlcg', 'lbfgs', 'lbfgsb') 
            stepper_type: Line search method ('parabolic', 'strongwolfe', 'linear')
            max_iterations: Default maximum FWI iterations
            grad_tolerance: Default gradient convergence tolerance  
            obj_tolerance: Objective function tolerance
            
            # Multi-frequency Control
            frequency_bands: List of (f_min, f_max) tuples for each stage
            iterations_per_band: Max iterations for each frequency band
            grad_tolerance_per_band: Gradient tolerance for each frequency band
            
            # Model Constraints
            vmin, vmax: Velocity bounds in m/s
            epsilon: Regularization parameter
            
            # Advanced Options
            stepper_params: Additional stepper parameters (dict)
            solver_params: Additional solver parameters (dict)
        """
        self.wave_equation = wave_equation
        self.observed_data = observed_data
        self.geometry_params = geometry_params
        
        # Store optimization parameters
        self.solver_type = solver_type
        self.stepper_type = stepper_type
        self.max_iterations = max_iterations
        self.grad_tolerance = grad_tolerance
        self.obj_tolerance = obj_tolerance
        
        # Model constraints
        self.vmin = vmin
        self.vmax = vmax
        self.epsilon = epsilon
        
        # Set up multi-frequency parameters
        if frequency_bands is None:
            self.frequency_bands = [(3.0, 8.0), (3.0, 12.0), (3.0, 20.0)]
        else:
            self.frequency_bands = frequency_bands
        
        n_bands = len(self.frequency_bands)
        
        # Set iterations per band
        if iterations_per_band is None:
            self.iterations_per_band = [max_iterations] * n_bands
        else:
            if len(iterations_per_band) != n_bands:
                raise ValueError(f"iterations_per_band length ({len(iterations_per_band)}) must match frequency_bands length ({n_bands})")
            self.iterations_per_band = iterations_per_band
        
        # Set gradient tolerance per band
        if grad_tolerance_per_band is None:
            self.grad_tolerance_per_band = [grad_tolerance] * n_bands
        else:
            if len(grad_tolerance_per_band) != n_bands:
                raise ValueError(f"grad_tolerance_per_band length ({len(grad_tolerance_per_band)}) must match frequency_bands length ({n_bands})")
            self.grad_tolerance_per_band = grad_tolerance_per_band
        
        # Store advanced parameters
        self.stepper_params = stepper_params or {}
        self.solver_params = solver_params or {}
        
    def _create_stepper(self, band_index: int = 0):
        """Create stepper with appropriate parameters."""
        # Get base parameters
        params = self.stepper_params.copy()
        
        if self.stepper_type == 'parabolic':
            return ParabolicStep(**params)
        elif self.stepper_type == 'strongwolfe':
            return StrongWolfe(**params)
        elif self.stepper_type == 'linear':
            return ParabolicStep(**params)  # Use parabolic as fallback
        else:
            raise ValueError(f"Unknown stepper type: {self.stepper_type}")
    
    def _create_solver(self, band_index: int = 0):
        """Create solver with appropriate parameters for frequency band."""
        # Create stopper with band-specific parameters
        stopper = BasicStopper(
            niter=self.iterations_per_band[band_index],
            tolg=self.grad_tolerance_per_band[band_index],
            tolobj=self.obj_tolerance
        )
        
        # Create stepper
        stepper = self._create_stepper(band_index)
        
        # Get solver parameters
        params = self.solver_params.copy()
        
        if self.solver_type == 'nlcg':
            return NLCGsolver(stopper, stepper, **params)
        elif self.solver_type == 'lbfgs':
            return LBFGSsolver(stopper, stepper, **params)
        elif self.solver_type == 'lbfgsb':
            # Import LBFGSB if needed
            try:
                from GenericSolver.pyNonLinearSolver import LBFGSBsolver
                return LBFGSBsolver(stopper, stepper, **params)
            except ImportError:
                print("LBFGSBsolver not available, falling back to LBFGS")
                return LBFGSsolver(stopper, stepper, **params)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
    
    def run_single_frequency(self, initial_model: vectorIC, 
                           band_index: int = 0, verbose: bool = True) -> Dict:
        """
        Run single-frequency FWI for a specific frequency band.
        
        Args:
            initial_model: Starting velocity model
            band_index: Index of frequency band to use
            verbose: Print iteration information
            
        Returns:
            Dictionary with inversion results
        """
        # Get frequency band parameters
        if band_index < len(self.frequency_bands):
            f_min, f_max = self.frequency_bands[band_index]
            center_freq = (f_min + f_max) / 2.0
        else:
            # Use default frequency if band_index out of range
            center_freq = self.geometry_params.get('f0', 15.0)
        
        # Create geometry with updated frequency
        geometry_params = self.geometry_params.copy()
        geometry_params['f0'] = center_freq
        
        # Create acoustic operator
        acoustic_op = self.wave_equation.create_operator(**geometry_params)
        
        # notable change: Create bounded and regularized FWI problem using local class
        problem = BoundedFWIProblem(
            model=initial_model,
            observed_data=self.observed_data,
            acoustic_operator=acoustic_op,
            geometry_params=geometry_params, # Pass params for preconditioning
            vmin=self.vmin, vmax=self.vmax
        )
        
        # Create solver for this frequency band
        solver = self._create_solver(band_index)
        
        # Print band information
        if verbose and band_index < len(self.frequency_bands):
            print(f"Frequency Band {band_index+1}: {f_min:.1f}-{f_max:.1f} Hz (center: {center_freq:.1f} Hz)")
            print(f"  Max iterations: {self.iterations_per_band[band_index]}")
            print(f"  Gradient tolerance: {self.grad_tolerance_per_band[band_index]:.1e}")
        
        # Run optimization
        print("Starting FWI optimization...")
        solver.run(problem, verbose=verbose)
        
        # Extract results
        final_model = problem.model.clone()
        final_obj = problem.get_obj(problem.model)
        final_grad = problem.get_grad(problem.model)
        final_grad_norm = final_grad.norm()
        
        results = {
            'final_model': final_model,
            'final_objective': final_obj,
            'final_gradient_norm': final_grad_norm,
            'iterations': len(solver.obj) if hasattr(solver, 'obj') else 0,
            'converged': final_grad_norm < self.grad_tolerance_per_band[band_index],
            'frequency_band': self.frequency_bands[band_index] if band_index < len(self.frequency_bands) else (0, center_freq),
            'center_frequency': center_freq
        }
        
        if verbose:
            print(f"FWI completed after {results['iterations']} iterations")
            print(f"Final objective: {results['final_objective']:.2e}")
            print(f"Final gradient norm: {results['final_gradient_norm']:.2e}")
            print(f"Converged: {results['converged']}")
        
        return results
    
    def run_multi_frequency(self, initial_model: vectorIC, verbose: bool = True) -> List[Dict]:
        """
        Run multi-frequency FWI with frequency continuation.
        
        Args:
            initial_model: Starting velocity model
            verbose: Print detailed information
            
        Returns:
            List of results dictionaries for each frequency band
        """
        current_model = initial_model.clone()
        all_results = []
        
        print(f"\nSTARTING MULTI-FREQUENCY FWI")
        print(f"{'='*60}")
        print(f"Number of frequency bands: {len(self.frequency_bands)}")
        print(f"Solver: {self.solver_type.upper()}")
        print(f"Stepper: {self.stepper_type}")
        print(f"Velocity bounds: {self.vmin}-{self.vmax} m/s")
        print(f"Regularization: {self.epsilon}")
        
        for i in range(len(self.frequency_bands)):
            print(f"\n{'-'*60}")
            
            # Run FWI for this frequency band
            results = self.run_single_frequency(current_model, band_index=i, verbose=verbose)
            all_results.append(results)
            
            # Use result as starting model for next frequency band
            current_model = results['final_model'].clone()
            
            # Print progress
            if verbose:
                improvement_percent = 0
                if i == 0:
                    initial_obj = results['final_objective']
                else:
                    improvement_percent = ((all_results[0]['final_objective'] - results['final_objective']) / 
                                         all_results[0]['final_objective'] * 100)
                
                print(f"Progress: Band {i+1}/{len(self.frequency_bands)} completed")
                if i > 0:
                    print(f"Objective improvement: {improvement_percent:.1f}%")
        
        print(f"\n{'='*60}")
        print("MULTI-FREQUENCY FWI COMPLETED")
        print(f"{'='*60}")
        
        return all_results

# def run_acoustic_fwi_demo(model_shape: Tuple[int, ...] = (81, 81),
#                          model_spacing: Tuple[float, ...] = (12.5, 12.5),
#                          n_sources: int = 4, n_receivers: int = 16,
#                          nt: int = 400, dt: Optional[float] = None,  # Let Devito calculate optimal dt
#                          # FWI Control Parameters
#                          solver_type: str = 'nlcg',
#                          stepper_type: str = 'parabolic', 
#                          # Multi-frequency parameters
#                          frequency_bands: Optional[List[Tuple[float, float]]] = None,
#                          iterations_per_band: Optional[List[int]] = None,
#                          grad_tolerance_per_band: Optional[List[float]] = None,
#                          # Model constraints
#                          vmin: float = 1400.0, vmax: float = 4000.0, epsilon: float = 0.01,
#                          # Execution control
#                          multi_frequency: bool = True, verbose: bool = True) -> Dict:
#     """
#     Complete acoustic FWI demonstration function.
    
#     This function creates models, generates synthetic data, and runs FWI.
#     It serves as the main demo function for the package.
    
#     Args:
#         model_shape: Shape of velocity model
#         model_spacing: Grid spacing
#         n_sources: Number of sources
#         n_receivers: Number of receivers  
#         nt: Number of time samples
#         dt: Time sampling interval
#         multi_frequency: Whether to run multi-frequency FWI
#         verbose: Print detailed information
        
#     Returns:
#         Dictionary with complete FWI results
#     """
#     from pydeviseis.wave_equations.acoustic import (
#         create_complex_demo_model, create_smooth_demo_model, create_acoustic_geometry
#     )
#     from pydeviseis.core.environment import auto_setup_environment
    
#     # Setup environment
#     print("Setting up Devito environment...")
#     env_info = auto_setup_environment(verbose=verbose)
    
#     # Create wave equation
#     wave_eq = AcousticWaveEquation(model_shape, model_spacing)
    
#     # Create models
#     print("Creating velocity models...")
#     true_model = create_complex_demo_model(*model_shape)
#     smooth_model = create_smooth_demo_model(*model_shape, velocity=1800.0)
    
#     # Create acquisition geometry
#     geometry = create_acoustic_geometry(model_shape, model_spacing, 
#                                       n_sources, n_receivers)
#     geometry_params = {
#         **geometry,
#         'nt': nt,
#         'dt': dt,
#         'f0': 15.0,
#         'multi_shot': True
#     }
    
#     # Generate synthetic observed data
#     print("Generating synthetic observed data...")
#     forward_op = wave_eq.create_operator(**geometry_params)
#     forward_op.set_background(true_model)
    
#     # Create data vector for multi-shot
#     observed_data = MultiShotVectorAdapter.create_multishot_data_vector(
#         n_sources, nt, n_receivers)
    
#     # Generate data
#     forward_op.forward(False, true_model, observed_data)
    
#     # Add noise if desired
#     noise_level = 0.05  # 5% noise
#     for shot_data in observed_data.vecs:
#         data_array = shot_data.getNdArray()
#         noise = np.random.normal(0, noise_level * shot_data.norm() / np.sqrt(shot_data.size), 
#                                data_array.shape)
#         data_array[:] += noise
    
#     # Create and run FWI
#     print("Setting up FWI...")
#     fwi = DevitoFWI(wave_eq, observed_data, geometry_params,
#                    solver_type='nlcg', stepper_type='parabolic',
#                    max_iterations=20, grad_tolerance=1e-6)
    
#     # Run FWI
#     if multi_frequency:
#         results = fwi.run_multi_frequency(smooth_model, verbose=verbose)
#         final_result = results[-1]  # Last frequency band result
#     else:
#         final_result = fwi.run_single_frequency(smooth_model, verbose=verbose)
#         results = [final_result]
    
#     # Package complete results
#     complete_results = {
#         'true_model': true_model,
#         'initial_model': smooth_model,
#         'observed_data': observed_data,
#         'final_model': final_result['final_model'],
#         'fwi_results': results,
#         'geometry_params': geometry_params,
#         'environment_info': env_info
#     }
    
#     print("\nFWI demonstration completed successfully!")
#     return complete_results