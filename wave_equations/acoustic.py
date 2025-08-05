"""
Acoustic wave equation wrapper using Devito backend.

This module provides a high-level interface for acoustic wave 
equation solving that integrates with the pysolver framework.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from GenericSolver.pyOperator import NonLinearOperator
from pydeviseis.core.devito_adapters import DevitoAcousticOperator, DevitoMultiShotOperator
from pydeviseis.core.vector_adapters import DevitoVectorAdapter

class AcousticWaveEquation:
    """High-level interface for acoustic wave equation modeling."""
    def __init__(self, model_shape: Tuple[int, ...], model_spacing: Tuple[float, ...],
                 model_origin: Optional[Tuple[float, ...]] = None,
                 space_order: int = 4, nbl: int = 40, kernel: str = 'OT2'):
        self.model_shape = model_shape
        self.model_spacing = model_spacing
        self.model_origin = model_origin or tuple(0.0 for _ in model_shape)
        self.space_order = space_order
        self.nbl = nbl
        self.kernel = kernel
    
    def create_operator(self, src_coordinates: np.ndarray, rec_coordinates: np.ndarray,
                       nt: int, dt: float, f0: float = 15.0,
                       multi_shot: bool = True):
        """Create a non-linear acoustic wave propagation operator."""
        common_params = {
            'model_shape': self.model_shape, 'model_spacing': self.model_spacing,
            'model_origin': self.model_origin, 'src_coordinates': src_coordinates,
            'rec_coordinates': rec_coordinates, 'nt': nt, 'dt': dt, 'f0': f0,
            'space_order': self.space_order, 'nbl': self.nbl, 'kernel': self.kernel
        }
        
        if multi_shot and len(src_coordinates) > 1:
            devito_op = DevitoMultiShotOperator(**common_params)
        else:
            devito_op = DevitoAcousticOperator(**common_params)
            
        # *** NOTEWORTHY FIX: Wrap the Devito operator in the NonLinearOperator class ***
        return NonLinearOperator(nl_op=devito_op,
                                 lin_op=devito_op,
                                 set_background_func=devito_op.set_background)

# The helper functions below are defined in the main notebook, so they are not strictly
# necessary in this file but are kept here for completeness.

def create_complex_demo_model(nx: int = 101, nz: int = 101, dx: float = 10.0, dz: float = 10.0) -> 'vectorIC':
    """
    Create a complex 2D velocity model for testing.
    """
    model_array = np.ones((nx, nz)) * 1500.0
    model_array[40:60, 40:60] = 2500.0
    model_array[:, 70:] = 2000.0
    model_array[:, 85:] = 2800.0
    np.random.seed(42)
    noise = np.random.normal(0, 50, (nx, nz))
    model_array += noise
    model_array = np.maximum(model_array, 1400.0)
    return DevitoVectorAdapter.numpy_to_vectorIC(model_array)

def create_acoustic_geometry(model_shape: Tuple[int, ...], 
                           model_spacing: Tuple[float, ...],
                           n_sources: int = 5, n_receivers: int = 20,
                           source_depth: float = 50.0, 
                           receiver_depth: float = 50.0) -> Dict[str, np.ndarray]:
    """
    Create acquisition geometry for acoustic modeling.
    """
    nx, nz = model_shape; dx, dz = model_spacing
    src_x = np.linspace(dx*5, dx*(nx-5), n_sources)
    src_z = np.ones(n_sources) * source_depth
    src_coordinates = np.column_stack([src_x, src_z])
    rec_x = np.linspace(dx*2, dx*(nx-2), n_receivers)
    rec_z = np.ones(n_receivers) * receiver_depth
    rec_coordinates = np.column_stack([rec_x, rec_z])
    return {'src_coordinates': src_coordinates, 'rec_coordinates': rec_coordinates}