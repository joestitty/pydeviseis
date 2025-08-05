"""
Vector format conversion utilities between Devito, numpy, and pySolver formats.
This is the foundation for all other integrations.
"""

import numpy as np
import GenericSolver.pyVector as Vec
from GenericSolver.pyVector import vectorIC, superVector
from typing import Tuple, Optional, Union, List


class DevitoVectorAdapter:
    """Handles conversions between numpy arrays and pySolver vectorIC objects."""
    
    @staticmethod
    def create_model_vector(shape: Tuple[int, ...], 
                          initial_value: float = 0.0) -> vectorIC:
        """
        Create a vectorIC for velocity models.
        
        Args:
            shape: Model shape (nx, nz) for 2D or (ny, nx, nz) for 3D
            initial_value: Initial value to fill the vector
            
        Returns:
            vectorIC object containing model
        """
        # Create numpy array with C-contiguous memory layout
        array = np.full(shape, initial_value, dtype=np.float64, order='C')
        return vectorIC(array)
    
    @staticmethod
    def create_data_vector(nt: int, nr: int, initial_value: float = 0.0) -> vectorIC:
        """
        Create a vectorIC for seismic data.
        
        Args:
            nt: Number of time samples
            nr: Number of receivers
            initial_value: Initial value to fill the vector
            
        Returns:
            vectorIC object containing data array
        """
        array = np.full((nt, nr), initial_value, dtype=np.float64, order='C')
        return vectorIC(array)
    
    @staticmethod
    def numpy_to_vectorIC(array: np.ndarray) -> vectorIC:
        """Convert numpy array to vectorIC."""
        # Ensure C-contiguous for vectorIC compatibility
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return vectorIC(array.astype(np.float64))
    
    @staticmethod
    def vectorIC_to_numpy(vec: vectorIC) -> np.ndarray:
        """Extract numpy array from vectorIC."""
        return vec.getNdArray()
    
    @staticmethod
    def create_model_from_array(array: np.ndarray) -> vectorIC:
        """Create model vectorIC from numpy array."""
        return DevitoVectorAdapter.numpy_to_vectorIC(array)


class MultiShotVectorAdapter:
    """Handles multi-shot data organization using superVector."""
    
    @staticmethod
    def create_multishot_data_vector(n_shots: int, nt: int, nr: int) -> superVector:
        """
        Create superVector for multi-shot data.
        
        Args:
            n_shots: Number of shots
            nt: Number of time samples per shot
            nr: Number of receivers per shot
            
        Returns:
            superVector containing one vectorIC per shot
        """
        shots = []
        for _ in range(n_shots):
            shot_vec = DevitoVectorAdapter.create_data_vector(nt, nr)
            shots.append(shot_vec)
        return superVector(shots)
    
    @staticmethod
    def shots_to_supervector(shot_arrays: List[np.ndarray]) -> superVector:
        """Convert list of shot arrays to superVector."""
        shots = []
        for shot_array in shot_arrays:
            shot_vec = DevitoVectorAdapter.numpy_to_vectorIC(shot_array)
            shots.append(shot_vec)
        return superVector(shots)
    
    @staticmethod
    def supervector_to_shots(data: superVector) -> List[np.ndarray]:
        """Extract list of numpy arrays from superVector."""
        return [DevitoVectorAdapter.vectorIC_to_numpy(shot) for shot in data.vecs]