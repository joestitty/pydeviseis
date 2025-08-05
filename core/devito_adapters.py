"""
Core Devito operators that interface with pyOperator framework.

This module provides operators that wrap Devito wave propagation
functionality while maintaining compatibility with pysolver's
optimization framework.
"""

import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

# pySolver imports with correct paths
from GenericSolver.pyOperator import Operator
from GenericSolver.pyVector import vectorIC, superVector

# Devito imports
from devito import Function, TimeFunction
from examples.seismic import Model, AcquisitionGeometry, RickerSource, Receiver
from examples.seismic.acoustic import AcousticWaveSolver

from .vector_adapters import DevitoVectorAdapter, MultiShotVectorAdapter

class DevitoAcousticOperator(Operator):
    """pyOperator wrapper around Devito's AcousticWaveSolver."""
    def __init__(self, model_shape, model_spacing, src_coordinates, rec_coordinates,
                 nt, f0=15.0, dt=None, model_origin=None,
                 space_order=4, nbl=40, kernel='OT2'):
        self.model_shape = model_shape
        self.model_spacing = model_spacing
        self.model_origin = model_origin or tuple(0.0 for _ in model_shape)
        self.src_coordinates = src_coordinates
        self.rec_coordinates = rec_coordinates
        self.nt = nt
        self.dt_user = dt
        self.f0 = f0
        self.space_order = space_order
        self.nbl = nbl
        self.kernel = kernel
        
        # Set up domain and range
        domain = DevitoVectorAdapter.create_model_vector(model_shape)
        nr = rec_coordinates.shape[0] if rec_coordinates.ndim == 2 else rec_coordinates.shape[1]
        range_vec = DevitoVectorAdapter.create_data_vector(nt, nr)
        super().__init__(domain, range_vec)
        
        # Initialize solver-related attributes
        self.devito_solver = None
        self.current_model_array = None

    def set_background(self, model: vectorIC):
        """
        Sets up or REBUILDS the Devito solver for a given background model.
        This ensures the absorbing boundaries are always correct.
        """
        model_array = DevitoVectorAdapter.vectorIC_to_numpy(model)
        
        # important info here
        # Instead of trying to partially update the model, we rebuild the
        # solver if the model has changed. This is the surest way to guarantee
        # that the damping parameters (bcs="damp") are correctly recalculated.
        if self.devito_solver is None or not np.array_equal(model_array, self.current_model_array):
            
            # Create the Devito Model with absorbing boundaries for the new velocity
            devito_model = Model(vp=model_array, origin=self.model_origin, shape=self.model_shape,
                                 spacing=self.model_spacing, space_order=self.space_order,
                                 nbl=self.nbl, dt=self.dt_user, bcs="damp")
            
            # Use the model's critical_dt if none was provided by the user
            dt_calc = self.dt_user or devito_model.critical_dt
            tn = (self.nt - 1) * dt_calc
            
            # Handle single-shot vs multi-shot source geometry
            src_coords = self.src_coordinates
            if src_coords.ndim == 2 and len(src_coords) > 1:
                 src_coords = src_coords[0:1]

            # Re-create the acquisition geometry with the new model
            geometry = AcquisitionGeometry(devito_model, self.rec_coordinates, src_coords,
                                           t0=0.0, tn=tn, f0=self.f0, src_type='Ricker')
            
            # Re-create the solver
            self.devito_solver = AcousticWaveSolver(devito_model, geometry,
                                                    space_order=self.space_order, kernel=self.kernel)
            
            self.current_model_array = model_array.copy()

    def forward(self, add: bool, model: vectorIC, data: vectorIC):
        """Forward modeling operation."""
        self.set_background(model)
        rec, _, _ = self.devito_solver.forward(dt=self.dt_user)
        
        # Extract the expected amount of data
        nt_expected = self.nt
        data_produced = rec.data[:nt_expected, :]
        
        if add:
            data.getNdArray()[:] += data_produced
        else:
            data.getNdArray()[:] = data_produced

    def jacobian(self, add: bool, model_pert: vectorIC, data_pert: vectorIC):
        """Jacobian (linearized forward) operation."""
        if self.devito_solver is None:
            raise RuntimeError("Solver not initialized. Call set_background first.")
            
        # Create perturbation function
        dm_func = Function(name='dm', grid=self.devito_solver.model.grid, space_order=0)
        
        # CRITICAL FIX: Handle grid.dim properly - it's an integer, not iterable
        ndim = self.devito_solver.model.grid.dim
        interior_slice = tuple(slice(self.nbl, -self.nbl) for _ in range(ndim))
        dm_func.data[interior_slice] = DevitoVectorAdapter.vectorIC_to_numpy(model_pert)
        
        # Run Jacobian
        rec, _, _, _ = self.devito_solver.jacobian(dmin=dm_func, dt=self.dt_user)
        
        # Extract data
        nt_expected = self.nt
        data_produced = rec.data[:nt_expected, :]
        
        if add:
            data_pert.getNdArray()[:] += data_produced
        else:
            data_pert.getNdArray()[:] = data_produced

    def adjoint(self, add: bool, model_grad: vectorIC, data_resid: vectorIC):
        """Adjoint (gradient) operation."""
        if self.devito_solver is None:
            raise RuntimeError("Solver not initialized. Call set_background first.")
            
        # Run forward to get saved wavefield
        _, u, _ = self.devito_solver.forward(save=True, dt=self.dt_user)
        
        # Create receiver for residual injection
        rec_residual = Receiver(name='rec_adjoint', grid=self.devito_solver.model.grid,
                                time_range=self.devito_solver.geometry.time_axis,
                                coordinates=self.rec_coordinates)
        
        # Set residual data
        nt_expected = self.nt
        rec_residual.data[:nt_expected, :] = data_resid.getNdArray()
        
        # Run gradient calculation
        grad, _ = self.devito_solver.jacobian_adjoint(rec_residual, u, dt=self.dt_user)
        
        # Extract interior gradient (excluding padding) - FIXED
        ndim = self.devito_solver.model.grid.dim
        interior_slice = tuple(slice(self.nbl, -self.nbl) for _ in range(ndim))
        grad_interior = grad.data[interior_slice]
        
        if add:
            model_grad.getNdArray()[:] += grad_interior
        else:
            model_grad.getNdArray()[:] = grad_interior

class DevitoMultiShotOperator(Operator):
    """Multi-shot acoustic operator for parallel FWI."""
    def __init__(self, model_shape, model_spacing, src_coordinates, rec_coordinates,
                 nt, dt, f0=15.0, model_origin=None,
                 space_order=4, nbl=40, kernel='OT2'):
        
        self.n_shots = len(src_coordinates)
        
        # Set up domain and range
        domain = DevitoVectorAdapter.create_model_vector(model_shape)
        nr = rec_coordinates.shape[-2] if rec_coordinates.ndim == 3 else rec_coordinates.shape[0]
        range_vec = MultiShotVectorAdapter.create_multishot_data_vector(self.n_shots, nt, nr)
        super().__init__(domain, range_vec)
        
        # Create individual shot operators
        self.shot_operators = []
        for i in range(self.n_shots):
            shot_src = src_coordinates[i:i+1]
            shot_rec = rec_coordinates[i] if rec_coordinates.ndim == 3 else rec_coordinates
            
            shot_op = DevitoAcousticOperator(
                model_shape=model_shape, model_spacing=model_spacing,
                src_coordinates=shot_src, rec_coordinates=shot_rec,
                nt=nt, dt=dt, f0=f0, model_origin=model_origin,
                space_order=space_order, nbl=nbl, kernel=kernel)
            self.shot_operators.append(shot_op)

    def set_background(self, model: vectorIC):
        """Set background model for all shot operators."""
        for shot_op in self.shot_operators:
            shot_op.set_background(model)

    def forward(self, add: bool, model: vectorIC, data: superVector):
        """Multi-shot forward modeling."""
        self.set_background(model)
        if not add:
            data.zero()
        for i, shot_op in enumerate(self.shot_operators):
            shot_op.forward(False, model, data.vecs[i])

    def jacobian(self, add: bool, model_pert: vectorIC, data_pert: superVector):
        """Multi-shot Jacobian operation."""
        # CRITICAL FIX: Handle first shot separately, then accumulate
        if not add:
            data_pert.zero()
        
        # Process each shot
        for i, shot_op in enumerate(self.shot_operators):
            shot_op.jacobian(add or (i > 0), model_pert, data_pert.vecs[i])

    def adjoint(self, add: bool, model_grad: vectorIC, data_resid: superVector):
        """Multi-shot adjoint operation."""
        if not add:
            model_grad.zero()
        
        # Accumulate gradients from all shots
        for i, shot_op in enumerate(self.shot_operators):
            shot_op.adjoint(add or (i > 0), model_grad, data_resid.vecs[i])