"""
FWI problem formulations using pysolver framework.

Custom implementation that properly inherits from Problem
and fixes isinstance issues.
"""

import numpy as np
from typing import Optional, Union
from GenericSolver.pyVector import vectorIC, superVector

# Import the Problem class correctly..
import GenericSolver.pyProblem as pyProblem
from GenericSolver.pyProblem import Problem

class FWIProblem(Problem):
    """
    Custom FWI problem that directly implements the non-linear L2 problem interface
    to ensure full compatibility with the GenericSolver framework.
    """
    def __init__(self, model: vectorIC, observed_data: Union[vectorIC, superVector],
                 acoustic_operator,
                 minBound: Optional[vectorIC] = None, 
                 maxBound: Optional[vectorIC] = None, **kwargs):
        
        # Call parent constructor with bounds
        super().__init__(minBound, maxBound)
        
        # Store the NonLinearOperator wrapper
        self.op = acoustic_operator
        self.data = observed_data
        
        # Set up internal vectors
        self.model = model
        self.grad = model.clone().zero()
        self.res = observed_data.clone().zero()
        self.dres = observed_data.clone().zero()
        self.dmodel = model.clone().zero()
        
        # Set linear flag
        self.linear = False
        
        # Initialize counters and flags
        self.fevals = 0
        self.gevals = 0
        self.obj = 0.0
        self.obj_updated = False
        self.grad_updated = False

    def get_obj(self, model):
        """Compute objective function: 1/2 ||F(m) - d||^2"""
        self.set_model(model)
        if not self.obj_updated:
            self.res = self.resf(self.model)
            self.obj = 0.5 * self.res.dot(self.res)
            self.obj_updated = True
            self.fevals += 1
        return self.obj

    def get_grad(self, model):
        """Compute gradient: J^T(F(m) - d)"""
        self.set_model(model)
        if not self.grad_updated:
            self.res = self.resf(self.model)
            self.grad = self.gradf(self.model, self.res)
            self.grad_updated = True
            self.gevals += 1
        return self.grad
    
    def get_model(self):
        """Return current model"""
        return self.model
    
    def set_model(self, model):
        """Set current model and reset update flags"""
        if not np.array_equal(model.getNdArray(), self.model.getNdArray()):
            self.model.copy(model)
            self.obj_updated = False
            self.grad_updated = False
        
    def resf(self, model):
        """Compute residual vector r = f(m) - d"""
        # op is the NonLinearOperator, nl_op is the Devito operator
        self.op.nl_op.forward(False, model, self.res)
        self.res.scaleAdd(self.data, 1., -1.)
        return self.res
        
    def gradf(self, model, res):
        """Compute gradient vector g = F'r = F'(f(m) - d)"""
        self.op.set_background(model)
        self.op.lin_op.adjoint(False, self.grad, res)
        return self.grad

    def dresf(self, model, dmodel):
        """Compute dres = Fdm"""
        self.op.set_background(model)
        self.op.lin_op.jacobian(False, dmodel, self.dres)
        return self.dres

class BoundedFWIProblem(FWIProblem):
    """FWI problem with velocity bounds."""
    def __init__(self, vmin: float = 1400.0, vmax: float = 4000.0, **kwargs):
        if 'model' in kwargs:
            model = kwargs['model']
            min_bound = model.clone(); min_bound.set(vmin)
            max_bound = model.clone(); max_bound.set(vmax)
            kwargs['minBound'] = min_bound
            kwargs['maxBound'] = max_bound
        
        # Remove any unsupported arguments from parent classes
        kwargs.pop('epsilon', None)
        kwargs.pop('prior_model', None)
        
        super().__init__(**kwargs)