# Module containing the definition of the operator necessary for the solver class
# It takes vector objects from the pyVector class

from __future__ import division, print_function, absolute_import
import time
from copy import deepcopy
import numpy as np
from pyVector import vector, superVector
import pyproximal as pp
import pyOperator as Op
import pyProblem as Problem
import sep_util
import sys
sys.path.append('/home/groups/biondo/jdstitt/sep/research/stitt_fwi/pyseis_diffusion/src/python')
from fwi_latent_diffusion import process_with_latent_diffusion

class ProxOperator:

    def __init__(self):
        raise NotImplementedError("This is an abstract class")

    def prox(self, input: vector, output:vector, tau):
        raise NotImplementedError("This is an abstract class")
    
class ProxOperatorNull(ProxOperator):
    """ 
        Null proximal operator -- operator that does nothing
    """

    def __init__(self):
        pass

    def prox(self, input: vector, output:vector, tau):
        pass
    
class ProxDstack(ProxOperator):
    """ 
        Proximal operator acting on superVector objects
        y1 = | A  0 |  x1
        y2   | 0  B |  x2
    """

    def __init__(self, prox_ops: list):
        self.ops = []
        if isinstance(prox_ops, list):
            for op in prox_ops:
                if op is None:
                    self.ops.append(ProxOperatorNull())
                elif isinstance(op, ProxOperator):
                    self.ops.append(op)
        else:
            raise TypeError('Argument must be either ProxOperator or list of ProxOperators')


    def prox(self, input: superVector, output: superVector, tau):
        for i, op in enumerate(self.ops):
            op.prox(input.vecs[i], output.vecs[i], tau)

class ProxOperatorExplicit(ProxOperator):
    """ 
        Proximal operator based on the explicit form (pyproximal)
    """

    # Default class methods/functions
    def __init__(self, prox_operator, epsilon=1):
        self.prox_op = prox_operator
        self.epsilon = epsilon

    def prox(self, input: vector, output:vector, tau):
        arr_in = input.getNdArray()
        arr_out = output.getNdArray()
        # epsilon scaling to make it equivalent to regularized problem formulation
        res = self.prox_op.prox(arr_in, tau * self.epsilon * self.epsilon / 2)
        arr_out[:] = res.reshape(arr_in.shape)[:]

class ProxOperatorImplicit(ProxOperator):
    """ 
        Proximal operator based on the implicit form (by solving the actual optimization problem)
        The proximal operator is defined as the solution of the following optimization problem:
            min ||Ax - b||^2 + epsilon * ||x - u||^2, or
            min ||f(x) - b||^2 + epsilon * ||x - u||^2
        where A is a linear, or f(x) is a non-linear operator,
        u is the vector at which the proximal operator is evaluated, 
        b is the data and epsilon is a regularization parameter
    """

    def __init__(self, model, data, op, solver, epsilon=1, warm=True):
        self.epsilon = epsilon
        if isinstance(op, Op.NonLinearOperator):
            self.problem = Problem.ProblemL2NonLinearReg(model, data, op, self.epsilon, prior_model=None)
        elif isinstance(op, Op.Operator):
            self.problem = Problem.ProblemL2LinearReg(model, data, op, self.epsilon, prior_model=None)
        else:
            raise TypeError("Provided operator should be of Operator class")
        self.solver = solver
        self.warm = warm

    def prox(self, input: vector, output:vector, tau):
        # if we choose not to start with the previous estimate of the model
        if not self.warm:
            self.problem.model.zero()
        # scale epsilon by the current step size tau
        if tau < 0:
            raise RuntimeError("Error in the proximal evaluation: step size is negative!")
        self.problem.epsilon = np.sqrt(1/(tau * self.epsilon))
        # set the prior in ||x - u||^2 regularization term
        self.problem.prior_model = input.clone()
        self.solver.run(self.problem, verbose=True)
        output.copy(self.solver.inv_model)


class ProxOperatorFastDiffusion(ProxOperator):
    """ 
        Proximal operator based on the fast explicit diffusion 
        by Sergey Fomel 
    """

    def __init__(self, op, nsteps, epsilon=1):
        self.op = Op.ChainOperator(op, op.H)
        self.epsilon = epsilon
        self.nsteps = nsteps

    def prox(self, input: vector, output:vector, tau):
        output.copy(input)
        t = output.clone()
        for k in range(self.nsteps, 0, -1):
            self.op.forward(False, output, t)
            tk = 1/(4*np.sin(np.pi*k/(self.nsteps+1))**2)
            output.scaleAdd(t, 1., -tk*self.eps)

# ------------------------------------------------------------------
#  (e)  Plug‑and‑play diffusion denoiser as an explicit prox op
# ------------------------------------------------------------------
class DiffusionDenoiser(ProxOperator):
    """
    Plug‑and‑play proximal operator that calls LatentDiffusionProcessor.process().
    """

    def __init__(self,
                 processor,
                 timestep_start: int,
                 vmin: float,
                 vmax: float,
                 patch_size: int = 256,
                 stride: int = 128,
                 merge_method: str = 'gaussian',
                 min_freq: float = 3.0,    # Added required argument
                 max_freq: float = 14.0,   # Added required argument
                 dx: float = 15.0,         # Added required argument for model sampling
                 dy: float = 15.0,         # Added required argument for model sampling
                 work_dir: str = None      # Added required argument
                 ):
        self.processor    = processor
        self.timestep     = timestep_start
        self.vmin, self.vmax = vmin, vmax
        self.patch_size   = patch_size
        self.stride       = stride
        self.merge_method = merge_method
        # Add the missing required parameters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.dx = dx
        self.dy = dy
        self.work_dir = work_dir if work_dir is not None else "."

    def prox(self, input: vector, output: vector, tau):
        """
        input  : current ADMM iterate (pyVector)
        output : output vector (pyVector)
        tau    : 1/rho (float), unused here
        """
        arr = input.getNdArray()   # 2D numpy array
        denoised = process_with_latent_diffusion(
            processor      = self.processor,
            velocity_model = arr,
            timestep       = self.timestep,
            device         = self.processor.device,
            vmin           = self.vmin,
            vmax           = self.vmax,
            return_noise   = False,
            merge_method   = self.merge_method,
            # Add the missing required arguments
            min_freq       = self.min_freq,
            max_freq       = self.max_freq,
            dx             = self.dx,
            dy             = self.dy,
            work_dir       = self.work_dir
        )
        # Overwrite the output vector in place
        output.getNdArray()[:] = denoised

        # (Optional) step the diffusion timestep down each ADMM iteration
        self.timestep = max(self.timestep - 1, 0)

# Modify the FWIDiffusionDenoiser class to strictly enforce velocity constraints
class FWIDiffusionDenoiser(pp.ProxOperator):
    """
    Plug‑and‑play proximal operator for FWI that calls LatentDiffusionProcessor.process().
    Handles padding/truncation and STRICTLY enforces velocity constraints.
    """
    def __init__(self, processor, wave_eq_solver, timestep_start, vmin, vmax,
                 min_freq, max_freq, dx, dy, work_dir=None,
                 patch_size=512, stride=256, merge_method='gaussian'):
        # Initialize base class
        super(pp.ProxOperator, self).__init__()
        
        # Store parameters
        self.processor = processor
        self.wave_eq_solver = wave_eq_solver
        self.timestep = timestep_start
        self.vmin, self.vmax = vmin, vmax
        self.patch_size = patch_size
        self.stride = stride
        self.merge_method = merge_method
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.dx = dx
        self.dy = dy
        self.work_dir = work_dir
        
        print(f"FWIDiffusionDenoiser initialized with velocity bounds: {self.vmin} - {self.vmax}")

    def prox(self, input_vec, output_vec, tau):
        """
        Apply the diffusion proximal operator, correctly handling padding/truncation.
        """
        # Get the padded model array
        padded_model = input_vec.getNdArray()
        
        # Print statistics before any processing
        print(f"Input model stats BEFORE truncation: min={np.min(padded_model)}, max={np.max(padded_model)}")
        
        # Truncate the model using wave equation solver's internal method
        unpadded_model = self.wave_eq_solver._truncate_model(padded_model)
        
        # Print statistics on unpadded model
        print(f"Unpadded model stats BEFORE clipping: min={np.min(unpadded_model)}, max={np.max(unpadded_model)}")
        
        # ENFORCE velocity constraints on unpadded model
        np.clip(unpadded_model, self.vmin, self.vmax, out=unpadded_model)
        
        print(f"Unpadded model stats AFTER clipping: min={np.min(unpadded_model)}, max={np.max(unpadded_model)}")
        
        # Apply diffusion to the unpadded model
        try:
            denoised_unpadded = process_with_latent_diffusion(
                processor=self.processor,
                velocity_model=unpadded_model,
                timestep=self.timestep,
                device=self.processor.device,
                vmin=self.vmin,
                vmax=self.vmax,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
                dx=self.dx,
                dy=self.dy,
                work_dir=self.work_dir,
                return_noise=False,
                merge_method=self.merge_method
            )
            
            # Print statistics after diffusion
            print(f"Denoised model stats BEFORE clipping: min={np.min(denoised_unpadded)}, max={np.max(denoised_unpadded)}")
            
            # ENFORCE velocity constraints again after diffusion
            np.clip(denoised_unpadded, self.vmin, self.vmax, out=denoised_unpadded)
            
            print(f"Denoised model stats AFTER clipping: min={np.min(denoised_unpadded)}, max={np.max(denoised_unpadded)}")
            
            # Pad the denoised model back
            denoised_padded = self.wave_eq_solver._pad_model(
                denoised_unpadded, 
                self.wave_eq_solver.model_padding, 
                self.wave_eq_solver._FAT
            )
            
            # Print statistics after padding
            print(f"Final padded model stats: min={np.min(denoised_padded)}, max={np.max(denoised_padded)}")
            
            # Set the output vector
            output_vec.getNdArray()[:] = denoised_padded
            
        except Exception as e:
            # If any error occurs, just apply clipBounds and return
            print(f"Error in diffusion process: {e}")
            print("Falling back to simple clipping...")
            
            # Just apply bounds and return
            np.clip(unpadded_model, self.vmin, self.vmax, out=unpadded_model)
            bounded_padded = self.wave_eq_solver._pad_model(
                unpadded_model, 
                self.wave_eq_solver.model_padding, 
                self.wave_eq_solver._FAT
            )
            output_vec.getNdArray()[:] = bounded_padded
        
        # Decrement the timestep for next iteration
        self.timestep = max(self.timestep - 1, 0)

class BoundedProxOperatorImplicit(ProxOperatorImplicit):
    """
    Extension of ProxOperatorImplicit that enforces velocity bounds during FWI
    """
    def __init__(self, model, data, op, solver, wave_solver, *, epsilon=1, warm=True, vmin=None, vmax=None):
        # Only pass the parameters that the parent class expects
        super().__init__(model=model, data=data, op=op, solver=solver, epsilon=epsilon, warm=warm)
        self.vmin = vmin
        self.vmax = vmax
        self.wave_eq_solver = wave_solver
        print(f"BoundedProxOperatorImplicit initialized with velocity bounds: {self.vmin} - {self.vmax}")
    
    def prox(self, input_vec, output_vec, tau):
        """Override prox to enforce bounds before and after FWI"""
        # Get input and ensure it meets velocity bounds
        input_arr = input_vec.getNdArray()
        
        # Enforce minimum velocity on input
        unpadded_input = self.wave_eq_solver._truncate_model(input_arr)
        print(f"FWI input model before bounds: min={np.min(unpadded_input):.2f}, max={np.max(unpadded_input):.2f}")
        
        # Apply bounds only if they are specified
        if self.vmin is not None or self.vmax is not None:
            # Use None as the default bounds for np.clip if either is not specified
            np.clip(unpadded_input, self.vmin, self.vmax, out=unpadded_input)
            
        print(f"FWI input model after bounds: min={np.min(unpadded_input):.2f}, max={np.max(unpadded_input):.2f}")
        
        # Pad it back and set in input vector
        bounded_padded = self.wave_eq_solver._pad_model(
            unpadded_input, 
            self.wave_eq_solver.model_padding, 
            self.wave_eq_solver._FAT
        )
        input_vec.getNdArray()[:] = bounded_padded
        
        # Call the parent method to actually do the FWI
        super().prox(input_vec, output_vec, tau)
        
        # Now enforce bounds on the output
        output_arr = output_vec.getNdArray()
        unpadded_output = self.wave_eq_solver._truncate_model(output_arr)
        print(f"FWI output model before bounds: min={np.min(unpadded_output):.2f}, max={np.max(unpadded_output):.2f}")
        
        # Apply bounds only if they are specified
        if self.vmin is not None or self.vmax is not None:
            # Use None as the default bounds for np.clip if either is not specified
            np.clip(unpadded_output, self.vmin, self.vmax, out=unpadded_output)
            
        print(f"FWI output model after bounds: min={np.min(unpadded_output):.2f}, max={np.max(unpadded_output):.2f}")
        
        # Pad it back and set in output vector
        bounded_padded = self.wave_eq_solver._pad_model(
            unpadded_output, 
            self.wave_eq_solver.model_padding, 
            self.wave_eq_solver._FAT
        )
        output_vec.getNdArray()[:] = bounded_padded