"""
Environment setup utilities for Devito-Pysolver integration.

This module handles GPU setup, environment configuration, and 
device management for optimal performance.
"""

import os
import shutil
import subprocess
import sys
from typing import List, Optional, Dict
import warnings


def setup_devito_gpu_environment(gpu_ids: Optional[List[int]] = None,
                                compiler: str = 'pgcc',
                                clear_cache: bool = True,
                                verbose: bool = True):
    """
    Setup Devito environment for GPU execution.
    
    Args:
        gpu_ids: List of GPU IDs to use (None for auto-detect)
        compiler: Compiler to use ('nvc' or 'gcc')
        clear_cache: Whether to clear Devito cache
        verbose: Print setup information
        
    Returns:
        Dictionary with environment setup information
    """
    if verbose:
        print("Setting up Devito environment for GPU execution...")
    
    # Set Devito environment variables for GPU
    devito_env = {
        'DEVITO_LANGUAGE': 'openacc',
        'DEVITO_ARCH': compiler,
        'DEVITO_PLATFORM': 'nvidiaX',
        'DEVITO_LOGGING': 'INFO' if verbose else 'WARNING',
        'CC': compiler,
        'CXX': compiler + '++' if compiler == 'nvc' else 'g++',
    }
    
    # Apply environment variables
    for key, value in devito_env.items():
        os.environ[key] = value
        if verbose:
            print(f"  {key} = {value}")
    
    # Setup GPU visibility
    gpu_info = setup_gpu_visibility(gpu_ids, verbose)
    
    # Clear Devito cache if requested
    if clear_cache:
        clear_devito_cache(verbose)
    
    # Verify GPU availability
    gpu_check = check_gpu_availability(verbose)
    
    setup_info = {
        'devito_env': devito_env,
        'gpu_info': gpu_info,
        'gpu_available': gpu_check['available'],
        'num_gpus': gpu_check['count'],
        'cache_cleared': clear_cache
    }
    
    if verbose:
        print("Devito GPU environment setup completed.")
        
    return setup_info


def setup_gpu_visibility(gpu_ids: Optional[List[int]] = None, 
                        verbose: bool = True) -> Dict:
    """
    Setup GPU visibility for CUDA/OpenACC.
    
    Args:
        gpu_ids: List of GPU IDs to make visible
        verbose: Print information
        
    Returns:
        Dictionary with GPU setup information
    """
    gpu_info = check_gpu_availability(verbose=False)
    
    if gpu_ids is None:
        # Auto-detect available GPUs
        if gpu_info['available']:
            gpu_ids = list(range(gpu_info['count']))
        else:
            gpu_ids = []
            
    if len(gpu_ids) > 0:
        # Set CUDA visible devices
        cuda_visible = ','.join(map(str, gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
        
        # Set OpenACC device
        os.environ['ACC_DEVICE_TYPE'] = 'nvidia'
        os.environ['ACC_DEVICE_NUM'] = str(gpu_ids[0])  # Primary GPU
        
        if verbose:
            print(f"GPU Setup:")
            print(f"  CUDA_VISIBLE_DEVICES = {cuda_visible}")
            print(f"  ACC_DEVICE_NUM = {gpu_ids[0]}")
            print(f"  Using {len(gpu_ids)} GPU(s): {gpu_ids}")
    else:
        if verbose:
            print("No GPUs specified or available - using CPU")
    
    return {
        'gpu_ids': gpu_ids,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'acc_device_num': os.environ.get('ACC_DEVICE_NUM', ''),
        'num_gpus_used': len(gpu_ids)
    }


def check_gpu_availability(verbose: bool = True) -> Dict:
    """
    Check GPU availability and get device information.
    
    Args:
        verbose: Print GPU information
        
    Returns:
        Dictionary with GPU availability info
    """
    gpu_info = {
        'available': False,
        'count': 0,
        'devices': [],
        'driver_version': None,
        'cuda_version': None
    }
    
    try:
        # Try nvidia-smi to get GPU info
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info['available'] = True
            gpu_info['count'] = len(lines)
            
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_info['devices'].append({
                        'id': int(parts[0]),
                        'name': parts[1],
                        'memory_mb': int(parts[2])
                    })
        
        # Get driver version
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', 
                               '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info['driver_version'] = result.stdout.strip().split('\n')[0]
    
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        # nvidia-smi not available or failed
        pass
    
    # Try to get CUDA version
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    gpu_info['cuda_version'] = line.split('release')[1].split(',')[0].strip()
                    break
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    if verbose and gpu_info['available']:
        print(f"GPU Information:")
        print(f"  Driver version: {gpu_info['driver_version']}")
        print(f"  CUDA version: {gpu_info['cuda_version']}")
        print(f"  Available GPUs: {gpu_info['count']}")
        for device in gpu_info['devices']:
            print(f"    GPU {device['id']}: {device['name']} ({device['memory_mb']} MB)")
    elif verbose:
        print("No NVIDIA GPUs detected or nvidia-smi not available")
    
    return gpu_info


def clear_devito_cache(verbose: bool = True):
    """
    Clear Devito compilation cache.
    
    Args:
        verbose: Print information about cache clearing
    """
    cache_dir = os.path.expanduser("~/.cache/devito")
    
    if os.path.exists(cache_dir):
        if verbose:
            print(f"Clearing Devito cache at {cache_dir}...")
        try:
            shutil.rmtree(cache_dir)
            if verbose:
                print("Cache cleared successfully.")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not clear cache - {e}")
    else:
        if verbose:
            print("No Devito cache found to clear.")


def setup_devito_cpu_environment(num_threads: Optional[int] = None,
                                compiler: str = 'gcc',
                                clear_cache: bool = True,
                                verbose: bool = True):
    """
    Setup Devito environment for CPU execution.
    
    Args:
        num_threads: Number of OpenMP threads (None for auto)
        compiler: Compiler to use
        clear_cache: Whether to clear cache
        verbose: Print setup information
        
    Returns:
        Dictionary with setup information
    """
    if verbose:
        print("Setting up Devito environment for CPU execution...")
    
    # Set Devito environment for CPU
    devito_env = {
        'DEVITO_LANGUAGE': 'openmp',
        'DEVITO_ARCH': compiler,
        'DEVITO_PLATFORM': 'cpu64',
        'DEVITO_LOGGING': 'INFO' if verbose else 'WARNING',
        'CC': compiler,
        'CXX': 'g++' if compiler == 'gcc' else compiler + '++',
    }
    
    # Set OpenMP threads
    if num_threads is None:
        num_threads = os.cpu_count()
    
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    devito_env['OMP_NUM_THREADS'] = str(num_threads)
    
    # Apply environment variables
    for key, value in devito_env.items():
        os.environ[key] = value
        if verbose:
            print(f"  {key} = {value}")
    
    if clear_cache:
        clear_devito_cache(verbose)
    
    if verbose:
        print(f"CPU environment setup completed with {num_threads} threads.")
    
    return {
        'devito_env': devito_env,
        'num_threads': num_threads,
        'cache_cleared': clear_cache
    }


def auto_setup_environment(prefer_gpu: bool = True,
                          gpu_ids: Optional[List[int]] = None,
                          verbose: bool = True) -> Dict:
    """
    Automatically setup the best available environment.
    
    Args:
        prefer_gpu: Prefer GPU if available
        gpu_ids: Specific GPU IDs to use
        verbose: Print setup information
        
    Returns:
        Dictionary with setup information
    """
    if verbose:
        print("Auto-detecting optimal Devito environment...")
    
    gpu_info = check_gpu_availability(verbose=False)
    
    if prefer_gpu and gpu_info['available']:
        if verbose:
            print("GPUs detected - setting up GPU environment")
        return setup_devito_gpu_environment(gpu_ids, verbose=verbose)
    else:
        if verbose:
            if prefer_gpu:
                print("No GPUs available - falling back to CPU")
            else:
                print("Using CPU as requested")
        return setup_devito_cpu_environment(verbose=verbose)


def print_environment_summary():
    """Print summary of current Devito environment."""
    print("\nCurrent Devito Environment:")
    print("=" * 40)
    
    relevant_vars = [
        'DEVITO_LANGUAGE', 'DEVITO_ARCH', 'DEVITO_PLATFORM',
        'DEVITO_LOGGING', 'CC', 'CXX', 'CUDA_VISIBLE_DEVICES',
        'ACC_DEVICE_TYPE', 'ACC_DEVICE_NUM', 'OMP_NUM_THREADS'
    ]
    
    for var in relevant_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    print("=" * 40)


# Example usage function
def setup_for_fwi_demo(gpu_ids: Optional[List[int]] = None,
                      prefer_gpu: bool = True,
                      verbose: bool = True) -> Dict:
    """
    Setup environment specifically optimized for FWI demonstrations.
    
    Args:
        gpu_ids: GPU IDs to use (None for auto-detect)
        prefer_gpu: Prefer GPU execution
        verbose: Print information
        
    Returns:
        Setup information dictionary
    """
    print("Setting up environment for FWI demonstration...")
    
    # Setup environment
    setup_info = auto_setup_environment(prefer_gpu, gpu_ids, verbose)
    
    # Import Devito after environment setup
    try:
        import devito
        setup_info['devito_version'] = devito.__version__
        if verbose:
            print(f"Devito version: {devito.__version__}")
    except ImportError:
        warnings.warn("Devito not available - please install Devito")
        setup_info['devito_version'] = None
    
    if verbose:
        print_environment_summary()
    
    return setup_info


if __name__ == "__main__":
    # Test the environment setup
    setup_info = setup_for_fwi_demo(verbose=True)
    print(f"\nSetup completed successfully!")
    print(f"GPU available: {setup_info.get('gpu_available', False)}")
    print(f"Number of GPUs: {setup_info.get('num_gpus', 0)}")