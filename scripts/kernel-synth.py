# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import functools
from pathlib import Path
from typing import Optional
import os

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)
from tqdm.auto import tqdm

LENGTH = 1024
KERNEL_BANK = [
    ExpSineSquared(periodicity=24 / LENGTH),  # H
    ExpSineSquared(periodicity=48 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=24 * 7 / LENGTH),  # H
    ExpSineSquared(periodicity=48 * 7 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 * 7 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=7 / LENGTH),  # D
    ExpSineSquared(periodicity=14 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=30 / LENGTH),  # D
    ExpSineSquared(periodicity=60 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=365 / LENGTH),  # D
    ExpSineSquared(periodicity=365 * 2 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=4 / LENGTH),  # W
    ExpSineSquared(periodicity=26 / LENGTH),  # W
    ExpSineSquared(periodicity=52 / LENGTH),  # W
    ExpSineSquared(periodicity=4 / LENGTH),  # M
    ExpSineSquared(periodicity=6 / LENGTH),  # M
    ExpSineSquared(periodicity=12 / LENGTH),  # M
    ExpSineSquared(periodicity=4 / LENGTH),  # Q
    ExpSineSquared(periodicity=4 * 10 / LENGTH),  # Q
    ExpSineSquared(periodicity=10 / LENGTH),  # Y
    DotProduct(sigma_0=0.0),
    DotProduct(sigma_0=1.0),
    DotProduct(sigma_0=10.0),
    RBF(length_scale=0.1),
    RBF(length_scale=1.0),
    RBF(length_scale=10.0),
    RationalQuadratic(alpha=0.1),
    RationalQuadratic(alpha=1.0),
    RationalQuadratic(alpha=10.0),
    WhiteKernel(noise_level=0.1),
    WhiteKernel(noise_level=1.0),
    ConstantKernel(),
]


def random_binary_map(a: Kernel, b: Kernel):
    """
    Applies a random binary operator (+ or *) with equal probability
    on kernels ``a`` and ``b``.

    Parameters
    ----------
    a
        A GP kernel.
    b
        A GP kernel.

    Returns
    -------
        The composite kernel `a + b` or `a * b`.
    """
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_maps)(a, b)


def sample_from_gp_prior(
    kernel: Kernel, X: np.ndarray, random_seed: Optional[int] = None
):
    """
    Draw a sample from a GP prior.

    Parameters
    ----------
    kernel
        The GP covaraince kernel.
    X
        The input "time" points.
    random_seed, optional
        The random seed for sampling, by default None.

    Returns
    -------
        A time series sampled from the GP prior.
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2
    gpr = GaussianProcessRegressor(kernel=kernel)
    ts = gpr.sample_y(X, n_samples=1, random_state=random_seed)

    return ts


def sample_from_gp_prior_efficient(
    kernel: Kernel,
    X: np.ndarray,
    random_seed: Optional[int] = None,
    method: str = "eigh",
):
    """
    Draw a sample from a GP prior. An efficient version that allows specification
    of the sampling method. The default sampling method used in GaussianProcessRegressor
    is based on SVD which is significantly slower that alternatives such as `eigh` and
    `cholesky`.

    Parameters
    ----------
    kernel
        The GP covaraince kernel.
    X
        The input "time" points.
    random_seed, optional
        The random seed for sampling, by default None.
    method, optional
        The sampling method for multivariate_normal, by default `eigh`.

    Returns
    -------
        A time series sampled from the GP prior.
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2

    cov = kernel(X)
    ts = np.random.default_rng(seed=random_seed).multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=cov, method=method
    )

    return ts


def generate_time_series(max_kernels: int = 5):
    """Generate a synthetic time series from KernelSynth.

    Parameters
    ----------
    max_kernels, optional
        The maximum number of base kernels to use for each time series, by default 5

    Returns
    -------
        A time series generated by KernelSynth.
    """
    while True:
        X = np.linspace(0, 1, LENGTH)

        # Randomly select upto max_kernels kernels from the KERNEL_BANK
        selected_kernels = np.random.choice(
            KERNEL_BANK, np.random.randint(1, max_kernels + 1), replace=True
        )

        # Combine the sampled kernels using random binary operators
        kernel = functools.reduce(random_binary_map, selected_kernels)

        # Sample a time series from the GP prior
        try:
            ts = sample_from_gp_prior(kernel=kernel, X=X)
        except np.linalg.LinAlgError as err:
            print("Error caught:", err)
            continue

        # The timestamp is arbitrary
        return {"start": np.datetime64("2000-01-01 00:00", "s"), "target": ts.squeeze()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num-series", type=int, default=1000_000)
    parser.add_argument("-J", "--max-kernels", type=int, default=5)
    #Add batch size
    parser.add_argument("--batch-size", type=int, default=10000)  
    #Add resume argument
    parser.add_argument("--resume", type=int, default=0, help="Batch to resume from")  
    
    args = parser.parse_args()
    path = str(Path(__file__).parent / "kernelsynth-data-batch-{}.arrow")
    
    batch_size = args.batch_size
    total_batches = args.num_series // batch_size
    
    # Start processing from the resume point

    for batch in tqdm(range(args.resume, total_batches)):
        batch_path = path.format(batch)
        
        # Skip the batch if it already exists (previously processed)
        if Path(batch_path).exists(batch_path):
            print(f"Batch {batch} already exists, skipping...")
            continue

    generated_dataset = Parallel(n_jobs=-1)(
        delayed(generate_time_series)(max_kernels=args.max_kernels)
        # for _ in tqdm(range(args.num_series))
        for _ in range(batch_size)
    )

    ArrowWriter(compression="lz4").write_to_file(
        generated_dataset,
        path=Path(batch_path),     
    )
    print(f"Batch {batch} saved.")
