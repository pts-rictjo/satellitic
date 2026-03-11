lic_ = """
   Copyright 2026 Richard Tjörnhammar

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import time
import jax
import jax.numpy as jnp
from jax import lax


"""
General N-dimensional Hilbert dimensionality reduction using JAX.
Pipeline:
    N-D coordinates -> Hilbert index -> M-D coordinates
Based on the Skilling Hilbert transform.
Non ensemble method
"""
# -----------------------------------------------------------
# N-D -> Hilbert index
# -----------------------------------------------------------

@jax.jit
def hilbert_index_nd(coords, bits):
    """
    Compute Hilbert index for N-D integer coordinates.

    coords : (N,) uint64
    bits   : bits per dimension
    """

    coords = coords.astype(jnp.uint64)
    n = coords.shape[0]
    x = coords.copy()

    # Gray transform
    def gray_step(i, x):

        bit = jnp.uint64(1) << (bits - 1 - i)

        def inner(j, x):

            cond = (x[j] & bit) != 0

            x = jnp.where(cond, x ^ (bit - 1), x)

            t = (x[0] ^ x[j]) & (bit - 1)

            x0 = x[0] ^ t
            xj = x[j] ^ t

            x = x.at[0].set(x0)
            x = x.at[j].set(xj)

            return x

        x = lax.fori_loop(1, n, inner, x)

        return x

    x = lax.fori_loop(0, bits, gray_step, x)

    # Build index
    def build_index(i, state):

        x, h = state
        b = bits - 1 - i

        digit = jnp.uint64(0)

        for d in range(n):
            digit |= ((x[d] >> b) & 1) << d

        h = (h << n) | digit

        return (x, h)

    x, h = lax.fori_loop(
        0,
        bits,
        build_index,
        (x, jnp.uint64(0))
    )

    return h


# -----------------------------------------------------------
# Batch version
# -----------------------------------------------------------

@jax.jit
def hilbert_index_nd_batch(points, bits):
    return jax.vmap(lambda p: hilbert_index_nd(p, bits))(points)


# -----------------------------------------------------------
# Hilbert index -> coordinates
# -----------------------------------------------------------

@jax.jit
def hilbert_coords_from_index(h, dims, bits):

    x = jnp.zeros(dims, dtype=jnp.uint64)

    def extract(i, state):

        x, h = state

        digit = h & ((1 << dims) - 1)

        for d in range(dims):
            bit = (digit >> d) & 1
            x = x.at[d].set(x[d] | (bit << i))

        h = h >> dims

        return (x, h)

    x, _ = lax.fori_loop(
        0,
        bits,
        extract,
        (x, h)
    )

    return x


@jax.jit
def hilbert_coords_batch(h, dims, bits):
    return jax.vmap(lambda x: hilbert_coords_from_index(x, dims, bits))(h)


# -----------------------------------------------------------
# Dimensionality reduction pipeline
# -----------------------------------------------------------

@jax.jit
def hilbert_project(points, bits, target_dims):

    mins = jnp.min(points, axis=0)
    maxs = jnp.max(points, axis=0)

    norm = (points - mins) / (maxs - mins + 1e-9)

    grid = jnp.floor(norm * (2**bits - 1)).astype(jnp.uint64)

    h = hilbert_index_nd_batch(grid, bits)

    coords = hilbert_coords_batch(h, target_dims, bits)

    return coords.astype(jnp.float32) / (2**bits)


# -----------------------------------------------------------
# Test program
# -----------------------------------------------------------

def run_test():

    key = jax.random.PRNGKey(0)

    n_points = 20000
    dims = 8
    target_dims = 2
    bits = 10

    print("Generating random data...")
    X = jax.random.normal(key, (n_points, dims))

    print(f"Input shape: {X.shape}")

    print("Running Hilbert projection...")

    t0 = time.time()
    Y = hilbert_project(X, bits, target_dims)
    jax.block_until_ready(Y)
    t1 = time.time()

    print("Output shape:", Y.shape)
    print("Time:", t1 - t0, "seconds")

    try:
        import matplotlib.pyplot as plt

        Y_np = jnp.array(Y)

        plt.figure(figsize=(6,6))
        plt.scatter(Y_np[:,0], Y_np[:,1], s=3)
        plt.title("Hilbert Projection Result")
        plt.show()

    except Exception:
        print("matplotlib not available, skipping plot")

# -----------------------------------------------------------
# Ensemble Hilbert method
#
"""
Fast Hilbert Ensemble Projection (N-D → 2D/3D) using JAX.
- Ensemble of random permutations reduces axis bias.
- Averaging multiple Hilbert projections produces high-quality embeddings.
- GPU-friendly, deterministic, extremely fast.
"""

# ------------------------------
# N-D -> Hilbert index
# ------------------------------

@jax.jit
def hilbert_index_nd_ens(coords, bits):
    coords = coords.astype(jnp.uint64)
    n = coords.shape[0]
    x = coords.copy()

    def gray_step(i, x):
        bit = jnp.uint64(1) << (bits - 1 - i)
        def inner(j, x):
            cond = (x[j] & bit) != 0
            x = jnp.where(cond, x ^ (bit - 1), x)
            t = (x[0] ^ x[j]) & (bit - 1)
            x = x.at[0].set(x[0] ^ t)
            x = x.at[j].set(x[j] ^ t)
            return x
        x = lax.fori_loop(1, n, inner, x)
        return x

    x = lax.fori_loop(0, bits, gray_step, x)

    def build_index(i, state):
        x, h = state
        b = bits - 1 - i
        digit = jnp.uint64(0)
        for d in range(n):
            digit |= ((x[d] >> b) & 1) << d
        h = (h << n) | digit
        return (x, h)

    x, h = lax.fori_loop(0, bits, build_index, (x, jnp.uint64(0)))
    return h

@jax.jit
def hilbert_index_nd_batch_ens(points, bits):
    return jax.vmap(lambda p: hilbert_index_nd_ens(p, bits))(points)

# ------------------------------
# Hilbert index -> coordinates
# ------------------------------

@jax.jit
def hilbert_coords_from_index_ens(h, dims, bits):
    x = jnp.zeros(dims, dtype=jnp.uint64)
    def extract(i, state):
        x, h = state
        digit = h & ((1 << dims) - 1)
        for d in range(dims):
            bit = (digit >> d) & 1
            x = x.at[d].set(x[d] | (bit << i))
        h = h >> dims
        return (x, h)
    x, _ = lax.fori_loop(0, bits, extract, (x, h))
    return x

@jax.jit
def hilbert_coords_batch_ens(h, dims, bits):
    return jax.vmap(lambda x: hilbert_coords_from_index_ens(x, dims, bits))(h)

# ------------------------------
# Hilbert Ensemble Projection
# ------------------------------

def ensemble_hilbert_project(X, bits=12, target_dims=2, ensemble_size=4, key=jax.random.PRNGKey(0)):
    N = X.shape[1]
    projected = []

    for i in range(ensemble_size):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)
        X_perm = X[:, perm]

        # Normalize and grid
        mins = jnp.min(X_perm, axis=0)
        maxs = jnp.max(X_perm, axis=0)
        grid = jnp.floor((X_perm - mins)/(maxs - mins + 1e-9) * (2**bits-1)).astype(jnp.uint64)

        # N-D Hilbert index -> target_dims
        h = hilbert_index_nd_batch_ens(grid, bits)
        coords = hilbert_coords_batch_ens(h, target_dims, bits)
        projected.append(coords.astype(jnp.float32) / (2**bits))

    # Average ensemble
    return jnp.mean(jnp.stack(projected, axis=0), axis=0)

# ------------------------------
# Test
# ------------------------------

def run_test_ensemble():
    key = jax.random.PRNGKey(42)
    n_points = 20000
    dims = 8
    target_dims = 2
    bits = 10
    ensemble_size = 6

    print("Generating random data...")
    X = jax.random.normal(key, (n_points, dims))
    print(f"Input shape: {X.shape}")

    print("Running ensemble Hilbert projection...")
    t0 = time.time()
    Y = ensemble_hilbert_project(X, bits, target_dims, ensemble_size, key)
    jax.block_until_ready(Y)
    t1 = time.time()

    print("Output shape:", Y.shape)
    print("Time:", t1 - t0, "seconds")

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        Y_np = jnp.array(Y)
        plt.figure(figsize=(6,6))
        plt.scatter(Y_np[:,0], Y_np[:,1], s=3, alpha=0.7)
        plt.title(f"Ensemble Hilbert Projection ({dims}D → {target_dims}D)")
        plt.show()
    except Exception:
        print("matplotlib not available, skipping plot")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

if __name__ == "__main__":
    run_test_ensemble()
    run_test()

    # alternative
    # stacked = jnp.concatenate(projected, axis=1)
    # Y_2D = PCA_reduce(stacked, 2)
