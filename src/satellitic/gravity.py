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
import jax
jax.config.update("jax_enable_x64", True)
from functools import partial
import jax.numpy as jnp
xp=jnp
bUseJax = True

@jax.jit
def jaxcel(r, m, params):

    G = params.G

    idx_massive = params.idx_massive
    idx_light   = params.idx_light

    satellite_indices = params.satellite_indices
    satellite_parent  = params.satellite_parent

    planet_indices = params.planet_indices
    planet_J2 = params.planet_J2
    planet_R  = params.planet_R
    planet_MU = params.planet_MU

    a = jnp.zeros_like(r)

    # -------------------------
    # Massive ↔ Massive
    # -------------------------

    rM = r[idx_massive]
    mM = m[idx_massive]

    def body_i(ri):

        dr = ri - rM
        r2 = jnp.sum(dr * dr, axis=1)

        inv_r3 = jnp.where(r2 > 0, r2**(-1.5), 0.0)

        return -G * jnp.sum(mM[:, None] * dr * inv_r3[:, None], axis=0)

    aM = jax.vmap(body_i)(rM)

    a = a.at[idx_massive].set(aM)

    # -------------------------
    # Light due to Massive
    # -------------------------

    rL = r[idx_light]

    dr = rL[:, None, :] - rM[None, :, :]
    r2 = jnp.sum(dr * dr, axis=2)

    inv_r3 = jnp.where(r2 > 0.0, r2**(-1.5), 0.0)

    aL = -G * jnp.sum(
        mM[None, :, None] * dr * inv_r3[:, :, None],
        axis=1
    )

    a = a.at[idx_light].set(aL)

    # -------------------------
    # J2 perturbation
    # -------------------------

    if satellite_indices.size > 0:

        r_planets = r[planet_indices]
        r_sats = r[satellite_indices]

        r_parent = r_planets[satellite_parent]

        r_rel = r_sats - r_parent

        x,y,z = r_rel[:,0], r_rel[:,1], r_rel[:,2]

        r2 = x*x + y*y + z*z
        r5 = r2 * r2 * jnp.sqrt(r2)

        J2p = planet_J2[satellite_parent]
        Rp  = planet_R[satellite_parent]
        MUp = planet_MU[satellite_parent]

        factor = 1.5 * J2p * MUp * Rp**2 / r5
        z2_r2 = (z*z)/r2

        ax = factor * x * (5*z2_r2 - 1)
        ay = factor * y * (5*z2_r2 - 1)
        az = factor * z * (5*z2_r2 - 3)

        a_j2 = jnp.stack([ax,ay,az],axis=1)

        a = a.at[satellite_indices].add(a_j2)

    return a


@jax.jit
def accel(r, m, params):

    G = params["G"]

    idx_massive = params["idx_massive"]
    idx_light   = params["idx_light"]

    satellite_indices = params["satellite_indices"]
    satellite_parent  = params["satellite_parent"]

    planet_indices = params["planet_indices"]
    planet_J2 = params["planet_J2"]
    planet_R  = params["planet_R"]
    planet_MU = params["planet_MU"]

    a = jnp.zeros_like(r)

    # -------------------------
    # Massive ↔ Massive
    # -------------------------

    rM = r[idx_massive]
    mM = m[idx_massive]

    def body_i(ri):

        dr = ri - rM
        r2 = jnp.sum(dr * dr, axis=1)

        inv_r3 = jnp.where(r2 > 0, r2**(-1.5), 0.0)

        return -G * jnp.sum(mM[:, None] * dr * inv_r3[:, None], axis=0)

    aM = jax.vmap(body_i)(rM)

    a = a.at[idx_massive].set(aM)

    # -------------------------
    # Light due to Massive
    # -------------------------

    rL = r[idx_light]

    dr = rL[:, None, :] - rM[None, :, :]
    r2 = jnp.sum(dr * dr, axis=2)

    inv_r3 = jnp.where(r2 > 0.0, r2**(-1.5), 0.0)

    aL = -G * jnp.sum(
        mM[None, :, None] * dr * inv_r3[:, :, None],
        axis=1
    )

    a = a.at[idx_light].set(aL)

    # -------------------------
    # J2 perturbation
    # -------------------------

    if satellite_indices.size > 0:

        r_planets = r[planet_indices]
        r_sats = r[satellite_indices]

        r_parent = r_planets[satellite_parent]

        r_rel = r_sats - r_parent

        x,y,z = r_rel[:,0], r_rel[:,1], r_rel[:,2]

        r2 = x*x + y*y + z*z
        r5 = r2 * r2 * jnp.sqrt(r2)

        J2p = planet_J2[satellite_parent]
        Rp  = planet_R[satellite_parent]
        MUp = planet_MU[satellite_parent]

        factor = 1.5 * J2p * MUp * Rp**2 / r5
        z2_r2 = (z*z)/r2

        ax = factor * x * (5*z2_r2 - 1)
        ay = factor * y * (5*z2_r2 - 1)
        az = factor * z * (5*z2_r2 - 3)

        a_j2 = jnp.stack([ax,ay,az],axis=1)

        a = a.at[satellite_indices].add(a_j2)

    return a



from functools import partial
from .special import CELL_OFFSETS, hash3, build_hash_grid

@partial(jax.jit, static_argnames=["config"])
def accel_hashgrid(r, m, config):

    N = r.shape[0]

    grid, cell = build_hash_grid(r, config) # HERE

    num_cells = grid.shape[0]

    # --------------------------------
    # For each particle: compute neighbor cells
    # --------------------------------
    neighbor_cells = cell[:, None, :] + CELL_OFFSETS[None, :, :]   # (N, 27, 3)

    neighbor_hash = hash3(neighbor_cells.reshape(-1, 3))
    neighbor_hash = jnp.mod(neighbor_hash, num_cells)
    neighbor_hash = neighbor_hash.reshape(N, 27)

    # --------------------------------
    # Gather candidates
    # --------------------------------
    candidates = grid[neighbor_hash]   # (N, 27, capacity)

    # Flatten candidates
    candidates = candidates.reshape(N, -1)  # (N, 27*capacity)

    # Mask invalid
    valid = candidates >= 0

    idx = jnp.where(valid, candidates, 0)

    rj = r[idx]          # (N, K, 3)
    mj = m[idx]          # (N, K)

    ri = r[:, None, :]   # (N, 1, 3)

    # --------------------------------
    # Compute interactions
    # --------------------------------
    dr = ri - rj
    d2 = jnp.sum(dr * dr, axis=-1) + 1e-9

    within_cutoff = d2 < config.cutoff**2
    mask = valid & within_cutoff

    inv_r3 = jnp.where(mask, 1.0 / (d2 * jnp.sqrt(d2)), 0.0)

    acc = -jnp.sum(
        (mj * inv_r3)[..., None] * dr,
        axis=1
    )

    return acc

@partial(jax.jit, static_argnames=["config"])
def accel_total(r, m, params, config):

    a = jnp.zeros_like(r)

    # -------------------------
    # Massive ↔ Massive
    # -------------------------
    a = a + accel_massive(r, m, params)

    # -------------------------
    # Light due to Massive
    # -------------------------
    a = a + accel_light_massive(r, m, params)

    # -------------------------
    # Satellite local interactions (JAX-safe)
    # -------------------------
    def sat_branch(a):
        a_local = accel_hashgrid(
            r[params.satellite_indices],
            m[params.satellite_indices],
            config
        )
        return a.at[params.satellite_indices].add(a_local)

    a = jax.lax.cond(
        config.has_satellites and config.use_collisions ,
        sat_branch,
        lambda a: a,
        a
    )

    # -------------------------
    # J2 perturbation
    # -------------------------
    a = a + accel_J2(r, params)

    return a



@jax.jit
def accel_massive(r, m, params):

    G = params.G
    idx = params.idx_massive

    rM = r[idx]
    mM = m[idx]

    def body_i(ri):
        dr = ri - rM
        r2 = jnp.sum(dr * dr, axis=1)
        inv_r3 = jnp.where(r2 > 0, r2**(-1.5), 0.0)
        return -G * jnp.sum(mM[:, None] * dr * inv_r3[:, None], axis=0)

    aM = jax.vmap(body_i)(rM)

    a = jnp.zeros_like(r)
    return a.at[idx].set(aM)


@jax.jit
def accel_light_massive(r, m, params):

    G = params.G

    idxL = params.idx_light
    idxM = params.idx_massive

    rL = r[idxL]
    rM = r[idxM]
    mM = m[idxM]

    dr = rL[:, None, :] - rM[None, :, :]
    r2 = jnp.sum(dr * dr, axis=2)

    inv_r3 = jnp.where(r2 > 0.0, r2**(-1.5), 0.0)

    aL = -G * jnp.sum(
        mM[None, :, None] * dr * inv_r3[:, :, None],
        axis=1
    )

    a = jnp.zeros_like(r)
    return a.at[idxL].set(aL)


@partial(jax.jit, static_argnames=["config"])
def accel_satellite_local(r_sat, m_sat, config):

    # reuse the hashgrid kernel
    return accel_hashgrid(r_sat, m_sat, config)


@jax.jit
def accel_J2(r, params):

    def compute(_):

        r_planets = r[params.planet_indices]
        r_sats = r[params.satellite_indices]

        r_parent = r_planets[params.satellite_parent]

        r_rel = r_sats - r_parent

        x,y,z = r_rel[:,0], r_rel[:,1], r_rel[:,2]

        r2 = x*x + y*y + z*z
        r5 = r2 * r2 * jnp.sqrt(r2)

        J2p = params.planet_J2[params.satellite_parent]
        Rp  = params.planet_R[params.satellite_parent]
        MUp = params.planet_MU[params.satellite_parent]

        factor = 1.5 * J2p * MUp * Rp**2 / r5
        z2_r2 = (z*z)/r2

        ax = factor * x * (5*z2_r2 - 1)
        ay = factor * y * (5*z2_r2 - 1)
        az = factor * z * (5*z2_r2 - 3)

        a_j2 = jnp.stack([ax,ay,az],axis=1)

        a = jnp.zeros_like(r)
        return a.at[params.satellite_indices].add(a_j2)

    return jax.lax.cond(
        params.satellite_indices.size > 0,
        compute,
        lambda _: jnp.zeros_like(r),
        operand=None
    )
