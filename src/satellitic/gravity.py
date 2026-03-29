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

