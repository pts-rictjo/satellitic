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
from .init import *

import jax
import jax.numpy as jnp

from .special       import morton_sort, normalize
from .neighbors     import build_neighbor_list, build_neighbor_list_from_cells, build_block_neighbor_list, VerletNeighborManager
from .gravity       import accel
from .collisions    import detect_collisions_from_neighbors, apply_collision_response, debris_growth_monitor, instability_warning


# ============================================================
# Velocity Verlet
# ============================================================

@jax.jit
def vverlet_step(r, v, a, m, radii, params, neighbors, dt):

    r1 = r + v*dt + 0.5*a*dt*dt
    a1 = accel(r1,m,params)
    v1 = v + 0.5*(a + a1)*dt

    return r1, v1, a1


@jax.jit
def vverlet_step_with_collisions(r, v, a, m, radii, params, neighbors, dt):

    r1 = r + v*dt + 0.5*a*dt*dt
    a1 = accel(r1, m, params)
    v1 = v + 0.5*(a + a1)*dt

    colliding = detect_collisions_from_neighbors(
        r1, v1, radii, m, neighbors
    )

    v2 = apply_collision_response(
        r1, v1, m, radii, neighbors, colliding,
        restitution=params.get("restitution", 1.0)
    )

    return r1, v2, a1


# ============================================================
# Simulator
# ============================================================

class MortonVerletSimulator:

    def __init__(
        self,
        cell_size          = 1e7,
        cutoff             = 2e7,
        max_neighbors      = 64,
        rebuild_frequency  = 20
    ):

        self.cell_size = cell_size
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.rebuild_frequency = rebuild_frequency

        self.step_counter = 0
        self.neighbors = None


    def rebuild(self,r):

        order, grid = morton_sort(r,self.cell_size)

        r_sorted = r[order]

        neighbors = build_neighbor_list(
            r_sorted,
            grid[order],
            self.cutoff,
            self.max_neighbors
        )

        self.neighbors = neighbors

        return r_sorted, order


    def step(self,r,v,m,radius,params,a,dt):

        if self.step_counter % self.rebuild_frequency == 0:

            r,order = self.rebuild(r)

            v = v[order]
            m = m[order]
            radius = radius[order]

        r,v,a = vverlet_step(
            r,v,a,m,radius,
            params,self.neighbors,dt
        )

        self.step_counter += 1

        return r,v,a


from functools import partial

class LocalizedVerletSimulator:

    def __init__(
        self,
        cutoff          = 2e7 ,
        skin            = 1e6 ,
        max_neighbors   = 64
    ):

        self.cutoff         = cutoff
        self.max_neighbors  = max_neighbors
        self.neigh          = VerletNeighborManager(cutoff, skin, max_neighbors = self.max_neighbors)
        self.neighbors      = None
        self.order          = None

        self.step_counter = 0
        self.neighbors = None


    def rebuild_if_needed(self, r, v, m, radius):

        if self.neighbors is None or self.neigh.needs_rebuild(r) :

            r_sorted, order = self.neigh.rebuild(r)

            v = v[order]
            m = m[order]
            radius = radius[order]

            self.neighbors = self.neigh.neighbors
            self.order = order

            return r_sorted, v, m, radius

        return r, v, m, radius

    def step(self, r, v, a, m, radius, params, dt):

        r, v, m, radius = self.rebuild_if_needed(
            r, v, m, radius
        )

        r, v, a = vverlet_step_with_collisions(
            r, v, a, m, radius, params,
            self.neighbors, dt
        )

        self.step_counter += 1
      
        return r, v, a
    

    #@partial(jax.jit, static_argnames=["steps_per_frame"])
    def multi_step( self, r, v, a, m, radius, params, dt, steps_per_frame ):

        # TODO fixme, smarter branching and not self function
        #       to include in all step scans (rebuilds neighbours)
        r, v, m, radius = self.rebuild_if_needed(
            r, v, m, radius
        )

        def body(carry, _):
            r, v, a = carry

            r, v, a = vverlet_step_with_collisions(
                r, v, a, m, radius, params,
                self.neighbors, dt
            )

            return (r, v, a), None

        (r, v, a), _ = jax.lax.scan(body, (r, v, a), None, length = steps_per_frame )
        return r, v, a


    #@partial(jax.jit, static_argnames=["steps_per_frame"])
    def multi_step_lr( self, r, v, a, m, radius, params, dt, steps_per_frame ):

        def body(carry, _):
            r, v, a = carry

            r, v, a = vverlet_step(
                r, v, a, m, radius, params,
                self.neighbors, dt
            )

            return (r, v, a), None

        (r, v, a), _ = jax.lax.scan(body, (r, v, a), None, length = steps_per_frame )
        return r, v, a

# ============================================================
# Better JAX compatibility -- WARNING UNDER DEVELOPMENT --
# ============================================================
from typing import NamedTuple

class SimState(NamedTuple):
    r: jnp.ndarray
    v: jnp.ndarray
    a: jnp.ndarray
    neighbors: jnp.ndarray
    ref_positions: jnp.ndarray   # for Verlet skin

@jax.jit
def rebuild_neighbors(r, params):

    order, grid = morton_sort(r, params["cell_size"])
    r_sorted = r[order]

    neighbors = build_neighbor_list(
        r_sorted,
        grid[order],
        params["cutoff"],
        params["max_neighbors"]
    )

    return r_sorted, neighbors, order

@jax.jit
def needs_rebuild(r, ref_r, skin):

    disp = jnp.linalg.norm(r - ref_r, axis=1)
    return jnp.max(disp) > 0.5 * skin


@jax.jit
def step_fn(state: SimState, params):

    r, v, a, neighbors, ref_r = state

    # -------------------------
    # Decide rebuild (NO Python if)
    # -------------------------
    rebuild = needs_rebuild(r, ref_r, params["skin"])

    def do_rebuild(_):
        r_new, neighbors_new, _ = rebuild_neighbors(r, params)
        return r_new, neighbors_new, r_new

    def no_rebuild(_):
        return r, neighbors, ref_r

    r, neighbors, ref_r = jax.lax.cond(
        rebuild,
        do_rebuild,
        no_rebuild,
        operand=None
    )

    # -------------------------
    # Integrate
    # -------------------------
    r1 = r + v * params["dt"] + 0.5 * a * params["dt"]**2
    a1 = accel(r1, params["m"], params)
    v1 = v + 0.5 * (a + a1) * params["dt"]

    # -------------------------
    # Collisions (optional)
    # -------------------------
    def collisions(args):
        r1, v1 = args

        colliding = detect_collisions_from_neighbors(
            r1, v1, params["radii"], params["m"], neighbors
        )

        v2 = apply_collision_response(
            r1, v1, params["m"], params["radii"],
            neighbors, colliding,
            restitution=params.get("restitution", 1.0)
        )

        return r1, v2

    def no_collisions(args):
        return args

    r1, v1 = jax.lax.cond(
        params["use_collisions"],
        collisions,
        no_collisions,
        (r1, v1)
    )

    new_state = SimState(
        r=r1,
        v=v1,
        a=a1,
        neighbors=neighbors,
        ref_positions=ref_r
    )

    return new_state, None


def run_jax_simulation(state0, params, nsteps):

    def body_fn(state, _):
        return step_fn(state, params)

    state_final, _ = jax.lax.scan(
        body_fn,
        state0,
        None,
        length=nsteps
    )

    return state_final

def initialize_state(r, v, m, params):

    a0 = accel(r, m, params)

    r_sorted, neighbors, _ = rebuild_neighbors(r, params)

    return SimState(
        r=r_sorted,
        v=v,
        a=a0,
        neighbors=neighbors,
        ref_positions=r_sorted
    )

@jax.jit
def run_chunk(state0, params, steps_per_chunk):

    def body_fn(state, _):
        new_state, _ = step_fn(state, params)
        return new_state, new_state  # store outputs

    state_final, traj = jax.lax.scan(
        body_fn,
        state0,
        None,
        length=steps_per_chunk
    )

    return state_final, traj


@jax.jit
def run_jax_simulation_trajectory(state0, params, nsteps):

    def body_fn(state, _):
        new_state, _ = step_fn(state, params)
        return new_state, state   # store previous state

    final_state, trajectory = jax.lax.scan(
        body_fn,
        state0,
        None,
        length=nsteps
    )

    return final_state, trajectory


def simulate_jax_stream(r, v, m, dt, Nsteps, steps_per_frame, params, radius=None):

    state = initialize_state(r, v, m, params)

    while True:
        state, traj = run_chunk(state, params, steps_per_frame)

        # Yield frame-by-frame
        for i in range(steps_per_chunk):
            yield (
                traj.r[i],
                traj.v[i],
                traj.a[i]
            )

# ============================================================
# Simpel Simulation generator improve needed
# ============================================================
def simulate( r, v, m, dt, Nsteps, steps_per_frame, params, radius=None ):
    if radius is None :
        radius = np.ones(len(r))
    sim = LocalizedVerletSimulator()

    a = accel( r , m , params )

    particle_weight = jnp.ones(len(m))
    active_mask     = jnp.ones(len(m))

    step_count = 0

    while True :

        if params.get('close range interactions',False) :
            r,v,a = sim.multi_step(
                r,v,a,m,radius,
                params,dt,steps_per_frame
            )
        else: # long range only
            r,v,a = sim.multi_step_lr(
                r,v,a,m,radius,
                params,dt,steps_per_frame
            )

        if params.get('debris',False) and params.get('close range interactions',False):
            total_fragments, warning = debris_growth_monitor(
                particle_weight,
                active_mask
            )
            if warning:
                instability_warning(total_fragments)

        step_count += steps_per_frame
        yield r , v , m, step_count
