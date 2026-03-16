lic_ = """
   Copyright 2025 Richard Tjörnhammar

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
def vverlet_step(r,v,a,m,radius,params,neighbors,dt):

    r1 = r + v*dt + 0.5*a*dt*dt
    a1 = accel(r1,m,params)
    v1 = v + 0.5*(a + a1)*dt

    return r1,v1,a1


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

        if self.neigh.needs_rebuild(r):

            r_sorted, order = self.neigh.rebuild(r)

            v = v[order]
            m = m[order]
            radius = radius[order]

            self.neighbors = self.neigh.neighbors
            self.order = order

            return r_sorted, v, m, radius

        return r, v, m, radius


    def step(self, r, v, m, radius, params, a, dt):

        r, v, m, radius = self.rebuild_if_needed(
            r, v, m, radius
        )

        r, v, a = vverlet_step_with_collisions(
            r, v, a, m, radius, params,
            self.neighbors, dt
        )

        self.step_counter += 1
      
        return r, v, a
    

# ============================================================
# Simulation generator
# ============================================================

def simulate( r, v, m, radius, params, dt ):

    sim = LocalizedVerletSimulator()

    a = accel(r,m,params)

    particle_weight = jnp.ones(len(m))
    active_mask = jnp.ones(len(m))


    while True :

        r,v,a = sim.step(
            r,v,m,radius,
            params,a,dt
        )

        total_fragments, warning = debris_growth_monitor(
            particle_weight,
            active_mask
        )

        if warning:
            instability_warning(total_fragments)

        yield r,v


