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
import jax.numpy as jnp

from .reduce import build_spatial_morton_index

# ============================================================
# Cell linked list neighbor builder
# ============================================================

def build_neighbor_list(r, grid, cutoff, max_neighbors):

    N = r.shape[0]

    neighbors = -jnp.ones((N, max_neighbors), dtype=jnp.int32)

    cutoff2 = cutoff * cutoff

    for i in range(N):

        count = 0

        for j in range(N):

            if i == j:
                continue

            dr = r[i] - r[j]
            d2 = jnp.dot(dr, dr)

            if d2 < cutoff2:

                if count < max_neighbors:
                    neighbors = neighbors.at[i, count].set(j)
                    count += 1

    return neighbors


def build_neighbor_list_from_cells(
    r,
    cutoff,
    max_neighbors,
    cell_size
):
    """
    GPU scalable cell-linked neighbor list.

    Complexity:
        build:  O(N)
        search: O(N * k)

    k ~ particles in nearby cells
    """

    N = r.shape[0]

    # ------------------------------------------------
    # Build grid coordinates
    # ------------------------------------------------

    cell = jnp.floor(r / cell_size).astype(jnp.int32)

    xmin = jnp.min(cell[:,0])
    ymin = jnp.min(cell[:,1])
    zmin = jnp.min(cell[:,2])

    cell = cell - jnp.array([xmin, ymin, zmin])

    nx = jnp.max(cell[:,0]) + 1
    ny = jnp.max(cell[:,1]) + 1
    nz = jnp.max(cell[:,2]) + 1

    # ------------------------------------------------
    # Hash cell index
    # ------------------------------------------------

    cell_id = cell[:,0] + nx*(cell[:,1] + ny*cell[:,2])

    order = jnp.argsort(cell_id)

    r_sorted = r[order]
    cell_sorted = cell[order]
    cell_id_sorted = cell_id[order]

    # ------------------------------------------------
    # Find cell start/end
    # ------------------------------------------------

    unique_cells, start = jnp.unique(cell_id_sorted, return_index=True)

    end = jnp.concatenate([start[1:], jnp.array([N])])

    # ------------------------------------------------
    # Neighbor offsets (27 cells)
    # ------------------------------------------------

    offsets = jnp.array([
        [-1,-1,-1],[-1,-1,0],[-1,-1,1],
        [-1,0,-1],[-1,0,0],[-1,0,1],
        [-1,1,-1],[-1,1,0],[-1,1,1],

        [0,-1,-1],[0,-1,0],[0,-1,1],
        [0,0,-1],[0,0,0],[0,0,1],
        [0,1,-1],[0,1,0],[0,1,1],

        [1,-1,-1],[1,-1,0],[1,-1,1],
        [1,0,-1],[1,0,0],[1,0,1],
        [1,1,-1],[1,1,0],[1,1,1],
    ], dtype=jnp.int32)

    cutoff2 = cutoff*cutoff

    neighbors = -jnp.ones((N,max_neighbors),dtype=jnp.int32)

    # ------------------------------------------------
    # Main search loop
    # ------------------------------------------------

    for i in range(N):

        count = 0

        cell_i = cell_sorted[i]

        for off in offsets:

            cell_n = cell_i + off

            if jnp.any(cell_n < 0):
                continue

            cid = cell_n[0] + nx*(cell_n[1] + ny*cell_n[2])
            match = jnp.where(unique_cells == cid)[0]

            if match.size == 0:
                continue

            s = start[match[0]]
            e = end[match[0]]

            for j in range(s,e):

                if j == i:
                    continue

                dr = r_sorted[i] - r_sorted[j]
                d2 = jnp.dot(dr,dr)

                if d2 < cutoff2:

                    if count < max_neighbors:

                        neighbors = neighbors.at[i,count].set(j)
                        count += 1

    # convert back to original ordering

    inv_order = jnp.argsort(order)
    neighbors = neighbors[inv_order]

    return neighbors

    
def build_block_neighbor_list(
    r,
    cutoff,
    max_neighbors,
    block_size=64
):
    """
    Hilbert-sorted block neighbor list
    """

    N = r.shape[0]

    n_blocks = (N + block_size - 1) // block_size

    neighbors = -jnp.ones((N,max_neighbors),dtype=jnp.int32)

    cutoff2 = cutoff * cutoff

    for b in range(n_blocks):

        start_i = b * block_size
        end_i = min((b+1)*block_size, N)

        # search nearby blocks
        for nb in range(max(0,b-2), min(n_blocks,b+3)):

            start_j = nb * block_size
            end_j = min((nb+1)*block_size, N)
            ri = r[start_i:end_i]
            rj = r[start_j:end_j]
            dr = ri[:,None,:] - rj[None,:,:]
            d2 = jnp.sum(dr*dr,axis=-1)
            mask = d2 < cutoff2

            for i in range(end_i-start_i):

                idx = start_i + i
                candidates = jnp.where(mask[i])[0] + start_j
                count = min(max_neighbors, candidates.shape[0])
                neighbors = neighbors.at[idx,:count].set(candidates[:count])

    return neighbors

class VerletNeighborManager:

    def __init__(self, cutoff, skin):

        self.cutoff = cutoff
        self.skin = skin
        self.rebuild_radius = cutoff + skin

        self.neighbors = None
        self.r_last = None


    def needs_rebuild(self, r):

        if self.r_last is None:
            return True

        disp = jnp.linalg.norm(r - self.r_last, axis=1)

        return jnp.max(disp) > self.skin * 0.5


    def rebuild(self, r):

        grid, order = build_spatial_morton_index(r, bits=21)

        r_sorted = r[order]

        neighbors = build_block_neighbor_list(
            r_sorted,
            self.rebuild_radius,
            64
        )

        self.neighbors = neighbors
        self.r_last = r

        return r_sorted, order
