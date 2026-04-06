import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from functools import partial
xp=jnp
bUseJax = True

from .reduce import hilbert_index_batch, morton_index_batch

# ============================================================
# NASA breakup model
# ============================================================

def nasa_breakup_fragments(
    key,
    pos,
    vel,
    total_mass,
    Lmin=0.01,
    Lmax=1.0,
    density=2700.0
):

    N_expected = 0.1 * (total_mass**0.75) * (Lmin**-1.71)

    n_frag = jnp.clip(jnp.int32(N_expected),4,200)

    key1,key2,key3 = jax.random.split(key,3)

    u = jax.random.uniform(key1,(n_frag,))

    alpha = 1.71
    beta = alpha - 1.0

    L = (
        (Lmin**(-beta) +
        u*(Lmax**(-beta) - Lmin**(-beta)))
        **(-1.0/beta)
    )

    volume = (4/3)*jnp.pi*(L/2)**3
    mass = density * volume

    mass = mass * (total_mass / jnp.sum(mass))

    dirs = jax.random.normal(key2,(n_frag,3))
    dirs = dirs / jnp.linalg.norm(dirs,axis=1,keepdims=True)

    frag_pos = pos + dirs * L[:,None]

    dv = jax.random.normal(key3,(n_frag,3))

    frag_vel = vel + dv * 5.0

    return frag_pos, frag_vel, mass, L


# ============================================================
# Fragment cloud compression
# ============================================================

def fragment_cloud_model(
    key,
    pos,
    vel,
    total_mass,
    max_fragments=64
):

    frag_r, frag_v, frag_m, frag_L = nasa_breakup_fragments(
        key,pos,vel,total_mass
    )

    n_frag = frag_r.shape[0]

    if n_frag > max_fragments:

        compression = n_frag // max_fragments + 1
        idx = jnp.arange(0,n_frag,compression)

        frag_r = frag_r[idx]
        frag_v = frag_v[idx]
        frag_m = frag_m[idx] * compression

        weight = jnp.ones_like(frag_m) * compression
        is_cloud = jnp.ones_like(frag_m,dtype=bool)

    else:

        weight = jnp.ones_like(frag_m)
        is_cloud = jnp.zeros_like(frag_m,dtype=bool)

    return frag_r,frag_v,frag_m,weight,is_cloud


# ============================================================
# Debris instability monitoring
# ============================================================
def debris_growth_monitor(
    particle_weight,
    active_mask,
    warning_threshold=1e6
):

    total_fragments = jnp.sum(
        particle_weight * active_mask
    )

    warning = total_fragments > warning_threshold

    return total_fragments, warning


# --------------------------------
# Collision handling
# --------------------------------

@partial(jax.jit, static_argnames=("cutoff_scale", "bits"))
def build_collision_index(r, radii, bits=10, cutoff_scale=2.0):
    """
    Build spatial index optimized for collision detection.
    
    Args:
        r: positions (N, 3)
        radii: particle radii (N,)
        bits: number of bits for spatial hash
        cutoff_scale: safety factor for search radius
    """
    # Effective search radius per particle (2x radius for safety margin)
    effective_radii = radii * cutoff_scale
    
    # Normalize positions to [0, 2^bits - 1] grid
    mins = jnp.min(r - effective_radii[:, None], axis=0)
    maxs = jnp.max(r + effective_radii[:, None], axis=0)
    
    # Scale to integer grid
    grid = jnp.floor(
        (r - mins) / (maxs - mins + 1e-9) * (2**bits - 1)
    ).astype(jnp.uint64)
    
    # Use Hilbert for better locality (or Morton for speed)
    hilbert_idx = hilbert_index_batch(grid, bits)
    
    # Sort by Hilbert index
    order = jnp.argsort(hilbert_idx)
    
    return order, grid, hilbert_idx, mins, maxs


@partial(jax.jit, static_argnames=("neighbor_window"))
def detect_collisions(r, v, radii, mass, order, neighbor_window=64):
    """
    Detect collisions using Hilbert-sorted particles.
    Only check neighbors within a window along the curve.
    """
    N = r.shape[0]
    
    # Sort particles by Hilbert order
    r_sorted = r[order]
    v_sorted = v[order]
    radii_sorted = radii[order]
    mass_sorted = mass[order]
    
    def scan_collisions(i, collisions):
        """Check particle i against neighbors i+1 to i+window"""
        
        # Get current particle
        r_i = r_sorted[i]
        v_i = v_sorted[i]
        r_i_val = radii_sorted[i]
        m_i = mass_sorted[i]
        
        # Define neighbor window (forward only, avoid double counting)
        start = i + 1
        end = jnp.minimum(i + neighbor_window, N)
        
        def check_neighbor(j, coll):
            r_j = r_sorted[j]
            v_j = v_sorted[j]
            r_j_val = radii_sorted[j]
            m_j = mass_sorted[j]
            
            # Vector from i to j
            dr = r_j - r_i
            dist = jnp.sqrt(jnp.sum(dr * dr) + 1e-12)  # avoid div by zero
            
            # Check if overlapping
            overlap = (r_i_val + r_j_val) - dist
            
            # Relative velocity
            dv = v_j - v_i
            
            # Check if approaching (optional - avoid separating pairs)
            approaching = jnp.dot(dv, dr) < 0
            
            # Collision condition
            colliding = (overlap > 0) & approaching
            
            # Update collision list
            coll = jax.lax.cond(
                colliding,
                lambda: coll.at[coll[0]].set(jnp.array([i, j, overlap, dist])),
                lambda: coll
            )
            
            # Increment counter if collision found
            coll = coll.at[0].add(jnp.where(colliding, 1, 0))
            
            return coll
        
        # Scan through neighbors
        collisions = lax.fori_loop(start, end, check_neighbor, collisions)
        
        return collisions
    
    # Initialize collision array: [count, (i, j, overlap, dist), ...]
    max_collisions = N * neighbor_window // 2  # Upper bound
    collisions = jnp.zeros((max_collisions + 1, 4), dtype=jnp.float32)
    collisions = collisions.at[0, 0].set(1)  # Store count at index 0
    
    # Scan all particles
    collisions = lax.fori_loop(0, N - 1, scan_collisions, collisions)
    
    return collisions

@jax.jit
def detect_collisions_from_neighbors(r, v, radii, mass, neighbors):
    """
    Collision detection using precomputed neighbor list.
    """

    N, K = neighbors.shape

    ri = r[:, None, :]
    rj = r[neighbors]

    vi = v[:, None, :]
    vj = v[neighbors]

    ri_rad = radii[:, None]
    rj_rad = radii[neighbors]

    dr = rj - ri
    dist2 = jnp.sum(dr * dr, axis=-1)

    rad_sum = ri_rad + rj_rad
    overlap = rad_sum * rad_sum - dist2

    dv = vj - vi
    approaching = jnp.sum(dv * dr, axis=-1) < 0

    colliding = (overlap > 0) & approaching & (neighbors >= 0)

    return colliding


def collision_event(
    key,
    ri,vi,mi,
    rj,vj,mj
):

    rel_v = jnp.linalg.norm(vi - vj)
    absorb, fragment, elastic = classify_collision(
        mi,mj,rel_v
    )

    # --------------------------------
    # absorption
    # --------------------------------

    ri2,vi2,mi2 = absorb_body(
        ri,vi,mi,
        rj,vj,mj
    )

    # --------------------------------
    # fragmentation
    # --------------------------------

    frag_r,frag_v,frag_m = generate_fragments(
        key,
        (ri+rj)/2,
        (vi+vj)/2,
        mi+mj,
        n_frag=8
    )

    return {
        "absorb": (ri2,vi2,mi2),
        "fragment": (frag_r,frag_v,frag_m),
        "elastic": None
    }


# ============================================================
# Simple Elastic collisions
# ============================================================
@jax.jit
def apply_elastic_collisions(r, v, mass, radii, collisions, restitution=1.0):
    """
    Apply elastic collision responses.
    Uses conservation of momentum and energy.
    """
    N = r.shape[0]
    n_collisions = collisions[0, 0].astype(jnp.int32)
    
    # Temporary buffers for velocity updates
    dv = jnp.zeros_like(v)
    
    def process_collision(k, dv):
        """Process single collision"""
        i = collisions[k, 0].astype(jnp.int32)
        j = collisions[k, 1].astype(jnp.int32)
        overlap = collisions[k, 2]
        
        # Get particle data
        r_i = r[i]
        r_j = r[j]
        v_i = v[i]
        v_j = v[j]
        m_i = mass[i]
        m_j = mass[j]
        
        # Collision normal (from i to j)
        dr = r_j - r_i
        n = dr / (jnp.linalg.norm(dr) + 1e-12)
        
        # Relative velocity
        v_rel = v_j - v_i
        vn = jnp.dot(v_rel, n)
        
        # Only process if approaching
        def apply_impulse(_):
            # Reduced mass
            mu = 2.0 / (1.0/m_i + 1.0/m_j)
            
            # Impulse magnitude (elastic collision)
            J = mu * vn * (1.0 + restitution)
            
            # Update velocities
            dv_i = - (J / m_i) * n
            dv_j =   (J / m_j) * n
            
            # Accumulate updates
            dv = dv.at[i].add(dv_i)
            dv = dv.at[j].add(dv_j)
            
            return dv
        
        # Only apply if approaching
        dv = jax.lax.cond(vn > 0, lambda: dv, apply_impulse, dv)
        
        return dv
    
    # Process all collisions
    dv = lax.fori_loop(1, n_collisions + 1, process_collision, dv)
    
    # Apply velocity updates
    v_new = v + dv
    
    # Optional: separate overlapping particles
    # (simplest - push apart along normal)
    def separate_particles(k, r):
        i = collisions[k, 0].astype(jnp.int32)
        j = collisions[k, 1].astype(jnp.int32)
        overlap = collisions[k, 2]
        
        dr = r[j] - r[i]
        n = dr / (jnp.linalg.norm(dr) + 1e-12)
        
        # Push apart equally
        correction = 0.5 * overlap * n
        r = r.at[i].add(-correction)
        r = r.at[j].add(correction)
        
        return r
    
    # Only separate if needed (can cause instability)
    # r_new = lax.fori_loop(1, n_collisions + 1, separate_particles, r)
    
    return v_new  #, r_new


@jax.jit
def resolve_collisions(r,v,m,radius,neighbors):

    ri = r[:,None,:]
    rj = r[neighbors]

    vi = v[:,None,:]
    vj = v[neighbors]

    mi = m[:,None]
    mj = m[neighbors]

    dr = ri - rj
    dv = vi - vj

    dist = jnp.linalg.norm(dr,axis=-1)

    rad = radius[:,None] + radius[neighbors]

    collision = dist < rad

    n = normalize(dr)

    vrel = jnp.sum(dv*n,axis=-1)

    approaching = vrel < 0

    mask = collision & approaching

    impulse = (2*vrel)/(1/mi + 1/mj)

    impulse = jnp.where(mask,impulse,0)

    delta_v = impulse[...,None]*n

    dv_i = -jnp.sum(delta_v/mi[...,None],axis=1)

    return v + dv_i



# ============================================================
# Collision outcome classifier
# ============================================================

@jax.jit
def classify_collision( mi, mj, rel_speed ):

    mass_ratio = jnp.maximum(mi, mj) / (jnp.minimum(mi, mj) + 1e-12)
    absorb = mass_ratio > 1000.0
    kinetic = 0.5 * (mi*mj)/(mi+mj) * rel_speed**2
    fragment = kinetic > 1e6
    elastic = ~(absorb | fragment)

    return absorb, fragment, elastic


@jax.jit
def absorb_body( ri,vi,mi,rj,vj,mj ):

    total_mass = mi + mj
    new_v = (mi*vi + mj*vj)/total_mass

    return ri,new_v,total_mass


def generate_fragments(key, pos, vel, total_mass, n_frag):

    key1,key2 = jax.random.split(key)

    # random directions
    dirs = jax.random.normal(key1,(n_frag,3))
    dirs = dirs / jnp.linalg.norm(dirs,axis=1,keepdims=True)

    # velocity dispersion
    speeds = jax.random.uniform(key2,(n_frag,)) * 10.0

    frag_v = vel + dirs * speeds[:,None]

    # equal mass split for skeleton
    frag_m = jnp.ones(n_frag) * (total_mass/n_frag)

    frag_r = pos + dirs * 0.1

    return frag_r, frag_v, frag_m

@jax.jit
def collision_response_hashgrid( r, v, m, radii, config ):
    print("NOT IMPLEMENTED: COLLISIONS ARE CURRENTLY UNDER DEVELOPMENT")
    exit(1)

def apply_collision_response( r, v, m, radii, neighbors, colliding,
        restitution = 1.0 ) :
    print("NOT IMPLEMENTED: COLLISIONS ARE CURRENTLY UNDER DEVELOPMENT")
    exit(1)

def instability_warning(total_fragments):
    print("\n Debris growth warning")
    print("Estimated fragments:",int(total_fragments))
    print("Possible Kessler cascade detected")

