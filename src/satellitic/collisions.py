import jax
import jax.numpy as jnp

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



def instability_warning(total_fragments):
    print("\n⚠ Debris growth warning")
    print("Estimated fragments:",int(total_fragments))
    print("Possible Kessler cascade detected")

