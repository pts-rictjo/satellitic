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


def instability_warning(total_fragments):

    print("\n⚠ Debris growth warning")
    print("Estimated fragments:",int(total_fragments))
    print("Possible Kessler cascade detected")
