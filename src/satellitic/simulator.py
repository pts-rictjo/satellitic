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

import numpy as np
from .constants     import constants_solar_system, solarsystem
from .constants     import celestial_types, build_run_system
from .constants     import build_params, StaticConfig, DynamicParams, SimStateNN, SimState
from .special       import morton_sort, normalize
from .neighbors     import build_neighbor_list, build_neighbor_list_from_cells, build_block_neighbor_list, VerletNeighborManager
from .gravity       import accel, jaxcel, accel_total
from .collisions    import detect_collisions_from_neighbors, collision_response_hashgrid, apply_collision_response, debris_growth_monitor, instability_warning

TYPE_PLANET     = celestial_types['Planet']
TYPE_STAR       = celestial_types['Star']
TYPE_MOON       = celestial_types['Moon']
TYPE_SATELLITE  = celestial_types['Satellit']
TYPE_OTHER      = celestial_types['Other']
TYPE_DEBRIS     = celestial_types['Debris']

def constants( sel = None ) :
    if sel is None :
        return ( constants_solar_system )
    else :
        return ( constants_solar_system[sel] )

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

from functools import partial
@partial(jax.jit, static_argnames=["config"])
def rebuild_neighbors(r, config: StaticConfig): # HERE

    order, grid = morton_sort(r, config.cell_size)
    r_sorted = r[order]

    neighbors = build_neighbor_list(
        r_sorted,
        grid[order],
        config.cutoff,
        config.max_neighbors
    )

    return r_sorted, neighbors, order

@jax.jit
def needs_rebuild(r, ref_r, skin):
    disp = jnp.linalg.norm(r - ref_r, axis=1)
    return jnp.max(disp) > 0.5 * skin

@partial(jax.jit, static_argnames=["config"])
def step_fn_NN(state, dyn: DynamicParams, config: StaticConfig):

    r, v, a, neighbors, ref_r = state

    # -------------------------
    # Neighbor rebuild decision
    # -------------------------
    rebuild = needs_rebuild(r, ref_r, config.skin)

    def do_rebuild(_):
        r_new, neighbors_new, _ = rebuild_neighbors(r, config)
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
    # Integrate (Velocity Verlet)
    # -------------------------
    dt = dyn.dt

    r1 = r + v * dt + 0.5 * a * dt**2
    a1 = jaxcel(r1, dyn.m, dyn)
    v1 = v + 0.5 * (a + a1) * dt

    # -------------------------
    # Collisions
    # -------------------------
    def collisions(args):
        r1, v1 = args

        colliding = detect_collisions_from_neighbors(
            r1, v1, dyn.radii, dyn.m, neighbors
        )

        v2 = apply_collision_response(
            r1, v1, dyn.m, dyn.radii,
            neighbors, colliding,
            restitution=dyn.restitution
        )

        return r1, v2

    def no_collisions(args):
        return args

    r1, v1 = jax.lax.cond(
        config.use_collisions,
        collisions,
        no_collisions,
        (r1, v1)
    )

    return (
        SimStateNN(r1, v1, a1, neighbors, ref_r),
        None
    )

from functools import partial
import jax

@partial(jax.jit, static_argnames=["config"])
def step_fn(state, dyn: DynamicParams, config: StaticConfig):

    r, v, a = state
    dt = dyn.dt

    r1 = r + v * dt + 0.5 * a * dt * dt
    a1 = accel_total(r1, dyn.m, dyn, config)
    v1 = v + 0.5 * (a + a1) * dt

    return SimState(r1, v1, a1), None


@partial(jax.jit, static_argnames=["config"])
def step_fn_with_sr(state, dyn: DynamicParams, config: StaticConfig):

    r, v, a = state
    dt = dyn.dt

    r1 = r + v * dt + 0.5 * a * dt * dt
    a1 = accel_total(r1, dyn.m, dyn, config)
    v1 = v + 0.5 * (a + a1) * dt

    # -------------------------
    # Optional collisions
    # -------------------------
    v1 = collision_response_hashgrid(
            r1, v1, dyn.m,
            dyn.radii, config
        )

    return SimState(r1, v1, a1), None


def initialize_state(r, v, dyn: DynamicParams, config: StaticConfig):
    a0 = accel_total(r, dyn.m, dyn, config)
    return SimState(
        r=r,
        v=v,
        a=a0
    )

def initialize_state_nn(r, v, dyn: DynamicParams, config: StaticConfig):

    a0 = jaxcel(r, dyn.m, dyn)
    r_sorted, neighbors, _ = rebuild_neighbors(r, config)

    return SimStateNN(
        r=r_sorted,
        v=v,
        a=a0,
        neighbors=neighbors,
        ref_positions=r_sorted
    )

@partial(jax.jit, static_argnames=["config", "steps_per_chunk"])
def run_chunk(state, dyn, config, steps_per_chunk):

    if config.use_collisions :
        def body_fn(state, _):
            return step_fn_with_sr(state, dyn, config)
    else:
        def body_fn(state, _):
            return step_fn        (state, dyn, config)

    state_final, traj = jax.lax.scan(
        body_fn,
        state,
        None,
        length=steps_per_chunk
    )
    return state_final, traj


@partial(jax.jit, static_argnames=["config", "steps_per_chunk"])
def run_chunkNN(state, dyn, config, steps_per_chunk):

    def body_fn(state, _):
        return step_fn_NN(state, dyn, config)

    state_final, traj = jax.lax.scan(
        body_fn,
        state,
        None,
        length=steps_per_chunk
    )

    return state_final, traj


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

# ==========================================
# not cpu streaming, stepping within jax
# ==========================================
def jax_chunked_simulator(
    run_parameters = { 'dt':1e0,
            'Nframes': 2 ,
            'steps_per_frame':100 ,
            'mass_epsilon':None ,
            'mass_rule':None,
            'debris':False ,
            'close range interactions' : False,
            'write positions'  : True  ,
            'write velocities' : False ,
            'write masses'     : False } ,
    system_topology     = solarsystem ,
    system_constants    = constants_solar_system ,
    satellite_topology  = None,
    visual_params       = None,
    bWriteTrajectory    = False,
    trajectory_filename = None,
    bVerbose            = False,
    bDebug              = False ):
    import jax
    import jax.numpy as jnp
    import numpy as np

    # --------------------------------------------------
    # Parse parameters
    # --------------------------------------------------
    dt = run_parameters.get("dt", 5.0)
    Nframes = run_parameters.get("Nframes", 1 )
    steps_per_frame = run_parameters.get("steps_per_frame", 100)

    if Nframes is None:
        raise ValueError("JAX simulator requires finite Nsteps")

    total_steps = Nframes * steps_per_frame
    sim_time    = 0.0

    # --------------------------------------------------
    # Build system
    # --------------------------------------------------
    run_system = build_run_system(
        system_topology,
        system_constants,
        satellite_topology
    )
    run_system.apply_barycentric_motion_correction()

    r, v, m, stypes, snames = run_system.phase_state()

    #
    # CREATION AND SETUP OF A LEDGER
    from .constants import InteractionLedger
    if not ( run_parameters.get('mass_epsilon',None) is None and run_parameters.get('mass_rule',None) is None ) :
        run_system.ledger = InteractionLedger( mass_rule = run_parameters.get('mass_rule',None) ,
                    mass_epsilon = run_parameters.get('mass_epsilon',None) )
    else :
        run_system.ledger = InteractionLedger()
    print ( "Using mass interaction rules: (mass_rule , mass_epsilon)" ,
               *run_system.ledger.interaction_rules() )
    ledger = run_system.ledger
    ledger .constants = constants
    ledger .set_phase_space( run_system.phase_space() )
    if run_system.satellites_object is not None :
        ledger .satellites_objects = [ [sobj[0],*sobj[1].get_index_pairs()] for sobj in run_system.satellites_object ]
    ledger .convert_partition_types(jnp)

    if True :
        print("Initial system built:", len(m), "bodies")

    if bVerbose :
        print("With masses : ",  m )
    # --------------------------------------------------
    # Convert to JAX
    # --------------------------------------------------
    r = jnp.asarray(r)
    v = jnp.asarray(v)
    m = jnp.asarray(m)

    # --------------------------------------------------
    # Build params (IMPORTANT: freeze structure)
    # --------------------------------------------------
    base = build_params(run_system)

    params = {
        **base,
        **run_parameters,
        "dt": dt,
        "m": m,
    }

    dyn = DynamicParams(
        G=base["G"],
        idx_massive=jnp.array(base["idx_massive"]),
        idx_light=jnp.array(base["idx_light"]),

        satellite_indices=jnp.array(base["satellite_indices"]),
        satellite_parent=jnp.array(base["satellite_parent"]),
        has_satellites = jnp.array(base["satellite_indices"]).shape[0] > 0,

        planet_indices=jnp.array(base["planet_indices"]),
        planet_J2=jnp.array(base["planet_J2"]),
        planet_R=jnp.array(base["planet_R"]),
        planet_MU=jnp.array(base["planet_MU"]),

        restitution=run_parameters.get("restitution", 1.0),
        radii=jnp.ones(len(r)),  # or real radii

        m=jnp.asarray(m),
        dt=run_parameters.get("dt", 1.0),
    )

    config = StaticConfig(
        cell_size=2e4,
        cutoff=5e4,
        skin=1e4,
        max_neighbors=128,
        grid_capacity=128, # NEW
        use_collisions=run_parameters.get("close range interactions", False),
        has_satellites=dyn.has_satellites
    )
    #
    ## LATER WHEN COLLISIONS
    #params['cell_size'] = 2e3     # 2 km
    #params['cutoff']    = 5e3     # 5 km
    #params['skin']      = 1e3     # 1 km
    #
    # --------------------------------------------------
    # Initialize JAX state
    # --------------------------------------------------
    state = initialize_state(r, v, dyn, config)

    # --------------------------------------------------
    # Trajectory writer (host-side only)
    # --------------------------------------------------
    writer = None
    if bWriteTrajectory:
        from .iotools import TrajectoryManager
        from functools import reduce

        if trajectory_filename is None:
            import time
            trajectory_filename = (
                "traj_" + time.asctime().replace(":", "-").replace(" ", "_") + ".trj"
            )

        flags = []
        if params.get("write positions", True):
            flags.append(TrajectoryManager.FLAG_POS)
        if params.get("write velocities", False):
            flags.append(TrajectoryManager.FLAG_VEL)
        if params.get("write masses", False):
            flags.append(TrajectoryManager.FLAG_MASS)

        iFlag = reduce(lambda x, y: x | y, flags, 0)

        writer = TrajectoryManager(
            trajectory_filename,
            particle_types = run_system.get_particle_types(),
            dt_frame = dt * steps_per_frame,
            version = 2,
            flags = iFlag,
            dynamic = params.get("debris", False),
        )

        writer.write_cdp(run_system)

    # --------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------
    n_chunks = Nframes
    print("Running JAX simulation (chunked)...")
    total_time = n_chunks*steps_per_frame*dt
    print("Will simulate :",total_time,"[secs] or",total_time/60./60./24.,"[days] using a timestep of",dt,"[secs]\n\n" )

    traj = []
    for chunk_id in range(n_chunks):

        # --- JAX execution (fully compiled) ---
        state, _ = run_chunk( state, dyn, config, steps_per_frame )
        traj.append(state)
        #
        # --- Update simulation time ---
        sim_time += dt * steps_per_frame
        #

        # --------------------------------------------------
        # SINGLE device → host transfer per chunk
        # --------------------------------------------------
        r_host = np.asarray(state.r)
        v_host = np.asarray(state.v)

        # --------------------------------------------------
        # Write trajectory (host-side)
        # --------------------------------------------------
        if writer is not None :
            writer.write_step( r = r_host, v = v_host, m = dyn.m )

        # --------------------------------------------------
        # Optional progress
        # --------------------------------------------------
        if (chunk_id % 10 == 0) or n_chunks<10 :
            print(
                f"\rSimulation time: {sim_time:12.3f} s or {sim_time/60./60./24.:12.3f} days | Step: {(chunk_id+1)*steps_per_frame}/{total_steps}",
              end = "",
              flush = True
            )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    if writer is not None:
        writer.close()

    if bDebug :
        print("\n\n",flush=False)
        for s in traj:
            print ( np.sqrt(np.sum((s.r-s.r[3])**2,axis=1) )/ (384399*1000) )

    print("\n\nSimulation complete.")

    return state



# ============================================================
# Simpel Simulation generator improvements needed
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
