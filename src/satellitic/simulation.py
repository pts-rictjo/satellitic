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
# -----------------------
# Top-level Newtonian dynamics
# -----------------------
import numpy as np
from .constants import constants_solar_system
from .constants import celestial_types

TYPE_PLANET     = celestial_types['Planet']
TYPE_STAR       = celestial_types['Star']
TYPE_MOON       = celestial_types['Moon']
TYPE_SATELLITE  = celestial_types['Satellit']
TYPE_OTHER      = celestial_types['Other']

import sys
def excepthook(type, value, tb):
    import traceback
    traceback.print_exception(type, value, tb)
sys.excepthook = excepthook

# nix-shell -p {python313Packages.numpy,python313Packages.matplotlib,python313Packages.jax,python313Packages.sgp4,python313Packages.vispy,python313Packages.pyqt5,libsForQt5.qtbase}

try :
        import jax
        jax.config.update("jax_enable_x64", True)
        bUseJax = True
        print("ImportSuccess:", "HAS JAX IN ENVIRONMENT")
except ImportError :
        print ( "ImportError:","JAX: WILL NOT USE IT")
        bUseJax = False
except OSError:
        print ( "OSError:","JAX: WILL NOT USE IT")
        bUseJax = False

class Starsystem ( object ) :
    def __init__( self , constants ) :
        self.components_ = None
        self.celestial_bodies_ = None
        self.celestial_types_ = None
        self.celestial_names_ = None
        self.constants_ = constants

    def planets( self ) :
        if not self.celestial_types_ is None :
            plnt_idx = np.where( self.celestial_types_ == 'Planet' )
            return self.celestial_names_[plnt_idx],plnt_idx

    def constants( self , sel = None ) :
        if sel is None :
            return ( self.constants_ )
        else :
            return ( self.constants_[sel] )

class Celestial( object ):
    def __init__( self ) :
        self.atmosphere_ = None
        self.mass_ = None
        self.radius_ = None
        self.surface_temperature_ = None
        self.composition_ = None

def constants( sel = None ) :
    if sel is None :
        return ( constants_solar_system )
    else :
        return ( constants_solar_system[sel] )

def backend_array(x, xp):
    return xp.asarray(x)

def leapfrog( dt , r , v , F, m ) :
    # LEGACY
    E = m * np.sum(v**2) * 0.5
    v = v + dt * F / m
    r = r + dt * v
    E = 0.5* ( E + m * np.sum(v**2) * 0.5 )
    return ( r, v, F )

def accel_direct( r , m ):
    G = constants('G')
    rr = vdist(r)
    dist = np.linalg.norm(rr, axis=2)
    np.fill_diagonal(dist, np.inf)
    return -G * np.sum(
        m[None,:,None] * rr / dist[:,:,None]**3,
        axis = 1
    )

# softening factor, setting it once
# soft smoothing for numerical stability
# smallest physical scale that does not
# need to be resolved. local overrides present
# the global value is overriden for tle satellites
# and asteriods/comets
eps_ = 1e4  # meters

if bUseJax :
    G_ = constants('G')
    import jax.numpy as jnp
    @jax.jit
    def accel_jax(r, m):
        N       = r.shape[0]
        mask    = 1.0 - jnp.eye(N, dtype=r.dtype)
        G       = G_
        rr      = r[:,None,:] - r[None,:,:]
        dist2   = jnp.sum(rr*rr, axis=2) + eps_*eps_ # soft smoothing
        inv_r3 = jnp.where(
            dist2 > 0,
            dist2**(-1.5),
            0.0
        ) * mask
        return ( -G * jnp.sum(
                m[None,:,None] * rr * inv_r3[:,:,None],
                axis=1 )  
            )

    J2	    = constants('J2')
    RE	    = constants('RE')
    MU_E	= constants('MU_E')

    # @jax.jit
    def accel_j2(r_rel):
        eps_ = 50
        """
        r_rel: (Nleo, 3) position relative to Earth center
        """
        x, y, z = r_rel[:,0], r_rel[:,1], r_rel[:,2]

        r2 = x*x + y*y + z*z + eps_*eps_
        r  = jnp.sqrt(r2)

        z2_r2 = (z*z) / r2
        factor = 1.5 * J2 * MU_E * RE*RE / r**5

        ax = factor * x * (5*z2_r2 - 1)
        ay = factor * y * (5*z2_r2 - 1)
        az = factor * z * (5*z2_r2 - 3)

        return jnp.stack([ax, ay, az], axis=1)

    @jax.jit
    def total_accel(r, m, idx_earth, idx_leo):
        a		= accel_jax(r, m)		# Newtonian accelerations
        if idx_earth is None or idx_leo is None :
            return a
        rE		= r[idx_earth]
        r_rel	= r[idx_leo] - rE
        a_j2	= accel_j2(r_rel)		# Adding in J2 corrections for LEO objects
        a		= a.at[idx_leo].add(a_j2)
        return a

def accel_soft( r , m , idx_earth=None, idx_leo=None ):
    # with numerical softening    
    G = constants('G')
    rr = vdist(r)
    dist = np.sqrt(np.sum(rr*rr, axis=2) + eps_*eps_)
    np.fill_diagonal(dist, np.inf)
    return -G * np.sum(
        m[None,:,None] * rr / dist[:,:,None]**3,
        axis=1
    )
#
# choose one acceleration form
accel = accel_soft
if bUseJax :
    accel = total_accel

def vverlet_a(dt, r, v, a, m , idx_earth=None, idx_leo=None ):
    r1 = r + v*dt + 0.5*a*dt*dt
    a1 = accel(r1, m, idx_earth, idx_leo)
    v1 = v + 0.5*(a + a1)*dt
    return r1, v1, a1

def vverlet_F( dt , r , v, F , m , mm ):
    Fdm	= (F.T / m).T * 0.5
    r1		= r + v*dt + Fdm*dt*dt
    v05	= v + Fdm*dt
    rr		= vdist( r )
    F1	= forceG( rr , mm )
    v1	= v05 + 0.5 * (F1.T / m).T * dt
    return r1 , v1 , F1

def forceG( rr , mm ) :
    G		= constants('G')
    r3		= np.sum(rr**2,axis=2)**1.5
    er3	= rr.T/r3
    er3[np.isnan(er3)] = 0
    return ( np.sum( G * mm * er3 , axis=2).T )

def vdist( r , type=0 ):
    if type==0:
        return ( r[:,None,:] - r[None,:,:] )
    else:
        return ( np.array([[ (r_-w_) for r_ in r] for w_ in r]) )

def productpairs_z( m ):
    return ( np.outer(m,m)*(1-np.eye(len(m))) )

def angular_momentum( r , v , m ):
    return ( np.sum(np.cross(r, m[:,None]*v), axis=0) )

def energy(r, v, m):
    KE = 0.5 * np.sum(m * np.sum(v*v, axis=1))
    rr = vdist(r)
    dist = np.linalg.norm(rr, axis=2)
    np.fill_diagonal(dist, np.inf)
    PE = -0.5 * constants('G') * np.sum(m[:,None]*m[None,:] / dist)
    return KE + PE, KE, PE

def add_particles(r, v, m, r_new, v_new, m_new):
    r = np.vstack([r, r_new])
    v = np.vstack([v, v_new])
    m = np.concatenate([m, m_new])
    return r, v, m

def classify_earth_satellites(r, idx_earth, R_max=1e8):
    # Determine J2 addition for satellites
    # Rudimentary simple first entry logic for asteroids etc
    r_rel = r - r[idx_earth]
    dist = np.linalg.norm(r_rel, axis=1)
    return dist < R_max

def satellite_masses(N, default=1000.0):
    return np.full(N, default)

def simulate(r, v, m, dt, Nsteps=None, steps_per_frame=10, idx_earth=None, idx_leo=None, system=None ):
    a = accel( r, m, idx_earth, idx_leo )
    if Nsteps is None :
        while True:
            for _ in range(steps_per_frame):
                r, v, a = vverlet_a(dt, r, v, a, m, idx_earth, idx_leo)
            yield r
    else :
        for step in range(Nsteps):
            for _ in range(steps_per_frame):
                r, v, a = vverlet_a(dt, r, v, a, m, idx_earth, idx_leo)
            yield r

def newtonian_simulator( bAnimated = True ,
    tle_file_name	= None  ,
    Nsteps			= None ,
    dt				= 5e1   , # sec.s.
    steps_per_frame = 100   ,
    use_the_force	= False ,
    trajectory_filename = "trajectory.trj", 
    bLegacy         = False , bVerbose=False ) :
    if bVerbose:
        print( """ time step examples for stable orbits
    Orbit	dt
    LEO	1–10 s
    MEO	30–60 s
    GEO	300 s
    """ )
    if not Nsteps is None :
        dt_frame = dt * steps_per_frame

    if bAnimated :
        from .visuals import choose_vispy_backend
        bGUI = choose_vispy_backend()
        if bGUI :
            # ---- VisPy 2D projected solar system plot ----
            from vispy import app, scene
            from vispy.scene import visuals
        else:
            print('Nonfatal error: Could not connect to backend GUI (pyqt5 etc). Will not run pyvis')
    
    const			= constants()
    #
    # Solar system setup
    G           = constants('G')
    D           = constants('DMoon')
    MSun        = constants('MSun')
    ME          = constants('MEarth')
    MM          = constants('MMoon')
    AU          = constants('AU')
    rE          = -MM / (ME + MM) * D
    rM          =  ME / (ME + MM) * D
    omega_EM    = np.sqrt( G * (ME + MM) / D**3) # EARTH-MOON angular velocity
    vE          = omega_EM * abs(rE)
    vM          = omega_EM * abs(rM)
    vES         = np.sqrt(constants('G') * MSun / AU) # EARTH-SUN orbital velocity
    #
    # Create start vectors
    r = [
        [0.0, 0.0, 0.0],              # Sun
        [AU + rE, 0.0, 0.0],          # Earth
        [AU + rM, 0.0, 0.0]           # Moon
    ]
    v = [
        [0.0, 0.0, 0.0],              # Sun
        [0.0, vES + vE, 0.0],         # Earth
        [0.0, vES + vM, 0.0]          # Moon
    ]
    m			= [MSun, ME, MM]
    idx_sun     = 0
    idx_earth	= 1
    idx_moon    = 2
    #
    xp	= np
    r   = backend_array(r, xp)
    v   = backend_array(v, xp)
    m   = backend_array(m, xp)
    if bVerbose:
        print('Collected : ' , r, v, m )
    #
    # Add random extra objects
    Nextra = 0
    if Nextra > 0 :
        r_extra = AU * (1 + 0.2*np.random.randn(Nextra,3))
        v_extra = np.cross(
            r_extra,
            np.array([0,0,1])
        ) # project onto 2D plane
        v_extra *= np.sqrt(constants('G')*MSun / np.linalg.norm(r_extra,axis=1))[:,None]
        m_extra = np.full(Nextra, 1e12)  # small bodies

        r = np.vstack([r, r_extra])
        v = np.vstack([v, v_extra])
        m = np.concatenate([m, m_extra])

        print ( np.shape(v) , np.shape(r) , np.shape(m) )

    Ncurrent = len(m)

    if not tle_file_name is None :
        from datetime import datetime
        from .iotools import read_tles, tles_to_states
        #
        epoch = datetime(2025, 1, 1)
        sats				= read_tles(tle_file_name)
        r_sat, v_sat, names	= tles_to_states(sats, epoch)
        r_sat += r[idx_earth]
        v_sat += v[idx_earth]
        m_sat			= satellite_masses(len(r_sat))
        Nsat			= len(m_sat)
        d_sat			= np.linalg.norm(r_sat,axis=1)
        idx_leo			= np.array(range( Ncurrent, Ncurrent+Nsat ))
        r, v, m = add_particles(r, v, m, r_sat, v_sat, m_sat)
    else :
        idx_leo = None

    Ncurrent = len(m)
    print('Simulating N-body Newtonian celestial dynamics')
    print( Ncurrent , 'body problem')
    print('Velocity verlet updates')
    if not idx_leo is None :
        print('Applying J2 corrections to LEO satellites')
    particle_type = np.full(Ncurrent, TYPE_OTHER, dtype=np.uint8)
    particle_type[idx_sun]   = TYPE_STAR
    particle_type[idx_earth] = TYPE_PLANET
    particle_type[idx_moon]  = TYPE_MOON
    if not idx_leo is None :
        particle_type[idx_leo]   = TYPE_SATELLITE
    #
    # Barycentric motion correction
    P = np.sum(m[:,None] * v, axis=0)
    v -= P / np.sum(m)
    if bLegacy:
        #
        # BOOTSTRAP TESTING
        rr	= vdist(r)
        mm	= productpairs_z(m)
        #
        # Initial step values
        F		= forceG(rr,mm)
    a		= accel(r, m, idx_earth, idx_leo)
    #
    if bVerbose and bLegacy:
        print ( np.shape((F.T/m).T) )
        print ( 'accelerations:', (F.T/m).T , a )
        print ( idx_earth, idx_leo )
        #
        # DO SINGLE UPDATE
        print ( 'DAYS:' , Nsteps*dt/60/60/24 )

    if bLegacy:
        print('WARNING OLD CODE REMNANT')
        if Nsteps is None :
            print('Call simulator with Nsteps>0')
            exit(1)
        #
        # LEGACY SIMULATOR
        traj   = []
        vverlet = vverlet_F if use_the_force else vverlet_a
        F = F if use_the_force else a
        for step in range(Nsteps):
            r, v, F = vverlet( dt , r , v, F , m , mm )
            traj.append([r,v,F])
        print('Did a simulation output ' + 'forces' if use_the_force else 'accelerations' )

        # LEGACY VISUALISATION
        # choose a particle (Moon is 2)
        iparticle = 2

        import matplotlib.pyplot as plt
        R = np.array([ t[0][iparticle] for t in traj ])
        P = np.array([ t[0][1] for t in traj ])
        print(R)

        x = [ r[0] for r in R-P ]
        y = [ r[1] for r in R-P ]
        z = [ r[2] for r in R-P ]

        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect('equal')
        ax.set_xlim(-1.2*AU, 1.2*AU)
        ax.set_ylim(-1.2*AU, 1.2*AU)
        ax.set_facecolor("black")

        plt.plot( x, y, 'k' )
        plt.plot( z, y, 'b' )
        plt.plot( x, z, 'b' )
        plt.show()

    # ---- Initial data ----
    r_np = np.asarray(r)
    N = r_np.shape[0]
    sizes = np.full(N, 2.0, dtype=np.float32)
    colors = np.full((N, 4), [0.5, 0.5, 0.5, 1.0], dtype=np.float32)

    if bAnimated and bGUI:
        # special bodies
        sizes[idx_sun]   = 20
        sizes[idx_earth] = 16
        sizes[idx_moon]  = 8

        colors[idx_sun]   = [1.0, 1.0, 0.0, 1.0]   # yellow
        colors[idx_earth] = [0.0, 0.4, 1.0, 1.0]   # blue
        colors[idx_moon]  = [1.0, 1.0, 1.0, 1.0]   # white
        
        # Canvas & view
        canvas = scene.SceneCanvas(
            keys='interactive',
            size=(800, 800),
            bgcolor='black',
            show=True
        )
        # Whole system
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.PanZoomCamera(aspect=1)
        view.camera.set_range(
            x=(-1.2*AU, 1.2*AU),
            y=(-1.2*AU, 1.2*AU)
        )

        # Scatter visual
        markers = visuals.Markers()
        markers.set_data(
            r_np[:, :3],
            face_color=colors,
            size=sizes
        )
        view.add(markers)
		
        if not idx_leo is None :
            # Earth-centric 3D canvas
            canvas_ec = scene.SceneCanvas(
                keys='interactive',
                size=(600, 600),
                bgcolor='black',
                show=True,
                title='Earth-centric view'
            )
            Re = constants('REarth')
            view_ec = canvas_ec.central_widget.add_view()
            view_ec.camera = scene.cameras.TurntableCamera(
                fov=45,
                distance=Re*5
            )
            earth_group = np.concatenate([[idx_moon], idx_leo])
            r_ec = r_np[earth_group] - r_np[idx_earth]
            markers_ec = visuals.Markers()
            markers_ec.set_data(
                r_ec,
                size=6,
                face_color=[1, 1, 1, 1]
            )
            view_ec.add(markers_ec)

            earth = visuals.Sphere(
                radius=Re,
                method='latitude',
                color=(0.1, 0.3, 1.0, 0.4)
            )
            view_ec.add(earth)
            R = constants('DMoon')  # ~ Moon distance
            view_ec.camera.set_range(
                x=(-R, R),
                y=(-R, R),
                z=(-R, R)
            )

    # Simulation generator
    sim = simulate(
        r, v, m,
        dt=dt , Nsteps=Nsteps ,
        steps_per_frame=steps_per_frame,
        idx_earth=idx_earth,
        idx_leo=idx_leo
    )

    if not Nsteps is None :
        from .iotools import TrajectoryManager
        writer = TrajectoryManager(
            trajectory_filename,
            particle_type=particle_type,
            N_steps=Nsteps, dt_frame=dt_frame
        )

        for step in range(Nsteps):
            r_new = next(sim)
            writer.write_step(np.asarray(r_new))

        writer.close()

    if bAnimated and bGUI :
        # Animation timer
        def update(event):
            r_new = next(sim)
            r_np = np.asarray(r_new)

            markers.set_data(
                r_np[:, :3],
                face_color=colors,
                size=sizes
            )
            if not idx_leo is None :
                # Earth-centric 3D view
                r_ec = r_np[earth_group] - r_np[idx_earth]
                markers_ec.set_data(
                    r_ec,
                    face_color=[1, 1, 1, 1],
                    size=6
                )

        timer = app.Timer(interval=1/30)
        timer.connect(update)
        timer.start()
        app.run()

# -----------------------
# Top-level pipeline
# -----------------------
def run_snapshot_simulation(
    out_dir: str = "simulator_output",
    groups: List[str] = CELESTRAK_GROUPS,
    local_tle_file: str = None,
    N_target: int = DEFAULT_N_TARGET,
    grid_nlat: int = DEFAULT_GRID_NLAT,
    grid_nlon: int = DEFAULT_GRID_NLON,
    model: str = DEFAULT_BEAM_MODEL,
    n_beams_per_sat: int = DEFAULT_N_BEAMS_PER_SAT,
    beam_half_angle_deg: float = DEFAULT_BEAM_HALF_ANGLE_DEG,
    beam_pattern: str = DEFAULT_BEAM_PATTERN,
    beam_max_tilt_deg: float = DEFAULT_BEAM_MAX_TILT_DEG,
    beam_gain_model: str = DEFAULT_BEAM_MODEL,
    gain_threshold: float = DEFAULT_GAIN_THRESHOLD,
    frequency_band: str = DEFAULT_FREQUENCY_BAND,
    preferred_bands: Dict[str, Tuple[float,float]] = PREFERRED_BANDS,
    chunk_sat: int = DEFAULT_CHUNK_SAT,
    chunk_ground: int = DEFAULT_CHUNK_GROUND,
    use_gpu_if_available: bool = USE_GPU_IF_AVAILABLE,
    compute_power_map: bool = False,
    save_tles_to_disk: bool = False,
    do_random_sampling:bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    # 1) gather TLEs (CelesTrak primary, local fallback)
    tles = []
    if local_tle_file is None :
        from .iotools import fetch_tle_group_celestrak, parse_tle_text
        for g in groups:
            try:
                print(f"Fetching TLEs for group '{g}' from CelesTrak...")
                raw = fetch_tle_group_celestrak(g)
                tles_group = parse_tle_text(raw)
                print(f"  parsed {len(tles_group)} TLEs from {g}")
                if save_tles_to_disk :
                    fo = open(f"{out_dir+'/'}{g}TLE.txt","w")
                    print ( raw , file=fo )
                    fo.close()
                tles.extend(tles_group)
            except Exception as e:
                print(f"  failed to fetch {g} from CelesTrak: {e}; continuing")

    if len(tles) == 0 :
        from .iotools import load_local_tles
        # local file
        print("No TLEs downloaded from CelesTrak; attempting to load local TLE file:", local_tle_file)
        try:
            tles = load_local_tles(local_tle_file)
            if len(tles) == 0:
                raise RuntimeError("No TLEs available: CelesTrak failed and local file not found/empty.")
        except Exception as e:
            print(f"  failed to obtain tle data from {local_tle_file} : {e}; continuing")
    # Trim to N_target
    if N_target is not None and len(tles) > N_target:
        if do_random_sampling :
            import random
            indices = random.sample( range(len(tles)) , N_target )
            tles = [ tles[ idx ] for idx in indices ]
        else :
            tles = tles[:N_target]
    print("Total TLEs to be used:", len(tles))

    # 2) propagate to epoch
    epoch = datetime.datetime.utcnow()
    print("Propagating TLEs to epoch (UTC):", epoch.isoformat())
    from .propagate import propagate_tles_to_epoch
    names, pos_teme_km, vel_teme_km_s, satrecs = propagate_tles_to_epoch(tles, epoch)
    print("  propagated:", pos_teme_km.shape[0], "satellites")

    # 3) TEME -> ECEF (km)
    from .convert import teme_to_ecef_km, ecef_to_geodetic_wgs84_km
    print("Converting TEME -> ECEF (km) (astropy fallback if available)")
    pos_ecef_km = teme_to_ecef_km(pos_teme_km, epoch)

    # 4) geodetic sub-satellite points (lat/lon/alt)
    lat_s_rad, lon_s_rad, alt_s_km = ecef_to_geodetic_wgs84_km(pos_ecef_km)

    # 5) Build ground grid
    print("Building ground grid (lat/lon)...")
    lat_vals = np.linspace(-60*RAD, 60*RAD, grid_nlat)
    lon_vals = np.linspace(-180*RAD, 180*RAD, grid_nlon)
    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing='ij')
    ground_lat_flat = lat2d.ravel()
    ground_lon_flat = lon2d.ravel()
    G = ground_lat_flat.size
    print(f"  ground grid: {grid_nlat} x {grid_nlon} = {G} points")

    # 6) decide on GPU usage
    use_gpu = use_gpu_if_available and CUPY_AVAILABLE
    if use_gpu_if_available and not CUPY_AVAILABLE:
        print("CuPy requested but not available; running on CPU (NumPy).")
    if use_gpu:
        print("CuPy detected and will be used for parts of computation (GPU).")

    # 7) aggregate beams to ground
    from .beam import aggregate_beams_to_ground
    print("Aggregating beams to ground (this can be slow for large N; tune chunks)...")
    t0 = time.time()
    total_counts, pref_counts, cofreq_map, power_dbw, Nvis = aggregate_beams_to_ground(
        sat_ecef_km=pos_ecef_km,
        sat_vel_eci_km_s=vel_teme_km_s,
        sat_names=names,
        ground_lat_rad=ground_lat_flat,
        ground_lon_rad=ground_lon_flat,
        model=model,
        n_beams_per_sat=n_beams_per_sat,
        beam_half_angle_deg=beam_half_angle_deg,
        beam_pattern=beam_pattern,
        beam_max_tilt_deg=beam_max_tilt_deg,
        beam_gain_model=beam_gain_model,
        gain_threshold=gain_threshold,
        frequency_band=frequency_band,
        preferred_bands=preferred_bands,
        chunk_sat=chunk_sat,
        chunk_ground=chunk_ground,
        use_gpu=use_gpu,
        compute_power_map=compute_power_map
    )
    t1 = time.time()
    print(f"Aggregation complete in {t1-t0:.1f} s")

    # reshape to 2D for plotting
    total_grid = total_counts.reshape(lat2d.shape)
    pref_grid = pref_counts.reshape(lat2d.shape)
    combined_cofreq_flat = np.zeros_like(total_counts)
    for f, arr in cofreq_map.items():
        combined_cofreq_flat += arr
    combined_cofreq_grid = combined_cofreq_flat.reshape(lat2d.shape)

    # save outputs
    out_total_png = os.path.join(out_dir, "total_beams_heatmap.png")
    out_pref_png = os.path.join(out_dir, "preferred_beams_heatmap.png")
    out_cofreq_png = os.path.join(out_dir, "cofreq_heatmap.png")

    print("Saving heatmaps...")
    from .visualise import plot_heatmap
    plot_heatmap(total_grid, lat_vals, lon_vals, out_total_png, title="Total beams")
    plot_heatmap(pref_grid, lat_vals, lon_vals, out_pref_png, title="Preferred-band beams")
    plot_heatmap(combined_cofreq_grid, lat_vals, lon_vals, out_cofreq_png, title="Co-frequency beams")

    # Save CSVs and grids
    from .convert import save_flat_csv
    out_nvis_csv = os.path.join(out_dir, "nvis_beams.csv")
    out_total_csv = os.path.join(out_dir, "total_beams.csv")
    out_pref_csv = os.path.join(out_dir, "preferred_beams.csv")
    out_cofreq_csv = os.path.join(out_dir, "cofreq_beams.csv")
    save_flat_csv(Nvis, out_nvis_csv, header="nvis_beams")
    save_flat_csv(total_counts, out_total_csv, header="total_beams")
    save_flat_csv(pref_counts, out_pref_csv, header="preferred_beams")
    save_flat_csv(combined_cofreq_flat, out_cofreq_csv, header="cofreq_beams")
    np.save(os.path.join(out_dir, "lat_grid.npy"), lat2d)
    np.save(os.path.join(out_dir, "lon_grid.npy"), lon2d)

    if compute_power_map and (power_dbw is not None):
        out_power_png = os.path.join(out_dir, "received_power_heatmap.png")
        power_grid = power_dbw.reshape(lat2d.shape)
        plot_heatmap(power_grid, lat_vals, lon_vals, out_power_png, title="Received power (dBW)")
        save_flat_csv(power_dbw, os.path.join(out_dir, "received_power.csv"), header="received_power_dBW")

    print("All outputs written to:", out_dir)
    return {
        "total_png"  : out_total_png  ,
        "pref_png"   : out_pref_png   ,
        "cofreq_png" : out_cofreq_png ,
        "total_csv"  : out_total_csv  ,
        "pref_csv"   : out_pref_csv   ,
        "cofreq_csv" : out_cofreq_csv ,
        "nvis_csv"   : out_nvis_csv   
    }

if __name__ == "__main__":
    try:
        out = run_snapshot_simulation(
            out_dir="sim_20251212_dev",
            groups=ALL_CELESTRAK_GROUPS,	# CELESTRAK_GROUPS,
            local_tle_file="tle_local.txt", 	# LOCAL_TLE_FALLBACK,
            N_target=10000,               	# set to 35000 for full-scale runs (ensure resources)
            grid_nlat=120,
            grid_nlon=240,
            model="multibeam",
            n_beams_per_sat=7,
            beam_half_angle_deg=0.8,
            beam_pattern="hex",
            beam_max_tilt_deg=10.0,
            beam_gain_model="gaussian",
            gain_threshold=0.25,
            frequency_band="E-band",
            preferred_bands=PREFERRED_BANDS,
            chunk_sat=256,
            chunk_ground=20000,
            use_gpu_if_available=False,   # set True if you installed cupy
            compute_power_map = True,
            do_random_sampling = True,
        )
        print("Simulation finished. Outputs:", out)
    except Exception as err:
        print("Error during simulation:", err)
        traceback.print_exc()

    import pandas as pd
    tdf = pd.concat( (	pd.read_csv(out['total_csv']),	pd.read_csv(out['pref_csv']),
			pd.read_csv(out['cofreq_csv']),	pd.read_csv(out['nvis_csv'])) )
    print ( tdf .describe() )

    newtonian_simulator ( bAnimated=False,Nsteps=1000,
        tle_file_name	= "/home/rictjo/Downloads/local_tles_smaller.txt" )
    print('Wrote a trajectory file')
    newtonian_simulator ( bAnimated=True,
        tle_file_name	= "/home/rictjo/Downloads/local_tles_smaller.txt" )
