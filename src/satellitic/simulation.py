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
