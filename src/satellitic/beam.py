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
# Beam gain models
# -----------------------
def gaussian_beam_gain(theta_rad: np.ndarray, half_angle_deg: float):
    bw = math.radians(half_angle_deg)
    sigma = bw / math.sqrt(2.0 * math.log(2.0))
    return np.exp(-0.5 * (theta_rad / sigma)**2)

def cosn_beam_gain(theta_rad: np.ndarray, half_angle_deg: float):
    bw = math.radians(half_angle_deg)
    denom = math.log(max(1e-12, math.cos(bw)))
    n = math.log(0.5) / denom if denom != 0 else 1.0
    g = np.cos(theta_rad)**n
    return np.clip(g, 0.0, 1.0)

def uniform_beam_gain(theta_rad: np.ndarray, half_angle_deg: float):
    bw = math.radians(half_angle_deg)
    return (theta_rad <= bw).astype(float)

def multi_beam_generator(n_beams: int,
                         beam_half_angle_deg: float,
                         pattern: str = "hex",
                         max_tilt_deg: float = 60.0,
                         frequency_band: str = "E-band",
                         rng: Optional[np.random.Generator] = None):
    desc_=""" -----------------------
 Multi-beam generator
 -----------------------
    Generate beam center directions in satellite body frame (z = nadir),
    beam half-angles and frequencies (Hz).
    """
    if rng is None:
        rng = np.random.default_rng()

    # frequencies in Hz
    if frequency_band == "E-band":
        freqs = np.linspace(71e9, 86e9, n_beams)
    elif frequency_band == "Ku":
        freqs = np.linspace(10.7e9, 14.5e9, n_beams)
    elif frequency_band == "Ka":
        freqs = np.linspace(17.7e9, 30e9, n_beams)
    else:
        freqs = np.linspace(10e9, 86e9, n_beams)

    if pattern == "random":
        tilt = math.radians(max_tilt_deg)
        cos_theta = rng.uniform(math.cos(tilt), 1.0, n_beams)
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = rng.uniform(0.0, 2.0*math.pi, n_beams)
        dirs = np.column_stack([sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta])
    elif pattern == "circular":
        theta = math.radians(max_tilt_deg)
        phi = np.linspace(0.0, 2.0*math.pi, n_beams, endpoint=False)
        dirs = np.column_stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)*np.ones(n_beams)])
    else:
        # hex tiling
        dirs_list = []
        rings = int(np.ceil(np.sqrt(n_beams)))
        count = 0
        for r in range(rings):
            frac = 0.0 if rings <= 1 else (r/(rings-1))
            theta = math.radians(max_tilt_deg) * frac
            n_in_ring = max(1, 6 * max(1, r))
            for k in range(n_in_ring):
                if count >= n_beams:
                    break
                phi = 2.0 * math.pi * (k / n_in_ring)
                dirs_list.append([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])
                count += 1
        dirs = np.array(dirs_list[:n_beams])

    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    half_angles = np.ones(n_beams) * beam_half_angle_deg
    return dirs, half_angles, freqs

# -----------------------
# Dispatcher: generate_beam_pattern
# -----------------------
def generate_beam_pattern(model: str,
                          sat_pos_ecef_m: np.ndarray,
                          sat_vel_eci_km_s: np.ndarray,
                          n_beams: int = 1,
                          beam_half_angle_deg: float = 1.5,
                          pattern: str = "hex",
                          max_tilt_deg: float = 60.0,
                          frequency_band: str = "E-band",
                          rng: Optional[np.random.Generator] = None):
    """
    Return:
      - boresights_ecef_unit (nb,3) unit vectors in ECEF frame
      - half_angles_deg (nb,)
      - freqs_hz (nb,)
    sat_pos_ecef_m: satellite position in ECEF (meters)
    sat_vel_eci_km_s: optional velocity (km/s) used for attitude - but we use ECEF-based nadir frame
    """
    # Build nadir-pointing body->ECEF rotation
    r = sat_pos_ecef_m.astype(float)
    r_hat = r / np.linalg.norm(r)
    z_body = -r_hat
    # choose a cross-track axis - use Earth's Z to define approximate along-track if velocity missing
    north = np.array([0.0, 0.0, 1.0])
    y_body = np.cross(z_body, north)
    ynorm = np.linalg.norm(y_body)
    if ynorm < 1e-10:
        y_body = np.array([0.0, 1.0, 0.0])
    else:
        y_body = y_body / ynorm
    x_body = np.cross(y_body, z_body)
    x_body = x_body / np.linalg.norm(x_body)
    R_b2ecef = np.column_stack([x_body, y_body, z_body])  # 3x3

    if model in ("gaussian", "cosn", "uniform"):
        # single boresight is nadir
        dirs_body = np.array([[0.0, 0.0, 1.0]])
        half_angles = np.array([beam_half_angle_deg])
        freqs = np.array([_default_freq_for_band(frequency_band)])
    elif model == "multibeam":
        dirs_body, half_angles, freqs = multi_beam_generator(n_beams=n_beams,
                                                             beam_half_angle_deg=beam_half_angle_deg,
                                                             pattern=pattern,
                                                             max_tilt_deg=max_tilt_deg,
                                                             frequency_band=frequency_band,
                                                             rng=rng)
    else:
        raise ValueError("Unknown beam model: " + str(model))
    # convert body dirs to ECEF boresight vectors
    boresights_ecef = (dirs_body @ R_b2ecef.T)
    boresights_ecef = boresights_ecef / np.linalg.norm(boresights_ecef, axis=1, keepdims=True)
    return boresights_ecef, half_angles, freqs

def _default_freq_for_band(band_name: str):
    if band_name.lower().startswith("e"):
        return 83e9
    if band_name.lower().startswith("ku"):
        return 12e9
    if band_name.lower().startswith("ka"):
        return 20e9
    return 12e9

def aggregate_beams_to_ground(
    sat_ecef_km: np.ndarray,
    sat_vel_eci_km_s: np.ndarray,
    sat_names: List[str],
    ground_lat_rad: np.ndarray,
    ground_lon_rad: np.ndarray,
    model: str = "multibeam",
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
    use_gpu: bool = False,
    compute_power_map: bool = False,
    eirp_dbw: float = 47.0  # example EIRP in dBW per beam (placeholder)
):
    desc_ = """ -----------------------
 Aggregation: beams -> ground (chunked; CPU & optional GPU support)
 -----------------------    
    Core aggregation routine. Returns:
     - total_beams_per_ground (G,)
     - preferred_beams_per_ground (G,)
     - cofreq_map: dict freq_hz -> (G,) counts
     - (optional) received_power_map (G,) in dBW if compute_power_map True (sum of linear powers)
    Notes:
     - sat_ecef_km: (N,3) satellite ECEF positions in km
     - sat_vel_eci_km_s: (N,3) provided but used only for attitude if needed
     - ground_lat_rad & ground_lon_rad are flattened arrays (G,)
    """
    use_gpu = use_gpu and CUPY_AVAILABLE
    xp = cp if use_gpu else np
    if compute_power_map :
        from .power import link_budget_received_db

    N = sat_ecef_km.shape[0]
    G = ground_lat_rad.size

    # Build ground ECEF (meters) and normals
    from .convert import geodetic_to_ecef_m
    alt0_m = np.zeros_like(ground_lat_rad)
    r_g_m  = geodetic_to_ecef_m(ground_lat_rad, ground_lon_rad, alt0_m)  # (G,3)
    n_g    = r_g_m / np.linalg.norm(r_g_m, axis=1, keepdims=True)

    # G = total number of ground points
    Nvis                  = np.zeros(G, dtype=int)	# needs reworking
    total_counts          = np.zeros(G, dtype=int)	# also uncorrected
    preferred_counts      = np.zeros(G, dtype=int)
    cofreq_map: Dict[float, np.ndarray] = {}
    if compute_power_map:
        # store linear power sums (watts) per ground point
        power_linear = np.zeros(G, dtype=float)
    else:
        power_linear = None

    # helper: is freq in any preferred band
    def is_preferred(freq_hz: float) -> bool:
        for (lo, hi) in preferred_bands.values():
            if freq_hz >= lo and freq_hz <= hi:
                return True
        return False

    # iterate sats in chunks
    for s0 in range(0, N, chunk_sat):
        s1 = min(N, s0 + chunk_sat)
        chunk_idxs = range(s0, s1)

        # pre-generate beams for chunk (list)
        beam_list = []  # entries: (sat_origin_m, boresight_unit_m (3,), half_angle_deg, freq_hz)
        for si in chunk_idxs:
            sat_ecef_m = sat_ecef_km[si] * KM2M
            sat_vel = sat_vel_eci_km_s[si] if sat_vel_eci_km_s is not None else None
            boresights_ecef, half_angles, freqs = generate_beam_pattern(
                model=model,
                sat_pos_ecef_m=sat_ecef_m,
                sat_vel_eci_km_s=sat_vel,
                n_beams=n_beams_per_sat,
                beam_half_angle_deg=beam_half_angle_deg,
                pattern=beam_pattern,
                max_tilt_deg=beam_max_tilt_deg,
                frequency_band=frequency_band
            )
            for b in range(boresights_ecef.shape[0]):
                beam_list.append((sat_ecef_m, boresights_ecef[b], float(half_angles[b]), float(freqs[b])))

        B = len(beam_list)
        if B == 0:
            continue

        # process ground in chunks
        for g0 in range(0, G, chunk_ground):
            g1 = min(G, g0 + chunk_ground)
            r_g_chunk = r_g_m[g0:g1]    # (gchunk, 3) meters
            n_g_chunk = n_g[g0:g1]
            counts_chunk = np.zeros(g1-g0, dtype=int)
            pref_chunk = np.zeros(g1-g0, dtype=int)
            freq_acc: Dict[float, np.ndarray] = {}
            pow_chunk_linear = np.zeros(g1-g0, dtype=float) if compute_power_map else None

            # loop beams (vectorizing across ground points)
            for (sat_origin_m, boresight_u, half_angle_deg, freq_hz) in beam_list:
                #
                v = r_g_chunk - sat_origin_m			# vektor satellit → mark
                dist = np.linalg.norm(v, axis=1)		# v_norm
                dist_safe = np.where(dist <= 0.0, 1e-6, dist)	# avoid divide-by-zero
                v_unit = v / dist_safe[:,None]			# v_unit = v / v_norm[:, None]
                #
                # beam-mask: θ mellan boresight och riktning från sat → mark
                cos_theta = np.einsum('ij,j->i', v_unit, boresight_u)	# boresight_u = satellitens boresight (nadir eller tiltad)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)		# begränsa värdespann
                theta = np.arccos(cos_theta)
                mask_beam = theta <= math.radians(half_angle_deg)
                if not np.any(mask_beam):
                    continue
                #
                # elevation (grader); elevation/horizon check: dot of v with local normal > 0 for visibility
                dotvn = np.einsum('ij,ij->i', v, n_g_chunk)
                min_elev_angle = 0.0				# horisonten
                min_elev_func  = np.sin(min_elev_angle)
                #
                # synlighetsmask
                visible_mask = dotvn >= min_elev_func
                final_mask = mask_beam & visible_mask
                if not np.any(final_mask):
                    continue

                # Nvis: alla synliga beams, oavsett frekvens
                Nvis[g0:g1][visible_mask] += 1

                # evaluate gain depending on model
                if model == "gaussian" or model == "multibeam":
                    gain_vals = gaussian_beam_gain(theta[final_mask], half_angle_deg)
                elif model == "cosn":
                    gain_vals = cosn_beam_gain(theta[final_mask], half_angle_deg)
                elif model == "uniform":
                    gain_vals = uniform_beam_gain(theta[final_mask], half_angle_deg)
                else:
                    gain_vals = gaussian_beam_gain(theta[final_mask], half_angle_deg)

                mask_gain = gain_vals >= gain_threshold
                if not np.any(mask_gain):
                    continue
                idxs = np.nonzero(final_mask)[0][mask_gain]
                counts_chunk[idxs] += 1
                if is_preferred(freq_hz):
                    pref_chunk[idxs] += 1
                #
                # Nco: per frekvensbin co-frequency accumulation
                freq_key = round(freq_hz, 3)				# bin på kHz-nivå, byt till 6 för MHz
                if freq_key not in freq_acc:
                    freq_acc[freq_key] = np.zeros(g1-g0, dtype=int)	# freq_acc[freq_hz] hz <-> key
                freq_acc[freq_key][idxs] += 1				# freq_acc[freq_hz][idxs] += 1

                # optional received power accumulation (very simple FSPL model from sat to ground)
                if compute_power_map:
                    # convert linear gain (0..1) to dBi or relative factor? Here treat gain_vals as normalized main-lobe factor.
                    # compute free-space path loss using mean distance for idxs
                    d_sel = dist[idxs]
                    # convert freq to Hz
                    pr_db = link_budget_received_db(eirp_dbw=eirp_dbw, freq_hz=freq_hz, distance_m=np.mean(d_sel), rx_gain_db=0.0, losses_db=0.0)
                    pr_lin = 10**(pr_db/10.0)  # convert dBW to watts approx (assuming dBW scale)
                    # scale by gain fraction (normalized)
                    # We'll add same pr_lin to each idxs weighted by normalized gain
                    for k, gi in enumerate(idxs):
                        pow_chunk_linear[gi] += pr_lin * gain_vals[mask_gain][k]

            # merge chunk results into global
            total_counts[g0:g1] += counts_chunk
            preferred_counts[g0:g1] += pref_chunk
            if compute_power_map:
                power_linear[g0:g1] += pow_chunk_linear
            for f, arr in freq_acc.items():
                if f not in cofreq_map:
                    cofreq_map[f] = np.zeros(G, dtype=int)
                cofreq_map[f][g0:g1] += arr

    if compute_power_map:
        # convert total linear watts to dBW per ground point (handle zeros)
        with np.errstate(divide='ignore'):
            power_dbw = 10.0 * np.log10(power_linear, where=(power_linear>0.0))
        power_dbw = np.where(power_linear>0.0, power_dbw, -999.0)
    else:
        power_dbw = None

    return total_counts, preferred_counts, cofreq_map, power_dbw, Nvis
