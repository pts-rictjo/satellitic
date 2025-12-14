from .global import *
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

# -----------------------
# Multi-beam generator
# -----------------------
def multi_beam_generator(n_beams: int,
                         beam_half_angle_deg: float,
                         pattern: str = "hex",
                         max_tilt_deg: float = 60.0,
                         frequency_band: str = "E-band",
                         rng: Optional[np.random.Generator] = None):
    """
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
