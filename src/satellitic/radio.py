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

# -----------------------
# Link-budget helper (placeholder)
# -----------------------
def free_space_path_loss_db(freq_hz: float, distance_m: float):
    """
    FSPL (dB) = 20 log10(4π d / λ)
    """
    c = 299792458.0
    lam = c / freq_hz
    with np.errstate(divide='ignore'):
        fspl = 20.0 * np.log10(4.0 * math.pi * distance_m / lam)
    return fspl

def link_budget_received_db(eirp_dbw: float, freq_hz: float, distance_m: float, rx_gain_db: float = 0.0, losses_db: float = 0.0):
    """
    Very simple link budget: Pr_dBW = EIRP_dBW - FSPL_dB + Gr_dB - losses
    EIRP_dBW desired in dBW.
    """
    fspl = free_space_path_loss_db(freq_hz, distance_m)
    pr = eirp_dbw - fspl + rx_gain_db - losses_db
    return pr

def reciever_noise( T_K , BW_Hz ) :
    kB		= 1.380649E-23				# m^2 kg s^-2 K^-1
    RX_N0	= 10*np.log10( kB * T_K / 1E-3 )	# mW [dBm]
    NRx		= RX_N0 + 10*np.log10(BW_Hz)		
    return ( NRx )


def compute_distance2_block(gs, sat, xp):
    gs2 = xp.sum(gs**2, axis=1, keepdims=True)
    sat2 = xp.sum(sat**2, axis=1)
    dot = gs @ sat.T
    return gs2 + sat2 - 2 * dot

def pfd_kernel_blockwise(
    sat_pos,
    gs_pos,
    eirp_dbw,
    Nco=80,
    xp=np,
    sat_block=2048
):
    Ng = gs_pos.shape[0]
    Nsat = sat_pos.shape[0]

    top_pfd = xp.full((Ng, Nco), -1e9, dtype=xp.float32)

    for i in range(0, Nsat, sat_block):
        sat_chunk = sat_pos[i:i+sat_block]
        eirp_chunk = eirp_dbw[i:i+sat_block]

        d2 = compute_distance2_block(gs_pos, sat_chunk, xp)

        pfd = eirp_chunk[None, :] - 10 * xp.log10(4 * xp.pi * d2)

        combined = xp.concatenate([top_pfd, pfd], axis=1)
        idx = xp.argpartition(combined, -Nco, axis=1)[:, -Nco:]
        top_pfd = xp.take_along_axis(combined, idx, axis=1)

    lin = xp.power(10.0, top_pfd / 10.0)
    return 10 * xp.log10(xp.sum(lin, axis=1))


def run_interference_mc_fast(
    traj,                    # memmap (Nt, Nsat, 3)
    ground_lat,
    ground_lon,
    eirp_ngso,               # perhaps rethink this 
    eirp_gso=None,           # perhaps rethink this 
    fs_noise_dbw=-136,
    Nco=80,
    Nmc=500,
    sample_gs=300,
    elevation_mask=(-5, 5),
    use_gpu=False
):
    xp = np
    if use_gpu:
        import cupy as cp
        xp = cp

    Nt, Nsat, _ = traj.shape

    # --- sample ground points ---
    idx = np.random.choice(len(ground_lat), sample_gs, replace=False)
    gs_lat = ground_lat[idx]
    gs_lon = ground_lon[idx]

    gs_pos = latlon_to_ecef(gs_lat, gs_lon)
    gs_pos = xp.asarray(gs_pos)

    IN_samples = []

    for mc in range(Nmc):

        # --- sample time ---
        t = np.random.randint(0, Nt)
        sat_pos = xp.asarray(traj[t])

        # --- elevation filter ---
        elev = compute_elevation(
            gs_pos.get() if use_gpu else gs_pos,
            sat_pos.get() if use_gpu else sat_pos
        )

        mask = (elev >= elevation_mask[0]) & (elev <= elevation_mask[1])

        # sätt bort osynliga
        eirp_eff = xp.asarray(eirp_ngso)
        eirp_eff = xp.where(mask.any(axis=0), eirp_eff, -1e9)

        # --- NGSO ---
        pfd_ngso = pfd_kernel_blockwise(
            sat_pos, gs_pos, eirp_eff,
            Nco=Nco, xp=xp
        )

        # --- GSO ---
        if eirp_gso is not None:
            pfd_gso = pfd_kernel_blockwise(
                sat_pos, gs_pos, xp.asarray(eirp_gso),
                Nco=min(20, Nco), xp=xp
            )
            pfd_total = combine_pfd(pfd_ngso, pfd_gso)
        else:
            pfd_total = pfd_ngso

        # --- FS antenna ---
        theta = 1.0  # placeholder (kan förbättras senare)
        G_rx = fs_antenna_gain(theta)

        I_dbw = pfd_total + G_rx
        IN = I_dbw - fs_noise_dbw
        IN_samples.append(IN)

    IN_samples = xp.concatenate(IN_samples)

    if use_gpu:
        IN_samples = IN_samples.get()

    return IN_samples

def evaluate_interference( IN , protection_LS=[-10,11] ):
    p_long  = np.mean( IN > protection_LS[0] )
    p_short = np.mean( IN > protection_LS[1] )

    return {
        "p_long": p_long,
        "p_short": p_short,
        "mean_IN": np.mean(IN),
        "p99": np.percentile(IN, 99)
    }


def find_pfd_mask_offset(traj, ground_lat, ground_lon, eirp_base):
    offsets = np.linspace(-20, 20, 25)
    results = []

    for off in offsets:
        IN = run_interference_mc_fast(
            traj,
            ground_lat,
            ground_lon,
            eirp_base + off
        )
        stats = evaluate_interference(IN)
        if stats["p_long"] < 0.2:
            results.append((off, stats))
    return results

# eirp(theta_off_axis) beam pattern todo fixme

def pfd_aggregation_kernel(
    sat_pos_ecef ,        # (Nsat, 3)
    gs_pos_ecef  ,        # (Ngs, 3)
    eirp_dbw ,            # (Nsat,)
    Nco = 80 ,
    xp = np               # np eller cupy
) :
    """
    Placeholder, första iterationen
    Returnerar agg PFD (dBW/m^2) per ground point
    """
    #
    # Senare även slant och atmo etc
    # --- vectorized distance ---
    # (Ngs, Nsat, 3)
    diff = sat_pos_ecef[None, :, :] - gs_pos_ecef[:, None, :]
    d2 = xp.sum(diff**2, axis=2)  # (Ngs, Nsat)
    #
    # --- Förluster
    #
    # --- PFD (dB) ---
    # Todo fixme
    # eirp_eff = eirp_dbw + tx_gain(theta_off_axis)
    pfd_db = eirp_eff[None, :] - 10 * xp.log10(4 * xp.pi * d2)
    #
    # Todo fixme
    mask = elevation > 0
    pfd_db = xp.where(mask, pfd_db, -1e9)
   
    # --- välj top Nco ---
    # partition är snabbare än sort
    idx = xp.argpartition(pfd_db, -Nco, axis=1)[:, -Nco:]

    # samla
    top_pfd = xp.take_along_axis(pfd_db, idx, axis=1)

    # --- summera linjärt ---
    top_lin = xp.power(10.0, top_pfd / 10.0)
    pfd_agg_lin = xp.sum(top_lin, axis=1)

    pfd_agg_db = 10 * xp.log10(pfd_agg_lin)

    return pfd_agg_db


def run_interference_mc(
    traj,                  # memmap: (Nt, Nsat, 3)
    particle_type,
    ground_lat,
    ground_lon,
    eirp_dbw,
    fs_noise_dbw,
    Nco=80,
    Nmc=1000,
    xp=np,
    sample_gs=200,
    bMap=True
):
    """
    Returnerar I/N samples
    """

    Nt, Nsat, _ = traj.shape

    if not bMap :
        # --- välj slumpmässiga mark punkter ---
        idx_gs = np.random.choice(len(ground_lat), sample_gs, replace=False)
    else :
        idx_gs = range(len(ground_lat))

    gs_lat = ground_lat[idx_gs]
    gs_lon = ground_lon[idx_gs]

    # Todo fixme
    gs_pos = latlon_to_ecef(gs_lat, gs_lon)  # (Ngs,3)

    IN_samples = []

    for k in range(Nmc):

        # --- slumpa tid ---
        t = np.random.randint(0, Nt)

        sat_pos = traj[t]  # (Nsat,3)

        # --- kernel ---
        pfd_agg = pfd_aggregation_kernel(
            sat_pos,
            gs_pos,
            eirp_dbw,
            Nco=Nco,
            xp=xp
        )

        # --- FS antenna gain ---
        # Todo fixme x2
        theta = compute_offaxis_angles(gs_pos, sat_pos)  # approx
        G_rx = fs_antenna_gain(theta)

        # approx: ta max eller medel
        G_eff = xp.mean(G_rx, axis=1)

        # --- interferens ---
        I_dbw = pfd_agg + G_eff

        # --- I/N ---
        IN = I_dbw - fs_noise_dbw

        IN_samples.append(IN)

    return xp.concatenate(IN_samples)
