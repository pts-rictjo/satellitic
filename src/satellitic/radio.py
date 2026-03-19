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
