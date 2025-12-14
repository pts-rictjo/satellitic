from .global import *

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
