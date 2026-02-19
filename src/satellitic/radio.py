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
