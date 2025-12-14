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
# Propagation: SGP4 to TEME/ECI
# -----------------------
def propagate_tles_to_epoch(tles: List[Tuple[str,str,str]], epoch_dt: datetime.datetime):
    """
    Propagate list of TLEs to epoch. Returns:
      - names: list[str]
      - pos_teme_km: ndarray (N,3)
      - vel_teme_km_s: ndarray (N,3)
      - satrecs: list of Satrec objects
    """
    names = []
    pos_list = []
    vel_list = []
    satrecs = []
    jd, fr = jday(epoch_dt.year, epoch_dt.month, epoch_dt.day,
                  epoch_dt.hour, epoch_dt.minute, epoch_dt.second + epoch_dt.microsecond*1e-6)
    for (name, l1, l2) in tles:
        sat = Satrec.twoline2rv(l1, l2)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            # skip satellites that cannot be propagated at epoch
            continue
        names.append(name)
        pos_list.append(np.array(r, dtype=float))
        vel_list.append(np.array(v, dtype=float))
        satrecs.append(sat)
    if len(pos_list) == 0:
        return names, np.zeros((0,3)), np.zeros((0,3)), satrecs
    return names, np.vstack(pos_list), np.vstack(vel_list), satrecs
