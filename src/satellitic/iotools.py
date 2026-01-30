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
from sgp4.api import Satrec
from sgp4.conveniences import jday_datetime
import numpy as np
from datetime import datetime
import struct
from .constants import celestial_types

TYPE_PLANET     = celestial_types['Planet']
TYPE_STAR       = celestial_types['Star']
TYPE_MOON       = celestial_types['Moon']
TYPE_SATELLITE  = celestial_types['Satellit']
TYPE_OTHER      = celestial_types['Other']

class TrajectoryManager:
    def __init__(self, filename, particle_type, N_steps, dt_frame):
        """
        particle_type : array-like, shape (N,), dtype uint8
        """
        self.f = open(filename, "wb")

        particle_type = np.asarray(particle_type, dtype=np.uint8)
        self.N = particle_type.size
        self.N_steps = N_steps

        # ---- header ----
        self.f.write(b"TRJ1")
        self.f.write(struct.pack("iid", self.N, self.N_steps, dt_frame))
        self.f.write(particle_type.tobytes())

    def write_step(self, r_np):
        """
        r_np : (N,3) float array
        """
        r32 = np.asarray(r_np, dtype=np.float32, order="C")
        self.f.write(r32.tobytes())

    def close(self):
        self.f.close()
        
    def read_trj(self,traj_file):
        with open("trajectory.trj", "rb") as f:
            magic = f.read(4)
            assert magic == b"TRJ1"

            N, Nt, dt_frame = struct.unpack("iid", f.read(8))

            particle_type = np.frombuffer(
                f.read(N), dtype=np.uint8
            )
            data = np.frombuffer(f.read(), dtype=np.float32)
            traj = data.reshape(Nt, N, 3)
        
        print("""position of particle i at timestep t in : traj[t, i]
planets = traj[:, particle_type == TYPE_PLANET]
sats    = traj[:, particle_type == TYPE_SATELLITE]""" )
        return traj, particle_type, dt_frame ,N ,Nt

def read_tles(filename):
    sats = []
    with open(filename) as f:
        lines = f.readlines()
    i=0
    while i + 2 < len(lines):
        name = lines[i].strip()
        l1   = lines[i+1].strip()
        l2   = lines[i+2].strip()
        sats.append((name, Satrec.twoline2rv(l1, l2)))
        i+=3

    return sats

def tle_to_state(name,sat, epoch):
    jd, fr = jday_datetime(epoch)
    e, r, v = sat.sgp4(jd, fr)

    if e == 0:
        r = np.array(r) * 1000.0   # km → m
        v = np.array(v) * 1000.0   # km/s → m/s
        return r, v
    elif e != 0:
        print(f"Skipping {name}: SGP4 error {e}")
        return None,None


def tles_to_states(sats, epoch):
    r_list = []
    v_list = []
    names  = []

    Nskipped = 0
    for name, sat in sats:
        r, v = tle_to_state( name, sat, epoch)
        if r is None or v is None :
            Nskipped+=1
            continue
        r_list.append(r)
        v_list.append(v)
        names.append(name)
    print('Skipped',Nskipped,'due to epoch error (degraded orbit information)')

    return (
        np.stack(r_list),
        np.stack(v_list),
        names
    )


def fetch_tle_group_celestrak(group: str, timeout: int = 30) -> str:
    url = f"https://celestrak.com/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

def parse_tle_text(raw: str) -> List[Tuple[str,str,str]]:
    desc_ = """
    Parse TLE text (3-line blocks: name, line1, line2).
    Returns list of (name, line1, line2).
    Robust to extra blank lines.
    """
    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip() != ""]
    tles = []
    i = 0
    while i + 2 < len(lines):
        name = lines[i].strip()
        l1 = lines[i+1].strip()
        l2 = lines[i+2].strip()
        if l1.startswith("1 ") and l2.startswith("2 "):
            tles.append((name, l1, l2))
            i += 3
        else:
            i += 1
    return tles

def load_local_tles(filepath: str) -> List[Tuple[str,str,str]]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()
    return parse_tle_text(data)

def download_tle_data (
    out_dir: str = "tle_downloads",
    groups: List[str] = ALL_CELESTRAK_GROUPS,
):
    os.makedirs(out_dir, exist_ok=True)
    # gather TLEs from CelesTrak primary
    tles  = []
    names = []

    for g in groups:
        try:
            print(f"Fetching TLEs for group '{g}' from CelesTrak...")
            raw = fetch_tle_group_celestrak(g)
            tles_group = parse_tle_text(raw)
            names.append( f"{out_dir+'/'}{g}TLE.txt" )
            print(f"  parsed {len(tles_group)} TLEs from {g}")
            fo = open(f"{out_dir+'/'}{g}TLE.txt","w")
            print ( raw , file=fo )
            fo.close()
            tles.extend(tles_group)
        except Exception as e:
            print(f"  failed to fetch {g} from CelesTrak: {e}; continuing")

def gather_tle_data (
    out_dir: str = "tle_downloads",
    groups: List[str] = ALL_CELESTRAK_GROUPS,
    local_tle_file: str = "tle_local.txt",
):
    os.makedirs(out_dir, exist_ok=True)
    # gather TLEs from CelesTrak primary
    tles  = []
    names = []

    for g in groups:
        try:
            print(f"Fetching TLEs for group '{g}' from CelesTrak...")
            raw = fetch_tle_group_celestrak(g)
            tles_group = parse_tle_text(raw)
            names.append( f"{out_dir+'/'}{g}TLE.txt" )
            print(f"  parsed {len(tles_group)} TLEs from {g}")
            fo = open(f"{out_dir+'/'}{g}TLE.txt","w")
            print ( raw , file=fo )
            fo.close()
            tles.extend(tles_group)
        except Exception as e:
            print(f"  failed to fetch {g} from CelesTrak: {e}; continuing")
    fo = open(local_tle_file,"w")
    fo .close()
    fo = open(local_tle_file,"a")
    for name in names :
        print ( name , ':', os.path.getsize(name) )
        with open(name,"r") as input :
            try:
                for line in input :
                    if len(line.replace(" ","").replace("\n","")) > 0 :
                        print ( line.replace( "\n" , "" ) , file=fo )
            except Exception as err:
                continue
    fo.close()

def collate_tle_data(
    out_dir: str        = "tle_downloads",
    groups: List[str]   = ALL_CELESTRAK_GROUPS,
    local_tle_file: str = "tle_local.txt",
):
    os.makedirs(out_dir, exist_ok=True)
    # gather TLEs from CelesTrak primary
    tles  = []
    names = []
    fo = open(local_tle_file,"w")
    fo .close()
    fo = open(local_tle_file,"a")
    for g in groups :
        name = f"{out_dir+'/'}{g}TLE.txt"
        print ( name , ':', os.path.getsize(name) )
        with open(name,"r") as input :
            try:
                for line in input :
                    if len(line.replace(" ","").replace("\n","")) > 0 :
                        print ( line.replace( "\n" , "" ) , file=fo )
            except Exception as err:
                continue
    fo.close()

if __name__ == "__main__":
    if True :
        download_tle_data()
        collate_tle_data()
