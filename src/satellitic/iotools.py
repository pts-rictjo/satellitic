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
TYPE_DEBRIS     = celestial_types['Debris']

class TrajectoryManagerLegacy:
    def __init__(self, filename, particle_types, dt_frame):
        import pickle as pwrite
        self.pwrite_    = pwrite
        self.filename_  = filename
        self.f          = open( self.filename_ , "wb" )

        particle_type   = np.asarray(particle_types, dtype=np.uint8)
        self.N          = particle_type.size
        self.dt_frame   = dt_frame
        self.N_steps_written = 0

        # ---- header ----
        self.f.write(b"TRJ1")

        # Placeholder N_steps = 0
        self.f.write(struct.pack("iid", self.N, 0, dt_frame))
        self.f.write(particle_type.tobytes())

    def write_step(self, r_np):
        r32 = np.asarray(r_np, dtype=np.float32, order="C")
        self.f.write(r32.tobytes())
        self.N_steps_written += 1

    def write_cdp( self, run_system ) :
        # Output some celestial dynamics parameters
        # and initial state
        # Needs a read/load function later...
        # state_var = pickle.load(f)
        #
        ofile = open(self.filename_.replace('.trj','.cdp'),'wb')
        self.pwrite_.dump( run_system.phase_state(), ofile )
        if run_system.satellites_object is not None :
            self.pwrite_.dump( run_system.ledger.satellites_objects , file=ofile )
        ofile.close()


    def close(self):
        # Seek back and update N_steps
        self.f.seek(4)  # after magic
        self.f.write(struct.pack("iid",
                             self.N,
                             self.N_steps_written,
                             self.dt_frame))
        self.f.close()
        
    def read_trj(self,traj_file) :
        with open(traj_file, "rb") as f:
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

    def read_trj_memmap( self, filename, N=None ) :
        f = open(filename, "rb")
        magic = f.read(4)
        assert magic == b"TRJ1"

        N_, Nt, dt = struct.unpack("iid", f.read(16))
        if N is None :
            N = N_

        particle_type = np.frombuffer(f.read(N), dtype=np.uint8)

        offset = 4 + 16 + N
        traj = np.memmap(
            filename,
            dtype=np.float32,
            mode="r",
            offset=offset,
            shape=(Nt, N, 3)
        )

        return traj, particle_type, dt, N, Nt


class TrajectoryManager:

    # ---- format constants ----
    MAGIC_V1 = b"TRJ1"   # legacy
    MAGIC_VX = b"TRJX"   # new

    FLAG_POS  = 1 << 0
    FLAG_VEL  = 1 << 1
    FLAG_MASS = 1 << 2
    FLAG_TYPE = 1 << 3
    FLAG_DYN  = 1 << 4

    def __init__(self, filename, particle_types=None, dt_frame=1.0,
                 version=2, flags=None, dynamic=False):

        self.filename_ = filename
        self.f = open(filename, "wb")

        self.version = version
        self.dt_frame = dt_frame
        self.N_steps_written = 0

        if version == 1:
            # ---- legacy mode ----
            self.N = len(particle_types)
            self.flags = self.FLAG_POS

            self.f.write(self.MAGIC_V1)
            self.f.write(struct.pack("iid", self.N, 0, dt_frame))

            pt = np.asarray(particle_types, dtype=np.uint8)
            self.f.write(pt.tobytes())

        else:
            # ---- new format ----
            self.f.write(self.MAGIC_VX)

            if flags is None:
                flags = self.FLAG_POS

            if dynamic:
                flags |= self.FLAG_DYN
                self.N = -1
            else:
                self.N = len(particle_types)

            self.flags = flags

            Nt_placeholder = 0

            self.f.write(struct.pack(
                "iiiid",
                version,
                flags,
                self.N,
                Nt_placeholder,
                dt_frame
            ))

            if not dynamic and particle_types is not None:
                pt = np.asarray(particle_types, dtype=np.uint8)
                self.f.write(pt.tobytes())

    #
    # Description
    #
    def description(self):
       desc_= """

For canonical simulations you can define:
tm = TrajectoryManager(
    "sim.trj",
    particle_types=types,
    dt_frame=1.0,
    version=2,
    flags=(
        TrajectoryManager.FLAG_POS |
        TrajectoryManager.FLAG_VEL |
        TrajectoryManager.FLAG_MASS
    )
)

For grand canonical simulations you can specify:
tm = TrajectoryManager(
    "dyn.trj",
    dt_frame=1.0,
    version=2,
    flags=TrajectoryManager.FLAG_POS | TrajectoryManager.FLAG_TYPE,
    dynamic=True
)

minimal write functional
tm.write_step(r, types=types_step)

       """
    # --------------------------------------------------
    # WRITE
    # --------------------------------------------------
    def write_step(self, r, v=None, m=None, types=None):
        f = self.f

        if self.version == 1:
            r32 = np.asarray(r, dtype=np.float32)
            f.write(r32.tobytes())
            self.N_steps_written += 1
            return

        r = np.asarray(r, dtype=np.float32)

        if self.flags & self.FLAG_DYN:
            N = r.shape[0]
            f.write(struct.pack("i", N))

        f.write(r.tobytes())

        if self.flags & self.FLAG_VEL:
            v = np.asarray(v, dtype=np.float32)
            f.write(v.tobytes())

        if self.flags & self.FLAG_MASS:
            m = np.asarray(m, dtype=np.float32)
            f.write(m.tobytes())

        if self.flags & self.FLAG_TYPE:
            t = np.asarray(types, dtype=np.uint8)
            f.write(t.tobytes())

        self.N_steps_written += 1

    # --------------------------------------------------
    # CLOSE
    # --------------------------------------------------
    def close(self):
        self.f.seek(0)

        if self.version == 1:
            self.f.seek(4)
            self.f.write(struct.pack(
                "iid",
                self.N,
                self.N_steps_written,
                self.dt_frame
            ))
        else:
            self.f.seek(4)
            self.f.write(struct.pack(
                "iiiid",
                self.version,
                self.flags,
                self.N,
                self.N_steps_written,
                self.dt_frame
            ))

        self.f.close()

    # --------------------------------------------------
    # READ
    # --------------------------------------------------
    def read_trj(self, filename):
        with open(filename, "rb") as f:
            magic = f.read(4)

            # ---- legacy ----
            if magic == self.MAGIC_V1:
                N, Nt, dt = struct.unpack("iid", f.read(16))
                pt = np.frombuffer(f.read(N), dtype=np.uint8)
                data = np.frombuffer(f.read(), dtype=np.float32)
                traj = data.reshape(Nt, N, 3)
                return {"r": traj}, pt, dt, N, Nt

            # ---- new ----
            elif magic == self.MAGIC_VX:
                version, flags, N, Nt, dt = struct.unpack("iiiid", f.read(24))
                dynamic = bool(flags & self.FLAG_DYN)
                particle_type = None
                if not dynamic and N > 0:
                    particle_type = np.frombuffer(
                        f.read(N), dtype=np.uint8
                    )
                steps = []

                for _ in range(Nt):
                    if dynamic:
                        N_step = struct.unpack("i", f.read(4))[0]
                    else:
                        N_step = N

                    r = np.frombuffer(
                        f.read(4 * 3 * N_step),
                        dtype=np.float32
                    ).reshape(N_step, 3)

                    step = {"r": r}
                    if flags & self.FLAG_VEL:
                        v = np.frombuffer(
                            f.read(4 * 3 * N_step),
                            dtype=np.float32
                        ).reshape(N_step, 3)
                        step["v"] = v

                    if flags & self.FLAG_MASS:
                        m = np.frombuffer(
                            f.read(4 * N_step),
                            dtype=np.float32
                        )
                        step["m"] = m

                    if flags & self.FLAG_TYPE:
                        t = np.frombuffer(
                            f.read(N_step),
                            dtype=np.uint8
                        )
                        step["type"] = t
                    steps.append(step)

                return steps, particle_type, dt, N, Nt

            else:
                raise ValueError("Unknown trajectory format")

    def read_trj_memmap_v2(self, filename):
        import numpy as np
        import struct

        f = open(filename, "rb")
        magic = f.read(4)

        # ---------------------------
        # LEGACY (TRJ1)
        # ---------------------------
        if magic == self.MAGIC_V1:
            N, Nt, dt = struct.unpack("iid", f.read(16))
            pt = np.frombuffer(f.read(N), dtype=np.uint8)
            offset = 4 + 16 + N

            traj = np.memmap(
                filename,
                dtype=np.float32,
                mode="r",
                offset=offset,
                shape=(Nt, N, 3)
            )
            return {"r": traj}, pt, dt, N, Nt

        # ---------------------------
        # NEW FORMAT (TRJX)
        # ---------------------------
        elif magic == self.MAGIC_VX :

            version, flags, N, Nt, dt = struct.unpack("iiiid", f.read(24))
            if flags & self.FLAG_DYN:
                raise ValueError("Memmap not possible for dynamic trajectories")

            # ---- particle types (optional global) ----
            particle_type = None
            header_size = 4 + 24

            if N > 0:
                particle_type = np.frombuffer(
                    f.read(N), dtype=np.uint8
                )
                header_size += N

            # ---- compute stride ----
            stride = 0

            has_vel  = bool(flags & self.FLAG_VEL)
            has_mass = bool(flags & self.FLAG_MASS)
            has_type = bool(flags & self.FLAG_TYPE)

            offset_r = stride
            stride += 3 * 4 * N

            if has_vel:
                offset_v = stride
                stride += 3 * 4 * N
            else:
                offset_v = None

            if has_mass:
                offset_m = stride
                stride += 4 * N
            else:
                offset_m = None

            if has_type:
                offset_t = stride
                stride += 1 * N
            else:
                offset_t = None

            # ---- memmap raw block ---- 
            raw = np.memmap(
                filename,
                dtype=np.uint8,
                mode="r",
                offset=header_size,
                shape=(Nt, stride)
            )

            # ---- views ----
            def view_field(offset, dtype, shape_per_step):
                if offset is None:
                    return None
                byte_count = np.dtype(dtype).itemsize * np.prod(shape_per_step)
                sub = raw[:, offset:offset + byte_count]
                return sub.view(dtype).reshape((Nt,) + shape_per_step)

            result = {}
            result["r"] = view_field(offset_r, np.float32, (N, 3))
            if has_vel:
                result["v"] = view_field(offset_v, np.float32, (N, 3))
            if has_mass:
                result["m"] = view_field(offset_m, np.float32, (N,))
            if has_type:
                result["type"] = view_field(offset_t, np.uint8, (N,))

            return result, particle_type, dt, N, Nt

        else:
            raise ValueError("Unknown trajectory format")
           



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
