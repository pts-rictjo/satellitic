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
"""
Generate scalable Walker-style satellite constellations and export TLEs
from pandas DataFrame definitions.
"""
from .init import *

import numpy as np
import pandas as pd

bUseSRS = False
try :
        import pyodbc
        bUseSRS = True
        print("ImportSuccess:", "HAS pyodbc IN ENVIRONMENT")
except ImportError :
        print ( "ImportError:","pyodbc: WILL NOT USE IT")
except OSError:
        print ( "OSError:","pyodbc: WILL NOT USE IT")

from datetime import datetime

MU = MU_EARTH_GRAV
R_EARTH = R_EARTH_KM

from .constants import cept_systems, systems_5Cs142dE_20241108, systems_5Cs142dE, recommended_system_names
# ---------------------------------------------------------------------
# Orbital utilities
# ---------------------------------------------------------------------

def mean_motion_rev_per_day(alt_km: float) -> float:
    """
    Convert circular-orbit altitude to mean motion (rev/day).
    """
    a = R_EARTH + alt_km
    n_rad_s = np.sqrt(MU / a**3)
    return n_rad_s * 86400.0 / (2.0 * np.pi)

# ---------------------------------------------------------------------
# Generate r,v directly from list entries
# ---------------------------------------------------------------------
def rot1(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ])

def rot3(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ])

def generate_constellation_state( system   ,
                                 R_PLANET  = 6371000.0 ,
                                 MU_PLANET = 3.986004418e14 ):
    """
    system: list of tuples
    (height_km, n_planes, sats_per_plane, inclination_deg, raan0_deg)

    returns:
      positions : (N,3)
      velocities: (N,3)
    """

    positions = []
    velocities = []

    for (h_km, n_planes, n_sat, inc_deg, raan0_deg) in system:

        r = R_PLANET + h_km * 1e3
        v = np.sqrt(MU_PLANET / r)

        inc = np.deg2rad(inc_deg)
        raan0 = np.deg2rad(raan0_deg)

        for p in range(n_planes):

            raan = raan0 + 2*np.pi * p / n_planes
            R = rot3(raan) @ rot1(inc)

            for s in range(n_sat):

                f = 2*np.pi * s / n_sat

                # orbital plane
                r_pqw = np.array([
                    r*np.cos(f),
                    r*np.sin(f),
                    0.0
                ])

                v_pqw = np.array([
                    -v*np.sin(f),
                     v*np.cos(f),
                     0.0
                ])

                r_eci = R @ r_pqw
                v_eci = R @ v_pqw

                positions.append(r_eci)
                velocities.append(v_eci)

    return np.array(positions), np.array(velocities)


# ---------------------------------------------------------------------
# TLE tooling
# ---------------------------------------------------------------------
def tle_checksum(line: str) -> int:
    cksum = 0
    for c in line:
        if c.isdigit():
            cksum += int(c)
        elif c == '-':
            cksum += 1
    return cksum % 10

def format_tle(
    satnum,
    epoch,
    inc,
    raan,
    ecc,
    argp,
    M,
    n,
    bstar=0.0,
    revnum=1
):
    year = epoch.year % 100
    doy = (epoch - datetime(epoch.year, 1, 1)).total_seconds() / 86400 + 1
    epoch_str = f"{year:02d}{doy:012.8f}"

    ecc_str = f"{ecc:.7f}".split('.')[1]

    bstar_str = f"{bstar:.5e}".replace('e', '').replace('+', '')

    l1 = (
        f"1 {satnum:05d}U 00000A   {epoch_str}  "
        f".00000000  00000-0 {bstar_str:>8} 0  999"
    )
    l1 += str(tle_checksum(l1))

    l2 = (
        f"2 {satnum:05d} "
        f"{inc:8.4f} {raan:8.4f} {ecc_str:7s} "
        f"{argp:8.4f} {M:8.4f} {n:11.8f}{revnum:5d}"
    )
    l2 += str(tle_checksum(l2))

    return l1, l2


def generate_constellation_tles_legacy003(
    df: pd.DataFrame,
    satnum_start: int = 10000,
    eccentricity: float = 1e-4,
    argp_deg: float = 0.0,
    bstar: float = 0.0
):

    MU = 398600.4418
    R_E = 6378.137

    tles = []
    satnum = satnum_start
    epoch = datetime.utcnow()

    epoch_day = (
        (epoch - datetime(epoch.year, 1, 1)).total_seconds() / 86400.0
    ) + 1

    for _, row in df.iterrows():

        n_planes = int(row.n_planes)
        sats_per_plane = int(row.sats_per_plane)

        # --- Orbital mechanics
        a = R_E + row.height_km
        n_rad_s = np.sqrt(MU / a**3)
        mean_motion = n_rad_s * 86400.0 / (2*np.pi)

        # GEO override
        if row.height_km > 35000:
            mean_motion = 1.0027

        # --- Inferred parameters
        F = int(np.round(n_planes / 3)) if n_planes > 1 else 0
        raan_span = 360.0 if n_planes > 1 else 0.0

        for p in range(n_planes):

            raan_deg = (
                row.raan0_deg +
                p * raan_span / n_planes
            ) % 360.0

            for s in range(sats_per_plane):

                mean_anomaly_deg = (
                    s * 360.0 / sats_per_plane +
                    p * F * 360.0 / (n_planes * sats_per_plane)
                ) % 360.0

                # --- realism noise
                ecc = np.random.uniform(1e-5, 5e-4)
                mean_anomaly_deg += np.random.uniform(-0.1, 0.1)
                raan_deg += np.random.uniform(-0.05, 0.05)

                tle1, tle2 = format_tle(
                    satnum=satnum,
                    epoch=epoch,
                    inc=row.inclination_deg,
                    raan=raan_deg,
                    ecc=ecc,
                    argp=0.0,
                    M=mean_anomaly_deg,
                    n=mean_motion,
                    bstar=bstar
                )

                tles.append({
                    "system": row.system,
                    "satnum": satnum,
                    "plane": p,
                    "slot": s,
                    "tle1": tle1,
                    "tle2": tle2
                })
                satnum += 1
    return pd.DataFrame(tles)


def generate_constellation_tles_legacy001(
    df: pd.DataFrame,
    satnum_start: int = 10000,
    eccentricity: float = 1e-4,
    argp_deg: float = 0.0,
    bstar: float = 0.0
) -> pd.DataFrame:
    """
    Generate TLEs for multiple constellation systems defined in a DataFrame.

    Required DataFrame columns:
        system
        height_km
        n_planes
        sats_per_plane
        inclination_deg
        raan0_deg

    Returns:
        DataFrame with one row per satellite:
            system, satnum, plane, slot, tle1, tle2
    """

    tles = []
    satnum = satnum_start
    epoch = datetime.utcnow()

    # Epoch in TLE fractional day-of-year format
    epoch_day = (
        (epoch - datetime(epoch.year, 1, 1)).days + 1
        + (epoch.hour + epoch.minute / 60 + epoch.second / 3600) / 24
    )

    for _, row in df.iterrows():
        n_planes = int(row.n_planes)
        sats_per_plane = int(row.sats_per_plane)

        mean_motion = mean_motion_rev_per_day(row.height_km)
        mean_motion_rad_min = mean_motion * 2.0 * np.pi / 1440.0

        for p in range(n_planes):
            raan_deg = (row.raan0_deg + p * 360.0 / n_planes) % 360.0

            for s in range(sats_per_plane):
                mean_anomaly_deg = (s * 360.0 / sats_per_plane) % 360.0

                tle1, tle2 = format_tle(
                                satnum=satnum,
                                epoch=epoch,
                                inc=row.inclination_deg,
                                raan=raan_deg,
                                ecc=eccentricity,
                                argp=argp_deg,
                                M=mean_anomaly_deg,
                                n=mean_motion,
                                bstar=bstar
                )

                tles.append({
                    "system": row.system,
                    "satnum": satnum,
                    "plane": p,
                    "slot": s,
                    "tle1": tle1,
                    "tle2": tle2
                })

                satnum += 1

    return pd.DataFrame(tles)

#
## SGP4 juggling related
def wrap360(x):
    return x % 360.0

def safe_eccentricity(height_km):
    # tighter than before → avoids SGP4 instability
    if height_km > 35000:   # GEO
        return 1e-6
    elif height_km > 10000: # MEO
        return np.random.uniform(1e-6, 5e-5)
    else:                   # LEO
        return np.random.uniform(1e-6, 1e-4)


def mean_motion_from_altitude( height_km,
				MU = 398600.4418, R_E = 6378.137 ):
    a = R_E + height_km
    n_rad_s = np.sqrt(MU / a**3)
    n_rev_day = n_rad_s * 86400.0 / (2*np.pi)
    # hard safety clamp (SGP4 stability)
    return float(np.clip(n_rev_day, 0.1, 18.0))

def generate_constellation_tles_legacy002(
    df: pd.DataFrame,
    satnum_start: int = 10000,
    eccentricity: float = 1e-4,
    argp_deg: float = 0.0,
    bstar: float = 0.0,
    max_retries: int = 3,
):
    from sgp4.api import Satrec

    tles = []
    satnum = satnum_start
    epoch = datetime.utcnow()

    for _, row in df.iterrows():

        n_planes = int(row.n_planes)
        sats_per_plane = int(row.sats_per_plane)

        # --- orbital parameters
        mean_motion = mean_motion_from_altitude(row.height_km)

        # GEO override (strict)
        is_geo = row.height_km > 35000
        if is_geo:
            mean_motion = 1.0027

        # inferred Walker phasing
        F = int(np.round(n_planes / 3)) if n_planes > 1 else 0
        raan_span = 360.0 if n_planes > 1 else 0.0

        for p in range(n_planes):

            base_raan = (
                row.raan0_deg +
                p * raan_span / n_planes
            )

            # small plane-level perturbation ONLY
            raan_plane = wrap360(base_raan + np.random.uniform(-0.02, 0.02))

            for s in range(sats_per_plane):

                success = False

                for _ in range(max_retries):

                    # --- Mean anomaly with Walker phasing
                    M = (
                        s * 360.0 / sats_per_plane +
                        p * F * 360.0 / (n_planes * sats_per_plane)
                    )

                    M += np.random.uniform(-0.05, 0.05)
                    M = wrap360(M)

                    # --- safe eccentricity
                    ecc = safe_eccentricity(row.height_km)

                    inc = float(np.clip(row.inclination_deg, 0.01, 179.99))
                    argp = 0.0 if is_geo else 0.0

                    try:
                        tle1, tle2 = format_tle(
                            satnum=satnum,
                            epoch=epoch,
                            inc=inc,
                            raan=raan_plane,
                            ecc=ecc,
                            argp=argp,
                            M=M,
                            n=mean_motion,
                            bstar=bstar
                        )

                        # --- Validate with SGP4 BEFORE accepting
                        sat = Satrec.twoline2rv(tle1, tle2)

                        # quick propagation check (epoch)
                        e, _, _ = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)

                        if e == 0:
                            success = True
                            break

                    except Exception:
                        continue

                if not success:
                    # skip only if truly impossible (very rare now)
                    continue

                tles.append({
                    "system": row.system,
                    "satnum": satnum,
                    "plane": p,
                    "slot": s,
                    "tle1": tle1,
                    "tle2": tle2
                })

                satnum += 1

    return pd.DataFrame(tles)


def repack_input( selection:list[str] , study_systems:list ) -> list :
    data = []
    for sel,mdata in zip(selection,study_systems):
        for row in mdata :
            data.append([sel,*row])
    return ( data )

def build_constellation_df( input:list ) -> pd.DataFrame :
    return ( pd.DataFrame( input ,
      columns = [
        "system" ,
        "height_km" ,
        "n_planes" ,
        "sats_per_plane" ,
        "inclination_deg" ,
        "raan0_deg"
      ]
    ) )
#
def generate_constellation_tles(
    df: pd.DataFrame,
    satnum_start: int = 10000,
    bstar: float = 0.0,
    MU:float = 398600.4418,
    R_E:float = 6378.137,
    daysec:float = 86400.0,
    mean_planet_motion:float = 1.0027,
    far_planet_height:float=35000.,
    bExtraChecked:bool=False
):
    from datetime import datetime, UTC
    import numpy as np

    def wrap360(x):
        return x % 360.0

    def mean_motion_from_altitude(height_km):
        a = R_E + height_km
        n_rad_s = np.sqrt(MU / a**3)
        n_rev_day = n_rad_s * daysec / (2*np.pi)
        #return float(np.clip(n_rev_day, 0.1, 17.9))
        #return float(np.clip(n_rev_day, 0.5, 16.0))
        return float(np.clip(n_rev_day, 1.0, 15.5))

    def safe_eccentricity_height(height_km):
        if height_km > far_planet_height:
            return 1e-6
        elif height_km > far_planet_height / 3.0:
            return np.random.uniform(1e-6, 5e-5)
        else:
            return np.random.uniform(1e-6, 1e-4)

    def safe_eccentricity_from_mean_motion(n):
        n_rad_s = n * 2*np.pi / daysec
        a = (MU / n_rad_s**2)**(1/3)
        min_perigee = R_E + 100.0  # 100 km safety
        max_e = 1.0 - (min_perigee / a)
        max_e = max(1e-6, max_e)
        return np.random.uniform(1e-6, max_e * 0.5)

    def safe_eccentricity():
        return np.random.uniform(0.0, 0.002)

    def format_tle_old(
        satnum, inc, raan, ecc, argp, M, n,
        epoch_str="26001.00000000" ):
        # --- Proper international designator (dummy but valid)
        int_desig = "24001A  "
        ecc_str = f"{int(ecc * 1e7):07d}"

        line1 = (
            f"1 {satnum:05d}U {int_desig}"
            f"{epoch_str} "
            f" .00000000  00000-0  00000-0 0  9990"
        )
        line2 = (
            f"2 {satnum:05d} "
            f"{inc:8.4f} "
            f"{raan:8.4f} "
            f"{ecc_str} "
            f"{argp:8.4f} "
            f"{M:8.4f} "
            f"{n:11.8f}00000"
        )
        return line1, line2

    def format_tle(satnum, inc, raan, ecc, argp, M, n, epoch_str="26001.00000000"):
        # Ensure satellite number is 5 digits
        if satnum > 99999:
            satnum = satnum % 100000   # or satnum - 100000
        satnum_str = f"{satnum:05d}"

        # ----- Line 1 (69 chars) -----
        # Initialize with spaces
        line1 = [' '] * 69
        line1[0] = '1'                     # column 1
        # columns 3-7: satellite number
        for i, ch in enumerate(satnum_str):
            line1[2 + i] = ch
        line1[7] = 'U'                     # column 8
        # columns 10-17: international designator
        intl = "24001A  "
        for i, ch in enumerate(intl):
            line1[9 + i] = ch
        # columns 19-32: epoch (14 chars)
        epoch = epoch_str.zfill(14)        # ensure 14 chars
        for i, ch in enumerate(epoch):
            line1[18 + i] = ch
        # columns 34-43: n_dot (10 chars)
        n_dot = " .00000000"
        for i, ch in enumerate(n_dot):
            line1[33 + i] = ch
        # columns 45-52: n_ddot (8 chars)
        n_ddot = " 00000-0"
        for i, ch in enumerate(n_ddot):
            line1[44 + i] = ch
        # columns 54-61: B* (8 chars)
        bstar = " 00000-0"
        for i, ch in enumerate(bstar):
            line1[53 + i] = ch
        # column 63: ephemeris type
        line1[62] = '0'
        # columns 65-68: element number (4 chars)
        elnum = " 9990"
        for i, ch in enumerate(elnum):
            line1[64 + i] = ch
        # column 69: checksum (compute later)
        line1_str = ''.join(line1[:68])   # first 68 chars
        line1_str += str(checksum(line1_str))
        # Replace the last character (already set) with the checksum
        line1[68] = line1_str[68]

        # ----- Line 2 (69 chars) -----
        line2 = [' '] * 69
        line2[0] = '2'
        # columns 3-7: satellite number
        for i, ch in enumerate(satnum_str):
            line2[2 + i] = ch
        # columns 9-16: inclination (8 chars, right-aligned)
        inc_str = f"{inc:8.4f}"
        for i, ch in enumerate(inc_str):
            line2[8 + i] = ch
        # columns 18-25: RAAN
        raan_str = f"{raan:8.4f}"
        for i, ch in enumerate(raan_str):
            line2[17 + i] = ch
        # columns 27-33: eccentricity (7 digits, no decimal)
        ecc_int = int(ecc * 1e7)
        ecc_str = f"{ecc_int:07d}"
        for i, ch in enumerate(ecc_str):
            line2[26 + i] = ch
        # columns 35-42: arg of perigee
        argp_str = f"{argp:8.4f}"
        for i, ch in enumerate(argp_str):
            line2[34 + i] = ch
        # columns 44-51: mean anomaly
        M_str = f"{M:8.4f}"
        for i, ch in enumerate(M_str):
            line2[43 + i] = ch
        # columns 53-63: mean motion (11 chars)
        n_str = f"{n:11.8f}"
        for i, ch in enumerate(n_str):
            line2[52 + i] = ch
        # columns 65-68: revolution number (4 chars)
        rev_num = "   0"
        for i, ch in enumerate(rev_num):
            line2[64 + i] = ch
        # Checksum
        line2_str = ''.join(line2[:68])
        line2_str += str(checksum(line2_str))
        line2[68] = line2_str[68]
        return ''.join(line1), ''.join(line2)

    def checksum(line_part: str) -> int:
        total = 0
        for ch in line_part:
            if ch.isdigit():
                total += int(ch)
            elif ch == '-':
                total += 1
        return total % 10

    tles = []
    satnum = satnum_start
    epoch = datetime.now(UTC)
    if True:
        epoch_str = "26001.00000000"
    else:
        epoch_str = None

    for _, row in df.iterrows():

        n_planes = int(row.n_planes)
        sats_per_plane = int(row.sats_per_plane)

        mean_motion = mean_motion_from_altitude(row.height_km)

        if bExtraChecked and not (0.1 < mean_motion < 17.9):
            continue

        # GEO override
        if row.height_km > far_planet_height:
            mean_motion = mean_planet_motion

        # Walker-like defaults
        F = int(np.round(n_planes / 3)) if n_planes > 1 else 0
        raan_span = 360.0 if n_planes > 1 else 0.0

        for p in range(n_planes):

            # Stable RAAN per plane (NO accumulation)
            base_raan = row.raan0_deg + p * raan_span / n_planes
            raan_plane = wrap360(base_raan + np.random.uniform(-0.02, 0.02))

            for s in range(sats_per_plane):

                # Mean anomaly with phasing
                M = (
                    s * 360.0 / sats_per_plane +
                    p * F * 360.0 / (n_planes * sats_per_plane)
                )
                M = wrap360(M + np.random.uniform(-0.05, 0.05))

                #ecc = safe_eccentricity(row.height_km)
                #ecc = safe_eccentricity(mean_motion)
                #ecc = safe_eccentricity()
                ecc = 0.0
                argp = 0.0

                inc = float(np.clip(row.inclination_deg, 0.01, 179.99))

                l1, l2 = format_tle(
                  satnum=satnum,
                  inc=inc,
                  raan=raan_plane,
                  ecc=ecc,
                  argp=argp,
                  M=M,
                  n=mean_motion
                )

                tles.append({
                    "system": row.system,
                    "satnum": satnum,
                    "plane": p,
                    "slot": s,
                    "tle1": l1,
                    "tle2": l2
                })
                satnum += 1
    return pd.DataFrame(tles)


def create_tle_from_system_selection( selection , systems_information = systems_5Cs142dE ,
					system_names = None , bVerbose=False ,
					output_file = None ) :
    if system_names is None :
        system_names = { s:s for s in selection }

    study_systems	= [ systems_information[sys]	for sys in selection ]

    constellation_df = build_constellation_df ( repack_input( selection , study_systems ) )

    if bVerbose:
        print ( constellation_df )
        print ( "Generating TLEs..." )

    tle_df = generate_constellation_tles(
        constellation_df ,
        satnum_start = 10000
    )

    print ( f"Generated {len(tle_df)} satellites\n" )

    # Swap in system names
    tle_df['system'] = [  v + '-' + system_names.get(v,'') for v in tle_df.loc[:,'system'].values ]

    if bVerbose :
        print ( tle_df )

    # Optional: write to file
    if output_file is not None :
        with open(output_file, "w") as f:
            for _, row in tle_df.iterrows():
                f.write(row.tle1 + "\n")
                f.write(row.tle2 + "\n")

    return tle_df



if bUseSRS:
    class SRSDatabase:
        def __init__(self, mdb_files):
            self.conns = []
            self.cursors = []

            for f in mdb_files:
                conn = pyodbc.connect(
                    fr"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={f}",
                    autocommit=True
                )
                conn.setdecoding(pyodbc.SQL_CHAR, "latin1")
                conn.setencoding("latin1")
                conn.add_output_converter(pyodbc.SQL_WVARCHAR, self._decode_utf16)

                self.conns.append(conn)
                self.cursors.append(conn.cursor())

        @staticmethod
        def _decode_utf16(raw):
            s = raw.decode("utf-16le", errors="ignore")
            return s.split("\x00")[0]

        def table_exists(self, cursor, table):
            """Robust Access ODBC-compatible test."""
            for t in cursor.tables(tableType="TABLE"):
                if t.table_name.lower() == table.lower():
                    return True
            return False

        def query(self, sql, table_hint=None):
            if table_hint:
                for cursor in self.cursors:
                    if self.table_exists(cursor, table_hint):
                        return pd.read_sql(sql, cursor.connection)
                raise ValueError(f"Table '{table_hint}' not found.")

            # Try sequentially when no hint
            for cursor in self.cursors:
                try:
                    return pd.read_sql(sql, cursor.connection)
                except:
                    pass
            raise RuntimeError("Query failed in all MDB files.")

        def show_columns(self, cursor, table):
            print(f"\n=== Columns in {table} ===")
            try:
                cur_cols = cursor.columns(table=table)
                for c in cur_cols:
                    print(" ", c.column_name)
            except Exception as e:
                print("FAILED:", e)

        def show_table(self,table):
            for cur in self.cursors:
                if self.table_exists(cur, table):
                    self.show_columns(cur, table)
                    break

    def get_active_constellations(db: SRSDatabase, sql=None , table_hint="orbit"):
        if sql is None :
            sql = r"""
SELECT
    ng.sat_name,
    ng.ntc_id,
    o.orbit_set_id,
    o.nbr_sat_pl,
    o.right_asc,
    o.inclin_ang,
    o.apog_km,
    o.perig_km,
    o.perig_arg,
    o.long_asc,
    o.op_ht_km
FROM
    [non_geo] AS ng,
    [orbit] AS o
WHERE
    ng.ntc_id = o.ntc_id
    AND o.nbr_sat_pl > 0
ORDER BY
    o.orbit_set_id,
    ng.sat_name;"""

        return db.query(sql, table_hint=table_hint)

def build_unique_satellite_rows( df , unique_keys = [
        "ntc_id", 
        "orbit_set_id",
        "right_asc",
        "inclin_ang",
        "perig_arg",
        "apog_km",
        "perig_km"
    ]):

    df_unique = df.groupby(unique_keys).first().reset_index()
    df_unique["seq"] = df_unique.groupby("sat_name").cumcount() + 1
    df_unique["sat_name_seq"] = df_unique["sat_name"] + "." + df_unique["seq"].astype(str)
    return ( df_unique )

def tle_checksum(line):
    """
    Compute checksum for a TLE line.
    Sum of all digits + count of '-' characters, modulo 10.
    """
    s = 0
    for c in line:
        if c.isdigit():
            s += int(c)
        elif c == '-':
            s += 1
    return str(s % 10)

def generate_tle_file_from_srs_df( df , filename="output.tle" ):
    """
    Generate a TLE file from the SRS-derived dataframe.
    Each row produces one TLE entry.

    Required cols: 
    sat_name_seq, inclin_ang, right_asc, perig_arg,
    apog_km, perig_km
    """

    # Earth constants
    R_EARTH = 6378.137            # km
    MU = 398600.4418              # km^3/s^2

    def normalize_angle(a):
        """Normalize angles into 0–360 range."""
        return float(a) % 360.0 if not np.isnan(a) else 0.0

    norm = lambda a:normalize_angle(a)

    with open(filename, "w") as f:
        for idx, row in df.iterrows():
            name   = str(row["sat_name_seq"])
            inc    = norm(row["inclin_ang"])
            raan   = norm(row["right_asc"])
            argp   = norm(row["perig_arg"])
            lngasc = norm(row["long_asc"]) if "long_asc" in df.columns else 0.0

            # --- Select best orbital distances ---
            if "apog_dist" in df.columns and row["apog_dist"]==row["apog_dist"]:
                rapo = float(row["apog_dist"])
                rper = float(row["perig_dist"])
            else:
                rapo = float(row["apog_km"]) + R_EARTH
                rper = float(row["perig_km"]) + R_EARTH

            # Semimajor axis
            a = (rapo + rper) / 2.0

            # Eccentricity
            ecc = (rapo - rper) / (rapo + rper)
            ecc = max(0.0, min(ecc, 0.9999999))

            # Mean motion (rev/day)
            n_rad_s = np.sqrt(MU / (a ** 3))
            n_rev_day = n_rad_s * 86400 / (2 * np.pi)

            # Mean anomaly based on SRS geometry
            M = norm(lngasc - (raan + argp))

            # TLE eccentricity formatting (7 digits, no decimal)
            ecc_str = f"{ecc:.7f}".split(".")[1]

            # NORAD-like satellite number (deterministic)
            satnum = (idx % 90000) + 10000

            # Epoch placeholder (year-day format)
            epoch = "24001.00000000"   # 1 Jan 2024 — fully valid TLE epoch

            # --- BUILD Line 1 ---
            line1 = (
                f"1 {satnum:05d}U 24001A   {epoch}  "
                f".00000000  00000-0  00000-0 0  0000"
            )
            line1 = line1 + tle_checksum(line1)

            # --- BUILD Line 2 ---
            line2 = (
                f"2 {satnum:05d} "
                f"{inc:8.4f} "
                f"{raan:8.4f} "
                f"{ecc_str:>7s} "
                f"{argp:8.4f} "
                f"{M:8.4f} "
                f"{n_rev_day:11.8f}00000"
            )
            line2 = line2 + tle_checksum(line2)

            # --- Write TLE block ---
            f.write(f"0 {name}\n")
            f.write(line1 + "\n")
            f.write(line2 + "\n")

    print(f"\nTLE file written: {filename}\n")

def load_excel_files(file1, file2, file3):
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    df3 = pd.read_excel(file3)
    return df1, df2, df3


def merge_data(df1, df2, df3):
    # Slå ihop på gemensamma nycklar
    key = ["orbit_set_id", "orb_id"]

    df = df1.merge(df2, on=key, how="inner", suffixes=("", "_2"))
    df = df.merge(df3, on=key, how="inner", suffixes=("", "_3"))

    return df


def extract_parameters(df):
    if df.empty:
        raise ValueError("Dataframe is empty after merge")

    def find_col(name):
        for c in df.columns:
            if name.lower() in c.lower():
                return c
        raise KeyError(f"Column {name} not found")

    orbit_col		= find_col("orbit_set_id")
    orb_col		= find_col("orb_id")
    height_col		= find_col("op_ht")
    height_exp_col	= find_col("op_ht_exp")
    sats_col		= find_col("nbr_sat_pl")
    incl_col		= find_col("inclin_ang")

    # RAAN optional
    try:
        raan_col = find_col("right_asc")
    except:
        try:
            raan_col = find_col("long_asc")
        except:
            raan_col = None

    results = []

    values	= { orbit_col: 0 }
    df		= df.fillna( value=values )

    group_cols = [
        orbit_col, height_col, incl_col, sats_col
    ]
    if height_exp_col:
        group_cols.append(height_exp_col)

    for _, group in df.groupby( group_cols ) : # orbit_col

        height_km	= group[height_col].iloc[0]

        try :
            height_km = (group[height_col] * (10 ** group[height_exp_col])).iloc[0]
        except :
            height_km = group[height_col].iloc[0]

        n_planes	= group[ orb_col] .nunique()
        sats_per_plane	= group[sats_col] .iloc[ 0 ]
        inclination_deg	= group[incl_col] .iloc[ 0 ] # not used

        raan0_deg = group[raan_col].iloc[0] if raan_col else 0
        if pd.isna(raan0_deg):
            raan0_deg = 0

        results.append([
            int(height_km),
            int(n_planes),
            int(sats_per_plane),
            int(inclination_deg),
            int(raan0_deg)
        ])

    return results

def build_structure(data):
    if not data:
        return []
    cols = list(zip(*data))
    return [ a for a in zip(*cols)]


def create_constellation_dict(file_triplets, labels=None):
    """
    file_triplets: lista av tuples [(f1,f2,f3), ...]
    labels: t.ex ['L','M','N']
    """

    if labels is None:
        labels = [f"set_{i}" for i in range(len(file_triplets))]

    result = {}

    for (files, label) in zip(file_triplets, labels):
        df1, df2, df3 = load_excel_files(*files)
        merged = merge_data(df1, df2, df3)
        params = extract_parameters(merged)
        structured = build_structure(params)

        result[label] = structured

    return result



if __name__ == '__main__' :
    # Example one : To write TLE definitions, using default paramaters and a selection. 
    # Note that the required parameters systems_information and system_names are set to
    # defaults systems_5Cs142dE_20241108 and recommended_system_names but can be any
    # viable dictionaries.
    # Issue the below commands to generate a default tle file:
    selection		= ['A','B','D']
    tle_df = create_tle_from_system_selection( selection , output_file = "constellation_systems-" + '-'.join(selection) + ".tle" )

    # example two : Here the functionallity is detailed in greater depth
    selection = ['A','B','I'] # ( A and M are variations of the same system )
    study_systems = [ systems_5Cs142dE_20241108[sys] for sys in selection ]
    print ( f'Will attempt to study systems {", ".join(selection)} corresponding to {", ".join([recommended_system_names[s] for s in selection])} respectively' )
    print ( "With the following data" , study_systems )
    print( repack_input(selection,study_systems) )

    # Example constellation definitions (Systems A–C)
    example_input = [
        # system, height_km, n_planes, sats_per_plane, inclination_deg, raan0_deg
        ["A", 525, 28, 120, 53.0, 0.0],
        ["B", 610, 36, 36, 42.0, 0.0],
        ["RT", 1200, 36, 40, 88.0, 0.0],
    ]
    constellation_df = build_constellation_df( example_input )
    print(constellation_df)

    constellation_df = build_constellation_df( repack_input( selection , study_systems ) )
    print(constellation_df)

    print("Generating TLEs...")
    tle_df = generate_constellation_tles(
        constellation_df,
        satnum_start=20000
    )

    print(f"Generated {len(tle_df)} satellites\n")

    # Show first few TLEs
    for _, row in tle_df.head(3).iterrows():
        print(row.tle1)
        print(row.tle2)
        print()

    # Optional: write to file
    output_file = "constellation.tle"
    with open(output_file, "w") as f:
        for _, row in tle_df.iterrows():
            f.write(row.tle1 + "\n")
            f.write(row.tle2 + "\n")

    print(f"TLEs written to {output_file}")

    # Example SRS DATABASE

    path_ = "./"

    mdb_files = [
        path_ + "srs3048_part1of4.mdb",
        path_ + "srs3048_part2of4.mdb",
        path_ + "srs3048_part3of4.mdb",
        path_ + "srs3048_part4of4.mdb",
    ]

    db = SRSDatabase(mdb_files)
    db .show_table("geo")
    db .show_table("orbit_set")

    df = get_active_constellations(db)

    print ( df )
    print ( "\nKonstellationer:", df["orbit_set_id"].unique() )

    print ( build_unique_satellite_rows( df ) )


    path = "Data/Satellit/OrbitSystems/Detailed orbit characteristics of systems in Table 1/"

    system_fns = lambda A,path : tuple([ path+A.join(["System "," Orbital Parameters "]) + str(i+1) + '.xlsx' for i in range(3) ])

    which_system = 'L'
    constellations = create_constellation_dict(
        [system_fns(which_system,path)],
        labels=[which_system]
    )

    print( constellations[which_system] )


    from satellitic.constellation import SRSDatabase, get_active_constellations

    path_ = '../../Data/Satellit/SRS/srs3048/'

    mdb_files = [
        path_ + "srs3048_part1of4.mdb",
        path_ + "srs3048_part2of4.mdb",
        path_ + "srs3048_part3of4.mdb",
        path_ + "srs3048_part4of4.mdb",
    ]

    db = SRSDatabase(mdb_files)
    db .show_table("geo")
    db .show_table("orbit_set")

    df = get_active_constellations(db)
    df = build_unique_satellite_rows( df )

    generate_tle_file_from_srs_df( df , filename="srs3048.tle" )

