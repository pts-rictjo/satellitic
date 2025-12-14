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

def teme_to_ecef_km(pos_teme_km: np.ndarray, epoch_dt: datetime.datetime):
    desc_ = """ -----------------------
 TEME -> ECEF conversion
 -----------------------
    Convert TEME positions (km) to ECEF positions (km).
    If astropy is available, use its built-in transforms (recommended).
    Otherwise fallback to a GMST rotation (approximate).
    pos_teme_km: (N,3) array
    """
    if pos_teme_km.shape[0] == 0:
        return pos_teme_km.copy()

    if ASTROPY_AVAILABLE:
        try:
            # Use astropy to transform from TEME (if supported) to ITRS (ECEF)
            # Astropy does not have direct 'TEME' in older versions; attempt using TEME via ITRS/other if available.
            # We construct SkyCoord with TEME representation if supported; else fall back.
            # Approach: build CartesianRepresentation in TEME using 'coord' frame if available
            t = atime.Time(epoch_dt, scale='utc')
            # Try direct TEME transform (some astropy versions implement TEME)
            try:
                teme_frame = ac.TEME(obstime=t)
                cart = ac.CartesianRepresentation(pos_teme_km * u.km)
                sc = ac.SkyCoord(cart, frame=teme_frame)
                itrs = sc.transform_to(ac.ITRS(obstime=t))
                x = itrs.x.to(u.km).value
                y = itrs.y.to(u.km).value
                z = itrs.z.to(u.km).value
                return np.column_stack([x,y,z])
            except Exception:
                # Fallback: convert TEME->GCRS->ITRS (if possible) via astropy transformations
                # Build a generic ITRS vector via built-in transformation pipeline
                # Note: This may not be available in all astropy versions; if it fails, fallback to GMST.
                try:
                    from astropy.coordinates import GCRS
                    # Create a GCRS coordinate approximately from TEME by applying frame rotation (not exact)
                    # Use a simple GMST rotation here as last fallback (below)
                    pass
                except Exception:
                    pass
        except Exception as e:
            # If any astropy error, fallback to GMST method
            print("Astropy TEME->ECEF conversion failed, falling back to GMST rotation:", str(e))

    # --- GMST rotation fallback (approximate) ---
    jd = datetime_to_julian_date(epoch_dt)
    gmst = julian_date_to_gmst_rad(jd)
    cosg = math.cos(gmst); sing = math.sin(gmst)
    R = np.array([[cosg, sing, 0.0], [-sing, cosg, 0.0], [0.0, 0.0, 1.0]])
    return (R @ pos_teme_km.T).T

# --- helpers for julian/gmst ---
def datetime_to_julian_date(dt: datetime.datetime) -> float:
    year = dt.year; month = dt.month; day = dt.day
    hour = dt.hour; minute = dt.minute; second = dt.second + dt.microsecond*1e-6
    if month <= 2:
        year -= 1; month += 12
    A = year // 100
    B = 2 - A + A // 4
    day_frac = (hour + minute/60.0 + second/3600.0) / 24.0
    jd = int(365.25*(year + 4716)) + int(30.6001*(month + 1)) + day + B - 1524.5 + day_frac
    return float(jd)

def julian_date_to_gmst_rad(jd: float) -> float:
    T = (jd - 2451545.0) / 36525.0
    gmst_seconds = 67310.54841 + (876600.0*3600.0 + 8640184.812866)*T + 0.093104*T*T - 6.2e-6*T*T*T
    gmst_seconds = gmst_seconds % 86400.0
    return (gmst_seconds / 86400.0) * 2.0 * math.pi

def ecef_to_geodetic_wgs84_km(r_ecef_km: np.ndarray):
    desc_=""" -----------------------
 ECEF <-> geodetic (WGS84)
 -----------------------    
    Vectorized ECEF (km) -> geodetic lat, lon, alt (lat, lon rad; alt in km)
    Iterative Bowring method.
    """
    if r_ecef_km.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    x = r_ecef_km[:,0] * 1000.0
    y = r_ecef_km[:,1] * 1000.0
    z = r_ecef_km[:,2] * 1000.0
    a = WGS84_A_M; e2 = WGS84_E2
    lon = np.arctan2(y, x)
    p = np.sqrt(x*x + y*y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(10):
        sinl = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sinl * sinl)
        alt = p / np.cos(lat) - N
        lat_new = np.arctan2(z, p * (1 - e2 * (N/(N+alt))))
        if np.max(np.abs(lat_new - lat)) < 1e-12:
            lat = lat_new; break
        lat = lat_new
    sinl = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sinl * sinl)
    alt = p / np.cos(lat) - N
    return lat, lon, alt * M2KM

def geodetic_to_ecef_m(lat_rad: np.ndarray, lon_rad: np.ndarray, alt_m: np.ndarray):
    desc_=""" -----------------------
 ECEF/Geodetic helper
 -----------------------"""
    a = WGS84_A_M; e2 = WGS84_E2
    sinl = np.sin(lat_rad); cosl = np.cos(lat_rad)
    N = a / np.sqrt(1 - e2*sinl*sinl)
    x = (N + alt_m) * cosl * np.cos(lon_rad)
    y = (N + alt_m) * cosl * np.sin(lon_rad)
    z = (N * (1 - e2) + alt_m) * sinl
    return np.stack([x,y,z], axis=-1)


def save_flat_csv(flat_arr: np.ndarray, filename: str, header: str = "value"):
    np.savetxt(filename, flat_arr, delimiter=",", header=header, comments='')
