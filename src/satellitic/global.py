import math, time, os, sys, traceback
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt

# SGP4
from sgp4.api import Satrec, jday

# Optional dependencies: astropy (for precise TEME->ECEF), cupy (GPU)
ASTROPY_AVAILABLE = False
try:
    import astropy.time as atime
    import astropy.coordinates as ac
    import astropy.units as u
    ASTROPY_AVAILABLE = True
except Exception:
    ASTROPY_AVAILABLE = False

CUPY_AVAILABLE = False
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

# -----------------------
# User-configurable parameters (tune these)
# -----------------------
# https://celestrak.org/NORAD/elements/
#
ALL_CELESTRAK_GROUPS = ["Intelsat", "SES", "Eutelsat", "Telesat", "Starlink", "OneWeb", "Qianfan", "Hulianwang Digui", "Kuiper", "Iridium NEXT", "Orbcomm", "Globalstar", "Amateur Radio", "SatNOGS" ]
#
# TLE sources & local fallback
CELESTRAK_GROUPS = ["starlink", "oneweb","Globalstar","Intelsat"]
LOCAL_TLE_FALLBACK = "local_tles.txt"   # file path for fallback if celestrak fails

# Simulation grid and scale
DEFAULT_N_TARGET  = 35000     # target number of TLEs to consider (set lower for testing)
DEFAULT_GRID_NLAT = 180       # lat resolution
DEFAULT_GRID_NLON = 360       # lon resolution

# Beam / antenna settings
DEFAULT_N_BEAMS_PER_SAT = 7
DEFAULT_BEAM_HALF_ANGLE_DEG = 0.8
DEFAULT_BEAM_PATTERN = "hex"   # hex / circular / random
DEFAULT_BEAM_MODEL = "gaussian"  # gaussian / cosn / uniform / multibeam
DEFAULT_BEAM_MAX_TILT_DEG = 10.0

# Gain & frequency settings
DEFAULT_GAIN_THRESHOLD = 0.25
PREFERRED_BANDS = {
    "E-uplink": (71e9, 76e9),
    "E-downlink": (81e9, 86e9),
    "Ku": (10.7e9, 14.5e9),
    "Ka": (17.7e9, 30e9)
}
DEFAULT_FREQUENCY_BAND = "E-band"  # used by multi-beam generator to assign frequencies

# Chunking and GPU
DEFAULT_CHUNK_SAT = 256
DEFAULT_CHUNK_GROUND = 20000
USE_GPU_IF_AVAILABLE = False  # change to True to auto-enable GPU when cupy is present

# Earth constants
WGS84_A_M = 6378137.0
WGS84_B_M = 6356752.314245
WGS84_E2 = 1 - (WGS84_B_M**2 / WGS84_A_M**2)
R_EARTH_KM = 6378.137
KM2M = 1000.0
M2KM = 1.0 / 1000.0
RAD = math.pi / 180.0
