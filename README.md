# Satellitic
A collection of tools for satellite assessments

[![License](https://img.shields.io/github/license/Qiskit/qiskit.svg?)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/satellitic?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/satellitic)

## Install
Install the package using :
```
pip install satellitic
```

## Example
In order to create a similar image as this:
![text](https://raw.githubusercontent.com/pts-rictjo/satellitic/674bec3d24d930ecb37ec6bdce9e4cd7c238a03e/examples/cofreq_heatmap.png)

run the below code
```
import satellitic.simulation as satsim

out = satsim.run_snapshot_simulation(
            out_dir="sim_20251212_dev",
            groups=ALL_CELESTRAK_GROUPS,	# CELESTRAK_GROUPS,
            local_tle_file="tle_local.txt", 	# LOCAL_TLE_FALLBACK,
            N_target=10000,               	# set to 35000 for full-scale runs (ensure resources)
            grid_nlat=120,
            grid_nlon=240,
            model="multibeam",
            n_beams_per_sat=7,
            beam_half_angle_deg=0.8,
            beam_pattern="hex",
            beam_max_tilt_deg=10.0,
            beam_gain_model="gaussian",
            gain_threshold=0.25,
            frequency_band="E-band",
            preferred_bands=PREFERRED_BANDS,
            chunk_sat=256,
            chunk_ground=20000,
            use_gpu_if_available=False,   # set True if you installed cupy
            compute_power_map = False,
            do_random_sampling = True,
        )
print("Simulation finished. Outputs:", out)

import pandas as pd
tdf = pd.concat( ( pd.read_csv(out['total_csv']),	pd.read_csv(out['pref_csv']), pd.read_csv(out['cofreq_csv']),	pd.read_csv(out['nvis_csv'])) )
print ( tdf .describe() )
```
