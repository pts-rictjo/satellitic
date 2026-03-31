# Satellitic
A collection of tools for satellite assessments

[![License](https://img.shields.io/github/license/Qiskit/qiskit.svg?)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/satellitic?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/satellitic)

Take note that while this package aims to introduce simulation methods many of the derived statistical properties does not require full simulations. The reason is that satellite movements are well modelled by assuming that they are ergodic. Thus time averages are equal to ensemble averages.

## Install
Install the package using :
```
pip install satellitic
```

## Example
In order to create a similar image as this:
![text](https://raw.githubusercontent.com/pts-rictjo/satellitic/674bec3d24d930ecb37ec6bdce9e4cd7c238a03e/examples/cofreq_heatmap.png)

place the content of this projects data folder in you run root and execute the below code
```
from satellitic.init import ALL_CELESTRAK_GROUPS,PREFERRED_BANDS
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

# Forcing qt5 in VisPy
```
>>> from vispy import app
... print(app.use_app('pyqt5', True))  # force PyQt5
```
# To view an orbital simulation
Note that the streaming 3D vispy visualisation cannot handle huge amounts of satellites. Test with a small tle file and then offload to the trajectory file
```
>>> from satellitic.simulation import newtonian_simulator
ImportSuccess: HAS JAX IN ENVIRONMENT
>>> newtonian_simulator( run_parameters  = { 'dt':5e1,
            'Nsteps':None ,
            'steps_per_frame':100 ,
            'mass_epsilon':None ,
            'mass_rule':None } ,
    satellite_topology  = {'Earth':'data/local_small_tle.txt'} )
```
To write a trajectory file you can specify
```
>>> newtonian_simulator( run_parameters  = { 'dt':5e1,
            'Nsteps':None ,
            'steps_per_frame':100 ,
            'mass_epsilon':None ,
            'mass_rule':None } ,
    satellite_topology  = {'Earth':'data/local_small_tle.txt'} ,
    bAnimated = True , bWriteTrajectory = True,
    trajectory_filename = "trajectory.trj", bVerbose = False )
```

# Creating a TLE file from default system definitions
In order to create TLE definitions for systems, any viable dictionary can be supplied to the below defined function. The dictionaries of the systems as defined in the [ITU defintions](https://www.itu.int/md/R23-WP5C-C-0142/en) are already included as defaults:
```
from satellitic.constellation import create_tle_from_system_selection
if __name__=='__main__':
    selection		= ['A','B','D']
    tle_df = create_tle_from_system_selection( selection , output_file = "constellation_systems-" + '-'.join(selection) + ".tle" )
```
## creating TLE:s from a SRS database
```
    from satellitic.constellation import SRSDatabase, get_active_constellations, build_unique_satellite_rows, generate_tle_file_from_srs_df

    path_ = "Data/Satellit/SRS/srs3048/"

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
```

