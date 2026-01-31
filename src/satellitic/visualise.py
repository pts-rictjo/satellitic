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
# backend & animation helpers
# -----------------------
def choose_vispy_backend():
    """Try to set a VisPy GUI backend in order of preference."""
    import vispy
    backend_set = False

    try:
        import PyQt5
        vispy.use('pyqt5')
        backend_set = True
    except ImportError:
        try:
            import PySide6
            vispy.use('pyside6')
            backend_set = True
        except ImportError:
            try:
                import glfw
                vispy.use('glfw')
                backend_set = True
            except ImportError:
                backend_set = False

    if not backend_set:
        print(
            "WARNING: No VisPy GUI backend found. "
            "Install pyqt5, pyside6, or glfw to visualize."
        )
    return backend_set

# -----------------------
# plotting & saving helpers
# -----------------------
def plot_heatmap(grid2d: np.ndarray, lat_vals_rad: np.ndarray, lon_vals_rad: np.ndarray, filename: str, title: str = ""):
    plt.figure(figsize=(12,6))
    extent = [math.degrees(lon_vals_rad.min()), math.degrees(lon_vals_rad.max()),
              math.degrees(lat_vals_rad.min()), math.degrees(lat_vals_rad.max())]
    plt.imshow(grid2d, origin='lower', extent=extent, aspect='auto', cmap='inferno')
    plt.colorbar(label='count')
    plt.title(title)
    plt.xlabel('Longitude (deg)'); plt.ylabel('Latitude (deg)')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


