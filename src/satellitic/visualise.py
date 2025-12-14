from .global import *
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
