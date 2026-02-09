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
import numpy as np

R_EARTH = 6371e3  # meter

def antenna_gain(theta):
    """
    Enkel antennmodell
    """
    G0 = 10**(35/10)      # 35 dBi
    theta_3db = np.deg2rad(2.5)
    return G0 * np.exp(-(theta/theta_3db)**2)

def random_unit_vectors(n):
    """Uniforma riktningar på en sfär"""
    u = np.random.rand(n)
    v = np.random.rand(n)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    return np.column_stack((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ))

def angle_between(a, b):
    """Vinkel mellan två vektorer"""
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))

def fspl_weight(r):
    """Avståndsviktning för EPFD"""
    return (R_EARTH / r)**2

def exclusion_mask(directions, axis, theta_ex):
    """
    True = tillåten att sända
    """
    angles = angle_between(directions, axis)
    return angles > theta_ex

def beam_mask(directions, earth_axis, theta_beam):
    """
    True = jordstationen ligger i strålningskonen
    """
    angles = angle_between(directions, earth_axis)
    return angles < theta_beam


def theta_ex_of_elevation(eps):
    """
    Konservativ modell:
    större exklusion vid låg elevation
    """
    return np.deg2rad(5 + 15*np.exp(-eps/np.deg2rad(15)))


def psi_max_from_elevation(e, R_E, r):
    """
    Maximum geocentric angle psi (rad) for a given elevation e (rad)
    """
    Re_r = R_E / r
    term = np.sqrt(1 - (Re_r*np.cos(e))**2)
    cos_psi = Re_r*np.cos(e)**2 + np.sin(e)*term
    cos_psi = np.clip(cos_psi, -1.0, 1.0)
    return np.arccos(cos_psi)

def draw_epfd(mask_level=-150):
    import matplotlib.pyplot as plt

    epfd_mean,epfd99,elevations = epfd_means_shells(     shells = [
        dict(N=1600, h=550e3, P=5),    # LEO
        dict(N=400,  h=1200e3, P=10),  # högre LEO
    ] )

    plt.figure()
    plt.plot(np.rad2deg(elevations), 10*np.log10(epfd_mean),
         label="Mean EPFD")

    # EPFD-mask
    epfd_mask = mask_level * np.ones_like(epfd_mean)
    plt.plot(np.rad2deg(elevations), epfd_mask,
         "--", label="EPFD-mask")

    plt.xlabel("Elevation angle (degrees)")
    plt.ylabel("EPFD (dBW/m²/Hz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Satellitskal (exempel)
def epfd_means_shells(     shells = [
        dict(N=1600, h=550e3 , P=5),    # LEO
        dict(N=400,  h=1200e3, P=10),   # högre LEO
    ], elevations = np.deg2rad(np.linspace(0, 90, 90)) ):
    print("TODO : Check means at some point.")

    epfd_mean = []
    epfds = []
    for eps in elevations:
        epfd_eps = 0.0
        for s in shells:
            r = R_EARTH + s["h"]
            shell_epfd = epfd_shell(
                N=s["N"],
                r=r,
                P=s["P"],
                G_func=antenna_gain,
                theta_ex=theta_ex_of_elevation(eps),
                theta_beam=np.deg2rad(8)
            )
            #print(eps,shell_epfd)
            epfd_eps += np.abs(shell_epfd)
        epfds.append(shell_epfd)
        epfd_mean.append(epfd_eps)
    epfd_99 = np.percentile(epfds, 99)
    epfd_mean = np.array(epfd_mean)
    return epfd_mean, epfd_99, elevations

def epfd_shell (
    N, r, P, G_func,
    theta_ex ,                      # degrees
    elevation_min_deg=8 ,           # degrees
    theta_beam = 8 ,                # degrees
    R_E = 6371e3,
    elevation_angle = None ,        # optional: radians
    n_quad=64 ,
    epfd_floor = 1e-20,             # linear units floor to avoid log(0)
    smooth_transition_deg = 2       # smooth low-angle transition in degrees
    ):
    theta_beam      = np.deg2rad(theta_beam)
    theta_ex_deg    = theta_ex
    theta_ex        = np.deg2rad(theta_ex)
    
    """
    Compute the mean EPFD from a satellite shell with smooth low-elevation behaviour.

    Parameters
    ----------
    N : int
        Number of satellites in the shell.
    r : float
        Orbital radius (m).
    P : float
        Transmit power spectral density (W/Hz).
    G_func : callable
        Antenna gain function of off-axis angle (radians).
    theta_ex : float
        Exclusion cone angle (radians).
    theta_beam : float
        Beamwidth angle toward Earth (radians).
    elevation_angle : float or None
        Optional elevation angle for smooth low-angle weighting (radians).
    n_quad : int
        Number of Gauss–Legendre points.
    epfd_floor : float
        Linear EPFD floor to avoid negative/zero in dB.
    smooth_transition_deg : float
        Smoothing width for low-elevation transition (degrees).

    Returns
    -------
    EPFD : float
        Linear EPFD (W/m^2/Hz)
    """
    from numpy.polynomial.legendre import leggauss
    # 1. Gauss–Legendre quadrature over theta
    x, w = leggauss(n_quad)
    theta = 0.5*(theta_beam - theta_ex)*x + 0.5*(theta_beam + theta_ex)
    weights = 0.5*(theta_beam - theta_ex)*w

    # 2. Integrand (antenna gain × sin(theta))
    integrand = G_func(theta) * np.sin(theta)
    integral = np.sum(integrand * weights)

    Nco = Nvis_Nco_shell_with_exclusion( N=N, r=r,
                    elevation_min_deg = elevation_min_deg ,
                    theta_ex_deg = theta_ex_deg ,
                    R_E = 6371e3 )['Nco']

    # 3. Compute linear EPFD
    epfd_lin = Nco * P * integral / (4*np.pi*r**2)

    # 4. Optional: smooth low-elevation weighting (numerical smoothing)
    if elevation_angle is not None:
        # Smooth transition function W(e)
        e_deg = np.rad2deg(elevation_angle)
        e_c = smooth_transition_deg  # center of smooth turn-on
        delta_e = smooth_transition_deg / 2
        W = 1.0 / (1 + np.exp(-(e_deg - e_c)/delta_e))
        epfd_lin *= W

    # 5. Ensure non-negative EPFD
    epfd_lin = np.clip(epfd_lin, epfd_floor, None)

    return epfd_lin


def Nvis_Nco_shell_with_exclusion(
    N, r,
    elevation_min_deg ,
    theta_ex_deg ,
    R_E = 6371e3
):
    """
    Mean visible and co-channel satellites for one NGSO shell,
    including GSO exclusion cone statistics.
    """
    # --- Visibility (e >= 0)
    f_vis = (1 - R_E/r) * 0.5
    Nvis  = N * f_vis

    # --- Elevation-limited
    e_min = np.deg2rad(elevation_min_deg)
    psi_e = psi_max_from_elevation(e_min, R_E, r)
    f_elev = (1 - np.cos( psi_e )) / 2

    # --- GSO exclusion. Cone is from satellite to earth
    theta_ex = np.deg2rad( theta_ex_deg )
    f_allowed = (1 + np.cos(theta_ex)) / 2

    # --- Co-channel
    Nco = Nvis * f_elev * f_allowed

    return {'Nvis':Nvis, 'Nco':Nco}


if __name__=='__main__' :
    draw_epfd()
