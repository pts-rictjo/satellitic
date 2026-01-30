import numpy as np

celestial_types = {
'Planet'    : 1 ,
'Star'      : 0 ,
'Moon'      : 3 ,
'Satellit'  : 4 ,
'Other'     : 5 }

universal_constants = { 'G' : 6.674 * 10**(-11)	, # m3/(kg s2)	# gravitational constant
		'c' : 2.9979 * 10**8		, # m/s		# speed of light
		'h' : 6.626 * 10**(-34) 	, # J-s		# plancks constant
		'MH': 1.673 * 10**(-27) 	, # kg		# mass of hydrogen atom
		'Me': 9.109 * 10**(-31) 	, # kg		# mass of an electron
		'Rinf': 1.0974 * 10**7  	, # m−1		# Rydbergs constant
		'sigm': 5.670 * 10**(-8)	, # J/(s·m2 deg4) # Stefan-Boltzmann constant
		'Lmax': 2.898 * 10**(-3)	, # m K		# Wien’s law constant (λmaxT)
		'eV': 1.602 * 10**(-19) 	, # J 		# electron volt (energy)
		'ETNT': 4.2 * 10**9 		, # J			# energy equivalent of 1 ton TNT 
        }

constants_solar_system = { 'G' : 6.674 * 10**(-11)	, # m3/(kg s2)	# gravitational constant
		'c' : 2.9979 * 10**8		, # m/s		# speed of light
		'h' : 6.626 * 10**(-34) 	, # J-s		# plancks constant
		'MH': 1.673 * 10**(-27) 	, # kg		# mass of hydrogen atom
		'Me': 9.109 * 10**(-31) 	, # kg		# mass of an electron
		'Rinf': 1.0974 * 10**7  	, # m−1		# Rydbergs constant
		'sigm': 5.670 * 10**(-8)	, # J/(s·m2 deg4) # Stefan-Boltzmann constant
		'Lmax': 2.898 * 10**(-3)	, # m K		# Wien’s law constant (λmaxT)
		'eV': 1.602 * 10**(-19) 	, # J 		# electron volt (energy)
		'ETNT': 4.2 * 10**9 		, # J			# energy equivalent of 1 ton TNT
		'AU': 1.496 * 10**11 		, # m 		# astronomical unit
		'ly': 9.461 * 10**15		, # m 		# Light-year
		'pc': 3.086 * 10**16		, # m 		# parsec
		'lpc': 3.262 			, # m		# * LIGHTYEARS ( also parsec )
		'y': 3.156 * 10**7		, # s			# sidereal year
		'MEarth': 5.974 * 10**24	, # kg 		# mass of Earth
		'REarth': 6.378 * 10**6	, # m 		# equatorial radius of Earth
		'vEarth': 1.119 * 10**4	, # m/s		# escape velocity of Earth
		'MSun': 1.989 * 10**30	, # kg 		# mass of Sun
		'RSun': 6.960 * 10**8		, # m 		# equatorial radius of Sun
		'LSun': 3.85*10**26		, # W 		# luminosity of Sun
		'S': 1.368 * 10**3 		, # W/m2 		# solar constant (flux of energy received at Earth)
    'DMoon': 384399*1000	, # m		# Distance from Moon <-> Earth
		'RMoon': 3474*1000*0.5	, # m		# Moon radius
		'TMoon': 29.5			, # days		# Moons earth orbit time (synodic month)
    'revMoon':27.3 			, # days		# Moons one complete revolution time
		'MMoon': 7.346 * 10**22 	, # kg		# Moon mass
    'J2' : 1.08262668e-3		, # J2 acceleration for LEO satellites
    'RE' : 6.378137e6		, # meters (equatorial)
    'MU_E' : 3.986004418e14	, # m^3 / s^2	# G*M
    'DLEO' : 1000e3		, # [ m ]
		# Hubble constant (H0) 	 approximately 20 km/s per million light-years, or approximately 70 km/s per megaparsec
	}
