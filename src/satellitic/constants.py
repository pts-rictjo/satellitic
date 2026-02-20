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

celestial_types = {
'Star'      : 0 ,
'Planet'    : 1 ,
'Moon'      : 2 ,
'Satellit'  : 3 ,
'Other'     : 4 }

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

def constants ( sel = None, constants_ = constants_solar_system  ) :
    if sel is None :
        return ( constants_ )
    else :
        return ( constants_[sel] )

AU      = constants('AU')
d2s     = 24*60*60 # days to secs

class Celestial( object ) :
    def __init__( self ) :
        self.atmosphere_            = None
        self.mass_                  = None
        self.radius_                = None
        self.surface_temperature_   = None
        self.composition_           = None
        self.name_                  = None
#
# TOPOLOGY INFORMATION
class Starsystem ( object ) :
    def __init__( self , constants ) :
        self.name_              = None
        self.components_        = None
        self.celestial_bodies_  = None
        self.celestial_types_   = None
        self.celestial_names_   = None
        self.indices_           = dict()
        self.constants_         = constants
        self.Ncurrent_          = None
        self.t_                 = None
        self.s_                 = None
        #
        # Phase space
        self.r_ = None
        self.v_ = None
        self.m_ = None

    def planets( self ) :
        if not self.celestial_types_ is None :
            plnt_idx = np.where( self.celestial_types_ == 'Planet' )
            return self.celestial_names_[plnt_idx],plnt_idx

    def make_arrays( self , xp = np ):
        if self.r_ is None or self.v_ is None or self.m_ is None :
            print('Warning: Could not retrieve phase space')
        else :
            self.r_ = xp.asarray(self.r_)
            self.v_ = xp.asarray(self.v_)
            self.m_ = xp.asarray(self.m_)
            self.celestial_names_ = xp.asarray(self.celestial_names_)

    def assign_from_dict( self , systemdictionary , xp=np ) :
        i = 0
        bInitVelocities = False
        vs_ = None
        systemitems = list(systemdictionary.items())
        for item in systemitems :
            if len(item[1]) != 5 :
                print( "Error in starsystem data format" )
                print( "       Assumes : Distance, Velocity, Mass, Radius, Type" )
                exit(1)

            if self.celestial_names_ is None :
                self.celestial_names_ = xp.asarray( [ item[0] ] ) 
            else :
                self.celestial_names_ = xp.hstack( [self.celestial_names_ , item[0]] )
            self.indices_[item[0]] = i
            i = i+1

            bChecked = len(item[1][0])==3 if isinstance(item[1][0], (str, list, tuple) ) else False
            if self.r_ is None :
                if bChecked :
                    rs_ = item[1][0]
                else :
                    rs_ = [item[1][0],0,0]
                self.r_ = xp.asarray( rs_ )
            else :
                if bChecked :
                    rs_ = item[1][0]
                else :
                    rs_ = [item[1][0],0,0]
                self.r_ = xp.vstack([self.r_, rs_ ])  

            if self.s_ is None :
                self.s_ = xp.asarray([ item[1][3] ])
            else :
                self.s_ = xp.hstack( [self.s_, item[1][3]] )

            if not item[1][1] is None :
                bChecked = len(item[1][1])==3 if isinstance(item[1][1], (str, list, tuple) ) else False
                if bChecked :
                    vn_ = xp.asarray( item[1][1] )
                else :
                    vn_ = xp.asarray( [0,item[1][1],0] )
            else :
                if bInitVelocities == False :
                    print('A velocity value was missing will set all based on guess')
                bInitVelocities = True

            if not bInitVelocities :
                if vs_ is None :
                    vs_ = xp.asarray( vn_ )
                else :
                    vs_ = xp.vstack([ vs_ , vn_ ]) 

            if self.m_ is None :
                self.m_ = xp.asarray([ item[1][2] ])
            else :
                self.m_ = xp.hstack( [self.m_, item[1][2]] )

            if self.celestial_types_ is None :
                self.celestial_types_ = [ item[1][4] ]
            else :
                self.celestial_types_ .append( item[1][4] )
            self.Ncurrent_ = len(self.m_)
        
        if bInitVelocities :
            print('Intial velocities not supplied, setting with orbit guess')
            self.guess_initial_velocities( systemitems ,xp=xp )
        else :
            self.v_ = vs_
        self.make_arrays(xp)

    def guess_initial_velocities(self, items, xp=np):
        N = len(items)
        v = np.zeros((N, 3))

        # Identify Sun
        idx_sun = None
        Nstars = 0
        for i, obi in enumerate(items):
            if obi[1][4] == celestial_types['Star']:
                idx_sun = i
                Nstars+=1
        if Nstars != 1 :
            print(f'Guess not taking {Nstars} into account (assumes 1)')
        M_sun = items[idx_sun][1][2]

        # First pass: planets orbit Sun
        for i, obi in enumerate(items):
            r = obi[1][0]
            m = obi[1][2]
            typ = obi[1][4]
            if typ == celestial_types['Star']:
                continue
            if typ == celestial_types['Planet']:
                v_mag = np.sqrt(G * M_sun / r)
                v[i,1] = v_mag

        # Second pass: Moons orbiting Planets
        for i, obi in enumerate(items):
            typ = obi[1][4]
            if typ != celestial_types['Moon']:
                 continue

            # find Planet with Moon
            for j, obj in enumerate(items):
                if obj[1][4] == celestial_types['Planet'] and obj[0] in obi[0]:
                    idx_planet = j
                    break

            M_planet = items[idx_planet][1][2]
            r_planet = items[idx_planet][1][0]
            r_moon = obi[1][0]
            d = r_moon - r_planet
            v_rel = np.sqrt(G * M_planet / d)

            # Moon velocity = Earth's velocity + relative orbital velocity
            v[i,1] = v[idx_planet,1] + v_rel

        # Barycentric correction (important)
        total_momentum = np.sum(
            [items[i][1][2] * v[i] for i in range(N)],
            axis=0
        )
        total_mass = sum(items[i][1][2] for i in range(N))
        v -= total_momentum / total_mass

        self.v_ = xp.asarray(v)

    def create_name_index(self,name):
        if self.celestial_names_ is None :
            print('error: must specify celestial_names_')
            return
        idx_ = np.where(self.celestial_names_==name)[0]
        if not len(idx_) > 0 :
            print('Error:', name ,'not in names' , idx_ );
            exit(1)
        if self.indices_ is None :
            self.indices_ = dict()                 
            self.indices_[name] = idx_ 
        elif not name in self.indices_ :
            self.indices_[name] = idx_
        return idx_

    def phase_space(self, name=None):
        if self.r_ is None or self.v_ is None or self.m_ is None :
            print('Warning: Could not retrieve phase space')
            return None
        elif name is None :
            return self.r_, self.v_, self.m_
        else :
            idx_ = None
            if not self.indices_ is None :
                if name in self.indices_ :
                    idx_ = self.indices_[name]
            if idx_ is None :
                idx_ = self.create_name_index( name )
            return self.r_[idx_], self.v_[idx_], self.m_[idx_]

    def apply_barycentric_motion_correction(self) :
        P = np.sum(self.m_[:,None] * self.v_, axis=0)
        self.v_ -= P / np.sum(self.m_)

    def phase_state(self):
        return self.r_, self.v_, self.m_, self.celestial_types_, self.celestial_names_

    def add_particles(self , r_new, v_new, m_new, types_new , names_new ):
        self.r_ = np.vstack( [self.r_ , r_new] )
        self.v_ = np.vstack( [self.v_ , v_new] )
        self.m_ = np.hstack( [self.m_ , m_new] )
        self.celestial_types_ = np.hstack( [self.celestial_types_, types_new] )
        self.celestial_names_ = np.hstack( [self.celestial_names_, names_new] )
        if self.indices_ is None :
            self.indices_ = dict()
        for i,name in zip( range(self.Ncurrent_ , len(m_new) + self.Ncurrent_ ),
                            names_new ):
            self.indices_[name] = i
        self.Ncurrent_ = len( self.m_ )
        
    def celestial_types(self):
        return self.celestial_types_

    def constants( self , sel = None ) :
        if sel is None :
            return ( self.constants_ )
        else :
            return ( self.constants_[sel] )

    def find_indices_of( self, partial_name ):
        return ( np.where( np.strings.find( self.celestial_names_ , partial_name ) == 0 )[0] )


class TLESatellites ( object ) :
    def __init__( self , tle_file_name=None , planet=None ,xp=np , tag='LEO' ) :
        self.components_    = None
        self.tle_file_name_ = tle_file_name
        self.planet_        = planet
        self.r_ = None
        self.v_ = None
        self.m_ = None
        self.tag_ = tag
        self.names_ = None
        self.types_ = None
        self.assign()
        self.generate_nametypes()
        self.make_arrays(xp)

    def satellite_masses(self, N, default=1000.0) :
        return np.full(N, default) #.reshape(-1,1)

    def generate_nametypes(self, head=None, xp=np ):
        if head is None :
            head = self.tag_
        self.names_ = []
        self.types_ = []
        for i in range(len(self.m_)):
            self.names_.append( head+str(i) )
            self.types_.append( celestial_types['Satellit'] )
        self.types_ = xp.asarray(self.types_)
        self.names_ = xp.asarray(self.names_) #.reshape(-1,1)

    def assign(self , date = [2025,1,1] ):
        if not self.tle_file_name_ is None :
            from datetime import datetime
            from tle_io import read_tles, tles_to_states
            #
            epoch = datetime( *date )
            sats				= read_tles( self.tle_file_name_ )
            r_sat, v_sat, names	= tles_to_states(sats, epoch)
            m_sat   = self.satellite_masses(len(r_sat))
            self.r_ = r_sat
            self.v_ = v_sat
            self.m_ = m_sat
        else:
            print ( 'error: you need to specify .set_tle(tle_file_name)')

    def add_planet_dependency(self, name, r_planet, v_planet ):
        self.r_ += r_planet
        self.v_ += v_planet
        if self.planet_ is None :
            self.planet_ = name
        elif not self.planet_ == name :
            print ( 'error: planet missmatch', name, self.planet_ )
        
    def set_tle(self,tle_file_name):
        self.tle_file_name_ = tle_file_name

    def make_arrays( self , xp = np ):
        if self.r_ is None or self.v_ is None or self.m_ is None :
            print('Warning: Could not retrieve phase space')
        else :
            self.r_ = xp.asarray(self.r_)
            self.v_ = xp.asarray(self.v_)
            self.m_ = xp.asarray(self.m_)

    def phase_space(self):
        if self.r_ is None or self.v_ is None or self.m_ is None :
            print('Warning: Could not retrieve phase space')
            return None
        else :
            return self.r_, self.v_, self.m_

    def phase_state(self):
        return self.r_, self.v_, self.m_, self.types_, self.names_

    def block_indices(self,Ncurrent):
            Nsat			= len(self.m_)
            idx_leo			= np.array(range( Ncurrent, Ncurrent+Nsat ))
            return idx_leo
AU = universal_constants['AU']
d2s = 24*60*60 # days to secs
solarsystem_notes = """https://www.jpl.nasa.gov/_edu/pdfs/scaless_reference.pdf"""
solarsystem_legacy = { # Distance, Radius, revolution time, mass, type
'Sun'      : [0        , 1391400e3*0.5, 0 , 1.989*10**30 , celestial_types['Star'] ] ,
'Mercury'  : [0.39*AU  , 4879e3*0.5   , 87.97 *d2s , 3.285e23 , celestial_types['Planet'] ] ,
'Venus'    : [0.72*AU  , 12104e3*0.5  , 224.7 *d2s , 4.867e24 , celestial_types['Planet'] ] ,
'Earth'    : [1*AU     , 12756e3*0.5  , 365.25*d2s , 5.972e24 , celestial_types['Planet'] ] ,
'EarthMoon': [AU + 384400e3, 3474e3*0.5, 29.5*d2s, 7.346*10**22 , celestial_types['Moon'] ] ,
'Mars'     : [1.52*AU  , 6792e3*0.5   , 687.00*d2s , 6.390e23 , celestial_types['Planet'] ] ,
'Jupiter'  : [5.20*AU  , 142984e3*0.5 , 4333.0*d2s , 1.898e27 , celestial_types['Planet'] ] ,
'Saturn'   : [9.54*AU  , 120536e3*0.5 , 10756 *d2s , 5.683e26 , celestial_types['Planet'] ] ,
'Uranus'   : [19.2*AU  , 51118e3*0.5  , 30687 *d2s , 8.681e25 , celestial_types['Planet'] ] ,
'Neptune'  : [30.06*AU , 49528e3*0.5  , 60190 *d2s , 1.024e26 , celestial_types['Planet'] ] }

solarsystem = { 
# Distance, Velocity, Mass, Radius, Type
'Sun'      : [0           , None, 1.989e30, 1391400e3*0.5 , celestial_types['Star']   ] ,
'Mercury'  : [0.39*AU     , None, 3.285e23,    4879e3*0.5 , celestial_types['Planet'] ] ,
'Venus'    : [0.72*AU     , None, 4.867e24,   12104e3*0.5 , celestial_types['Planet'] ] ,
'Earth'    : [1.0 *AU     , None, 5.972e24,   12756e3*0.5 , celestial_types['Planet'] ] ,
'EarthMoon': [149982269700, None, 7.346e22,   3474e3*0.5  , celestial_types['Moon']   ] ,
'Mars'     : [1.52*AU     , None, 6.390e23,   6792e3*0.5  , celestial_types['Planet'] ] ,
'Jupiter'  : [5.20*AU     , None, 1.898e27, 142984e3*0.5  , celestial_types['Planet'] ] ,
'Saturn'   : [9.54*AU     , None, 5.683e26, 120536e3*0.5  , celestial_types['Planet'] ] ,
'Uranus'   : [19.2*AU     , None, 8.681e25,  51118e3*0.5  , celestial_types['Planet'] ] ,
'Neptune'  : [30.06*AU    , None, 1.024e26,  49528e3*0.5  , celestial_types['Planet'] ] }
#
# AU and AU per DAY
rsolar = np.array([ [0,0,0]  ,
            [-0.25033210,-0.187321750,0],
            [ 0.01747780,-0.662421103,0],
            [-0.90919162, 0.35929260 ,0],
            [ 1.20301883, 0.72707130 ,0],
            [ 3.73307699, 3.05242482 ,0],
            [ 6.16443306, 6.36677540 ,0],
            [14.57964662,-12.36891079,0],
            [16.95491140,-22.88713989,0] ])

vsolar = np.array([ [0,0,0],
            [-0.024388  ,-0.01850225,0],
            [0.02008547 , 0.00083655,0],
            [-0.00708584,-0.01455634,0],
            [-0.007198  ,-0.015228  ,0],
            [-0.00712445, 0.01166307,0],
            [-0.00508654, 0.00549364,0],
            [-0.00442682, 0.00339406,0],
            [0.00264751 , 0.00248746,0],
            [0.00256865 , 0.00168183,0] ])

def build_run_system( solarsystem   ,
        constants_solar_system      ,
        satellite_topology = None ):
    solsystem = Starsystem( constants_solar_system )
    solsystem .assign_from_dict( solarsystem )
    if not satellite_topology is None :
        for item in satellite_topology.items() :
            satellites = TLESatellites( tle_file_name = item[1] ,
                                planet = item[0] )
            satellites.add_planet_dependency( item[0] , *solsystem.phase_space(name = item[0])[:2] )
            solsystem.add_particles( *satellites.phase_state() )
    return solsystem

if __name__=='__main__' :
    print ( 'HERE' )
    solsystemet = Starsystem( constants_solar_system )
    solsystemet .assign_from_dict( solarsystem )
    print ( solsystemet.phase_space() )
    print ( solsystemet.celestial_types() )
    earth_satellites = TLESatellites( tle_file_name = "/home/rictjo/Downloads/local_tles_smaller.txt" ,
                                planet = 'Earth' )
    print ( *solsystemet.phase_space(name = 'Earth')[:2] )
    earth_satellites.add_planet_dependency( 'Earth' , *solsystemet.phase_space(name = 'Earth')[:2] )
    print ( earth_satellites.phase_space() )
    print ( earth_satellites.phase_state() )
    solsystemet.add_particles( *earth_satellites.phase_state() )
    print( solsystemet.phase_state() )
    what='LEO'
    print( what, solsystemet.find_indices_of( what ) )
