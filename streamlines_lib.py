# streamlines_lib.py (make sure this file is in the same folder)
import numpy as np
from scipy.optimize import fsolve
import math
from matplotlib.path import Path

from scipy.integrate import solve_ivp
import configparser
#import sys



# ---------- System parameters in CGS ----------

# ---------- CGS constants and conversions ----------
M_sun_g = 1.98847e33      # grams
year_s = 365.25 * 24 * 3600.0  # seconds in a year
m_H = 1.6735575e-24  # g
# Gravitational constant in cgs (cm^3 g^-1 s^-2)
G_cgs = 6.67430e-8  # cm^3 g^-1 s^-2
DAY = 86400.
G=G_cgs

# ---------- Physical system parameters in CGS ----------
# Physical semi-major axis (set here in cm). Change to your desired value.
# Example: 1 AU = 1.495978707e13 cm, so 2e11 m -> 2e13 cm. Use appropriate value.



# --- default system parameters (keep names as-is) ---
M_ob = 20.0 * M_sun_g
M_ns = 1.4 * M_sun_g
R_ns = 1e6
a_cm = 1.0e13 # example ~1.34 AU in cm; modify as needed; user can provide 1) Porb, 2) a_cm
a = a_cm
e = 0.8
inclination_deg = 30
R_star = 0.2 * a  #default, user can set Ropt
Rsun = 7e10
omega_obs = 180 # observer's direction in degrees (0 = along x-axis, 90 = along y-axis, 180 = opposite x-axis)

# 5 geometry parameters: affect profile NH(phase)
# 1) beta - affect  angle of accretion wake; because v  _orb/v_wind  defines this angle
# 2) streamlines_rg_fac - defines width of the accretion wake and level of NH
# 
# 3) wake_factor_density  is important for NH level variation
#
#  4) wake_extension defines the length of accretion wake - affects profile NH(phase) but only if it is very small
#  5) beta affects wind velocity and density of the wind -> NH?

# Wind mass-loss rate (Msun/yr -> g/s) — default conversion already applied
Mdot_wind_msun_per_yr = 1e-10
Mdot_wind_g_s = Mdot_wind_msun_per_yr * M_sun_g / year_s
beta = 0.5

# calculation parameters (defaults)
phase_def = 0.5
wake_factor_density = 3
wake_extension = 0.2 # units of period
streamlines_rg_fac = 3
roch_lobe_limit = 0.7 # 0.3
#inclin_aw = 170.0 # degrees of the accretion wake NOT USED
#width_aw = 70.0 #NOT USED
zeta_def = 0.5 # Davidson Ostriker 1973


along_streamlines = 0
dynamic_width = 1 #flag if wake width depends on the phase
wake_phase_width = 0.2 # used if dynamic_width = 0
streamlines_lim_def = 2.5 * a # used if along_streamlines = 1

map_n_pix = 400
N_points_to_find_intersections_with_orbit = 200
GEOMETRY = 1  # trail is between pos_ns - wake_width .. pos_ns + wake_width
#GEOMETRY = 2  # trail is between pos_ns - wake_width .. pos_ns

# --- Read the configuration file and override defaults if present ---
config = configparser.ConfigParser()
config.read('XRB.ini') # silent if missing

# Helper: attempt to read a float, falling back to current value
def _cfg_getfloat(section, option, current):
    try:
        return config.getfloat(section, option)
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        return current

def _cfg_getint(section, option, current):
    try:
        return config.getint(section, option)
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        return current

def _cfg_getstr(section, option, current):
    try:
        return config.get(section, option)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return current

# General section (only override when provided)
src = _cfg_getstr('General', 'src', globals().get('src', None))

# NOTE: we keep the same variable names and expect the INI to use the same units as the defaults.
# If your INI uses solar masses for Mopt, it must supply values already converted to grams (or you must add conversion).
mob = _cfg_getfloat('General', 'Mopt', M_ob / M_sun_g) # try to allow either absolute grams or Msun

# minimal behavior: if user provides Mopt as a plain number, assume it's in the same units as your default expression:
# your default used 20.0 * M_sun_g, so to be minimally intrusive we accept either:
# - Mopt given as grams (very large number) - then user should set properly in INI
# - Mopt given as solar masses - in that case we multiply by M_sun_g (see below).
# To keep behavior minimal and safe, detect plausible Msun values (<= 1000) and convert:
if mob is not None:
# mob currently holds the numeric value read from INI; decide if it's Msun or grams
    if mob <= 1000: # treat as solar masses if small (heuristic)
        M_ob = mob * M_sun_g
    else:
        M_ob = mob

# inclination (degrees)
inclination_deg = _cfg_getfloat('General', 'inclination_deg', inclination_deg)

omega_obs = _cfg_getfloat('General', 'omega_obs', omega_obs) # observer's direction in degrees (0 = along x-axis, 90 = along y-axis, 180 = opposite x-axis)

# orbital period (keep as-is; user must supply in expected units)
Porb_day = _cfg_getfloat('General', 'Porb_day', globals().get('Porb_day', None))
a_cm = np.power( G*(M_ob + M_ns) *  (Porb_day*DAY / (2*np.pi))**2 ,(1/3))


# eccentricity
e = _cfg_getfloat('General', 'eccentricity', e)


# accretion wake parameters:

wake_phase_width = _cfg_getfloat('Wake', 'wake_phase_width', wake_phase_width)

wake_factor_density = _cfg_getfloat('Wake', 'wake_factor_density', wake_factor_density)

wake_extension = _cfg_getfloat('Wake', 'wake_extension', wake_extension)

phase_def = _cfg_getfloat('Wake', 'phase_def', phase_def)
streamlines_lim_def = _cfg_getfloat('Wake', 'streamlines_lim_def', streamlines_lim_def)
streamlines_rg_fac = _cfg_getfloat('Wake', 'streamlines_rg_fac', streamlines_rg_fac)

roch_lobe_limit = _cfg_getfloat('Wake', 'roch_lobe_limit', roch_lobe_limit)
GEOMETRY = _cfg_getint('Wake', 'wake_geometry_choice', GEOMETRY)

# inclin_aw = _cfg_getfloat('Wake', 'inclin_aw', inclin_aw)
# width_aw = _cfg_getfloat('Wake', 'width_aw', width_aw)


# Mdot: prefer direct g/s if provided; also accept an alternative Msun/yr key if present
Mdot_wind_g_s = _cfg_getfloat('General', 'Mdot_wind_g_s', Mdot_wind_g_s)

# optional alternative key: Mdot_wind_msun_per_yr (human friendly) — convert if present
try:
    md_msunyr = config.get('General', 'Mdot_wind_msun_per_yr')
    if md_msunyr is not None:
        md_val = float(md_msunyr)
        Mdot_wind_msun_per_yr = md_val
        Mdot_wind_g_s = md_val * M_sun_g / year_s
except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
    pass

Mdot_wind = Mdot_wind_g_s
Mdot_wind_msun_per_yr = Mdot_wind_g_s / M_sun_g * year_s
# Keep beta, R_ns, a_cm, and aliases
beta = _cfg_getfloat('General', 'beta', beta)
R_ns = _cfg_getfloat('General', 'R_ns', R_ns)

# semimajor axis: prefer a_cm key if provided; keep alias a
a_cm = _cfg_getfloat('General', 'a_cm', a_cm)
a = a_cm



# Recompute R_star if user provided one, otherwise keep default fraction of a
R_star = _cfg_getfloat('General', 'Ropt', 0.2 * a/Rsun) * Rsun

# Calculation parameters (Calc section optional)



map_n_pix = _cfg_getint('Calc', 'map_n_pix', map_n_pix)

NH_min_plot = _cfg_getfloat('Plot', 'NH_min_plot', None)
NH_max_plot = _cfg_getfloat('Plot', 'NH_max_plot', None)


#  parameters combinations
v_p = np.sqrt(2 * G * M_ob / R_star)
Msum = G * (M_ob + M_ns)
R_star_cm = R_star
R_max_def = 10.*a

# print("Mopt/Msun = ", M_ob/M_sun_g)
# print ("Ropt/Rsun = ",R_star/Rsun)
# print("Ropt/a = ",R_star/a)
# print("Mdot_wind = ", Mdot_wind_g_s / M_sun_g * year_s, "Msun/yr")
# print("beta_wind = ", beta)

v_inf = 2.6 * v_p  #https://ui.adsabs.harvard.edu/abs/1995ApJ...455..269L/abstract stars hotter than 21 KK, class < B1
# take v_inf  if user provided one, otherwise keep default v_inf
v_inf = _cfg_getfloat('General', 'v_inf_km_s', v_inf/1e5) *1e5

print( "v_inf = ", v_inf*1e-5, " km/s")
#print("Mdot_wind/v_inf = ", Mdot_wind_g_s / v_inf, Mdot_wind_g_s / v_inf/a/m_H)


N_steps_from_observer = int(3000.*(R_max_def/1e14))



# Solve Kepler helper
def solve_kepler(M, e, guess=None):
    if guess is None:
        guess = M
    E = fsolve(lambda x: x - e * np.sin(x) - M, guess)[0]
    return E



def orbital_state(phase, e=e, a=a, Msum=Msum, R_star=R_star):

    M_anom = 2. * np.pi * phase
    E_anom = solve_kepler(M_anom, e)
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E_anom / 2))

    r_inst = a * (1 - e**2) / (1 + e * np.cos(nu))

    # vis-viva equation by Newton:
    # v_orb_mag  is the relative speed of the two bodies,
    # r is the distance between the two bodies' centers of mass
    #
    v_orb_mag = np.sqrt(Msum * (2/r_inst - 1/a))

    # Msum = G*(M1+M2)
    # r_inst -  current separation
    # a - sei-major axis
    gamma = np.arctan((e * np.sin(nu)) / (1 + e * np.cos(nu)))
    angle_to_ob = nu + np.pi
    pos_ob = np.array([r_inst * np.cos(angle_to_ob), r_inst * np.sin(angle_to_ob)])


    v_orb_angle = nu + np.pi/2 + gamma
    v_orb_vec = np.array([v_orb_mag * np.cos(v_orb_angle), v_orb_mag * np.sin(v_orb_angle)])

    return {
        'E_anom': E_anom,
        'nu': nu,
        'r_inst': r_inst,
        'gamma': gamma,
        'pos_ob': pos_ob,
        'v_orb_vec': v_orb_vec,
        'phase': phase,
    }


def full_system_state(phase, wake_width=wake_phase_width, **kwargs):
    """
    Computes the orbital physics AND the complex wake geometry.
    """
    # 1. Get the standard orbital math
    state = orbital_state(phase, **kwargs)
    
    # 2. Compute the wake using the state we just created
    # This keeps the ODE integration logic outside the core physics loop
    state['poly_wake'] = get_accretion_wake_poly(state, wake_width)
    
    return state
    
def  v_wind (dist) :
    with np.errstate(invalid='ignore', divide='ignore'):
        # v_w_mag = np.where(dist >= R_star, v_inf * np.sqrt(1 - R_star / dist), 0.0)
        v_w_mag = np.where(dist >= R_star, v_inf * np.power(1 - R_star / dist, beta), 0.0)
    return v_w_mag


def get_past_ns_pos(current_phase, past_phase):
    """
    Calculates the past position of the NS relative to its current
    position (0,0) at phase 'current_phase'.
    """
    curr_st = orbital_state(current_phase)
    past_st = orbital_state(past_phase)

    # Since orbital_state returns pos_ob = (OB - NS),
    # let's define absolute NS position as -pos_ob_global

    dx = curr_st['pos_ob'][0] - past_st['pos_ob'][0]
    dy = curr_st['pos_ob'][1] - past_st['pos_ob'][1]
    #print (current_phase, past_phase,dx,dy,past_st['pos_ob'][0],past_st['pos_ob'][1])
    return dx,dy



def get_past_intersection_line(current_phase,  n_points=300, x_ns=0, y_ns=0):
    """
    Calculates the current positions of wind particles that once intersected
    the NS, starting from an arbitrary current NS position (x_ns, y_ns).
    """
    P_sec = Porb_day * DAY
    lookback_times = np.linspace(0, wake_extension * P_sec, int(n_points))
    
    line_x, line_y = [], []

    # Current state
    curr_st = orbital_state(current_phase)
    # Current OB position in absolute coordinates
    pos_ob_now = np.array([x_ns, y_ns]) + curr_st['pos_ob']

    # ODE: dr/dt = v_wind(r)
    def wind_deriv(t, r):
        return v_wind(r)

    for tau in lookback_times:
        if tau == 0:
            line_x.append(x_ns)
            line_y.append(y_ns)
            #print (current_phase,line_x,line_y)

            continue

        past_phase = (current_phase - (tau / P_sec)) % 1.0
        past_st = orbital_state(past_phase)

        # The vector from OB to NS at the past time:
        # Since past_st['pos_ob'] = OB_past - NS_past,
        # the vector OB -> NS is exactly -past_st['pos_ob']
        r_past_vec_ob_to_ns = -past_st['pos_ob']
        r_start = past_st['r_inst']

        # Integrate radial expansion
        sol = solve_ivp(wind_deriv, [0, tau], [r_start], method='RK45', rtol=1e-4)
        r_final = sol.y[0][-1]

        # Unit vector for radial expansion (direction from OB to the old NS position)
        unit_vector = r_past_vec_ob_to_ns / r_start

        # Current position of that particle relative to the CURRENT OB star
        r_particle_rel_ob = unit_vector * r_final

        # Absolute position = Current OB + relative vector
        abs_pos = pos_ob_now + r_particle_rel_ob

        line_x.append(abs_pos[0])
        line_y.append(abs_pos[1])

    return np.array(line_x), np.array(line_y)

def wake_width_cm (phase) : 
    # TODO can be determined  together with _CURRENT_WAKE
    st = orbital_state(phase)
    pos_ob_cm = np.array(st['pos_ob']) 
    rOB_NS = np.sqrt(pos_ob_cm[0]**2+pos_ob_cm[1]**2) 
    rg = compute_accretion_radius_at_phase(phase) # accretion radius for NS at  NS's location
    # print (streamlines_rg_fac*rg/a, rOB_NS*roch_lobe_limit/a)
    # exit()
    return min (streamlines_rg_fac*rg, rOB_NS*roch_lobe_limit)

def get_accretion_wake (current_phase, phase_width) :

    print("      getting accretion wake")
        
    if (dynamic_width==0):
        previo_phase = current_phase-phase_width/2  
        future_phase = current_phase+phase_width/2


        xp1,yp1= get_past_ns_pos(current_phase, previo_phase)
        xp2,yp2= get_past_ns_pos(current_phase, future_phase)

    global N_points_to_find_intersections_with_orbit
    if dynamic_width:
        


        max_retries = 3
        n_points = N_points_to_find_intersections_with_orbit
        factor = 5
        found = False

        for attempt in range(max_retries):
            pts = find_local_orbit_intersections(
                current_phase,
                r0=wake_width_cm(current_phase),
                n_points=n_points
            )

            if len(pts) == 2:
                xp1, yp1 = pts[0]['x'], pts[0]['y']
                xp2, yp2 = pts[1]['x'], pts[1]['y']
                previo_phase = pts[0]['phase']
                future_phase = pts[1]['phase']
                found = True
                break
            else:
                print(f"Attempt {attempt+1}/{max_retries}: "
                    f"too many or 0 intersections: {len(pts)}, "
                    f"retrying with n_points={n_points * factor}")
                n_points *= factor

        if not found:
            print(f"Failed to find exactly 2 intersections after {max_retries} attempts. "
                f"Last attempt returned {len(pts)} points.")
            # Handle failure: skip this phase, raise an error, or use fallback
            # For example:
            # continue  # if inside an outer loop
            # raise RuntimeError("Could not find 2 orbit intersections")

        else:
            lx0, ly0 = get_past_intersection_line(current_phase, x_ns=0, y_ns=0)
            lx1, ly1 = get_past_intersection_line(previo_phase, x_ns=xp1, y_ns=yp1)
            lx2, ly2 = get_past_intersection_line(future_phase, x_ns=xp2, y_ns=yp2)
            N_points_to_find_intersections_with_orbit = n_points #change gloabl value
    
    if GEOMETRY == 2 :
        return lx1,ly1,lx0,ly0
    if GEOMETRY == 1 :    
        return lx1,ly1,lx2,ly2





def get_accretion_wake_poly(current_state, phase_width, flag=1):
    """
    Calculates the wake using an existing state dictionary.
    """
    current_phase = current_state['phase']

    lx1, ly1, lx2, ly2 = get_accretion_wake (current_phase, phase_width)

    path1 = np.column_stack([lx1, ly1])
    path2 = np.column_stack([lx2, ly2])

    # Close the polygon at the current NS position (0,0)
    #return np.vstack([path1, path2[::-1]])
    if flag==1:
        return np.vstack([[[0, 0]], path1, path2[::-1], [[0,0]]])
        #return np.vstack([[[0,0]], path1, path2[::-1]])
    else :
        return lx1, ly1, lx2, ly2

#Polynome_Wake =  1 #get_accretion_wake_poly(orbital_state(phase_def), wake_phase_width)

_CURRENT_WAKE = None 

def set_active_wake(phase, width=wake_phase_width):
    """Call this once to 'load' the wake into the library memory."""
    global _CURRENT_WAKE
    state = orbital_state(phase)
    _CURRENT_WAKE = get_accretion_wake_poly(state, width)

def get_rel_velocity_field(X, Y, pos_ob, v_orb_vec, R_star=R_star, v_inf=v_inf):
    dx = X - pos_ob[0]
    dy = Y - pos_ob[1]
    dist = np.sqrt(dx**2 + dy**2)
    v_w_mag = v_wind (dist)
    safe_dist = np.where(dist == 0, 1.0, dist)
    v_w_x = v_w_mag * (dx / safe_dist)
    v_w_y = v_w_mag * (dy / safe_dist)
    vx_rel = v_w_x - v_orb_vec[0]
    vy_rel = v_w_y - v_orb_vec[1]
    return vx_rel, vy_rel


def get_rel_velocity_modul(phase, R_star=R_star, v_inf=v_inf):
    # Get orbital state
    st = orbital_state(phase)
    pos_ob = st['pos_ob']
    v_orb_vec = st['v_orb_vec']

    VX_ns, VY_ns = get_rel_velocity_field(np.array([[0.0]]), np.array([[0.0]]), pos_ob, v_orb_vec)
    v_rel_ns = np.sqrt(np.clip(VX_ns[0,0]**2 + VY_ns[0,0]**2, 0.0, None))

    # If the above returns zero due to numerical issues, estimate v_rel as |v_orb| + v_inf (fallback)
    if v_rel_ns <= 0.0:
        v_rel_ns = np.linalg.norm(v_orb_vec) + v_inf
    return v_rel_ns

def compute_accretion_radius_at_phase(phase, r_sample_offset=1e-6):
    # Get orbital state
    st = orbital_state(phase)
    v_orb_vec = st['v_orb_vec']
    v_rel_ns =  get_rel_velocity_modul(phase, R_star=R_star, v_inf=v_inf)

    # If the above returns zero due to numerical issues, estimate v_rel as |v_orb| + v_inf (fallback)
    if v_rel_ns <= 0.0:
        v_rel_ns = np.linalg.norm(v_orb_vec) + v_inf

    # Capture radius Rg = 2 G M_ns / v_rel^2
    Rg = 2.0 * G * M_ns / (v_rel_ns**2)
    return Rg

def compute_Mdot_accreted_at_phase_general(phase, Mdot_wind):
    """
    Compute Mdot_accreted for a given orbital phase using the capture forMsumla.
    - phase: orbital phase in [0,1)
    - Mdot_wind: mass-loss rate of OB star (same units you choose)
    - r_sample_offset: small offset to evaluate wind velocity safely if needed
    Returns: Mdot_accreted (same units as Mdot_wind)
    """
    # Get orbital state
    st = orbital_state(phase)
    pos_ob = st['pos_ob']      # OB position relative to NS (NS at origin)
    v_orb_vec = st['v_orb_vec']  # NS orbital velocity vector (in NS frame)
    r_inst = st['r_inst']
    v_rel_ns =  get_rel_velocity_modul(phase)


    # Capture radius Rg = 2 G M_ns / v_rel^2 (if v_wind is unperturbed)
    Rg = compute_accretion_radius_at_phase(phase, r_sample_offset=1e-6)

    zeta =  zeta_def # Davidson Ostriker 1973
    # Accreted rate: (1/4) * Mdot_wind * (Rg / r_inst)^2 - if v_rel ~ v_wind
    Mdot_acc = 0.25 * Mdot_wind * (Rg / r_inst)**2 * v_rel_ns/v_wind (r_inst) * zeta




    return Mdot_acc, v_rel_ns, Rg


def compute_Mdot_accreted_at_phase(phase, Mdot_wind, flag=1):
    """
    Compute Mdot_accreted for a given orbital phase using the capture forMsumla.
    - phase: orbital phase in [0,1)
    - Mdot_wind: mass-loss rate of OB star (same units you choose)
    - r_sample_offset: small offset to evaluate wind velocity safely if needed
    Returns: Mdot_accreted (same units as Mdot_wind)
    """
    # Get orbital state
    st = orbital_state(phase)
    pos_ob = st['pos_ob']      # OB position relative to NS (NS at origin)
    v_orb_vec = st['v_orb_vec']  # NS orbital velocity vector (in NS frame)
    r_inst = st['r_inst']




    # Capture radius Rg = 2 G M_ns / v_rel^2
    Rg = compute_accretion_radius_at_phase(phase, r_sample_offset=1e-6)

    # Accreted rate: (1/4) * Mdot_wind * (Rg / r_inst)^2 - if v_rel ~ v_wind
    Mdot_acc = 0.25 * Mdot_wind * (Rg / r_inst)**2

    v_rel_ns =  get_rel_velocity_modul(phase)

    if flag==1 :
        return Mdot_acc, v_rel_ns, Rg
    if flag==2:
        return Mdot_acc

def compute_Lx_at_phase(phase, Mdot_wind):
    # see Lx_period.py for further details
    mdot =  float(compute_Mdot_accreted_at_phase(phase, Mdot_wind,flag=2)) # for  'vrel = v_wind'

    return G * M_ns * mdot/R_ns

def rho_wind_at_point_no_shadow(x_cm, y_cm, pos_ob_cm, Mdot_wind_g_s):
     dx = x_cm - pos_ob_cm[0]
     dy = y_cm - pos_ob_cm[1]
     r = math.hypot(dx, dy)
     if r <= R_star_cm:
         return np.inf
     v_r = v_wind(r)
     if v_r <= 0.0:
         return 0.0


     rho = Mdot_wind_g_s / (4.0 * math.pi * r**2 * v_r)

     return rho

def rho_wind_at_point_3d(phase, x_cm, y_cm, z_cm, pos_ob_cm, Mdot_wind_g_s):
    # NS and OBstar is in the plane  z=0
    dx = x_cm - pos_ob_cm[0]
    dy = y_cm - pos_ob_cm[1]
    dz = z_cm 
    r = np.sqrt(dx*dx+dy*dy+dz*dz)  # distance between the point and OB star
    #rOB_plane = np.sqrt(dx*dx+dy*dy) # projected  distance between point and OB star
    if r <= R_star_cm:
        return np.inf
    v_r = v_wind(r)
    if v_r <= 0.0:
        return 0.0
    rho = Mdot_wind_g_s / (4.0 * math.pi * r**2 * v_r)

    streamline_semi_width = wake_width_cm (phase)
    st = orbital_state(phase)
    pos_ob_cm = np.array(st['pos_ob']) 
    rOB_NS = np.sqrt(pos_ob_cm[0]**2+pos_ob_cm[1]**2) 
    # if is_in_conus(phase, x_cm, y_cm) :
    #    return 0.1*rho
    # is_in_custom_conus(phase, x_cm, y_cm) :# is_in_custom_conus (phase, x_cm, y_cm) :
    # if check_between_streamlines (x_cm, y_cm, phase, x1, y1, x2, y2, lim=streamlines_lim_def):
    inside = check_point_fast_streamlines_3d(x_cm, y_cm, z_cm, phase, pos_ob_cm[0], pos_ob_cm[1], streamline_semi_width, rOB_NS, lim=streamlines_lim_def)
    if inside :

        return rho*wake_factor_density
    else :
        return rho

def rho_wind_at_point(phase, x_cm, y_cm, pos_ob_cm, Mdot_wind_g_s):

     dx = x_cm - pos_ob_cm[0]
     dy = y_cm - pos_ob_cm[1]
     #r = math.hypot(dx, dy)
     r = np.sqrt(dx*dx+dy*dy)
     if r <= R_star_cm:
         return np.inf
     v_r = v_wind(r)
     if v_r <= 0.0:
         return 0.0
    #  r_inst = np.sqrt(pos_ob_cm[0]**2+pos_ob_cm[1]**2)
     rho = Mdot_wind_g_s / (4.0 * math.pi * r**2 * v_r)

     streamline_semi_width =  wake_width_cm (phase) 
     x1=-streamline_semi_width
     y1=-streamline_semi_width
     x2=streamline_semi_width
     y2=streamline_semi_width

     # if is_in_conus(phase, x_cm, y_cm) :
     #    return 0.1*rho
     # is_in_custom_conus(phase, x_cm, y_cm) :# is_in_custom_conus (phase, x_cm, y_cm) :
     # if check_between_streamlines (x_cm, y_cm, phase, x1, y1, x2, y2, lim=streamlines_lim_def):
     inside_wake = check_point_fast_streamlines(x_cm, y_cm, phase, x1, y1, x2, y2,  lim=streamlines_lim_def)
     if inside_wake :
         return rho * wake_factor_density
     else :
         return rho
# TODO: mplement a fully vectorized rho_map function (replacing np.vectorize)

def rho_wind_at_NS(phase, Mdot_wind_g_s):
    # Get orbital state
    st = orbital_state(phase)
    r_inst = st['r_inst']

    if r_inst <= R_star_cm:
        return np.inf
    v_r = v_wind(r_inst)
    if v_r <= 0.0:
        return 0.0
    rho = Mdot_wind_g_s / (4.0 * math.pi * r_inst**2 * v_r)
    return rho








def compute_Nh_3D(phase, inclination, observer_phi=0.0, R_max=R_max_def):
    """
    Integrates Nh along a 3D LOS.
    inclination: degrees, 90 is in-plane, 0 is pole-on.
    observer_phi: The angle in the XY plane (similar to your previous observer_dir).
    """
    st = orbital_state(phase)
    pos_ob_cm = st['pos_ob'] # This is [x_ob, y_ob] in the plane

    # Convert inclination to radians
    inc_rad = np.radians(inclination)
    phi_rad = np.radians(observer_phi)

    
    # 1. Observer position in 3D
    obs_x =  np.sin(inc_rad) * np.cos(phi_rad)
    obs_y =  np.sin(inc_rad) * np.sin(phi_rad)
    obs_z =  np.cos(inc_rad)

    
    # 2. Vector FROM observer TO Neutron Star (at 0,0,0)
    # The length of this vector is R_max
    vec_x, vec_y, vec_z = -obs_x, -obs_y, -obs_z

    
    
    n_steps = N_steps_from_observer
    ds = 1.0 / n_steps
    
    nh = 0.0
    flag_stop=0
    for i in range(n_steps):
        # Distance from observer toward NS
        s = (i + 0.5) * ds

        # 3D position along the ray
        # P(s) = Obs + s * (Unit vector toward NS)
        # Since vec is length R_max, unit vector is vec / R_max
        px = R_max *(obs_x + s * vec_x)
        py = R_max *(obs_y + s * vec_y)
        pz = R_max *(obs_z + s * vec_z)
        
        # 3. Distance from current point P to the OB star center
        # OB star is at (pos_ob_cm[0], pos_ob_cm[1], 0)
        dx = px - pos_ob_cm[0]
        dy = py - pos_ob_cm[1]
        dz = pz - 0.0 # OB star is in the orbital plane

        dist_to_ob = np.sqrt(dx**2 + dy**2 + dz**2)
        if (phase>=0.2) and (phase<0.23):
            flag_stop=1
            #print (phase, dist_to_ob/R_star,"<>", dist_to_ob/a, px/a,py/a,pz/a)

        # 4. Check for stellar eclipse
        if dist_to_ob < R_star:
            print (f'\n************* ECLIPSE  at {phase:.2f} *******************\n')
            return np.nan

        # 5. Density calculation
        # Use a 3D density function that accepts the full position
        rho = rho_wind_at_point_3d(phase, px, py, pz, pos_ob_cm, Mdot_wind_g_s)
        # If your function requires (x, y), calculate effective 2D coords relative to OB:
        # eff_x = pos_ob_cm[0] + dx
        # eff_y = pos_ob_cm[1] + dy
        # (The Z-component is folded into the radial distance check)

        n_H = rho / m_H
        nh += n_H * ds * R_max 

    # if flag_stop==1:
    #     exit()
    return nh



def compute_Nh_for_phase(phase,observer_dir=np.array([-1.0, 0.0]),R_max=R_max_def ):
    st = orbital_state(phase)
    #print ("Phase=",phase)
    #set_poly_wake (get_accretion_wake_poly(st, phase_width=wake_phase_width))

    pos_ob_cm = st['pos_ob']

    n_steps = 1000 
    ds = 1. / n_steps

    # Test intersection of LOS segment with stellar disk of OB star
    # intersects, tvals = los_intersects_star(pos_ob_cm, observer_dir=observer_dir, R_max=R_max_def)
    # if intersects:
    #     # If the intersections are such that the star center lies on the observer side between observer and NS,
    #     # the NS is eclipsed — we treat any intersection as eclipse.
    #     return np.nan

    # Otherwise integrate density along LOS from observer to NS
    nh = 0.0
    #print("phase=",phase)
    for i in range(n_steps):

        s = (i + 0.5) * ds  # distance along from observer toward NS
        # Position: P = O + t*L with t=s (since L = -observer_dir and we set ds accordingly)
        x = (observer_dir[0] ) + (-observer_dir[0]) * s
        y = (observer_dir[1] ) + (-observer_dir[1]) * s
        
        # Distance from current point P to the OB star center
        # OB star is at (pos_ob_cm[0], pos_ob_cm[1])
        dx = R_max * x - pos_ob_cm[0]
        dy = R_max * y - pos_ob_cm[1]
        dist_to_ob = np.sqrt(dx**2 + dy**2)
        # Check for stellar eclipse (twice? just in case?)
        if dist_to_ob < R_star:
            return np.nan
        rho = rho_wind_at_point(phase, x * R_max, y * R_max, pos_ob_cm, Mdot_wind_g_s)
        if not np.isfinite(rho):
            # we encountered stellar interior -> treat as eclipse
            return np.nan
        n_H = rho / m_H
        nh += n_H * ds * R_max
    
    return nh


def los_intersects_star(pos_ob_cm, observer_dir=np.array([-1.0, 0.0]), R_max=R_max_def):
    # Vector from observer to star center: w = C - O
    O = observer_dir * R_max  # observer position
    C = pos_ob_cm
    w = C - O
    # Project w onto LOS direction (-observer_dir) because LOS points toward origin
    # Direction vector along LOS from observer to NS is L = -observer_dir (unit)
    L = -observer_dir
    # Compute quadratic for intersection: |O + t*L - C|^2 = R_star^2, with t in [0, R_max]
    # Let D = O - C. Then |D + t L|^2 = R_star^2 -> (L·L) t^2 + 2 D·L t + D·D - R^2 = 0
    D = O - C
    a_q = np.dot(L, L)  # =1
    b_q = 2.0 * np.dot(D, L)
    c_q = np.dot(D, D) - R_star_cm**2
    disc = b_q**2 - 4.0 * a_q * c_q
    if disc < 0.0:
        return False, None  # no real intersection
    sqrt_disc = math.sqrt(disc)
    t1 = (-b_q - sqrt_disc) / (2.0 * a_q)
    t2 = (-b_q + sqrt_disc) / (2.0 * a_q)
    # We have intersections at parameters t1, t2 along LOS measured from observer position toward NS.
    # Valid eclipse occurs if any t in [0, R_max] AND the corresponding point lies between observer and NS (i.e., along segment)
    hits = []
    for t in (t1, t2):
        if 0.0 <= t <= R_max:
            # compute point P = O + t L; check that the point is between observer and NS (it will be if 0<=t<=R_max)
            hits.append(t)
    if len(hits) == 0:
        return False, None
    # There is intersection; since one intersection will be closer to observer and the other farther, the star blocks LOS if
    # at least one intersection is at positive t less than the distance from observer to NS (which is R_max).
    return True, (t1, t2)


def get_streamline_path(phase, x0, y0, lim, t_max=2000000, flag=2):
    """
    Calculates the full coordinates (x, y) of a streamline passing through (x0, y0).
    Returns: Nx2 numpy array of [x, y] coordinates.
    """
    # 1. Get orbital state
    st = orbital_state(phase % 1.0)
    pos_ob = st['pos_ob']
    v_orb_vec = st['v_orb_vec']

    # 2. Define the velocity field ODE
    def v_field(t, p):
        vx, vy = get_rel_velocity_field(p[0], p[1], pos_ob, v_orb_vec)
        return [float(vx), float(vy)]

    # 3. Stop if we hit the OB star or leave the plot bounds
    def stop_condition(t, p):
        dist_to_ob = np.sqrt((p[0]-pos_ob[0])**2 + (p[1]-pos_ob[1])**2)
        if dist_to_ob < R_star or np.abs(p[0]) > lim  or np.abs(p[1]) > lim :
            return 0
        return 1
    stop_condition.terminal = True

    # 4. Integrate Forward and Backward
    # We use rtol=1e-4 and a small first_step to ensure the solver starts moving
    options = {'events': stop_condition, 'rtol': 1e-4, 'first_step': 10.0}

    sol_b = solve_ivp(v_field, [0, -t_max], [x0, y0], **options)
    sol_f = solve_ivp(v_field, [0, t_max], [x0, y0], **options)

    # 5. Clean and Combine
    # sol_b.y is [x_values, y_values]. We flip 'b' so the path is continuous.
    x_path = np.concatenate([sol_b.y[0][::-1], sol_f.y[0]])
    y_path = np.concatenate([sol_b.y[1][::-1], sol_f.y[1]])

    #print("xpath=", x_path)
    if flag==1:
        return x_path, y_path
    if flag==2:
        return np.column_stack((x_path, y_path))


# def set_poly_wake1 (phase):
#     st = full_system_state (phase)
#     Polynome_Wake  = st['poly_wake']
#     #print ("!!!!!!")
#     exit()
    

# def set_poly_wake (polynome):
#     print ("!!!!")
#     Polynome_Wake = full_
#     print ("!!!!!!")
#     exit()
    

def check_point_fast_streamlines(x, y, phase, x1, y1, x2, y2, lim):
    if _CURRENT_WAKE is None:
        return 0
    
    # 1. Create a Path object (do this once per phase to save time)
    # Ideally, store the Path object in _CURRENT_WAKE instead of raw coordinates
    wake_path = Path(_CURRENT_WAKE)
    
    # 2. Use the built-in 'contains_point' method
    # It returns True/False, so we convert to 1/0
    return 1 if wake_path.contains_point((x, y)) else 0



def check_point_fast_streamlines_3d(x, y, z, phase, x_OB, y_OB, streamline_radius, rOB_NS_plane,
lim):
    """
    Return True if (x,y,z) is inside the 3D volume between two streamlines.
    Conventions:
    - NS is at (0,0,0).
    - OB is at (x_OB, y_OB, 0).
    - get_streamline_path(...) returns an (N,2) array of (x,y) in same frame (NS at origin).
    - streamline_radius: base cross-section radius at rOB_NS_plane (cm).
    - rOB_NS_plane: distance between OB and NS in the orbital plane (== sqrt(x_OB^2 + y_OB^2)).
    - lim: passed to get_streamline_path.

    Returns True if inside volume, False otherwise.
    """

    # validate rOB_NS_plane
    if rOB_NS_plane <= 0.0:
        return False

    if along_streamlines :
        # choose two seed points (these are just to select which streamlines to retrieve)
        x1, y1 = -streamline_radius, -streamline_radius
        x2, y2 = +streamline_radius, +streamline_radius

        path1 = np.asarray(get_streamline_path(phase, x1, y1, lim, flag=2))
        path2 = np.asarray(get_streamline_path(phase, x2, y2, lim, flag=2))

        if path1.size == 0 or path2.size == 0:
            return False
        if path1.ndim != 2 or path2.ndim != 2 or path1.shape[1] < 2 or path2.shape[1] < 2:
            return False

        # Build polygon between streamlines: forward path1 then reversed path2
        poly = np.vstack([path1, path2[::-1, :]])
    else :
        if  _CURRENT_WAKE is not None:
            poly = _CURRENT_WAKE
       
    wake_path = Path(poly)
    #return 1 if wake_path.contains_point((x, y)) else 0   
    # # Translate point to same frame: here NS at origin so no change.
    # # Compute r_proj: distance from OB to (x,y) in plane
    dx_ob = x - x_OB
    dy_ob = y - y_OB
    r_proj = math.hypot(dx_ob, dy_ob)

    # address z-thickness:
    R_cross = streamline_radius * (r_proj / rOB_NS_plane)
    # print ("R_cross=",R_cross/a)
    # If you prefer to cap R_cross or impose min value, do so here.
    # print (abs(z)<=R_cross)
    
    return abs(z) <= R_cross if wake_path.contains_point((x, y)) else 0





def point_in_polygon_2d(x, y, poly):
    """
    poly: (N,2) array of polygon vertices (closed or open; this will handle either)
    returns True if (x,y) is inside polygon (non-zero winding rule via parity).
    """
    inside = False
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi)
    if intersect:
        inside = not inside
        j = i
    return inside


def check_point_ultra_fast____old(x, y, phase, x1, y1, x2, y2, lim):
    path1 = get_streamline_path(phase, x1, y1, lim, flag=2)
    path2 = get_streamline_path(phase, x2, y2, lim, flag=2)

    # Logic: At the same 'radial distance' from the NS,
    # is the angle of our point (x,y) between the angles of path1 and path2?
    r_target = np.sqrt(x**2 + y**2)
    theta_target = np.arctan2(y, x)

    # Find points on streamlines at similar distance
    r1 = np.sqrt(path1[:,0]**2 + path1[:,1]**2)
    r2 = np.sqrt(path2[:,0]**2 + path2[:,1]**2)

    # Interpolate the angles of the streamlines at the point's radius
    theta1 = np.interp(r_target, r1, np.arctan2(path1[:,1], path1[:,0]))
    theta2 = np.interp(r_target, r2, np.arctan2(path2[:,1], path2[:,0]))

    # Sort them to ensure range check works
    t_min, t_max = sorted([theta1, theta2])

    return 1 if t_min <= theta_target <= t_max else 0

def check_between_streamlines(x, y, phase, x1, y1, x2, y2, lim):
    """
    Returns 1 if (x,y) is between streamlines 1 and 2, else 0.
    Works for single points, vectors, or meshgrids.
    """
    # 1. Get the boundary paths
    path1 = get_streamline_path(phase, x1, y1, lim,flag=2)
    path2 = get_streamline_path(phase, x2, y2, lim,flag=2)

    # 2. Close the polygon (Path 1 forward, then Path 2 backward)
    poly_verts = np.concatenate([path1, path2[::-1]])
    stream_polygon = Path(poly_verts)

    # 3. Prepare the input points (handles scalars and arrays)
    x_arr = np.atleast_1d(x)
    y_arr = np.atleast_1d(y)
    points = np.column_stack((x_arr.flatten(), y_arr.flatten()))

    # 4. Perform the test and convert to 0/1
    inside = stream_polygon.contains_points(points)
    result = inside.astype(int)

    # 5. Reshape back to the original input shape
    if np.isscalar(x):
        return int(result[0])
    return result.reshape(np.shape(x))







def get_past_ns_orbit(current_phase,  n_points=300):
    """
    Calculates the past positions of the NS relative to its current
    position (0,0) at phase 'current_phase'.
    """
    P_sec = Porb_day * DAY
    lookback_times = np.linspace(0, 1.1 * P_sec, int(n_points))

    # We define the current NS as (0,0).
    # To find past positions relative to current, we need the
    # vector difference in a fixed global frame.
    curr_st = orbital_state(current_phase)
    # Since orbital_state returns pos_ob = (OB - NS),
    # let's define absolute NS position as -pos_ob_global

    past_x, past_y = [], []

    for tau in lookback_times:
        past_phase = (current_phase - (tau / P_sec)) % 1.0
        past_st = orbital_state(past_phase)

        # Vector from current NS to past NS:
        # pos_ns_past - pos_ns_now
        # In this library's relative logic:
        # (pos_ob_now - pos_ob_past)
        dx = curr_st['pos_ob'][0] - past_st['pos_ob'][0]
        dy = curr_st['pos_ob'][1] - past_st['pos_ob'][1]

        past_x.append(dx)
        past_y.append(dy)

    return np.array(past_x), np.array(past_y)


import numpy as np
from scipy.interpolate import interp1d


def find_local_orbit_intersections(current_phase, r0, phase_window=0.5, n_points=N_points_to_find_intersections_with_orbit):
    """
    Searches forward and backward from current_phase to find the 
    intersections with a circle of radius r0.
    """
    P_sec = Porb_day * 86400
    # Search from -phase_window to +phase_window relative to current_phase
    offsets = np.linspace(-phase_window, phase_window, n_points)
    
    curr_st = orbital_state(current_phase)
    
    past_pts = [] # To store (x, y, phase) for crossings
    
    # Calculate relative positions for the window
    rel_x = []
    rel_y = []
    phases = []
    
    for dp in offsets:
        p = (current_phase + dp) % 1.0
        st = orbital_state(p)
        # Position relative to current NS at (0,0)
        rel_x.append(curr_st['pos_ob'][0] - st['pos_ob'][0])
        rel_y.append(curr_st['pos_ob'][1] - st['pos_ob'][1])
        phases.append(p)
    
    rel_x = np.array(rel_x)
    rel_y = np.array(rel_y)
    distances = np.sqrt(rel_x**2 + rel_y**2)
    


    # Find crossings of r0
    diff = distances - r0
    idx_crossings = np.where(np.diff(np.sign(diff)))[0]
    
    intersections = []
    for idx in idx_crossings:
        # Linear interpolation for high precision
        f = (r0 - distances[idx]) / (distances[idx+1] - distances[idx])
        
        ix = rel_x[idx] + f * (rel_x[idx+1] - rel_x[idx])
        iy = rel_y[idx] + f * (rel_y[idx+1] - rel_y[idx])
        ip = phases[idx] + f * (phases[idx+1] - phases[idx])
        
        intersections.append({'x': ix, 'y': iy, 'phase': ip % 1.0})

    return intersections

def find_orbit_circle_intersections(past_x, past_y, lookback_times, r0):
    """
    Finds the intersections of the past NS orbit with a circle of radius r0.
    Returns: list of (x, y) coordinates and list of lookback times.
    """
    # 1. Calculate distance from the current NS (0,0) for each point
    distances = np.sqrt(past_x**2 + past_y**2)
    
    # 2. Find indices where the path crosses the radius r0
    # We look for sign changes in (distance - r0)
    diff = distances - r0
    # np.sign change detection
    idx_crossings = np.where(np.diff(np.sign(diff)))[0]
    
    intersections_xy = []
    intersections_tau = []
    
    for idx in idx_crossings:
        # To be precise, we interpolate between the two points 
        # where the sign change happened.
        t1, t2 = lookback_times[idx], lookback_times[idx+1]
        d1, d2 = distances[idx], distances[idx+1]
        
        # Linear interpolation factor
        # r0 = d1 + fraction * (d2 - d1)
        fraction = (r0 - d1) / (d2 - d1)
        
        # Interpolate lookback time
        tau_intersect = t1 + fraction * (t2 - t1)
        
        # Interpolate coordinates
        x_intersect = past_x[idx] + fraction * (past_x[idx+1] - past_x[idx])
        y_intersect = past_y[idx] + fraction * (past_y[idx+1] - past_y[idx])
        
        intersections_xy.append((x_intersect, y_intersect))
        intersections_tau.append(tau_intersect)
        
        # User requested 2 intersections
        if len(intersections_xy) == 2:
            break
            
    return np.array(intersections_xy), np.array(intersections_tau)

# def is_in_custom_conus(phase, x, y, beta_aw=inclin_aw, psi = width_aw):
#     """
#     Checks if point (x, y) is inside a cone starting at the NS.
#
#     Args:
#         phase: Orbital phase [0, 1)
#         x, y: Coordinates (can be meshgrid arrays)
#         beta_aw: Angle offset from the NS->OB line (degrees)
#         psi: Full width of the cone (radians)
#     """
#     # 1. Get the direction to the OB star
#
#     beta_aw = beta_aw/180*np.pi
#     psi = psi/180*np.pi
#     st = orbital_state(phase % 1.0)
#     pos_ob = st['pos_ob']
#     theta_ob = np.arctan2(pos_ob[1], pos_ob[0])
#
#     # 2. Define the centerline of our custom cone
#     # We rotate the NS-OB line by beta
#     theta_center = theta_ob + beta_aw
#
#     # 3. Calculate the angle of the points (x, y) relative to NS
#     theta_point = np.arctan2(y-pos_ob[1]*0.5, x-pos_ob[0]*0.5)
#
#     # 4. Angular check using cosine similarity
#     # Point is in cone if the angular distance is <= psi/2
#     half_width = psi / 2.0
#
#     # Use cosine to handle the 2*pi wrap-around automatically
#     angle_diff_cos = np.cos(theta_point - theta_center)
#     inside_cone = angle_diff_cos >= np.cos(half_width)
#
#     return inside_cone

def is_in_conus(phase, x, y):
    """
    Returns True (or a boolean mask) if the point (x, y) is inside the
    tangent cone and further from the NS than the OB star.
    """
    # 1. Get orbital state and OB star parameters
    st = orbital_state(phase % 1.0)
    pos_ob = st['pos_ob']


    # 2. Geometry of the OB star relative to NS (origin)
    d_ob = np.sqrt(pos_ob[0]**2 + pos_ob[1]**2)
    theta_center = np.arctan2(pos_ob[1], pos_ob[0])

    # 3. Geometry of the point (x, y) relative to NS
    r_point = np.sqrt(x**2 + y**2)
    theta_point = np.arctan2(y, x)

    # 4. Calculate half-opening angle alpha
    # If the NS is inside the star (unlikely), alpha isn't defined
    if R_star >= d_ob:
        return np.zeros_like(x, dtype=bool)

    alpha = np.arcsin(R_star / d_ob)

    # 5. Check Angle Condition: Is the point within the cone's spread?
    # We use cosine of the difference to avoid 2*pi wrapping issues
    angle_diff_cos = np.cos(theta_point - theta_center)
    inside_angle = angle_diff_cos >= np.cos(alpha)

    # 6. Check Distance Condition: Is the point "beyond" the OB star?
    # We use d_ob as the threshold to ensure we are in the 'shadow' region
    beyond_star = r_point >= d_ob

    return inside_angle & beyond_star

import numpy as np

def is_in_curved_conus(phase, x, y):
    """
    Determines if point (x, y) is inside the curved shadow/cone of the OB star.
    Logic: If we trace the local relative velocity vector BACKWARDS,
    does it intersect the OB star's radius?
    """
    # 1. Get the system state from your library
    st = orbital_state(phase % 1.0)
    pos_ob = st['pos_ob']
    v_orb_vec = st['v_orb_vec']

    # 2. Get the local relative velocity field at the point(s)
    # This automatically handles the "curved" streamlines logic
    vx_rel, vy_rel = get_rel_velocity_field(x, y, pos_ob, v_orb_vec)

    # 3. Define the 'Upstream' vector (where the wind is coming from)
    ux = -vx_rel
    uy = -vy_rel

    # 4. Ray-Circle Intersection Math
    # Ray starts at P(x, y), moves in direction U(ux, uy)
    # Circle is at C(pos_ob), radius R_star
    cx, cy = pos_ob
    dx = x - cx
    dy = y - cy

    # Quadratic coefficients: At^2 + Bt + C = 0
    # A = dot(U, U)
    # B = 2 * dot(D, U) where D = P - C
    # C_quad = dot(D, D) - R^2
    A = ux**2 + uy**2
    B = 2 * (dx * ux + dy * uy)
    C_quad = dx**2 + dy**2 - R_star**2

    # Discriminant: If < 0, the ray misses the star entirely
    discriminant = B**2 - 4 * A * C_quad

    # 5. Logical Conditions
    # - Must have a real intersection (discriminant >= 0)
    # - Intersection must be in the 'forward' upstream direction (B < 0)
    # - Point must be outside the OB star (C_quad > 0)
    # - Point must be 'beyond' the OB star relative to the NS
    dist_ns_to_point = np.sqrt(x**2 + y**2)
    dist_ns_to_ob = np.sqrt(pos_ob[0]**2 + pos_ob[1]**2)

    in_shadow = (discriminant >= 0) & (B < 0) & (C_quad > 0)
    is_beyond = dist_ns_to_point > dist_ns_to_ob

    return in_shadow & is_beyond


def is_in_ballistic_shadow(phase, x, y):
    """
    Determines if point (x, y) is in the shadow cast by the OB star,
    accounting for orbital aberration (the 'curve').
    """
    st = orbital_state(phase % 1.0)
    pos_ob = st['pos_ob']
    v_orb_vec = st['v_orb_vec']

    # # 1. Vector from OB star to the point (x, y)
    dx = x - pos_ob[0]
    dy = y - pos_ob[1]
    dist_from_ob = np.sqrt(dx**2 + dy**2)
    #
    # # 2. Local wind velocity magnitude at that distance


    vx_rel, vy_rel = get_rel_velocity_field(x, y, st['pos_ob'], st['v_orb_vec'])

    # 4. Check if the point (x, y) lies on a ray that 'started'
    # from the surface of the OB star.
    # To simplify: a point is in the shadow if the angle of its
    # position relative to the OB star matches the 'aberrated' angle.

    # Angle of the point relative to OB star
    theta_point = np.arctan2(dy, dx)

    # Angle of the relative velocity (the path the wind takes)
    theta_flow = np.arctan2(vy_rel, vx_rel)

    # In a true ballistic shadow, these angles should be nearly identical
    # because the wind travels from the OB star to (x,y) along that vector.
    angle_diff = np.cos(theta_point - theta_flow)

    # We define the shadow boundary by the star's angular size as seen from the point
    angular_radius = np.arcsin(np.clip(R_star / dist_from_ob, 0, 1))
    #angular_radius  =0.11

    # Condition: The flow vector must point away from the OB star
    # and the angular deviation must be within the star's footprint.

    # distance from NS
    r_point = np.sqrt(x**2 + y**2)

    # is_shadow = (angle_diff < np.cos(angular_radius)) & (dist_from_ob > R_star)
    is_shadow = (angle_diff < 0.9*np.cos(angular_radius))  & (r_point<dist_from_ob)

    return is_shadow




def get_past_intersection_line_linear(current_phase, Porb_day, n_points=300):
    """
    Calculates the current positions of wind particles that once
    intersected the NS at various points in its past orbit.

    """
    P_sec = Porb_day * 86400
    # Look back up to one full orbit
    lookback_times = np.linspace(0, 1.8 * P_sec, n_points)

    line_x, line_y = [], []
    curr_st = orbital_state(current_phase)
    pos_ob_now = curr_st['pos_ob']

    for tau in lookback_times:
        past_phase = (current_phase - (tau / P_sec)) % 1.0
        past_st = orbital_state(past_phase)

        # NS position relative to OB star at that past time is -pos_ob
        r_past_vec = -past_st['pos_ob']
        r_mag_past = past_st['r_inst']

        # Wind velocity at the distance where the NS was
        # Note: using ws.v_wind since it's in your library
        v_w = v_wind(r_mag_past)
        dist_traveled = v_w * tau

        unit_vector = r_past_vec / r_mag_past
        r_particle_rel_ob = r_past_vec + (unit_vector * dist_traveled)

        # Convert back to NS-centric coords (where current NS is 0,0)
        abs_x = pos_ob_now[0] + r_particle_rel_ob[0]
        abs_y = pos_ob_now[1] + r_particle_rel_ob[1]

        line_x.append(abs_x)
        line_y.append(abs_y)

    return np.array(line_x), np.array(line_y)



