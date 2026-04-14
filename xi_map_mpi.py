import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.path import Path

import streamlines_lib as ws
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def compute_log_xi_map_mpi(phase, XX, YY,  use_accretion_Lx=True, Mdot_override=None, Lx_override=None, mu_mean=0.6):
    """
    Compute log10(xi) map for a given orbital phase.

    Parameters
    - phase: orbital phase in [0,1)
    - XX, YY: 2D arrays giving grid coordinates in units of 'a' (same convention used elsewhere)
    - use_accretion_Lx: if True, attempt to compute L_x from ws.compute_Lx_at_phase(phase, ws.Mdot_wind_g_s)
                        otherwise, use Lx_override if provided.
    - Mdot_override: if provided (in g/s), use this Mdot when computing L_x from accretion formula.
    - Lx_override: if provided, use this value for L_x (cgs erg/s) instead of accretion Lx.
    - mu_mean: mean molecular weight in amu (default 0.6)

    Returns:
    - log10_xi: 2D array containing log10(xi) with NaN where undefined (inside donor, zero/invalid density)
    - rho_grid: density grid in g/cm^3 (NaN inside donor)
    - n_grid: number density grid in cm^-3 (NaN inside donor)
    - pos_ob_cm: donor center position in cm (2-element array)
    - L_x_used: value of L_x used (erg/s)
    """

    # Convert grid to cgs for physics calls
    XX_cm = XX * ws.a
    YY_cm = YY * ws.a

    # choose Mdot for rho calculations (use library default unless overridden)
    Mdot_wind_g_s = Mdot_override if (Mdot_override is not None) else ws.Mdot_wind_g_s

    #st = ws.full_system_state(phase)
    ws.set_active_wake(phase)
    pos_ob_cm = np.array(st['pos_ob'])           # in cm
    # 2. Add the wake to the state dictionary AFTER the orbit is solved
    #ws.set_poly_wake1 (phase)

    # Split rows among ranks
    my_rows_indices = np.array_split(np.arange(YY.shape[0]), size)[rank]
    my_XX_cm = XX_cm[my_rows_indices]
    my_YY_cm = YY_cm[my_rows_indices]

    # Local computation
    my_rho_list = []
    for i in range(len(my_rows_indices)):
        print(i,"/",len(my_rows_indices))
        row = [ws.rho_wind_at_point(phase, x, y, pos_ob_cm, Mdot_wind_g_s)
               for x, y in zip(my_XX_cm[i], my_YY_cm[i])]

        my_rho_list.append(row)

    my_rho_array = np.array(my_rho_list)







    # Gather all pieces back to Rank 0
    all_rho_chunks = comm.gather(my_rho_array, root=0)

    # INITIALIZE rho_grid on all ranks so it exists in the local scope
    rho_grid = None

    if rank == 0:
        # Only Rank 0 builds the full grid
        rho_grid = np.vstack(all_rho_chunks)

    # IMPORTANT: Broadcast the full rho_grid from Rank 0 to all other ranks
    # so they can all calculate n_grid and xi_grid
    rho_grid = comm.bcast(rho_grid, root=0)

    # Now ALL ranks have rho_grid and can proceed
    rho_grid[~np.isfinite(rho_grid)] = np.nan

    # orbital state and donor/NS positions in code units and cgs

    # compute rho grid using vectorized pointwise function (returns np.inf inside donor in lib)
    #rho_grid = _rho_point_vec(XX_cm, YY_cm, pos_ob_cm, Mdot_wind_g_s)




    # IMPORTANT: Broadcast the full rho_grid from Rank 0 to all other ranks
    # so they can all calculate n_grid and xi_grid
    rho_grid = comm.bcast(rho_grid, root=0)

    # Now ALL ranks have rho_grid and can proceed
    # Replace non-finite/infinite values returned for stellar interior with np.nan
    rho_grid[~np.isfinite(rho_grid)] = np.nan

    # number density n = rho / (mu * m_p)
    n_grid = rho_grid / (mu_mean * ws.m_H)



    # decide L_x to use
    L_x_used = None
    if Lx_override is not None:
        L_x_used = Lx_override
    else:
        if use_accretion_Lx:
            try:
                Mdot_for_Lx = Mdot_override if (Mdot_override is not None) else ws.Mdot_wind_g_s
                # compute_Lx_at_phase in your library returns G*M_ns * Mdot_acc / R_ns (see library)
                Lx_try = ws.compute_Lx_at_phase(phase, Mdot_for_Lx)
                if np.isfinite(Lx_try) and Lx_try > 0.0:
                    L_x_used = Lx_try
                else:
                    # fallback to default constant L_x if present in caller namespace
                    L_x_used = None
            except Exception:
                L_x_used = None
        # If still not set, attempt to fall back to a constant in the ws module (if user defined)
        if L_x_used is None:
            L_x_used = getattr(ws, 'L_x_default', None)
            if L_x_used is None:
                raise RuntimeError("No Lx available: provide Lx_override or set ws.L_x_default or enable accretion Lx.")

    # distances from NS (NS placed at origin in your conventions)
    RR_cm = np.sqrt(XX_cm**2 + YY_cm**2)
    R_ns_floor = getattr(ws, 'R_ns', 1.0e6)   # cm
    r_ns_safe = np.where(RR_cm <= R_ns_floor, R_ns_floor, RR_cm)

    # compute xi where valid: xi = L_x / (n * r_ns^2)
    xi_grid = np.full_like(RR_cm, np.nan, dtype=float)
    valid = np.isfinite(n_grid) & (n_grid > 0.0)

    xi_grid[valid] = L_x_used / ( n_grid[valid] * (r_ns_safe[valid]**2) )

    # mask explicitly inside donor star using library R_star_cm if available
    R_star_cm_lib = getattr(ws, 'R_star_cm', getattr(ws, 'R_star', None) * ws.a_cm)
    dx_star = XX_cm - pos_ob_cm[0]
    dy_star = YY_cm - pos_ob_cm[1]
    r_from_OB = np.sqrt(dx_star**2 + dy_star**2)
    inside_mask = (r_from_OB <= R_star_cm_lib)
    xi_grid[inside_mask] = np.nan
    n_grid[inside_mask] = np.nan
    rho_grid[inside_mask] = np.nan

    # return log10(xi) (masked invalids) and other useful arrays
    with np.errstate(invalid='ignore', divide='ignore'):
        log10_xi = np.where(np.isfinite(xi_grid) & (xi_grid > 0.0), np.log10(xi_grid), np.nan)

    return log10_xi, rho_grid, n_grid, pos_ob_cm, L_x_used


def plot_log_xi_map(log10_xi, XX, YY,
                    pos_ob_cm,
                    phase,
                    L_x_used,
                    outname=None,
                    contour_level=2.5,
                    cmap='inferno'):
    """
    Plot a log10(xi) map (XX,YY in units of a).

    Parameters:
    - log10_xi: 2D array from compute_log_xi_map (NaN where invalid)
    - XX, YY: 2D arrays of grid coords (units of a)
    - pos_ob_cm: donor position in cm (from compute function)
    - phase: orbital phase (for title)
    - L_x_used: L_x used (for title)
    - outname: if provided, save figure to this filename
    - contour_level: contour value in log10(xi) to overlay
    - cmap: colormap name
    """
    fig, ax = plt.subplots(figsize=(8,7))
    pcm = ax.pcolormesh(XX, YY, log10_xi, shading='auto', cmap=cmap)
    cb = fig.colorbar(pcm, ax=ax, label='log10(xi) [log(erg cm s^-1)]')

    # robust color limits based on percentiles ignoring NaNs
    try:
        vmin = np.nanpercentile(log10_xi, 1)
        vmax = np.nanpercentile(log10_xi, 99)
        pcm.set_clim(vmin, vmax)
    except Exception:
        pass

    # Overlay contour (handle empty results safely)
    logxi_masked = np.ma.masked_invalid(log10_xi)
    with np.errstate(invalid='ignore'):
        cs = ax.contour(XX, YY, logxi_masked, levels=[contour_level], colors=['cyan'], linewidths=1.5)

    legend_handles = []

    # If contour produced artists with paths, label it; otherwise make a proxy handle
    # Use the new Matplotlib API for contours (avoids .collections warning)
    if cs.get_paths():
        # In newer Matplotlib, we label the contour set directly
        cs.set_label(f'log10(xi) = {contour_level}')
        # We add the ContourSet to the legend handles
        legend_handles.append(Line2D([0], [0], color='cyan', lw=1.5, label=f'log10(xi) = {contour_level}'))
    else:
        proxy_contour = Line2D([0], [0], color='cyan', lw=1.5, label=f'log10(xi) = {contour_level}')
        legend_handles.append(proxy_contour)

    # Mark NS (origin) and donor center (convert donor cm -> units of a for plotting)
    ax.scatter([0.0], [0.0], color='cyan', s=30, label='NS (origin)')
    ax.scatter([pos_ob_cm[0]/ws.a_cm], [pos_ob_cm[1]/ws.a_cm], color='white', s=30, label='OB center')

    # donor circle in plot units (radius in code units = R_star_cm / a_cm)
    R_star_plot = getattr(ws, 'R_star_cm', getattr(ws, 'R_star', None) * ws.a_cm) / ws.a_cm
    donor_circle = plt.Circle((pos_ob_cm[0]/ws.a_cm, pos_ob_cm[1]/ws.a_cm),
                              R_star_plot, color='white', fill=False, lw=1.2)
    ax.add_patch(donor_circle)

    
    wake_circle = plt.Circle((pos_ob_cm[0]/ws.a_cm, pos_ob_cm[1]/ws.a_cm),
                              ws.wake_width_cm (phase)/ws.a , color='blue', fill=False, lw=1.2, label='Wake size') 
    ax.add_patch(wake_circle)

    # Use linestyle='None' to remove the connecting line in the legend entry
    legend_handles.append(Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='cyan', markersize=6, 
                             label='NS (origin)', linestyle='None'))

    legend_handles.append(Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='white', markersize=6, 
                             label='OB center', linestyle='None'))

    # For an "empty" circle (hollow), set markerfacecolor to 'None' or 'none'
    legend_handles.append(Line2D([0], [0], marker='o', color='blue', 
                             markerfacecolor='None', markeredgecolor='blue', 
                             markersize=6, label='Wake size', linestyle='None'))

     # 2. Path the NS took to get to current position
    ox, oy = ws.get_past_ns_orbit(phase, ws.Porb_day)

    # Past Orbit Line
    plt.plot(ox/ws.a_cm, oy/ws.a_cm, 'k:', alpha=0.5, label='Past NS Orbit Path')

    
    ax = plt.gca() # Get current axis
    ws.set_active_wake(phase) 
    
   
    wake_patch = Polygon(ws._CURRENT_WAKE/ws.a_cm, closed=True, facecolor='black', edgecolor='black', fill=None, alpha=0.3, label='Accretion Wake')

    ax.add_patch(wake_patch)
    legend_handles.append(wake_patch)
    #st = ws.full_system_state(phase)
    # --- ADDING THE LINE OF SIGHT (LOS) ---
   
    # Convert degrees to radians
    omega_rad = np.radians(ws.omega_obs)
        
    # Calculate vector components. 
    # Length 'L' determines how far the LOS arrow stretches across the plot.
    L = XX.max() * 0.8 
    dx = L * np.cos(omega_rad)
    dy = L * np.sin(omega_rad)
       
    # Draw the LOS vector starting from the NS (0,0)
    # Using quiver for an arrow, or plot for a dashed line
    ax.quiver(0, 0, dx, dy, color='yellow', angles='xy', scale_units='xy', 
                  scale=1, width=.005)
        
    # Optional: Add a dashed line across the whole map
    ax.plot([0, dx*2], [0, dy*2], color='yellow', linestyle='--', alpha=0.5)

    show_visibility  = 0
    if show_visibility :
        # 1. Vectorize the function so it accepts arrays XX, YY
        check_vec = np.vectorize(ws.check_point_fast_streamlines)
        # 2. Run it on the whole grid
        inside_grid = check_vec(XX*ws.a, YY*ws.a, phase,0, 0, 1,1, ws.streamlines_lim_def)
        # 3. Plot as a 2D image
        plt.pcolormesh(XX, YY, inside_grid, cmap='Greys', alpha=0.5, shading='auto')
        plt.title("Wake Mask (1=Inside, 0=Outside)")        


    

    ax.set_xlabel('x / a')
    ax.set_ylabel('y / a')

    ax.set_title(f'log10(xi) map at phase={phase:.3f}\nL_x={L_x_used:.3e} erg/s')
    ax.set_aspect('equal', 'box')

    # set axis limits from XX,YY
    try:
        ax.set_xlim(XX.min(), XX.max())
        ax.set_ylim(YY.min(), YY.max())
    except Exception:
        pass

    ax.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout()

    if outname is not None:
        plt.savefig(outname, dpi=200, bbox_inches='tight')
        print(f"Saved {outname}")
    #plt.show()


#---------- main ----------
if __name__ == "__main__":

    # Accept phase from command line, otherwise default to 0.0
    if len(sys.argv) >= 2:
        phase = float(sys.argv[1])
    else:
        phase = ws.phase_def

    

    # build grid in units of 'a' (same as your earlier code)
    extent_factor = 3.2
    n_pix = ws.map_n_pix
    xs = np.linspace(-extent_factor*ws.a/ws.a, extent_factor*ws.a/ws.a, n_pix)  # simply [-extent, +extent] in units of a
    # simpler: use units of a directly
    extent_in_a = extent_factor  # because XX/YY are in units of a
    xs = np.linspace(-extent_in_a, extent_in_a, n_pix)
    ys = xs.copy()
    XX, YY = np.meshgrid(xs, ys)


    L_x_used = ws.compute_Lx_at_phase(phase, ws.Mdot_wind_g_s)

    st = ws.full_system_state(phase)
    # 2. Add the wake to the state dictionary AFTER the orbit is solved
    #poly = ws.get_accretion_wake_poly(st, phase_width=ws.wake_phase_width)
    # print ("1:",st['poly_wake'])
    # print ("2:",ws.Polynome_Wake)
   
    
    log10_xi, rho_grid, n_grid, pos_ob_cm, L_x_used = compute_log_xi_map_mpi(phase, XX, YY,
                                                                        use_accretion_Lx=True,
                                                                        Mdot_override=None,
                                                                        Lx_override=L_x_used)
    
    if rank == 0:
        plot_log_xi_map(log10_xi, XX, YY, pos_ob_cm, phase, L_x_used, outname=f'xi_map_phase_{phase:.3f}_geo{ws.GEOMETRY}.png', contour_level=2.5)



# ******************

