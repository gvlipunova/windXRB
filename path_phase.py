import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlines_lib as ws

from scipy.integrate import solve_ivp






def get_past_intersection_line_linear(current_phase, Porb_day, n_points=300):
    """
    Calculates the current positions of wind particles that once
    intersected the NS at various points in its past orbit.

    Calculates the current positions of wind particles that intersected
    the NS at various points in the past.
     The code above assumes the wind velocity $v_w$ is constant during the particle's flight from $r_{past}$ to $r_{now}$. Since the wind accelerates (the $\beta$ law), for very high precision, you would solve $t = \int_{r_{start}}^{r_{end}} \frac{dr}{v(r)}$. However, because the NS is usually far from the OB surface, the wind is near terminal velocity and the linear approximation $dist = v \cdot \tau$ is very effective.

    """
    P_sec = Porb_day * 86400
    # Look back up to one full orbit
    lookback_times = np.linspace(0, 1.8 * P_sec, n_points)

    line_x, line_y = [], []
    curr_st = ws.orbital_state(current_phase)
    pos_ob_now = curr_st['pos_ob']

    for tau in lookback_times:
        past_phase = (current_phase - (tau / P_sec)) % 1.0
        past_st = ws.orbital_state(past_phase)

        # NS position relative to OB star at that past time is -pos_ob
        r_past_vec = -past_st['pos_ob']
        r_mag_past = past_st['r_inst']

        # Wind velocity at the distance where the NS was
        # Note: using ws.v_wind since it's in your library
        v_w = ws.v_wind(r_mag_past)
        dist_traveled = v_w * tau

        unit_vector = r_past_vec / r_mag_past
        r_particle_rel_ob = r_past_vec + (unit_vector * dist_traveled)

        # Convert back to NS-centric coords (where current NS is 0,0)
        abs_x = pos_ob_now[0] + r_particle_rel_ob[0]
        abs_y = pos_ob_now[1] + r_particle_rel_ob[1]

        line_x.append(abs_x)
        line_y.append(abs_y)

    return np.array(line_x), np.array(line_y)

if __name__ == "__main__":
    # 1. Handle Input Phase
    try:
        if len(sys.argv) >= 2:
            input_phase = float(sys.argv[1])
        else:
            input_phase = float(ws.phase_def)
    except ValueError:
        print("Error: Phase must be a number.")
        sys.exit(1)

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # 3. Get orbital state for current phase
    st = ws.orbital_state(input_phase)

    # 4. Calculate the "Streak-line" (Past intersection line)
    # Using Porb_day from the library config

    lx, ly = ws.get_past_intersection_line(input_phase, ws.Porb_day)

    lx1, ly1, lx2,ly2 = ws.get_accretion_wake(input_phase,ws.wake_phase_width)


     # 2. Path the NS took to get to current position
    ox, oy = ws.get_past_ns_orbit(input_phase, ws.Porb_day)

    # Past Orbit Line
    plt.plot(ox, oy, 'k:', alpha=0.5, label='Past NS Orbit Path')

    # 5. Plotting
    # Plot OB star
    ax.plot(st['pos_ob'][0], st['pos_ob'][1], 'ro', markersize=12, label='OB Star')

    # Plot NS (fixed at origin in this frame)
    ax.plot(0, 0, 'ko', markersize=5, label='Neutron Star')

    #ax.plot(lx, ly, 'b--', linewidth=1.1, alpha=0.9, label='Accretion Streamline (Past NS path)')

    # Plot the line where past-intersected particles are now
    ax.plot(lx1, ly1, 'b--', linewidth=1.5, alpha=0.7, label='Accretion Streamline (Past NS path)')
    ax.plot(lx2, ly2, 'b--', linewidth=1.5, alpha=0.7, label='Accretion Streamline (Past NS path)')


    # Formatting
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_title(f"Wind Intersection Line - Orbital Phase: {input_phase:.3f}")
    ax.legend()
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    out_pdf="Path.pdf"
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"Plot saved as '{out_pdf}' for phase={input_phase:.3f}")
    plt.show()
