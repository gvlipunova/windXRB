import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlines_lib as ws

from scipy.integrate import solve_ivp



if __name__ == "__main__":
    # 1. Handle Input Phase
    try:
        if len(sys.argv) >= 2:
            input_phase = float(sys.argv[1])
        else:
            input_phase = float(ws.phase_def)
        if len(sys.argv) >= 3:
            ws.wake_extension = float(sys.argv[2])
        else:
            ws.wake_extension = float(ws.wake_extension)
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

    # calculate outline of the wake:
    lx1, ly1, lx2,ly2 = ws.get_accretion_wake(input_phase,ws.wake_extension)


     # 2. Path the NS took to get to current position
    ox, oy = ws.get_past_ns_orbit(input_phase, ws.Porb_day)

    # Past Orbit Line
    plt.plot(ox, oy, 'k:', alpha=0.5, label='Past NS Orbit Path')

    # 5. Plotting
    # Plot OB star
    ax.plot(st['pos_ob'][0], st['pos_ob'][1], 'ro', markersize=12, label='OB Star')

    # Plot NS (fixed at origin in this frame)
    ax.plot(0, 0, 'ko', markersize=5, label='Neutron Star')

    ax.plot(lx, ly, 'b--', linewidth=3.1, alpha=0.9, label='Accretion Streamline')

    # Plot the line where past-intersected particles are now
    #ax.plot(lx1, ly1, 'b--', linewidth=1.5, alpha=0.7, label='Accretion Streamline (Past NS path)')
    #ax.plot(lx2, ly2, 'b--', linewidth=1.5, alpha=0.7, label='Accretion Streamline (Past NS path)')


    # Formatting
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_title(f"Locus of particles which met NS as it rotated once around OB star counting from phase {input_phase:.1f}")
    ax.legend()
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    out_pdf="Path.pdf"
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"Plot saved as '{out_pdf}' for phase={input_phase:.3f}")
    plt.show()
