#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


import streamlines_lib as ws

def plot_Mdot_over_phase(Mdot_wind, phases=None, out_pdf='Lx_vs_phase.pdf'):
    if phases is None:
        phases = np.linspace(0.0, 2.0, 200)

    Lx_vals1 = np.zeros_like(phases)
    Lx_vals2 = np.zeros_like(phases)
    Mdot_vals1 = np.zeros_like(phases)
    Mdot_vals2 = np.zeros_like(phases)
    vrels = np.zeros_like(phases)
    Rgs = np.zeros_like(phases)


    for i, ph in enumerate(phases):
        Mdot_vals1[i], vrels[i], Rgs[i] = ws.compute_Mdot_accreted_at_phase(ph, Mdot_wind)
        Mdot_vals2[i], vrels[i], Rgs[i] = ws.compute_Mdot_accreted_at_phase_general(ph, Mdot_wind)
        Lx_vals1[i] = Mdot_vals1[i] * ws.G * ws.M_ns/ ws.R_ns
        Lx_vals2[i] = Mdot_vals2[i] * ws.G * ws.M_ns/ ws.R_ns
    # Plot Mdot vs phase
    plt.figure(figsize=(8,5))
    plt.plot(phases, Lx_vals1, '-b', lw=2,label='vrel = v_wind')
    plt.plot(phases, Lx_vals2, '-g', lw=2,)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    plt.xlabel('Orbital phase')
    plt.ylabel('NS Luminosity (erg/s)')
    plt.title(f'{ws.src}  Mdot_wind={Mdot_wind:.3g}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved plot to {out_pdf}")

    # Also return arrays for further analysis
    return phases, Mdot_vals1, vrels, Rgs






if __name__ == "__main__":
    # Example usage:
    # Set wind mass-loss rate; replace with your preferred value and units.
    # Example physical value: 1e-6 Msun/yr (but ensure units consistent with G and masses in ws)
    Mdot_wind = ws.Mdot_wind

    phases, Mdot_vals, vrels, Rgs = plot_Mdot_over_phase(Mdot_wind, phases=np.linspace(0,2,200))
    plt.show()
