#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import streamlines_lib as ws

def plot_Mdot_over_phase(Mdot_wind, phases=None, out_pdf='Mdot_vs_phase.pdf'):
    if phases is None:
        phases = np.linspace(0.0, 1.0, 100)

    Mdot_vals = np.zeros_like(phases)
    vrels = np.zeros_like(phases)
    Rgs = np.zeros_like(phases)

    for i, ph in enumerate(phases):
        Mdot_vals[i], vrels[i], Rgs[i] = ws.compute_Mdot_accreted_at_phase(ph, Mdot_wind)

    # Plot Mdot vs phase
    plt.figure(figsize=(8,5))
    plt.plot(phases, Mdot_vals, '-b', lw=2)
    plt.xlabel('Orbital phase')
    plt.ylabel('Mdot_accreted (same units as Mdot_wind)')
    plt.title(f'{ws.src} Accreted mass rate vs phase (Mdot_wind={Mdot_wind:.3g})')
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved plot to {out_pdf}")

    # Also return arrays for further analysis
    return phases, Mdot_vals, vrels, Rgs






if __name__ == "__main__":
    # Example usage:
    # Set wind mass-loss rate; replace with your preferred value and units.
    # Example physical value: 1e-6 Msun/yr (but ensure units consistent with G and masses in ws)
    Mdot_wind = 1e-6

    phases, Mdot_vals, vrels, Rgs = plot_Mdot_over_phase(Mdot_wind, phases=np.linspace(0,1,100))
    plt.show()
