#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import streamlines_lib as ws

# 1. Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Global settings (Available to all ranks)
inclination = ws.inclination_deg
Mdot_msun = ws.Mdot_wind_msun_per_yr

def safe_legend():
    """Adds a legend only if labeled artists exist to avoid UserWarnings."""
    if plt.gca().get_legend_handles_labels()[1]:
        plt.legend()

def compute_Nh_vs_phases_mpi(phases_full):
    # Split the phases into chunks for each rank
    phases_chunks = np.array_split(phases_full, size)
    my_phases = phases_chunks[rank]

    # Each core computes its own slice
    my_Nh = np.zeros(len(my_phases))
    
    for i, ph in enumerate(my_phases):
        # 1. Update the library's internal state/wake for this phase
        ws.set_active_wake(ph)
        
        # 2. Compute the column density
        # Using the 3D calculation from your latest library version
        my_Nh[i] = ws.compute_Nh_3D(ph, inclination, observer_phi=ws.omega_obs,R_max = ws.R_max_def)
        
        # Progress tracking
        print(f"[Rank {rank}] Processing phase {ph:.3f} ({i+1}/{len(my_phases)})")

    # 3. Gather all local Nh arrays back to rank 0
    all_Nh_chunks = comm.gather(my_Nh, root=0)

    if rank == 0:
        # Concatenate list of arrays into one long array
        return np.concatenate(all_Nh_chunks)
    else:
        return None

# ---------- main ----------
if __name__ == "__main__":
    # Define the full phase grid on all ranks
    num_phases = 125
    phases_full = np.linspace(0.0, 1.0, num_phases)

    # Run the parallel computation
    Nh_vals = compute_Nh_vs_phases_mpi(phases_full)

    # 4. Only Rank 0 handles the output
    if rank == 0:
        # Save raw data for safety
        np.savetxt(f"Nh_data_e{ws.e}.txt", np.column_stack([phases_full, Nh_vals]))

        plt.figure(figsize=(9, 5))
        
        # Plot valid points
        plt.plot(phases_full, Nh_vals, '-', marker='.', color='C0', label='N_H Trace')
        
        # Handle Eclipses (NaN values)
        eclipsed = np.isnan(Nh_vals)
        if np.any(eclipsed):
            # Place markers for eclipsed phases at the bottom of the plot
            ymin = np.nanmin(Nh_vals[~eclipsed]) if np.any(~eclipsed) else 1e20
            plt.plot(phases_full[eclipsed], np.full(np.sum(eclipsed), ymin * 0.8), 
                     'rx', label='Eclipsed (NS unseen)')

        plt.xlabel('Orbital phase (0..1)')
        plt.ylabel('N_H (atoms cm$^{-2}$)')
        if ws.NH_min_plot and ws.NH_max_plot:
            plt.ylim(ws.NH_min_plot, ws.NH_max_plot)
        plt.yscale('log')
        
        title_line1 = f"MPI Parallel N_H vs Phase (Cores: {size})"
        title_line1 = f"{ws.src}"
        title_line2 = rf"$\dot{{M}}={Mdot_msun:.1e} M_\odot/yr, e={ws.e}, i={ws.inclination_deg}^\circ, \Omega_{{\rm obs}}={ws.omega_obs:.1f}^\circ$"

        title_fig = title_line1 + "\n" + title_line2
        
        plt.title(title_fig)
        
        plt.grid(alpha=0.3, which='both')
        safe_legend()
        plt.tight_layout()
        
        pdf_name = f"Nh_vs_phase_mpi_e{ws.e}_incl{inclination}.pdf"
        plt.savefig(pdf_name, bbox_inches='tight')
        print(f"\n--- Computation Complete ---")
        print(f"Total points: {len(phases_full)}")
        print(f"Saved plot to: {pdf_name}")


        # --- Figure 2: Normalized N_H ---
        Nh_norm_factor = ws.Mdot_wind_g_s / (ws.v_inf * ws.a * ws.m_H)
        Nh_normalized = Nh_vals / Nh_norm_factor

        plt.figure(figsize=(9, 5))
        plt.plot(phases_full, Nh_normalized, '-', marker='.', color='C1')
        plt.xlabel('Orbital phase (0..1)')
        plt.ylabel(r'N_H / ($\dot{M}_w \,/\, v_\infty \, a \, m_H$)')
        plt.yscale('log')
        plt.title(title_fig)


        if np.any(eclipsed):
            ymin_norm = np.nanmin(Nh_normalized[~eclipsed]) if np.any(~eclipsed) else 1.0
            plt.plot(phases_full[eclipsed],
                    np.full(np.sum(eclipsed), ymin_norm * 0.5),
                    'rx', label='Eclipsed (NS unseen)')
        safe_legend()
        plt.grid(alpha=0.3, which='both')
        plt.savefig(f"Nh_normalized_vs_phase_mpi_e{ws.e}.pdf")
        plt.savefig(f"Nh_norm_vs_phase_mpi_e{ws.e}.pdf")


         # --- Figure 3: Normalized N_H ---
        Nh_norm_factor1 = ws.Mdot_wind_g_s* ws.wake_factor_density *ws.streamlines_rg_fac / (ws.v_inf * ws.a * ws.m_H )
        Nh_normalized1 = Nh_vals / Nh_norm_factor1

        plt.figure(figsize=(9, 5))
        plt.plot(phases_full, Nh_normalized1, '-', marker='.', color='C1')
        plt.xlabel('Orbital phase (0..1)')
        plt.ylabel(r'N_H / ($\dot{M}_w \,/\, v_\infty \, a \, m_H$)')
        plt.yscale('log')
        plt.title(title_fig)


        if np.any(eclipsed):
            ymin_norm = np.nanmin(Nh_normalized1[~eclipsed]) if np.any(~eclipsed) else 1.0
            plt.plot(phases_full[eclipsed],
                    np.full(np.sum(eclipsed), ymin_norm * 0.5),
                    'rx', label='Eclipsed (NS unseen)')
        plt.legend()
        plt.grid(alpha=0.3, which='both')
        plt.savefig(f"Nh_normalized_vs_phase_mpi_e{ws.e}.pdf")
        plt.savefig(f"Nh_norm_vs_phase_mpi_e{ws.e}.pdf")


        # 1. Dynamically gather all parameters from the 'ws' module
        # We filter out internal python attributes (__name__), modules, and functions
        header_parts = ["System Parameters and Geometry:"]
        for key, value in vars(ws).items():
            if not key.startswith("__") and not callable(value):
                # Optional: check if it's a simple type (float, int, str) to avoid huge arrays
                if isinstance(value, (int, float, str, np.number)):
                    header_parts.append(f"{key} = {value}")

        # 2. Append the column names as the last line of the header
        header_parts.append("-" * 30)
        header_parts.append("phase  Nh_cm2  Nh_normalized Nh_normalized1")

        # 3. Join them into a single multi-line string
        full_header = "\n".join(header_parts)
        # 4. Save the file
        datafilename = f"data/Nh_table_e{ws.e}_geo{ws.GEOMETRY}_d{ws.wake_factor_density}_i{ws.inclination_deg}_rgf{ws.streamlines_rg_fac}_wext{ws.wake_extension}_beta{ws.beta}_obs{ws.omega_obs}.dat"
        np.savetxt(
            datafilename,
            np.column_stack([phases_full, Nh_vals, Nh_normalized, Nh_normalized1]),
            header=full_header,
            fmt="%.6f  %.6e  %.6e  %.6e",
            comments="# "
        )
        print(f"Saved data to: {datafilename}")
