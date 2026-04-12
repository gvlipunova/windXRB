#!/usr/bin/env python
"""
Read all Nh_table_e*.dat files from a directory and overplot them.

Usage:
    python plot_all_Nh.py /path/to/data/directory
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

font_size = 18
###########################################3
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : font_size}

plt.rc('font', **font)
####################################################

def main():
    # --- Parse command-line argument ---
    if len(sys.argv) < 2:
        print("Usage: python plot_all_Nh.py <directory>")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not os.path.isdir(data_dir):
        print(f"Error: '{data_dir}' is not a valid directory.")
        sys.exit(1)

    # --- Find all matching files ---
    pattern = os.path.join(data_dir, "Nh_table_e*.dat")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files matching '{pattern}' found.")
        sys.exit(1)

    print(f"Found {len(files)} file(s) in '{data_dir}':")
    for f in files:
        print(f"  {os.path.basename(f)}")

    # --- Figure 1: Raw N_H ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # --- Figure 2: Normalized N_H ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # --- Figure 3: Another Normalized N_H ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    for filepath in files:
        basename = os.path.basename(filepath)

        # Extract eccentricity from filename for label
        # Expected format: Nh_table_e<value>.dat
        try:
            e_str = basename.replace("Nh_table_e", "").replace(".dat", "")
            label = f"e = {e_str}"
        except:
            label = basename

        # Read data
        try:
            data = np.loadtxt(filepath)
        except Exception as ex:
            print(f"Warning: could not read '{basename}': {ex}")
            continue

        if data.ndim != 2 or data.shape[1] < 3:
            print(f"Warning: unexpected format in '{basename}', skipping.")
            continue

        phase = data[:, 0]
        Nh = data[:, 1]
        Nh_norm = data[:, 2]
        Nh_norm1 = data[:, 3]

        # Mask eclipsed points (NaN)
        valid = ~np.isnan(Nh)

        # Plot raw N_H
        ax1.plot(phase[valid], Nh[valid], '-', marker='.', markersize=3,
                 label=label, alpha=0.8)
        if np.any(~valid):
            ax1.plot(phase[~valid],
                     np.full(np.sum(~valid), np.nanmin(Nh[valid]) * 0.5),
                     'x', color='red', markersize=4)

        # Plot normalized N_H
        valid_norm = ~np.isnan(Nh_norm)
        ax2.plot(phase[valid_norm], Nh_norm[valid_norm], '-', marker='.',
                 markersize=3, label=label, alpha=0.8)
        if np.any(~valid_norm):
            ax2.plot(phase[~valid_norm],
                     np.full(np.sum(~valid_norm),
                             np.nanmin(Nh_norm[valid_norm]) * 0.5),
                     'x', color='red', markersize=4)

        # Plot another normalized N_H
        valid_norm1 = ~np.isnan(Nh_norm1)
        ax3.plot(phase[valid_norm1], Nh_norm1[valid_norm1], '-', marker='.',
                 markersize=3, label=label, alpha=0.8)
        if np.any(~valid_norm1):
            ax3.plot(phase[~valid_norm1],
                     np.full(np.sum(~valid_norm1),
                             np.nanmin(Nh_norm1[valid_norm1]) * 0.5),
                     'x', color='red', markersize=4)

    # --- Format Figure 1 ---
    ax1.set_xlabel('Orbital phase (0..1)', fontsize=font_size)
    ax1.set_ylabel(r'N$_H$ (atoms cm$^{-2}$)', fontsize=font_size)
    ax1.set_yscale('log')
    ax1.set_xlim(0, 1)
    ax1.set_title('Column Density vs Orbital Phase')
    ax1.legend(fontsize=font_size-4, loc='best')
    ax1.grid(alpha=0.3, which='both')
    fig1.tight_layout()
    outfile1 = os.path.join(data_dir, "Nh_all_comparison.pdf")
    fig1.savefig(outfile1)
    print(f"\nSaved: {outfile1}")

    # --- Format Figure 2 ---
    ax2.set_xlabel('Orbital phase (0..1)', fontsize=font_size)
    ax2.set_ylabel(r'N$_H$ / ($\dot{M}_w / v_\infty \, a \, m_H$)', fontsize=font_size)
    ax2.set_yscale('log')
    ax2.set_xlim(0, 1)
    ax2.set_title('Normalized Column Density vs Orbital Phase')
    ax2.legend(fontsize=font_size-4, loc='best')
    ax2.grid(alpha=0.3, which='both')
    fig2.tight_layout()
    outfile2 = os.path.join(data_dir, "Nh_normalized_all_comparison.pdf")
    fig2.savefig(outfile2)
    print(f"Saved: {outfile2}")


    # --- Format Figure 3 ---
    ax3.set_xlabel('Orbital phase (0..1)', fontsize=font_size)
    ax3.set_ylabel(r'$N_H  \rho_w \, \Delta_{\rm wake}/ (\dot{M}_w / v_\infty \, a \, m_H  )$', fontsize=font_size)
    ax3.set_yscale('log')
    ax3.set_xlim(0, 1)
    ax3.set_title('Normalized Column Density vs Orbital Phase')
    ax3.legend(fontsize=font_size-4, loc='best')
    ax3.grid(alpha=0.3, which='both')
    fig3.tight_layout()
    outfile3 = os.path.join(data_dir, "Nh_normalized2_all_comparison.pdf")
    fig3.savefig(outfile3)
    print(f"Saved: {outfile3}")

    plt.show()


if __name__ == "__main__":
    main()
