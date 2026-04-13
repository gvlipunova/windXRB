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
def extract_param_from_header(filepath, param_name, flag_conf_file=False):
    """
    Scans the header of a file (lines starting with #) to find a specific parameter.
    Returns 'param_name = value' as a string if found, otherwise returns None.
    """
    
    try:
        with open(filepath, 'r') as f:
            for line in f:

                if not line.startswith('#') and not flag_conf_file:
                    break  # Stop searching if we hit data rows
                
                # Check if the parameter name is in the line (e.g., "# e = 0.462")
                if param_name in line and '=' in line:
                    # Split by '=' and clean up whitespace/comment hash
                    parts = line.split('=')
                    value = parts[1].strip()
                    
                    # Check if key matches exactly to avoid partial matches (e.g., 'a' in 'phase_def')
                    key = parts[0].replace('#', '').strip()
                    if key == param_name:
                        return value
    except Exception as e:
        print(f"Error reading header of {filepath}: {e}")
    return None

def get_params_from_header(filepath, param_list):
    """
    Reads the header once and extracts all requested parameters.
    Returns a dictionary of {param_name: value}.
    """
    results = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                if '=' in line:
                    parts = line.split('=')
                    key = parts[0].replace('#', '').strip()
                    if key in param_list:
                        results[key] = parts[1].strip()
    except Exception as e:
        print(f"Error reading header of {filepath}: {e}")
    return results

def main():
    # 1. Parse Arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Case 1 (Filenames): python plot_all_Nh.py <directory>")
        print("  Case 2 (Parameters): python plot_all_Nh.py <directory> <param1> <param2> ...")
        sys.exit(1)
    data_dir = sys.argv[1]
    # Extract all arguments after the directory as a list of parameters
    params_to_extract = sys.argv[2:]

    if not os.path.isdir(data_dir):
        print(f"Error: '{data_dir}' is not a valid directory.")
        sys.exit(1)

    # --- Find all matching files ---
    pattern = os.path.join(data_dir, "Nh_table*.dat")
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

    src = extract_param_from_header("XRB.ini", "src", flag_conf_file=True)# or "Unknown Source"

    for filepath in files:
        basename = os.path.basename(filepath)

        # --- GENERATE LABEL ---
        if not params_to_extract:
            # Case 1: Only dirname passed, use filename
            label = basename
        else:
            # Case 2: Extract specific parameters
            
            header_data = get_params_from_header(filepath, params_to_extract)
            # Build string: "param1=val1, param2=val2"
            # Only include parameters actually found in the header
            label_items = [f"{p}={header_data[p]}" for p in params_to_extract if p in header_data]
            label = ", ".join(label_items) if label_items else basename

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
        try:
            Nh_norm1 = data[:, 3]
        except IndexError:
            Nh_norm1 = None 

        # Mask eclipsed points (NaN)
        valid = ~np.isnan(Nh)

        # Plot raw N_H
        ax1.plot(phase[valid], Nh[valid],linestyle='-', marker='.', markersize=3,
                 label=label, alpha=0.8)
        if np.any(~valid):
            ax1.plot(phase[~valid],
                     np.full(np.sum(~valid), np.nanmin(Nh[valid]) * 0.5),
                     'x', color='red', markersize=4)

        # Plot normalized N_H
        valid_norm = ~np.isnan(Nh_norm)
        ax2.plot(phase[valid_norm], Nh_norm[valid_norm],linestyle='-', marker='.',
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

    

    ax1.set_title(f'Column Density vs Orbital Phase for {src}')
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


    plot_normalized_alt  = 0
    if plot_normalized_alt:
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
