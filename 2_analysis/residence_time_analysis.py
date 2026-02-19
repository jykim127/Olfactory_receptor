#!/usr/bin/env python3
import argparse
import numpy as np
import csv
import sys
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs): return iterable

def parse_args():
    p = argparse.ArgumentParser(description="GROMACS Contact-based Residence Time Analysis")
    p.add_argument("--tpr", required=True, help="Topology file (.tpr)")
    p.add_argument("--xtc", required=True, help="Trajectory file (.xtc)")
    p.add_argument("--ligand_sel", required=True, help='Selection for ligand (e.g. "resname LIG")')
    p.add_argument("--receptor_sel", default="protein", help='Selection for receptor (default: "protein")')
    p.add_argument("--cutoff_nm", type=float, default=0.4, help="Contact cutoff in nm (default: 0.4)")
    p.add_argument("--gap_frames", type=int, default=2, help="Bridge interruptions up to X frames (default: 2)")
    p.add_argument("--stride", type=int, default=1, help="Stride over frames (default: 1)")
    p.add_argument("--out_csv", default="contact_residence_time.csv", help="Output CSV filename")
    return p.parse_args()

def get_heavy_atoms(u, selection_str):
    """Select atoms excluding hydrogens for more robust contact analysis."""
    sel = u.select_atoms(f"({selection_str}) and not (element H or name H*)")
    if len(sel) == 0:
        print(f"Error: No atoms found for selection '{selection_str}'")
        sys.exit(1)
    return sel

def compute_contacts(u, lig, rec, cutoff_nm, stride):
    cutoff_A = cutoff_nm * 10.0
    times_ps = []
    contact_flags = []

    print(f"--- Starting Analysis (Total Frames: {len(u.trajectory[::stride])}) ---")
    for ts in tqdm(u.trajectory[::stride], desc="Processing Frames"):
        times_ps.append(ts.time)
        # Fast distance calculation considering Periodic Boundary Conditions (PBC)
        pairs = capped_distance(lig.positions, rec.positions, 
                                max_cutoff=cutoff_A, box=ts.dimensions, 
                                return_distances=False)
        contact_flags.append(len(pairs) > 0)
    
    return np.array(times_ps), np.array(contact_flags)

def bridge_gaps(contact, gap_frames):
    """Fill short False gaps between True segments."""
    if gap_frames <= 0: return contact
    c = contact.copy()
    for i in range(1, len(c) - 1):
        if not c[i]:
            left = max(0, i - gap_frames)
            right = min(len(c), i + gap_frames + 1)
            if np.any(c[left:i]) and np.any(c[i+1:right]):
                c[i] = True
    return c

def get_residence_stats(times_ps, contact):
    """Extract continuous contact blocks and calculate durations."""
    blocks = []
    if not np.any(contact): return blocks

    # Identify state changes
    diff = np.diff(contact.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if contact[0]: starts = np.insert(starts, 0, 0)
    if contact[-1]: ends = np.append(ends, len(contact) - 1)

    for s, e in zip(starts, ends):
        duration = times_ps[e] - times_ps[s]
        # For single-frame events, use the trajectory time step
        if duration == 0 and len(times_ps) > 1:
            duration = times_ps[1] - times_ps[0]
            
        blocks.append({
            "start_time_ps": times_ps[s],
            "end_time_ps": times_ps[e],
            "duration_ns": duration / 1000.0,
            "n_frames": int(e - s + 1)
        })
    return blocks

def main():
    args = parse_args()
    u = mda.Universe(args.tpr, args.xtc)
    
    lig = get_heavy_atoms(u, args.ligand_sel)
    rec = get_heavy_atoms(u, args.receptor_sel)

    times, contact_raw = compute_contacts(u, lig, rec, args.cutoff_nm, args.stride)
    contact_bridged = bridge_gaps(contact_raw, args.gap_frames)
    blocks = get_residence_stats(times, contact_bridged)

    print("\n" + "="*50)
    print(f"Ligand: {len(lig)} atoms | Receptor: {len(rec)} atoms")
    print(f"Cutoff: {args.cutoff_nm} nm")
    print(f"Gap Bridging: {args.gap_frames} frames")
    print("-" * 50)

    if not blocks:
        print("Result: No contact events detected.")
    else:
        durations = [b['duration_ns'] for b in blocks]
        print(f"Total Binding Events: {len(blocks)}")
        print(f"Mean Residence Time: {np.mean(durations):.4f} ns")
        print(f"Max Residence Time: {np.max(durations):.4f} ns")
        print(f"Total Contact Occupancy: {np.sum(contact_bridged)/len(contact_bridged)*100:.2f} %")

        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=blocks[0].keys())
            writer.writeheader()
            writer.writerows(blocks)
        print(f"\nDetailed data saved to: {args.out_csv}")
    print("="*50)

if __name__ == "__main__":
    main()