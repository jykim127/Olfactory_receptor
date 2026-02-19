import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess


plt.clf()
plt.close('all')

receptor_data = {
    'OR1A1':  {'r3': 122, 'r6': 219},
    'OR1A2':  {'r3': 122, 'r6': 207},
    'OR2AG2': {'r3': 122, 'r6': 209},
    'OR5A2':  {'r3': 124, 'r6': 228},
    'OR14A2': {'r3': 121, 'r6': 231}
}

stage_order = ['PDB', 'CONF', 'EM', 'NVT', 'NPT', 'MD', '20ns', '100ns']
base_dir = os.getcwd()
results = []

print("\n" + "="*50)
print("STARTING NEW ANALYSIS (Ver. Order Fixed)")
print("="*50)

for folder, res in receptor_data.items():
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path): continue
    
  
    static_stages = [
        ('PDB', f"{folder}.pdb"),
        ('CONF', 'conf.gro'), 
        ('EM', 'EM.tpr'), 
        ('NVT', 'NVT.tpr'), 
        ('NPT', 'NPT.tpr'), 
        ('MD', 'MD.tpr')
    ]
    
    for stage_name, file_name in static_stages:
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            file_path = os.path.join(folder_path, file_name.lower())
            
        if os.path.exists(file_path):
            output_xvg = f"temp_{folder}_{stage_name}.xvg"
            if file_path.lower().endswith('.tpr'):
                cmd = f"gmx distance -s {file_path} -select 'resnr {res['r3']} and name CA plus resnr {res['r6']} and name CA' -oall {output_xvg} -quiet"
            else:
                cmd = f"gmx distance -s {file_path} -f {file_path} -select 'resnr {res['r3']} and name CA plus resnr {res['r6']} and name CA' -oall {output_xvg} -quiet"
            
            try:
                subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
                if os.path.exists(output_xvg):
                    with open(output_xvg, 'r') as xvg_f:
                        lines = [l for l in xvg_f.readlines() if not l.startswith(('@', '#'))]
                        if lines:
                            dist_val = float(lines[-1].split()[1]) * 10
                            results.append({'Receptor': folder, 'Stage': stage_name, 'Distance': dist_val})
                            print(f"{folder:<10} | {stage_name:<6} | {dist_val:>10.2f} Å")
                    os.remove(output_xvg)
            except Exception: pass

   
    xtc_path = os.path.join(folder_path, 'md_fit.xtc')
    tpr_path = os.path.join(folder_path, 'MD.tpr')
    if os.path.exists(xtc_path) and os.path.exists(tpr_path):
        for time_point in [20, 100]:
            stage_name = f"{time_point}ns"
            output_xvg = f"temp_{folder}_{stage_name}.xvg"
            cmd = f"gmx distance -s {tpr_path} -f {xtc_path} -tu ns -b {time_point} -e {time_point} -select 'resnr {res['r3']} and name CA plus resnr {res['r6']} and name CA' -oall {output_xvg} -quiet"
            try:
                subprocess.run(cmd, shell=True, capture_output=True, executable='/bin/bash')
                if os.path.exists(output_xvg):
                    with open(output_xvg, 'r') as xvg_f:
                        lines = [l for l in xvg_f.readlines() if not l.startswith(('@', '#'))]
                        if lines:
                            dist_val = float(lines[-1].split()[1]) * 10
                            results.append({'Receptor': folder, 'Stage': stage_name, 'Distance': dist_val})
                            print(f"{folder:<10} | {stage_name:<6} | {dist_val:>10.2f} Å")
                    os.remove(output_xvg)
            except Exception: pass


if not results:
    print("[Error] No data found.")
else:
    df = pd.DataFrame(results)
    df['Stage'] = pd.Categorical(df['Stage'], categories=stage_order, ordered=True)
    df = df.sort_values(['Receptor', 'Stage'])

    fig, ax = plt.subplots(figsize=(14, 7))
    for receptor in df['Receptor'].unique():
        subset = df[df['Receptor'] == receptor]
    
        ax.plot(subset['Stage'].astype(str), subset['Distance'], marker='o', markersize=8, label=receptor, linewidth=2)

    ax.axhline(y=5, color='red', linestyle='--', alpha=0.3, label='Inactive boundary')
    ax.set_ylabel('Distance (Å)', fontsize=12)
    ax.set_title('TM3-TM6 Distance Process Summary (CONF Fixed)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    

    new_filename = 'Summary_Distance_Final_v2.png'
    plt.tight_layout()
    plt.savefig(new_filename, dpi=300)
    print(f"\n[Success] New image saved as: {new_filename}")
