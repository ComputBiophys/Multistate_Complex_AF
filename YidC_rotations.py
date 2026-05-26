import os
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, Superimposer
from matplotlib.lines import Line2D

first_structure = None
first_centroids = None
first_b_axis = None  
first_b_chain_ca = None

def get_ca_atoms(chain):
    return [atom for atom in chain.get_atoms() if atom.get_name() == 'CA']

def calculate_centroid(atoms):
    coordinates = np.array([atom.get_coord() for atom in atoms])
    return np.mean(coordinates, axis=0)

def calculate_principal_axis(atoms):
    coordinates = np.array([atom.get_coord() for atom in atoms])
    cov_matrix = np.cov(coordinates.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    return eigenvectors[:, np.argmax(eigenvalues)]

def process_first_conformation(pdb_file):
    global first_structure, first_centroids, first_b_axis, first_b_chain_ca
    
    parser = PDBParser(QUIET=True)
    first_structure = parser.get_structure("first", pdb_file)
    
    chains = {
        'A': first_structure[0]['A'],
        'B': first_structure[0]['B'],
        'C': first_structure[0]['C']
    }
    
    ca_atoms = {
        'A': get_ca_atoms(chains['A']),
        'B': get_ca_atoms(chains['B']),
        'C': get_ca_atoms(chains['C'])
    }
    
    first_centroids = {
        'A': calculate_centroid(ca_atoms['A']),
        'B': calculate_centroid(ca_atoms['B']),
        'C': calculate_centroid(ca_atoms['C'])
    }
    
    first_b_axis = calculate_principal_axis(ca_atoms['B'])
    first_b_chain_ca = ca_atoms['B'] 

def superimpose_to_first(structure):
    current_b_ca = get_ca_atoms(structure[0]['B'])
    
    if len(current_b_ca) != len(first_b_chain_ca):
        raise ValueError("The number of Cα in Chain B is different")
    
    superimposer = Superimposer()
    superimposer.set_atoms(first_b_chain_ca, current_b_ca)
    superimposer.apply(structure.get_atoms()) 
    
    return structure

def get_standardized_projections(structure):
    chains = {
        'A': structure[0]['A'],
        'B': structure[0]['B'],
        'C': structure[0]['C']
    }
    
    centroids = {
        'A': calculate_centroid(get_ca_atoms(chains['A'])),
        'B': calculate_centroid(get_ca_atoms(chains['B'])),
        'C': calculate_centroid(get_ca_atoms(chains['C']))
    }
    
    projections = {
        'A': np.array([centroids['A'][0], centroids['A'][1]]),
        'B': np.array([centroids['B'][0], centroids['B'][1]]),
        'C': np.array([centroids['C'][0], centroids['C'][1]])
    }
    
    return projections

def calculate_oriented_angle(p1, p2, center):
    v1 = p1 - center 
    v2 = p2 - center  
    
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    
    angle_diff = angle1 - angle2
    angle_deg = np.degrees(angle_diff)
    
    return angle_deg if angle_deg >= 0 else angle_deg + 360

def process_pdb_file(pdb_file, is_first=False):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        
        for chain_id in ['A', 'B', 'C']:
            if chain_id not in structure[0]:
                raise ValueError(f"Lacking chain {chain_id}")
        
        if is_first:
            process_first_conformation(pdb_file)
            projections = {
                'A': np.array([first_centroids['A'][0], first_centroids['A'][1]]),
                'B': np.array([first_centroids['B'][0], first_centroids['B'][1]]),
                'C': np.array([first_centroids['C'][0], first_centroids['C'][1]])
            }
        else:

            superimposed_struct = superimpose_to_first(structure)
            projections = get_standardized_projections(superimposed_struct)
        
        angle = calculate_oriented_angle(
            projections['A'], 
            projections['C'], 
            projections['B']
        )
        
        return {
            'filename': os.path.basename(pdb_file),
            'projections': projections,
            'angle': angle,
            'is_first': is_first
        }
        
    except Exception as e:
        print(f"Dealing with {pdb_file} Error: {str(e)}")
        return None

def visualize_results(results, output_dir=None, custom_colors=None):
    if not results:
        return
    
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(3.28, 3.28)
    
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axhline(y=0, color='gray', linewidth=1.5, alpha=0.3)
    ax.axvline(x=0, color='gray', linewidth=1.5, alpha=0.3)
    
    first_proj = next(res['projections'] for res in results if res['is_first'])
    b_pos = first_proj['B']
    c_pos = first_proj['C']

    a_0=b_pos[0]
    a_1=b_pos[1]
    c_pos[0]=c_pos[0]-a_0
    c_pos[1]=c_pos[1]-a_1
    
    b_pos[0]=b_pos[0]-a_0
    b_pos[1]=b_pos[1]-a_1

    ax.scatter(b_pos[0], b_pos[1], c='#6495ed', s=150, marker='o', 
              edgecolors='black', linewidth=1.5, label='SecY')
    ax.scatter(c_pos[0], c_pos[1], c='#8fbc8f', s=150, marker='s', 
              edgecolors='black', linewidth=1.5, label='SecE')
    
    ax.plot([b_pos[0], c_pos[0]], [b_pos[1], c_pos[1]], 
            'black', linewidth=1.5, alpha=1, label='SecY→SecE')
    
    if custom_colors is not None:

        if len(custom_colors) != len(results):
            raise ValueError(f"The number of colors（{len(custom_colors)}）and angles（{len(results)}）are different")
        colors = custom_colors
    else:

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (result, color) in enumerate(zip(results, colors)):
        proj = result['projections']
        angle = result['angle']
        filename = f"{i+1}"
        

        ax.plot([b_pos[0], proj['A'][0]-a_0], [b_pos[1], proj['A'][1]-a_1], 
                color=color, linestyle='--', linewidth=1.5, zorder=0)
        
        ax.scatter(proj['A'][0]-a_0, proj['A'][1]-a_1, color=color, s=200, marker='H', 
                  edgecolors='black', linewidth=1.5, zorder=0)
        
        bc_length = np.linalg.norm(c_pos - b_pos)
        radius = bc_length
        
        
        start_angle = np.arctan2(c_pos[1] - b_pos[1], c_pos[0] - b_pos[0])
        end_angle = start_angle + np.radians(angle)
        
        theta = np.linspace(0, 2*np.pi, 100)
        arc_x = b_pos[0] + radius * np.cos(theta)
        arc_y = b_pos[1] + radius * np.sin(theta)
        
        ax.plot(arc_x, arc_y, color="black", linewidth=1.5, alpha=0.3, linestyle='--')
        
        mid_angle = start_angle + np.radians(angle / 2)
        text_x = b_pos[0] + radius * 1.1 * np.cos(mid_angle)
        text_y = b_pos[1] + radius * 1.1 * np.sin(mid_angle)
        
    ax.scatter([], [], c='white', s=200, marker='H', edgecolors='black', linewidth=1.5, label='YidC')

    ax.set_xlabel('X (Å)', fontsize=15)
    ax.set_ylabel('Y (Å)', fontsize=15)
    ax.axis('equal')
    plt.tick_params(labelsize=15)

    plt.ylim(-55, 55)
    plt.xlim(-55, 55) 
    plt.xticks([-50, -25, 0, 25, 50])
    plt.yticks([-50, -25, 0, 25, 50])

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path_svg = os.path.join(output_dir, 'YidC_rotations.svg')
        plt.savefig(save_path_svg, dpi=300, bbox_inches='tight')

    else:
        plt.show()
    plt.close()

def load_angles_and_visualize(angle_file, output_dir=None, custom_colors=None):

    try:
        with open(angle_file, 'r') as f:
            angles = [float(line.strip()) for line in f if line.strip()]

    except Exception as e:

        return
    
    b_pos = np.array([0.0, 0.0])
    c_pos = np.array([5.0, 0.0])
    bc_length = np.linalg.norm(c_pos - b_pos)
    
    results = []
    for i, angle in enumerate(angles):
        angle_rad = np.radians(angle)
        a_pos = np.array([
            b_pos[0] + bc_length * np.cos(angle_rad),
            b_pos[1] + bc_length * np.sin(angle_rad)
                     
            
        ])
        results.append({
            'projections': {'A': a_pos, 'B': b_pos, 'C': c_pos},
            'angle': angle,
            'is_first': (i == 0)
        })
    
    visualize_results(results, output_dir, custom_colors=custom_colors)

def main(pdb_dir, output_dir=None, custom_colors=None):

    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.lower().endswith('.pdb')])
    
    if not pdb_files:
        print(f"There is no PDB file in {pdb_dir}")
        return
    
    results = []
    
    first_path = os.path.join(pdb_dir, pdb_files[0])
    first_result = process_pdb_file(first_path, is_first=True)
    if first_result:
        results.append(first_result)

    for pdb_file in pdb_files[1:]:
        file_path = os.path.join(pdb_dir, pdb_file)
        result = process_pdb_file(file_path, is_first=False)
        if result:
            results.append(result)

    if results:
        for res in sorted(results, key=lambda x: x['angle']):
            print(f"{res['filename'].replace('.pdb','')}: {res['angle']:.2f}°")

    visualize_results(results, output_dir, custom_colors)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='YidC rotations')
    parser.add_argument('pdb_dir', help='Directory of PDB files')
    parser.add_argument('-o', '--output', help='Output', default=None)
    args = parser.parse_args()
    
    custom_colors = [
        '#ffa07a',
        '#8B5A2B',
        '#e42747',
        '#FF6B6B',
        '#f70084',
        '#00AA7F',
        '#5f0e3d'
    ]
    
    try:
        import Bio
    except ImportError:
        print("Please install the packages first (pip install biopython numpy matplotlib)")
        exit(1)
    
    main(args.pdb_dir, args.output, custom_colors=custom_colors)
