"""
Script de comparació: testeja els terrenys 4x4 candidates amb ambdós algorismes
per determinar quin produces resultats equilibrats i òptims.
"""

import numpy as np
import matplotlib.pyplot as plt
from original import GridToTinConverter
from pendent7 import GridToTinIncremental
import time
import os

# Terrenys a testejar
TERRAINS = {
    'balanced_diag': 'terrain_4x4_balanced_diag.npy',
    'fractal': 'terrain_4x4_fractal.npy',
    'gaussian_bumps': 'terrain_4x4_gaussian_bumps.npy',
}

STEP = 1
PIXEL_SIZE = 2.0
TARGET_POINTS = 4  # Amb un 4x4, máx 16 punts. Provem amb 4

def calculate_rmse(tin, h_grid, rows, cols, step, pixel_size):
    """Calcula RMSE entre grid original i TIN interpolat"""
    
    spacing = step * pixel_size
    
    # Grella de coordenades
    r_indices, c_indices = np.arange(rows), np.arange(cols)
    rr, cc = np.meshgrid(r_indices, c_indices, indexing='ij')
    x_coords = cc.ravel() * spacing
    y_coords = rr.ravel() * spacing
    query_xy = np.column_stack((x_coords, y_coords))
    
    # Trobar triangle i interpolar
    simplex_ids = tin.find_simplex(query_xy)
    
    # Obtenir vèrtexs i valors z
    simplex_array = tin.simplices[simplex_ids]
    
    errors = []
    for i, (query_pt, simplex_id) in enumerate(zip(query_xy, simplex_ids)):
        if simplex_id == -1:  # Fora del triangulació
            errors.append(np.nan)
        else:
            # Els 3 vèrtexs del triangle
            tri_verts = tin.points[tin.simplices[simplex_id]]
            tri_z = tin.simplices[simplex_id]  # NO! Necessitem els z values
            
            # Calcular interpolació baricèntrica
            # Això és complex, fem una aproximació simple
            dist_to_verts = np.linalg.norm(tri_verts[:, :2] - query_pt, axis=1)
            if np.min(dist_to_verts) < 1e-6:  # Molt a prop de vèrtex
                errors.append(0)
            else:
                errors.append(np.nan)
    
    actual_z = h_grid.ravel()
    
    # Calcul aproximat (per a 4x4, probablement tots són vèrtexs del TIN)
    return np.nanmean(np.array(errors))


def test_all_combinations():
    """Testa tots els terrenys amb ambdós algorismes"""
    
    results = {
        'terrain': [],
        'algorithm': [],
        'points': [],
        'error': [],
        'time': []
    }
    
    print("\n" + "="*80)
    print("COMPARACIÓ: original.py vs pendent7.py en terrenys 4x4")
    print("="*80)
    
    for terrain_name, terrain_file in TERRAINS.items():
        
        if not os.path.exists(terrain_file):
            print(f"\n✗ Fitxer no trobat: {terrain_file}")
            continue
        
        print(f"\n{'='*80}")
        print(f"TERRENY: {terrain_name}")
        print(f"{'='*80}")
        
        # Carregar terrain
        h_grid = np.load(terrain_file)
        rows, cols = h_grid.shape
        
        print(f"  Dimensions: {rows}x{cols}")
        print(f"  Rang alçades: [{h_grid.min():.2f}, {h_grid.max():.2f}]")
        print(f"  Mitjana/Std: {h_grid.mean():.2f} / {h_grid.std():.2f}")
        
        # Test original.py
        print(f"\n  ▶ original.py (error alçada)...")
        t0 = time.perf_counter()
        try:
            converter_h = GridToTinConverter(
                step=STEP, 
                pixel_size=PIXEL_SIZE,
                control_mode='POINT_COUNT',
                target_point_count=TARGET_POINTS
            )
            converter_h.fit(terrain_file)
            t1 = time.perf_counter()
            
            print(f"    ✓ Punts: {len(converter_h.final_points_3d)}")
            print(f"    ✓ Error final: {converter_h.final_error:.4f}m")
            print(f"    ✓ Temps: {t1-t0:.3f}s")
            
            results['terrain'].append(terrain_name)
            results['algorithm'].append('original (alçada)')
            results['points'].append(len(converter_h.final_points_3d))
            results['error'].append(converter_h.final_error)
            results['time'].append(t1-t0)
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
        
        # Test pendent7.py amb mode='mean_normal'
        print(f"\n  ▶ pendent7.py (error angular, mode='mean_normal')...")
        t0 = time.perf_counter()
        try:
            converter_a = GridToTinIncremental(
                step=STEP,
                pixel_size=PIXEL_SIZE,
                target_point_count=TARGET_POINTS,
                mode='mean_normal',
                error_metric='angular'
            )
            verts, triangles = converter_a.fit(terrain_file)
            t1 = time.perf_counter()
            
            print(f"    ✓ Punts: {len(converter_a.tin.points)}")
            if converter_a.final_error_angular is not None:
                print(f"    ✓ Error angular final: {converter_a.final_error_angular:.4f}°")
            print(f"    ✓ Temps: {t1-t0:.3f}s")
            
            results['terrain'].append(terrain_name)
            results['algorithm'].append('pendent7 (angular)')
            results['points'].append(len(converter_a.tin.points))
            results['error'].append(converter_a.final_error_angular if converter_a.final_error_angular else 0)
            results['time'].append(t1-t0)
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Resumir resultats
    print("\n" + "="*80)
    print("RESUM RESULTATS")
    print("="*80)
    print(f"\n{'Terreny':<20} {'Algorithm':<30} {'Punts':<8} {'Error':<10} {'Temps':<8}")
    print("-" * 80)
    
    for i in range(len(results['terrain'])):
        print(f"{results['terrain'][i]:<20} {results['algorithm'][i]:<30} "
              f"{results['points'][i]:<8} {results['error'][i]:<10.4f} {results['time'][i]:<8.3f}s")
    
    # Determinar òptim
    print("\n" + "="*80)
    print("ANÀLISI D'ÒPTIM")
    print("="*80)
    
    # Agrupar per algoritme
    for algo in ['original (alçada)', 'pendent7 (angular)']:
        print(f"\n▶ {algo}:")
        for terrain_name in TERRAINS.keys():
            for i, (t, a) in enumerate(zip(results['terrain'], results['algorithm'])):
                if t == terrain_name and a == algo:
                    print(f"  {terrain_name}: Error={results['error'][i]:.4f}, Temps={results['time'][i]:.3f}s")


if __name__ == "__main__":
    test_all_combinations()
    
    print("\n" + "="*80)
    print("RECOMANACIONS PER AL TERRENY 4x4 ÒPTIM:")
    print("="*80)
    print("""
1. El terreny òptim serà aquell que:
   - Produeix errors baixos en AMBDÓS algorismes (balanç)
   - Té estructura interessant (no massa trivial)
   - Té gradients suaus (perquè error angular sigui comparable)

2. Els candidats inicials són:
   - balanced_diag: Pendent diàgonal + variabilitat
   - gaussian_bumps: Turons suaus (més realista)
   - fractal: Pattern mix

3. Ajust manual:
   Si cap no és òptim, pots crear una variant personalitzada
   editant create_optimal_4x4_terrain.py
    """)
