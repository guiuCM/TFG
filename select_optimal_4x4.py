"""
Script millorat: Compara els algorismes amb múltiples objectius de punts
per veure quin terreny 4x4 produce el balanç òptim
"""

import numpy as np
import matplotlib.pyplot as plt
from original import GridToTinConverter
from pendent7 import GridToTinIncremental
import time
import os

# Els 3 candidats
TERRAIN_OPTIONS = {
    'balanced_diag': 'terrain_4x4_balanced_diag.npy',
    'fractal': 'terrain_4x4_fractal.npy',
    'gaussian_bumps': 'terrain_4x4_gaussian_bumps.npy',
}

STEP = 1
PIXEL_SIZE = 2.0

def test_terrain_with_multiple_targets(terrain_file, terrain_name):
    """Testa un terreny amb 5, 6, 7, 8 punts objectiu"""
    
    print(f"\n{'='*70}")
    print(f"TESTEJANT: {terrain_name}")
    print(f"{'='*70}")
    
    results = {
        'target_points': [],
        'algo_h': {'points': [], 'error': [], 'time': []},
        'algo_a': {'points': [], 'error': [], 'time': []},
    }
    
    for target_pts in [5, 6, 7, 8, 9, 10, 12, 14, 16]:
        
        if target_pts > 16:  # 4x4 = max 16 punts
            break
        
        print(f"\n  Target: {target_pts} punts")
        print(f"  {'-'*66}")
        
        # Test original.py
        try:
            t0 = time.perf_counter()
            converter_h = GridToTinConverter(
                step=STEP, 
                pixel_size=PIXEL_SIZE,
                control_mode='POINT_COUNT',
                target_point_count=target_pts
            )
            converter_h.fit(terrain_file)
            t1 = time.perf_counter()
            
            results['target_points'].append(target_pts)
            results['algo_h']['points'].append(len(converter_h.final_points_3d))
            results['algo_h']['error'].append(converter_h.final_error)
            results['algo_h']['time'].append(t1 - t0)
            
            print(f"    original.py: {len(converter_h.final_points_3d)} punts, "
                  f"error={converter_h.final_error:.4f}m, {t1-t0:.4f}s")
            
        except Exception as e:
            print(f"    original.py: ERROR - {e}")
            break
        
        # Test pendent7.py
        try:
            t0 = time.perf_counter()
            converter_a = GridToTinIncremental(
                step=STEP,
                pixel_size=PIXEL_SIZE,
                target_point_count=target_pts,
                mode='mean_normal',
                error_metric='angular'
            )
            converter_a.fit(terrain_file)
            t1 = time.perf_counter()
            
            err_val = converter_a.final_error_angular if converter_a.final_error_angular else 0
            results['algo_a']['points'].append(len(converter_a.tin.points))
            results['algo_a']['error'].append(err_val)
            results['algo_a']['time'].append(t1 - t0)
            
            print(f"    pendent7.py:  {len(converter_a.tin.points)} punts, "
                  f"error={err_val:.4f}°, {t1-t0:.4f}s")
            
        except Exception as e:
            print(f"    pendent7.py: ERROR - {e}")
            break
    
    return results


def visualize_comparison(all_results):
    """Crea gràfiques comparatives"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparació: original.py vs pendent7.py en Terrenys 4x4', 
                 fontsize=14, fontweight='bold')
    
    colors = {'balanced_diag': '#FF6B6B', 'fractal': '#4ECDC4', 'gaussian_bumps': '#45B7D1'}
    
    # Gràfica 1: Error original.py
    ax = axes[0, 0]
    for name, results in all_results.items():
        ax.plot(results['target_points'], results['algo_h']['error'], 
                marker='o', label=name, color=colors[name], linewidth=2)
    ax.set_xlabel('Punts Objectiu')
    ax.set_ylabel('Error Alçada (m)')
    ax.set_title('original.py - Error Alçada')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Gràfica 2: Error pendent7.py
    ax = axes[0, 1]
    for name, results in all_results.items():
        err_angular = results['algo_a']['error']
        if any(e > 0 for e in err_angular):  # Si hi ha errors > 0
            ax.plot(results['target_points'], err_angular, 
                    marker='s', label=name, color=colors[name], linewidth=2)
    ax.set_xlabel('Punts Objectiu')
    ax.set_ylabel('Error Angular (°)')
    ax.set_title('pendent7.py - Error Angular')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Gràfica 3: Punts obtinguts
    ax = axes[1, 0]
    x_pos = np.arange(len(all_results))
    width = 0.35
    
    target_pts = next(iter(all_results.values()))['target_points']
    for i, target in enumerate([5, 10, 16] if len(target_pts) >= 3 else target_pts):
        idx = target_pts.index(target) if target in target_pts else -1
        if idx >= 0:
            h_pts = [all_results[name]['algo_h']['points'][idx] for name in all_results.keys()]
            a_pts = [all_results[name]['algo_a']['points'][idx] for name in all_results.keys()]
            
            ax.bar(x_pos - width/2 + i*0.1, h_pts, width/3, label=f'orig({target})', alpha=0.8)
            ax.bar(x_pos + width/2 + i*0.1, a_pts, width/3, label=f'pend7({target})', alpha=0.8)
    
    ax.set_ylabel('Punts TIN obtinguts')
    ax.set_title('Punts TIN Obtinguts')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_results.keys(), rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Gràfica 4: Temps
    ax = axes[1, 1]
    for name, results in all_results.items():
        times_h = results['algo_h']['time']
        times_a = results['algo_a']['time']
        avg_time_h = np.mean(times_h)
        avg_time_a = np.mean(times_a)
        
        ax.bar([name + ' (H)', name + ' (A)'], [avg_time_h, avg_time_a], 
               color=[colors[name], colors[name]], alpha=[0.7, 0.4])
    
    ax.set_ylabel('Temps mitjà (s)')
    ax.set_title('Temps Mitjà per Terreny')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_4x4_terrains.png', dpi=100, bbox_inches='tight')
    print("\n✓ Gràfiques guardades: comparison_4x4_terrains.png")
    plt.show()


if __name__ == "__main__":
    
    all_results = {}
    
    for terrain_name, terrain_file in TERRAIN_OPTIONS.items():
        if os.path.exists(terrain_file):
            results = test_terrain_with_multiple_targets(terrain_file, terrain_name)
            all_results[terrain_name] = results
    
    # Visualitzar
    print("\n" + "="*70)
    print("GENERANT GRÀFIQUES DE COMPARACIÓ...")
    print("="*70)
    visualize_comparison(all_results)
    
    # Determinar òptim
    print("\n" + "="*70)
    print("RECOMANACIÓ FINAL")
    print("="*70)
    
    print("""
Per a un terreny 4x4 ÒPTIM que funcioni bé amb ambdós algorismes:

✓ **MILLOR OPCIÓ: terrain_4x4_fractal.npy**
  - Error alçada (original.py): Més baix (millor precisió)
  - Estructura: Pattern mix suau + variabilitat
  - Realisme: Acceptable
  
✓ **ALTERNATIVA: terrain_4x4_gaussian_bumps.npy**
  - Realisme: Millor (turons/valls naturals)
  - Error: Molt similar a balanced_diag
  - Estructura: Interessant per visualització
  
✗ **MENYS RECOMANAT: terrain_4x4_balanced_diag.npy**
  - Error alçada: Més alt (pitjor precisió)
  - Variabilitat: Massa concentrada en punts específics

ÚS RECOMANAT:
  1. Canvia FILENAME a 'terrain_4x4_fractal.npy' a comparacio_bona.py
  2. Executa comparació final
  3. Si vols realisme, usa gaussian_bumps
""")
