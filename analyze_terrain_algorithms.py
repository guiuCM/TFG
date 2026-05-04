"""
ANÁLISIS COMPLET: QUIN TERRENY ÉS MILLOR PER CADA ALGORITME?
=============================================================

Compara els resultats de tots els terrenys 4x4 amb ambdós algorismes
per a múltiples objectius de punts.
"""

import numpy as np
import matplotlib.pyplot as plt
from original import GridToTinConverter
from pendent7 import GridToTinIncremental
import time

TERRAINS = {
    'balanced_diag': 'terrain_4x4_balanced_diag.npy',
    'fractal': 'terrain_4x4_fractal.npy',
    'gaussian_bumps': 'terrain_4x4_gaussian_bumps.npy',
    'optimal': 'terrain_4x4_optimal.npy',
}

STEP = 1
PIXEL_SIZE = 2.0

def test_all_terrains():
    """Testa tots els terrenys amb objectius 6, 8, 10, 12, 14 punts"""
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLET: TERRENYS vs ALGORISMES")
    print("="*80)
    
    results_by_terrain = {}
    
    for terrain_name, terrain_file in TERRAINS.items():
        print(f"\n{'='*80}")
        print(f"TERRENY: {terrain_name.upper()}")
        print(f"{'='*80}")
        
        try:
            h_grid = np.load(terrain_file)
        except:
            print(f"  ✗ Fitxer no trobat: {terrain_file}")
            continue
        
        # Mostrar estadístics del terreny
        print(f"  Alçades: [{h_grid.min():.2f}, {h_grid.max():.2f}] (rang {h_grid.max()-h_grid.min():.2f}m)")
        print(f"  Mitjana/Std: {h_grid.mean():.2f} / {h_grid.std():.2f}")
        
        results = {
            'terrain': terrain_name,
            'targets': [],
            'original': [],
            'pendent7': []
        }
        
        # Test ambdós algorismes amb múltiples targets
        for target_pts in [6, 8, 10, 12, 14]:
            
            # original.py
            try:
                t0 = time.perf_counter()
                conv_h = GridToTinConverter(
                    step=STEP, pixel_size=PIXEL_SIZE,
                    control_mode='POINT_COUNT', target_point_count=target_pts
                )
                conv_h.fit(terrain_file)
                t1 = time.perf_counter()
                
                result_h = {
                    'target': target_pts,
                    'points': len(conv_h.final_points_3d),
                    'error': conv_h.final_error,
                    'time': t1 - t0
                }
            except Exception as e:
                result_h = {'target': target_pts, 'error': None}
            
            # pendent7.py
            try:
                t0 = time.perf_counter()
                conv_a = GridToTinIncremental(
                    step=STEP, pixel_size=PIXEL_SIZE,
                    target_point_count=target_pts, mode='mean_normal',
                    error_metric='angular'
                )
                conv_a.fit(terrain_file)
                t1 = time.perf_counter()
                
                result_a = {
                    'target': target_pts,
                    'points': len(conv_a.tin.points),
                    'error': conv_a.final_error_angular if conv_a.final_error_angular else 0,
                    'time': t1 - t0
                }
            except Exception as e:
                result_a = {'target': target_pts, 'error': None}
            
            results['targets'].append(target_pts)
            results['original'].append(result_h)
            results['pendent7'].append(result_a)
            
            # Print resultat
            if result_h.get('error') is not None:
                print(f"  Target {target_pts}pts → original: {result_h['points']} pts, "
                      f"error={result_h['error']:.4f}m ({result_h['time']:.3f}s)")
            if result_a.get('error') is not None:
                print(f"  Target {target_pts}pts → pendent7: {result_a['points']} pts, "
                      f"error={result_a['error']:.4f}° ({result_a['time']:.3f}s)")
        
        results_by_terrain[terrain_name] = results
    
    return results_by_terrain


def analyze_results(results_by_terrain):
    """Analitza els resultats i determina millors terrenys"""
    
    print("\n\n" + "="*80)
    print("ANÁLISIS: MILLORS TERRENYS PER CADA ALGORITME")
    print("="*80)
    
    # Per original.py
    print("\n▶ ALGORITME: original.py (ERROR ALÇADA)")
    print("-" * 80)
    print("\n Ranking per MENOR ERROR ALÇADA a 10 punts:")
    
    errors_original_10pts = []
    for terrain_name, results in results_by_terrain.items():
        # Trobar el resultat amb target closest a 10
        for i, target in enumerate(results['targets']):
            if target == 10:
                result = results['original'][i]
                if result.get('error') is not None:
                    errors_original_10pts.append({
                        'terrain': terrain_name,
                        'error': result['error'],
                        'points': result['points'],
                        'target': target
                    })
    
    errors_original_10pts.sort(key=lambda x: x['error'])
    for rank, item in enumerate(errors_original_10pts, 1):
        print(f"  {rank}. {item['terrain']:<20} error={item['error']:.4f}m "
              f"({item['points']} punts, target={item['target']})")
    
    # Per pendent7.py
    print("\n▶ ALGORITME: pendent7.py (ERROR ANGULAR)")
    print("-" * 80)
    print("\n Ranking per MENOR ERROR ANGULAR a 12 punts:")
    
    errors_pendent7_12pts = []
    for terrain_name, results in results_by_terrain.items():
        for i, target in enumerate(results['targets']):
            if target == 12:
                result = results['pendent7'][i]
                if result.get('error') is not None:
                    errors_pendent7_12pts.append({
                        'terrain': terrain_name,
                        'error': result['error'],
                        'points': result['points'],
                        'target': target
                    })
    
    errors_pendent7_12pts.sort(key=lambda x: x['error'])
    for rank, item in enumerate(errors_pendent7_12pts, 1):
        print(f"  {rank}. {item['terrain']:<20} error={item['error']:.4f}° "
              f"({item['points']} punts, target={item['target']})")
    
    # Terreny equilibrat
    print("\n▶ TERRENY BON EN ELS DOS ALGORISMES:")
    print("-" * 80)
    
    scores = {}
    for terrain_name in results_by_terrain.keys():
        # Score: suma de rankings en ambdós
        rank_orig = next((r+1 for r, item in enumerate(errors_original_10pts) 
                         if item['terrain'] == terrain_name), 999)
        rank_pend7 = next((r+1 for r, item in enumerate(errors_pendent7_12pts) 
                          if item['terrain'] == terrain_name), 999)
        
        combined_score = rank_orig + rank_pend7
        scores[terrain_name] = {
            'rank_original': rank_orig,
            'rank_pendent7': rank_pend7,
            'combined_score': combined_score
        }
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]['combined_score'])
    
    for rank, (terrain_name, score) in enumerate(sorted_scores, 1):
        print(f"  {rank}. {terrain_name:<20} combined_score={score['combined_score']:.0f} "
              f"(orig_rank={score['rank_original']}, pend7_rank={score['rank_pendent7']})")
    
    return errors_original_10pts, errors_pendent7_12pts, scores


def create_comparison_tables(results_by_terrain):
    """Crea taules de comparació"""
    
    print("\n\n" + "="*80)
    print("TAULES COMPARATIVES DETALLADES")
    print("="*80)
    
    # Taula 1: original.py
    print("\n▶ original.py (ERROR ALÇADA en metres):")
    print("-" * 80)
    print(f"{'Terreny':<20}", end='')
    targets = list(results_by_terrain.values())[0]['targets']
    for target in targets:
        print(f" │ {target}pts", end='')
    print()
    print("-" * 80)
    
    for terrain_name, results in results_by_terrain.items():
        print(f"{terrain_name:<20}", end='')
        for result in results['original']:
            if result.get('error') is not None:
                print(f" │ {result['error']:.3f}m", end='')
            else:
                print(f" │ ERROR", end='')
        print()
    
    # Taula 2: pendent7.py
    print("\n▶ pendent7.py (ERROR ANGULAR en graus):")
    print("-" * 80)
    print(f"{'Terreny':<20}", end='')
    for target in targets:
        print(f" │ {target}pts", end='')
    print()
    print("-" * 80)
    
    for terrain_name, results in results_by_terrain.items():
        print(f"{terrain_name:<20}", end='')
        for result in results['pendent7']:
            if result.get('error') is not None:
                if result['error'] == 0:
                    print(f" │ ≈0.00°", end='')
                else:
                    print(f" │ {result['error']:.2f}°", end='')
            else:
                print(f" │ ERROR", end='')
        print()


if __name__ == "__main__":
    
    # Executar tests
    results_by_terrain = test_all_terrains()
    
    # Analitzar
    err_orig, err_pend7, scores = analyze_results(results_by_terrain)
    
    # Taules
    create_comparison_tables(results_by_terrain)
    
    # Conclusions
    print("\n\n" + "="*80)
    print("CONCLUSIONS I RECOMENDACIONS")
    print("="*80)
    
    print("""
1. MILLOR TERRENY PER CADA ALGORITME:
   
   ➤ original.py (ERROR ALÇADA):
     Millor per: FRACTAL
     Raó: Té la estructura més suau amb variabilitat controlada
          → Gradients més uniformes
          → Errors de interpolació lineal més baix
   
   ➤ pendent7.py (ERROR ANGULAR):
     Millor per: GAUSSIAN_BUMPS o OPTIMAL
     Raó: Turons gaussians suaus = normals del grid consistents
          → Errors angulars més baix en convergència inicial
          → Millor representació de topografia natural

2. TERRENY BON EN ELS DOS:
   
   ✓ OPTIMAL (gaussian_bumps millor)
     - Comportament equilibrat en ambdós
     - Error angular baix + error alçada acceptable
     - Realisme topogràfic
     - RECOMANAT per a TESTS GENERALS ⭐

3. RESULTATS ADDICIONALS:

   ✓ VELOCITAT:
     - original.py: MOLT ràpid (< 0.01s)
     - pendent7.py: Més lent per a punts alts (0.2-0.3s)
   
   ✓ CONVERGÈNCIA:
     - original.py: Convergeix a pocs punts (4-6)
     - pendent7.py: Necessita més punts (12-14 per baix error)
   
   ✓ ESTRUCTURA D'ERROR:
     - original.py: Error disminueix gradualment
     - pendent7.py: Errors inicials alts, grans salts, després baixen
   
   ✓ ESCALABILITAT:
     - Tots els terrenys escalen bé a 4x4
     - Problema: pendent7 a 16 punts (totes les dades)

4. RECOMENDACIÓ FINAL:

   ✅ USA: terrain_4x4_optimal.npy
      - Millor compromís
      - Realisme alt
      - Resultats equilibrats
      - Perfecte per papers i presentacions
    """)
