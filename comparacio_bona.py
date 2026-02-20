import numpy as np
import matplotlib.pyplot as plt
from original import GridToTinConverter
from pendent6 import GridToTinIncremental
import time


FILENAME = 'bassiero.npy'
STEP = 1
PIXEL_SIZE = 2.0
TARGET_POINTS = 2000

def calculate_stats_for_model(model_tin, h_grid, rows, cols, step, pixel_size):
    spacing = step * pixel_size
    
    # Grella de coordenades
    r_indices, c_indices = np.arange(rows), np.arange(cols)
    rr, cc = np.meshgrid(r_indices, c_indices, indexing='ij')
    x_coords = cc.ravel() * spacing
    y_coords = rr.ravel() * spacing
    query_xy = np.column_stack((x_coords, y_coords))
    
    # Trobar triangle per a cada punt
    simplex_ids = model_tin.find_simplex(query_xy)
    
    # Pendent real (magnitud del gradient)
    dy, dx = np.gradient(h_grid, spacing, spacing)
    real_slope = np.sqrt(dx**2 + dy**2).ravel()
    
    ids = simplex_ids
    slopes = real_slope
    
    # Agrupar per ID de triangle
    order = np.argsort(ids)
    ids_sorted = ids[order]
    slopes_sorted = slopes[order]
    
    unique_ids, split_idx = np.unique(ids_sorted, return_index=True)
    groups = np.split(slopes_sorted, split_idx[1:])
    
    means = [np.mean(g) for g in groups if len(g) > 0]
    stds = [np.std(g) for g in groups if len(g) > 0]
    
    return means, stds



if __name__ == "__main__":
    
    print("COMPARACIÓ original.PY vs PENDENT.PY\n")
    # original.py
    t0 = time.perf_counter()
    converter_h = GridToTinConverter(step=STEP, control_mode='POINT_COUNT', target_point_count=TARGET_POINTS)
    converter_h.fit(FILENAME)
    t1 = time.perf_counter()
    print(f"   Temps: {t1 - t0:.2f}s, Punts: {len(converter_h.final_points_3d)}, Error final: {converter_h.final_error:.2f}m")
    
    # Carrega grid
    h_full = np.load(FILENAME)
    h_grid_h = h_full[::STEP, ::STEP]
    rows_h, cols_h = h_grid_h.shape
    
    means_h, stds_h = calculate_stats_for_model(converter_h.tin, 
                                                 h_grid_h, rows_h, cols_h, STEP, PIXEL_SIZE)
    
    # pendent.py
    t0 = time.perf_counter()
    converter_a = GridToTinIncremental(step=STEP, pixel_size=PIXEL_SIZE, target_point_count=TARGET_POINTS)
    verts, triangles = converter_a.fit(FILENAME)
    t1 = time.perf_counter()
    print(f"   Temps: {t1 - t0:.2f}s, Punts: {len(converter_a.tin.points)}")
    
    means_a, stds_a = calculate_stats_for_model(converter_a.tin,
                                                 converter_a.h_grid, converter_a.rows, converter_a.cols, STEP, PIXEL_SIZE)
    
    #boxplots
    print("\n>> Generant boxplots...")
    fig = plt.figure(figsize=(14, 6))
    plt.suptitle(f"Comparació original.py vs Pendent6.py ({TARGET_POINTS} punts, step={STEP})", fontsize=14)
    
    # Mitjana Pendent
    ax1 = plt.subplot(1, 2, 1)
    bp1 = ax1.boxplot([means_h, means_a], labels=['original (Alçada)', 'Pendent6 (Angle)'], patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    ax1.scatter([1] * len(means_h), means_h, alpha=0.3, s=20, color='blue')
    ax1.scatter([2] * len(means_a), means_a, alpha=0.3, s=20, color='red')
    ax1.set_title("Mitjana Pendent per Triangle", fontsize=12)
    ax1.set_ylabel("Pendent mig")
    ax1.grid(True, alpha=0.3)
    
    # Desviació Estàndard
    ax2 = plt.subplot(1, 2, 2)
    bp2 = ax2.boxplot([stds_h, stds_a], labels=['original (Alçada)', 'Pendent6 (Angle)'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    ax2.scatter([1] * len(stds_h), stds_h, alpha=0.3, s=20, color='blue')
    ax2.scatter([2] * len(stds_a), stds_a, alpha=0.3, s=20, color='red')
    ax2.set_title("Desviació Estàndard Pendent (Consistència)", fontsize=12)
    ax2.set_ylabel("Std Dev Pendent")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacio_simple.png', dpi=150)
    print("✓ Gràfica guardada: 'comparacio_simple.png'")
    
    # Report
    print("\n" + "="*60)
    print("ESTADÍSTIQUES DELS BOXPLOTS")
    print("="*60)
    print(f"{'MÈTRICA':<20} | {'Mitjana (mín-màx)':<25} | {'Std (mín-màx)':<25}")
    print("-"*60)
    print(f"{'original (Alçada)':<20} | {np.mean(means_h):.3f} ({np.min(means_h):.3f}-{np.max(means_h):.3f}) | {np.mean(stds_h):.3f} ({np.min(stds_h):.3f}-{np.max(stds_h):.3f})")
    print(f"{'Pendent6 (Angle)':<20} | {np.mean(means_a):.3f} ({np.min(means_a):.3f}-{np.max(means_a):.3f}) | {np.mean(stds_a):.3f} ({np.min(stds_a):.3f}-{np.max(stds_a):.3f})")
    print("="*60)
    
    plt.show()
