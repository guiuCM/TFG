import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from classes import GridToTinConverter
from pendent6 import GridToTinIncremental
import time


FILENAME = 'bassiero.npy'
STEP = 1
PIXEL_SIZE = 2.0
TARGET_POINTS = 500


def interpolate_tin_heights(tin, z_values, rows, cols, step, pixel_size):
    """
    Interpola les alçades del TIN a tots els punts del grid.
    Retorna una matriu 2D amb les alçades interpolades.
    """
    spacing = step * pixel_size
    z_array = np.array(z_values)
    
    # Crear coordenades de tots els punts del grid
    r_indices, c_indices = np.arange(rows), np.arange(cols)
    rr, cc = np.meshgrid(r_indices, c_indices, indexing='ij')
    x_coords = cc.ravel() * spacing
    y_coords = rr.ravel() * spacing
    query_xy = np.column_stack((x_coords, y_coords))
    
    # Trobar triangle per a cada punt
    simplex_ids = tin.find_simplex(query_xy)
    
    # Inicialitzar alçades amb NaN
    z_interpolated = np.full(len(query_xy), np.nan)
    
    valid = simplex_ids != -1
    if not np.any(valid):
        return z_interpolated.reshape(rows, cols)
    
    p_val = query_xy[valid]
    s_val = simplex_ids[valid]
    
    # Coordenades baricèntriques
    b = tin.transform[s_val, :2]
    c = tin.transform[s_val, 2]
    w = np.einsum('ijk,ik->ij', b, p_val - c)
    w = np.c_[w, 1 - w.sum(axis=1)]
    
    # Índexs dels vèrtexs
    verts_indices = tin.simplices[s_val]
    z_tri = z_array[verts_indices]
    
    # Interpolació
    z_interpolated[valid] = np.einsum('ij,ij->i', w, z_tri)
    
    return z_interpolated.reshape(rows, cols)


def calculate_slope_from_heights(h_grid, spacing):
    """
    Calcula el pendent (magnitud del gradient) a partir d'una matriu d'alçades.
    """
    dy, dx = np.gradient(h_grid, spacing, spacing)
    return np.sqrt(dx**2 + dy**2)


def plot_comparison_colormaps(h_grid_real, h_grid_tin, slope_real, slope_tin, title_prefix, filename):
    """
    Genera colormaps de diferències d'alçada i pendent.
    """
    # Diferències
    diff_height = h_grid_tin - h_grid_real
    diff_slope = slope_tin - slope_real
    
    # Màscara per NaN (punts fora del TIN)
    valid_mask = ~np.isnan(h_grid_tin)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{title_prefix} - Comparativa amb {TARGET_POINTS} punts", fontsize=14)
    
    # --- Fila 1: Alçades ---
    # Alçada real
    im1 = axes[0, 0].imshow(h_grid_real, cmap='terrain', origin='lower')
    axes[0, 0].set_title('Alçada Real (Grid)')
    plt.colorbar(im1, ax=axes[0, 0], label='Alçada (m)')
    
    # Alçada TIN interpolada
    im2 = axes[0, 1].imshow(h_grid_tin, cmap='terrain', origin='lower')
    axes[0, 1].set_title('Alçada TIN (Interpolada)')
    plt.colorbar(im2, ax=axes[0, 1], label='Alçada (m)')
    
    # Diferència d'alçades (centrada a 0)
    max_diff_h = np.nanmax(np.abs(diff_height))
    if max_diff_h > 0:
        norm_h = TwoSlopeNorm(vmin=-max_diff_h, vcenter=0, vmax=max_diff_h)
    else:
        norm_h = None
    im3 = axes[0, 2].imshow(diff_height, cmap='RdBu_r', norm=norm_h, origin='lower')
    axes[0, 2].set_title(f'Diferència Alçada (TIN - Real)\nRMSE: {np.sqrt(np.nanmean(diff_height**2)):.2f} m')
    plt.colorbar(im3, ax=axes[0, 2], label='Diferència (m)')
    
    # --- Fila 2: Pendents ---
    # Pendent real
    vmax_slope = max(np.nanmax(slope_real), np.nanmax(slope_tin))
    im4 = axes[1, 0].imshow(slope_real, cmap='viridis', origin='lower', vmin=0, vmax=vmax_slope)
    axes[1, 0].set_title('Pendent Real')
    plt.colorbar(im4, ax=axes[1, 0], label='Pendent (m/m)')
    
    # Pendent TIN
    im5 = axes[1, 1].imshow(slope_tin, cmap='viridis', origin='lower', vmin=0, vmax=vmax_slope)
    axes[1, 1].set_title('Pendent TIN')
    plt.colorbar(im5, ax=axes[1, 1], label='Pendent (m/m)')
    
    # Diferència de pendents (centrada a 0)
    max_diff_s = np.nanmax(np.abs(diff_slope))
    if max_diff_s > 0:
        norm_s = TwoSlopeNorm(vmin=-max_diff_s, vcenter=0, vmax=max_diff_s)
    else:
        norm_s = None
    im6 = axes[1, 2].imshow(diff_slope, cmap='RdBu_r', norm=norm_s, origin='lower')
    axes[1, 2].set_title(f'Diferència Pendent (TIN - Real)\nRMSE: {np.sqrt(np.nanmean(diff_slope**2)):.4f}')
    plt.colorbar(im6, ax=axes[1, 2], label='Diferència pendent')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"✓ Gràfica guardada: '{filename}'")
    
    # Estadístiques
    print(f"\n  Estadístiques {title_prefix}:")
    print(f"    Alçada - RMSE: {np.sqrt(np.nanmean(diff_height**2)):.4f} m")
    print(f"    Alçada - MAE:  {np.nanmean(np.abs(diff_height)):.4f} m")
    print(f"    Alçada - Max:  {np.nanmax(np.abs(diff_height)):.4f} m")
    print(f"    Pendent - RMSE: {np.sqrt(np.nanmean(diff_slope**2)):.6f}")
    print(f"    Pendent - MAE:  {np.nanmean(np.abs(diff_slope)):.6f}")
    print(f"    Pendent - Max:  {np.nanmax(np.abs(diff_slope)):.6f}")
    
    return diff_height, diff_slope


if __name__ == "__main__":
    
    print("=" * 60)
    print("COMPARACIÓ COLORMAP: ALÇADES I PENDENTS")
    print("=" * 60)
    
    # Carregar grid real
    h_full = np.load(FILENAME)
    h_grid_real = h_full[::STEP, ::STEP]
    rows, cols = h_grid_real.shape
    spacing = STEP * PIXEL_SIZE
    
    # Pendent real
    slope_real = calculate_slope_from_heights(h_grid_real, spacing)
    
    print(f"\nGrid: {rows}x{cols} punts, spacing={spacing}m")
    
    # --- MODEL 1: Classes.py (error d'alçada) ---
    print("\n>> Executant Classes.py (criteri: alçada)...")
    t0 = time.perf_counter()
    converter_h = GridToTinConverter(step=STEP, control_mode='POINT_COUNT', target_point_count=TARGET_POINTS)
    converter_h.fit(FILENAME)
    t1 = time.perf_counter()
    print(f"   Temps: {t1 - t0:.2f}s, Punts: {len(converter_h.final_points_3d)}")
    
    # Interpolar alçades del TIN
    h_tin_classes = interpolate_tin_heights(
        converter_h.tin, 
        [p[2] for p in converter_h.final_points_3d],  # Z values
        rows, cols, STEP, PIXEL_SIZE
    )
    slope_tin_classes = calculate_slope_from_heights(h_tin_classes, spacing)
    
    # --- MODEL 2: Pendent6.py (error angular) ---
    print("\n>> Executant Pendent6.py (criteri: angle)...")
    t0 = time.perf_counter()
    converter_a = GridToTinIncremental(step=STEP, pixel_size=PIXEL_SIZE, target_point_count=TARGET_POINTS)
    verts, triangles = converter_a.fit(FILENAME)
    t1 = time.perf_counter()
    print(f"   Temps: {t1 - t0:.2f}s, Punts: {len(converter_a.tin.points)}")
    
    # Interpolar alçades del TIN
    h_tin_pendent = interpolate_tin_heights(
        converter_a.tin,
        converter_a.tin_z_values,
        rows, cols, STEP, PIXEL_SIZE
    )
    slope_tin_pendent = calculate_slope_from_heights(h_tin_pendent, spacing)
    
    # --- GENERAR COLORMAPS ---
    print("\n>> Generant colormaps...")
    
    diff_h_classes, diff_s_classes = plot_comparison_colormaps(
        h_grid_real, h_tin_classes, slope_real, slope_tin_classes,
        "Classes.py (Alçada)", "colormap_classes.png"
    )
    
    diff_h_pendent, diff_s_pendent = plot_comparison_colormaps(
        h_grid_real, h_tin_pendent, slope_real, slope_tin_pendent,
        "Pendent6.py (Angle)", "colormap_pendent6.png"
    )
    
    # --- COMPARACIÓ DIRECTA ---
    print("\n>> Generant comparació directa...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Comparació Directa: Classes vs Pendent6 ({TARGET_POINTS} punts)", fontsize=14)
    
    # Diferència alçades - Classes
    max_h = max(np.nanmax(np.abs(diff_h_classes)), np.nanmax(np.abs(diff_h_pendent)))
    norm_h = TwoSlopeNorm(vmin=-max_h, vcenter=0, vmax=max_h)
    
    im1 = axes[0, 0].imshow(diff_h_classes, cmap='RdBu_r', norm=norm_h, origin='lower')
    axes[0, 0].set_title(f'Classes.py - Error Alçada\nRMSE: {np.sqrt(np.nanmean(diff_h_classes**2)):.2f} m')
    plt.colorbar(im1, ax=axes[0, 0], label='Diferència (m)')
    
    im2 = axes[0, 1].imshow(diff_h_pendent, cmap='RdBu_r', norm=norm_h, origin='lower')
    axes[0, 1].set_title(f'Pendent6.py - Error Alçada\nRMSE: {np.sqrt(np.nanmean(diff_h_pendent**2)):.2f} m')
    plt.colorbar(im2, ax=axes[0, 1], label='Diferència (m)')
    
    # Diferència pendents
    max_s = max(np.nanmax(np.abs(diff_s_classes)), np.nanmax(np.abs(diff_s_pendent)))
    norm_s = TwoSlopeNorm(vmin=-max_s, vcenter=0, vmax=max_s)
    
    im3 = axes[1, 0].imshow(diff_s_classes, cmap='RdBu_r', norm=norm_s, origin='lower')
    axes[1, 0].set_title(f'Classes.py - Error Pendent\nRMSE: {np.sqrt(np.nanmean(diff_s_classes**2)):.4f}')
    plt.colorbar(im3, ax=axes[1, 0], label='Diferència pendent')
    
    im4 = axes[1, 1].imshow(diff_s_pendent, cmap='RdBu_r', norm=norm_s, origin='lower')
    axes[1, 1].set_title(f'Pendent6.py - Error Pendent\nRMSE: {np.sqrt(np.nanmean(diff_s_pendent**2)):.4f}')
    plt.colorbar(im4, ax=axes[1, 1], label='Diferència pendent')
    
    plt.tight_layout()
    plt.savefig('colormap_comparacio.png', dpi=150)
    print("✓ Gràfica guardada: 'colormap_comparacio.png'")
    
    # --- RESUM FINAL ---
    print("\n" + "=" * 60)
    print("RESUM COMPARATIU")
    print("=" * 60)
    print(f"{'Mètrica':<25} | {'Classes.py':<15} | {'Pendent6.py':<15}")
    print("-" * 60)
    print(f"{'RMSE Alçada (m)':<25} | {np.sqrt(np.nanmean(diff_h_classes**2)):<15.4f} | {np.sqrt(np.nanmean(diff_h_pendent**2)):<15.4f}")
    print(f"{'MAE Alçada (m)':<25} | {np.nanmean(np.abs(diff_h_classes)):<15.4f} | {np.nanmean(np.abs(diff_h_pendent)):<15.4f}")
    print(f"{'Max Alçada (m)':<25} | {np.nanmax(np.abs(diff_h_classes)):<15.4f} | {np.nanmax(np.abs(diff_h_pendent)):<15.4f}")
    print(f"{'RMSE Pendent':<25} | {np.sqrt(np.nanmean(diff_s_classes**2)):<15.6f} | {np.sqrt(np.nanmean(diff_s_pendent**2)):<15.6f}")
    print(f"{'MAE Pendent':<25} | {np.nanmean(np.abs(diff_s_classes)):<15.6f} | {np.nanmean(np.abs(diff_s_pendent)):<15.6f}")
    print(f"{'Max Pendent':<25} | {np.nanmax(np.abs(diff_s_classes)):<15.6f} | {np.nanmax(np.abs(diff_s_pendent)):<15.6f}")
    print("=" * 60)
    
    plt.show()
