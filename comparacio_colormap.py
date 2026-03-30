import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from original import GridToTinConverter
from pendent7 import GridToTinIncremental
import time


FILENAME = 'bassiero.npy'
STEP = 1
PIXEL_SIZE = 2.0
TARGET_POINTS = 500

#passa de nou la malla a pixels, per a que les coordenades del TIN estiguin en metres i no en índexs de la matriu
def interpolate_tin_heights(tin, z_values, rows, cols, step, pixel_size):

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
    #Calcula el pendent (magnitud del gradient) a partir d'una matriu d'alçades.
    dy, dx = np.gradient(h_grid, spacing, spacing)
    return np.sqrt(dx**2 + dy**2)


def calculate_slope_from_tin_normals(tin, z_values, rows, cols, step, pixel_size):
    #Calcula el pendent directament des de les normals dels triangles del TIN, sense interpolar alçades primer.
    spacing = step * pixel_size
    
    # Crear coordenades de tots els punts del grid
    r_indices, c_indices = np.arange(rows), np.arange(cols)
    rr, cc = np.meshgrid(r_indices, c_indices, indexing='ij')
    x_coords = cc.ravel() * spacing
    y_coords = rr.ravel() * spacing
    query_xy = np.column_stack((x_coords, y_coords))
    
    # Trobar triangle per a cada punt
    simplex_ids = tin.find_simplex(query_xy)
    
    # Calcular normals de tots els triangles
    z_array = np.array(z_values)
    tris = tin.simplices
    pts = tin.points
    
    p0 = np.column_stack((pts[tris[:, 0], 0], pts[tris[:, 0], 1], z_array[tris[:, 0]]))
    p1 = np.column_stack((pts[tris[:, 1], 0], pts[tris[:, 1], 1], z_array[tris[:, 1]]))
    p2 = np.column_stack((pts[tris[:, 2], 0], pts[tris[:, 2], 1], z_array[tris[:, 2]]))
    
    u = p1 - p0
    v = p2 - p0
    normals = np.cross(u, v)
    
    # Normalitzar
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    
    # Assegurar que apunta amunt
    flip = normals[:, 2] < 0
    normals[flip] *= -1
    
    # Calcular pendent per cada triangle: slope = sqrt(nx² + ny²) / nz
    # Atenció: això és magnitud del gradient, equivalent a tan(angle)
    triangle_slopes = np.sqrt(normals[:, 0]**2 + normals[:, 1]**2) / (normals[:, 2] + 1e-10)
    
    # Assignar pendent del triangle a cada píxel
    slopes_flat = np.full(len(query_xy), np.nan)
    valid = simplex_ids != -1
    slopes_flat[valid] = triangle_slopes[simplex_ids[valid]]
    
    return slopes_flat.reshape(rows, cols)


def plot_comparison_colormaps(h_grid_real, h_grid_tin, slope_real, slope_tin, title_prefix, filename):
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
    
    # --- MODEL 1: original.py (error d'alçada) ---
    print("\n>> Executant original.py (criteri: alçada)...")
    t0 = time.perf_counter()
    converter_h = GridToTinConverter(step=STEP, pixel_size=PIXEL_SIZE, control_mode='POINT_COUNT', target_point_count=TARGET_POINTS)
    converter_h.fit(FILENAME)
    t1 = time.perf_counter()
    print(f"   Temps: {t1 - t0:.2f}s, Punts: {len(converter_h.final_points_3d)}")
    
    # DEBUG: Mostrar coordenades
    print(f"   [DEBUG] Coordenades original.py:")
    print(f"     X range: {converter_h.final_points_3d[:, 0].min():.2f} - {converter_h.final_points_3d[:, 0].max():.2f}")
    print(f"     Y range: {converter_h.final_points_3d[:, 1].min():.2f} - {converter_h.final_points_3d[:, 1].max():.2f}")
    print(f"     Z range: {converter_h.final_points_3d[:, 2].min():.2f} - {converter_h.final_points_3d[:, 2].max():.2f}")
    
    # Interpolar alçades del TIN
    h_tin_original = interpolate_tin_heights(
        converter_h.tin, 
        [p[2] for p in converter_h.final_points_3d],  # Z values
        rows, cols, STEP, PIXEL_SIZE
    )
    # Per original.py calculem pendent del grid interpolat (com sempre)
    slope_tin_original = calculate_slope_from_heights(h_tin_original, spacing)
    
    # --- MODEL 2: Pendent7.py (error angular) ---
    MODE_PENDENT7 = 'mean_normal'  # canvia a 'point' o 'triangle' per comparar altres modes

    print(f"\n>> Executant Pendent7.py (mode={MODE_PENDENT7})...")
    t0 = time.perf_counter()
    converter_a = GridToTinIncremental(step=STEP, pixel_size=PIXEL_SIZE, target_point_count=TARGET_POINTS, mode=MODE_PENDENT7)
    verts, triangles = converter_a.fit(FILENAME)
    t1 = time.perf_counter()
    print(f"   Temps: {t1 - t0:.2f}s, Punts: {len(converter_a.tin.points)}")
    
    # DEBUG: Mostrar coordenades
    print(f"   [DEBUG] Coordenades pendent7.py:")
    print(f"     X range: {converter_a.tin.points[:, 0].min():.2f} - {converter_a.tin.points[:, 0].max():.2f}")
    print(f"     Y range: {converter_a.tin.points[:, 1].min():.2f} - {converter_a.tin.points[:, 1].max():.2f}")
    print(f"     Z range: {min(converter_a.tin_z_values):.2f} - {max(converter_a.tin_z_values):.2f}")
    
    # Interpolar alçades del TIN
    h_tin_pendent = interpolate_tin_heights(
        converter_a.tin,
        converter_a.tin_z_values,
        rows, cols, STEP, PIXEL_SIZE
    )
    # Per pendent6.py calculem pendent DIRECTAMENT de les normals del TIN
    slope_tin_pendent = calculate_slope_from_tin_normals(
        converter_a.tin,
        converter_a.tin_z_values,
        rows, cols, STEP, PIXEL_SIZE
    )
    
    # --- GENERAR COLORMAPS ---
    print("\n>> Generant colormaps...")
    
    diff_h_original, diff_s_original = plot_comparison_colormaps(
        h_grid_real, h_tin_original, slope_real, slope_tin_original,
        "original.py (Alçada)", "colormap_original.png"
    )
    
    diff_h_pendent, diff_s_pendent = plot_comparison_colormaps(
        h_grid_real, h_tin_pendent, slope_real, slope_tin_pendent,
        f"Pendent7.py ({MODE_PENDENT7})", f"colormap_pendent7_{MODE_PENDENT7}.png"
    )
    
    # --- COMPARACIÓ DIRECTA ---
    print("\n>> Generant comparació directa...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Comparació Directa: original vs Pendent7 [{MODE_PENDENT7}] ({TARGET_POINTS} punts)", fontsize=14)
    
    # Diferència alçades - original
    max_h = max(np.nanmax(np.abs(diff_h_original)), np.nanmax(np.abs(diff_h_pendent)))
    norm_h = TwoSlopeNorm(vmin=-max_h, vcenter=0, vmax=max_h)
    
    im1 = axes[0, 0].imshow(diff_h_original, cmap='RdBu_r', norm=norm_h, origin='lower')
    axes[0, 0].set_title(f'original.py - Error Alçada\nRMSE: {np.sqrt(np.nanmean(diff_h_original**2)):.2f} m')
    plt.colorbar(im1, ax=axes[0, 0], label='Diferència (m)')
    
    im2 = axes[0, 1].imshow(diff_h_pendent, cmap='RdBu_r', norm=norm_h, origin='lower')
    axes[0, 1].set_title(f'Pendent7.py [{MODE_PENDENT7}] - Error Alçada\nRMSE: {np.sqrt(np.nanmean(diff_h_pendent**2)):.2f} m')
    plt.colorbar(im2, ax=axes[0, 1], label='Diferència (m)')
    
    # Diferència pendents
    max_s = max(np.nanmax(np.abs(diff_s_original)), np.nanmax(np.abs(diff_s_pendent)))
    norm_s = TwoSlopeNorm(vmin=-max_s, vcenter=0, vmax=max_s)
    
    im3 = axes[1, 0].imshow(diff_s_original, cmap='RdBu_r', norm=norm_s, origin='lower')
    axes[1, 0].set_title(f'original.py - Error Pendent\nRMSE: {np.sqrt(np.nanmean(diff_s_original**2)):.4f}')
    plt.colorbar(im3, ax=axes[1, 0], label='Diferència pendent')
    
    im4 = axes[1, 1].imshow(diff_s_pendent, cmap='RdBu_r', norm=norm_s, origin='lower')
    axes[1, 1].set_title(f'Pendent7.py [{MODE_PENDENT7}] - Error Pendent\nRMSE: {np.sqrt(np.nanmean(diff_s_pendent**2)):.4f}')
    plt.colorbar(im4, ax=axes[1, 1], label='Diferència pendent')
    
    plt.tight_layout()
    plt.savefig('colormap_comparacio.png', dpi=150)
    print("✓ Gràfica guardada: 'colormap_comparacio.png'")
    
    # --- RESUM FINAL ---
    print("\n" + "=" * 60)
    print("RESUM COMPARATIU")
    print("=" * 60)
    print(f"{'Mètrica':<25} | {'original.py':<15} | {f'Pendent7 [{MODE_PENDENT7}]':<15}")
    print("-" * 60)
    print(f"{'RMSE Alçada (m)':<25} | {np.sqrt(np.nanmean(diff_h_original**2)):<15.4f} | {np.sqrt(np.nanmean(diff_h_pendent**2)):<15.4f}")
    print(f"{'MAE Alçada (m)':<25} | {np.nanmean(np.abs(diff_h_original)):<15.4f} | {np.nanmean(np.abs(diff_h_pendent)):<15.4f}")
    print(f"{'Max Alçada (m)':<25} | {np.nanmax(np.abs(diff_h_original)):<15.4f} | {np.nanmax(np.abs(diff_h_pendent)):<15.4f}")
    print(f"{'RMSE Pendent':<25} | {np.sqrt(np.nanmean(diff_s_original**2)):<15.6f} | {np.sqrt(np.nanmean(diff_s_pendent**2)):<15.6f}")
    print(f"{'MAE Pendent':<25} | {np.nanmean(np.abs(diff_s_original)):<15.6f} | {np.nanmean(np.abs(diff_s_pendent)):<15.6f}")
    print(f"{'Max Pendent':<25} | {np.nanmax(np.abs(diff_s_original)):<15.6f} | {np.nanmax(np.abs(diff_s_pendent)):<15.6f}")
    print("=" * 60)
    
    plt.show()
