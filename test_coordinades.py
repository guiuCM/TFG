import numpy as np
from original import GridToTinConverter
from pendent6 import GridToTinIncremental

# Paràmetres
FILENAME = 'bassiero.npy'
STEP = 1
PIXEL_SIZE = 2.0
TARGET_POINTS = 10  # Només 10 punts per veure-ho clar

print("=" * 60)
print("TEST DE COORDENADES")
print("=" * 60)

# --- ORIGINAL.PY ---
print("\n>> Testing original.py...")
converter_orig = GridToTinConverter(step=STEP, pixel_size=PIXEL_SIZE, 
                                    control_mode='POINT_COUNT', target_point_count=TARGET_POINTS)
converter_orig.fit(FILENAME)

print(f"Nombre de punts: {len(converter_orig.final_points_3d)}")
print("Primers 5 punts (X, Y, Z):")
for i in range(min(5, len(converter_orig.final_points_3d))):
    x, y, z = converter_orig.final_points_3d[i]
    print(f"  Punt {i}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

# --- PENDENT6.PY ---
print("\n>> Testing pendent6.py...")
converter_pend = GridToTinIncremental(step=STEP, pixel_size=PIXEL_SIZE, target_point_count=TARGET_POINTS)
verts, triangles = converter_pend.fit(FILENAME)

print(f"Nombre de punts: {len(converter_pend.tin.points)}")
print(f"Nombre de Z values: {len(converter_pend.tin_z_values)}")
print("Primers 5 punts (X, Y) i Z:")
for i in range(min(5, len(converter_pend.tin.points))):
    x, y = converter_pend.tin.points[i]
    z = converter_pend.tin_z_values[i]
    print(f"  Punt {i}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

# --- COMPARACIÓ ---
print("\n>> Comparació de rangs:")
h_full = np.load(FILENAME)
h_grid = h_full[::STEP, ::STEP]
rows, cols = h_grid.shape
max_x_esperat = (cols - 1) * STEP * PIXEL_SIZE
max_y_esperat = (rows - 1) * STEP * PIXEL_SIZE

print(f"Grid: {rows} x {cols}")
print(f"Coordenades esperades:")
print(f"  X: 0 a {max_x_esperat:.2f}")
print(f"  Y: 0 a {max_y_esperat:.2f}")
print(f"  Z: {h_grid.min():.2f} a {h_grid.max():.2f}")

print(f"\noriginal.py:")
print(f"  X: {converter_orig.final_points_3d[:, 0].min():.2f} a {converter_orig.final_points_3d[:, 0].max():.2f}")
print(f"  Y: {converter_orig.final_points_3d[:, 1].min():.2f} a {converter_orig.final_points_3d[:, 1].max():.2f}")
print(f"  Z: {converter_orig.final_points_3d[:, 2].min():.2f} a {converter_orig.final_points_3d[:, 2].max():.2f}")

print(f"\npendent6.py:")
print(f"  X: {converter_pend.tin.points[:, 0].min():.2f} a {converter_pend.tin.points[:, 0].max():.2f}")
print(f"  Y: {converter_pend.tin.points[:, 1].min():.2f} a {converter_pend.tin.points[:, 1].max():.2f}")
print(f"  Z: {min(converter_pend.tin_z_values):.2f} a {max(converter_pend.tin_z_values):.2f}")

print("\n" + "=" * 60)
