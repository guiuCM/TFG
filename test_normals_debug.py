import numpy as np
from pendent6 import GridToTinIncremental

# Crear conversor
conv = GridToTinIncremental(step=1, pixel_size=2.0)
conv._load_data('bassiero.npy')

print("=" * 60)
print("TEST DE NORMALS DEL GRID")
print("=" * 60)

# Agafar un punt al mig
r, c = 750, 750
normal = conv.normal_grid[r, c]
print(f"\nPunt (row={r}, col={c}):")
print(f"  Normal: nx={normal[0]:.4f}, ny={normal[1]:.4f}, nz={normal[2]:.4f}")
print(f"  Magnitud: {np.linalg.norm(normal):.4f} (hauria de ser 1.0)")

# Calcular pendent a partir de la normal
slope_from_normal = np.degrees(np.arctan(np.sqrt(normal[0]**2 + normal[1]**2) / normal[2]))
print(f"  Pendent calculat des de la normal: {slope_from_normal:.2f} graus")

# Calcular pendent directament del gradient
spacing = conv.step * conv.pixel_size
dz_dy, dz_dx = np.gradient(conv.h_grid, spacing, spacing)
slope_direct = np.degrees(np.arctan(np.sqrt(dz_dx[r,c]**2 + dz_dy[r,c]**2)))
print(f"  Pendent calculat directament: {slope_direct:.2f} graus")

# Hauria de coincidir
print(f"  Diferència: {abs(slope_from_normal - slope_direct):.6f} graus")

# Comprovar varis punts
print("\n" + "=" * 60)
print("MOSTREIG DE NORMALS EN DIFERENTS PUNTS")
print("=" * 60)

test_points = [
    (0, 0, "Cantonada sup-esq"),
    (750, 750, "Centre"),
    (1499, 1499, "Cantonada inf-dreta"),
    (100, 750, "Vora esquerra"),
]

for r, c, desc in test_points:
    normal = conv.normal_grid[r, c]
    z = conv.h_grid[r, c]
    slope = np.degrees(np.arctan(np.sqrt(normal[0]**2 + normal[1]**2) / normal[2]))
    print(f"{desc:20s} | Z={z:7.2f} | Pendent={slope:5.2f}° | Normal=({normal[0]:6.3f}, {normal[1]:6.3f}, {normal[2]:6.3f})")
