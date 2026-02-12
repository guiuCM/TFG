import numpy as np
from pendent6 import GridToTinIncremental

# Test amb només 20 punts
converter = GridToTinIncremental(step=1, pixel_size=2.0, target_point_count=20)

# Carregar dades
import numpy as np
h_full = np.load('bassiero.npy')
h_grid = h_full[::1, ::1]
print(f"Grid shape: {h_grid.shape}")
print(f"Z range del grid complet: {h_grid.min():.2f} - {h_grid.max():.2f}")

# Executar
verts, triangles = converter.fit('bassiero.npy')

print(f"\nZ range del TIN: {min(converter.tin_z_values):.2f} - {max(converter.tin_z_values):.2f}")
print(f"Nombre de punts al TIN: {len(converter.tin.points)}")

# Verificar les primeres iteracions: comprovar si els errors són correctes
print("\n" + "="*60)
print("ANÀLISI: Per què els errors són tan grans?")
print("="*60)

# Agafem un punt candidat aleatori i calculem el seu error
if len(converter.tin.points) > 5:
    # Coordenades d'un punt del mig del grid
    test_row, test_col = 750, 750
    test_idx = test_row * converter.cols + test_col
    
    print(f"\nPunt de prova: fila={test_row}, col={test_col}, idx={test_idx}")
    print(f"Coordenades: X={test_col * 2.0:.2f}, Y={test_row * 2.0:.2f}")
    print(f"Z real: {h_grid[test_row, test_col]:.2f}")
    
    # Normal real
    real_normal = converter.normal_grid[test_row, test_col]
    print(f"Normal real: {real_normal}")
    
    # Trobar en quin triangle està
    x = test_col * 2.0
    y = test_row * 2.0
    simplex_id = converter.tin.find_simplex(np.array([[x, y]]))[0]
    
    if simplex_id != -1:
        print(f"Triangle ID: {simplex_id}")
        
        # Vèrtexs del triangle
        tri_verts = converter.tin.simplices[simplex_id]
        print(f"Índexs dels vèrtexs: {tri_verts}")
        
        # Coordenades XYZ dels vèrtexs
        for i, v_idx in enumerate(tri_verts):
            xy = converter.tin.points[v_idx]
            z = converter.tin_z_values[v_idx]
            print(f"  Vèrtex {i}: X={xy[0]:.2f}, Y={xy[1]:.2f}, Z={z:.2f}")
        
        # Calcular normal del triangle
        tris_xy = converter.tin.points[tri_verts]
        tris_z = np.array(converter.tin_z_values)[tri_verts]
        
        p0 = np.array([tris_xy[0, 0], tris_xy[0, 1], tris_z[0]])
        p1 = np.array([tris_xy[1, 0], tris_xy[1, 1], tris_z[1]])
        p2 = np.array([tris_xy[2, 0], tris_xy[2, 1], tris_z[2]])
        
        u = p1 - p0
        v = p2 - p0
        tin_normal = np.cross(u, v)
        tin_normal = tin_normal / np.linalg.norm(tin_normal)
        
        if tin_normal[2] < 0:
            tin_normal *= -1
        
        print(f"\nNormal del TIN: {tin_normal}")
        
        # Càlcul de l'angle
        dot = np.dot(real_normal, tin_normal)
        print(f"Dot product: {dot:.4f}")
        angle_rad = np.arccos(np.clip(dot, -1, 1))
        angle_deg = np.degrees(angle_rad)
        print(f"Angle d'error: {angle_deg:.2f} graus")
        
        # Interpretació
        if angle_deg > 90:
            print("⚠️ PROBLEMA: L'angle és > 90°, les normals apunten en direccions oposades!")
        elif angle_deg > 45:
            print("⚠️ ATENCIÓ: L'angle és gran (> 45°)")
        else:
            print("✓ L'angle sembla raonable")
    else:
        print("El punt està fora del TIN")
