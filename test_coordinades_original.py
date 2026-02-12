import numpy as np

# Simular el que fa original.py
h = np.load('bassiero.npy')
print(f"Shape del fitxer original: {h.shape}")

step = 1
pixel_size = 2.0

h_sampled = h[::step, ::step]
rows, cols = h_sampled.shape
print(f"Shape després del submostreig: {rows} x {cols}")

# SISTEMA ANTIC (INCORRECTE)
print("\n" + "="*60)
print("SISTEMA ANTIC (linspace):")
x_old = np.linspace(0, h.shape[1], cols)
y_old = np.linspace(0, h.shape[0], rows)
print(f"  X: de {x_old[0]:.2f} a {x_old[-1]:.2f}")
print(f"  Y: de {y_old[0]:.2f} a {y_old[-1]:.2f}")
print(f"  Distància entre punts X: {x_old[1] - x_old[0]:.2f} metres")
print(f"  Distància entre punts Y: {y_old[1] - y_old[0]:.2f} metres")

# SISTEMA NOU (CORRECTE)
print("\n" + "="*60)
print("SISTEMA NOU (spacing):")
spacing = step * pixel_size
r_indices = np.arange(rows)
c_indices = np.arange(cols)
rr, cc = np.meshgrid(r_indices, c_indices, indexing='ij')
xx = cc * spacing
yy = rr * spacing

print(f"  X: de {xx[0,0]:.2f} a {xx[0,-1]:.2f}")
print(f"  Y: de {yy[0,0]:.2f} a {yy[-1,0]:.2f}")
print(f"  Distància entre punts X: {spacing:.2f} metres")
print(f"  Distància entre punts Y: {spacing:.2f} metres")

print("\n" + "="*60)
print("CONCLUSIÓ:")
print(f"  El pixel_size especificat és: {pixel_size} metres")
print(f"  El sistema NOU respecta aquest valor: {spacing == pixel_size}")
print(f"  El sistema ANTIC NO el respectava: {(x_old[1] - x_old[0]) == pixel_size}")
print("="*60)
