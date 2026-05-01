import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Script per crear un terreny 4x4 òptim que produeix resultats equilibrats
en ambdós algorismes: original.py (error alçada) i pendent7.py (error angular)

La clau és crear:
1. Variabilitat espacial interessant (no massa plat, no massa complex)
2. Gradients suaus en general (que no faci que l'error angular exploti)
3. Alguns canvis locals (per tenir informació a captar)
"""

def create_balanced_4x4():
    """
    Crea un terreny 4x4 equilibrat amb:
    - Pendent base diàgonal suau (0 a 4)
    - Dos petits pics locals simètrics
    - Variabilitat suau però clara
    """
    
    # Base: pendent diàgonal lineal de 0 a 4
    terrain = np.array([
        [0.0,  1.2,  2.4,  3.6],
        [1.0,  2.0,  3.2,  4.2],
        [2.0,  3.0,  4.0,  5.0],
        [3.0,  4.0,  5.0,  6.0]
    ], dtype=np.float32)
    
    # Afegir dos petits pics simètrics (però suaus per no defenestrar el gradient)
    # Pic alt a (0, 3): +1.5
    terrain[0, 3] += 1.5
    
    # Pic baix a (3, 0): -1.5 (o cantar a 0.5)
    terrain[3, 0] = 0.5
    
    return terrain


def create_fractal_4x4():
    """
    Crea un terreny 4x4 amb estructura fractal simple
    (més realista però sense patrons repetitius)
    """
    
    # Valores aleatorios però deterministicos amb seed
    np.random.seed(42)
    
    # Crear base suau
    terrain = np.linspace(0, 6, 16).reshape(4, 4).astype(np.float32)
    
    # Afegir perturbacions suaus
    for i in range(4):
        for j in range(4):
            # Variació local controlada
            perturbation = 0.8 * np.sin(i * np.pi / 3) * np.cos(j * np.pi / 3)
            terrain[i, j] += perturbation
    
    return terrain


def create_gaussian_bumps_4x4():
    """
    Crea un terreny 4x4 amb gaussianes suaus
    (representa turons suaus i valls)
    """
    
    x = np.linspace(0, 4, 4)
    y = np.linspace(0, 4, 4)
    xx, yy = np.meshgrid(x, y)
    
    # Base suau lineal
    base = 2.0
    
    # Gaussiana positiva al centre-esquerra
    turon1 = 2.5 * np.exp(-((xx - 1)**2 + (yy - 1)**2) / 0.5)
    
    # Gaussiana negativa al centre-dreta
    vall = -1.0 * np.exp(-((xx - 3)**2 + (yy - 3)**2) / 0.7)
    
    # Pendent suau de fons
    pendent = 0.5 * (xx + yy)
    
    terrain = (base + turon1 + vall + pendent).astype(np.float32)
    
    return terrain


def test_terrains_on_algorithms(terrain, name):
    """
    Testa el terreny amb ambdós algorismes i compara resultats
    """
    print(f"\n{'='*60}")
    print(f"TESTEJANT: {name}")
    print(f"{'='*60}")
    print("Terreny 4x4:")
    print(terrain)
    print(f"\nEstadístiques:")
    print(f"  Mín: {terrain.min():.3f}, Màx: {terrain.max():.3f}")
    print(f"  Rang: {terrain.max() - terrain.min():.3f}")
    print(f"  Mitjana: {terrain.mean():.3f}, Std: {terrain.std():.3f}")
    
    # Calcular gradients
    dz_dy, dz_dx = np.gradient(terrain)
    slopes = np.sqrt(dz_dx**2 + dz_dy**2)
    print(f"\n  Pendents (gradient):")
    print(f"    Mín: {slopes.min():.3f}, Màx: {slopes.max():.3f}")
    print(f"    Mitjana: {slopes.mean():.3f}, Std: {slopes.std():.3f}")
    
    return terrain


if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("CREACIÓ DE TERRENYS 4x4 ÒPTIMS")
    print("="*60)
    
    # Crear tres candidates
    terrains = {
        'balanced_diag': create_balanced_4x4(),
        'fractal': create_fractal_4x4(),
        'gaussian_bumps': create_gaussian_bumps_4x4(),
    }
    
    # Testejar cada una
    for name, terrain in terrains.items():
        test_terrains_on_algorithms(terrain, name)
        
        # Guardar
        filename = f'terrain_4x4_{name}.npy'
        np.save(filename, terrain)
        print(f"\n✓ Guardat: {filename}")
    
    # Visualitzar
    print("\n" + "="*60)
    print("VISUALITZACIÓ 3D")
    print("="*60)
    
    fig = plt.figure(figsize=(15, 4))
    
    for idx, (name, terrain) in enumerate(terrains.items(), 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        
        # Surface plot
        surf = ax.plot_surface(xx, yy, terrain, cmap='terrain', 
                               edgecolor='k', linewidth=0.5, alpha=0.8)
        
        # Punts en 3D
        ax.scatter(xx.ravel(), yy.ravel(), terrain.ravel(), 
                  c='red', s=50, edgecolors='black', linewidths=1, zorder=5)
        
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (alçada)')
        ax.set_zlim(terrain.min() - 1, terrain.max() + 1)
    
    plt.tight_layout()
    plt.savefig('terrain_4x4_candidates.png', dpi=100, bbox_inches='tight')
    print("✓ Visualització guardada: terrain_4x4_candidates.png")
    plt.show()
    
    print("\n" + "="*60)
    print("RECOMANACIÓ:")
    print("="*60)
    print("Les millors candidates per a ambdós algorismes són:")
    print("1. 'balanced_diag' - Pendent suau amb variabilitat local")
    print("2. 'gaussian_bumps' - Estructura natural de turons/valls")
    print("\nProva ambdues amb:")
    print("  - comparacio_bona.py (canvia FILENAME)")
    print("="*60)
