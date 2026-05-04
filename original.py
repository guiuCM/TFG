import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import time
import os


class GridToTinConverter:
    
    def __init__(self, step=25, pixel_size=2.0, control_mode='POINT_COUNT', 
                 target_error_percentage=0.5, target_point_count=500):
        
        self.step = step
        self.pixel_size = pixel_size
        self.control_mode = control_mode
        self.target_error_percentage = target_error_percentage
        self.target_point_count = target_point_count
        self.all_points_3d = None
        self.num_total_points = 0
        self.elevation_range = 0
        self.rows = 0
        self.cols = 0
        self.slope_grid = None
        
        # Resultats
        self.final_points_3d = None
        self.tin = None
        self.final_error = 0.0

    def _load_and_sample_grid(self, npy_file_path):
        try:
            h = np.load(npy_file_path)
            #h.shape = (1500, 1500)
        except FileNotFoundError:
            print(f"Error: No s'ha trobat '{npy_file_path}'")
            return
            
        #print(f"Elevació Mín: {h.min()}, Elevació Màx: {h.max()}")
        self.elevation_range = h.max() - h.min()
        
        # Submostreig igual que pendent6.py
        h_sampled = h[::self.step, ::self.step]
        self.rows, self.cols = h_sampled.shape
        
        # Sistema de coordenades igual que pendent6.py
        spacing = self.step * self.pixel_size
        r_indices = np.arange(self.rows)
        c_indices = np.arange(self.cols)
        rr, cc = np.meshgrid(r_indices, c_indices, indexing='ij')
        
        # x = col * spacing, y = row * spacing
        xx = cc * spacing
        yy = rr * spacing

        self.slope_grid = np.degrees(np.arctan(np.sqrt(np.gradient(h_sampled, spacing, axis=0)**2 + np.gradient(h_sampled, spacing, axis=1)**2)))
        
        self.all_points_3d = np.vstack([xx.ravel(), yy.ravel(), h_sampled.ravel()]).T
        self.num_total_points = self.all_points_3d.shape[0]
        #print(f"Grid submostrejat a {self.num_total_points} punts (step={self.step}).")

    def _triangle_median_slopes(self, temp_tin):
        """Calcula la mediana dels pendents per a cada triangle del TIN."""
        if temp_tin is None or len(temp_tin.simplices) == 0:
            return np.array([])

        rows_grid, cols_grid = np.indices((self.rows, self.cols))
        spacing = self.step * self.pixel_size
        x = cols_grid * spacing
        y = rows_grid * spacing
        sample_points = np.column_stack((x.ravel(), y.ravel()))
        slope_values = self.slope_grid.ravel()

        medians = np.empty(len(temp_tin.simplices), dtype=float)
        for tri_idx, simplex in enumerate(temp_tin.simplices):
            triangle = temp_tin.points[simplex]
            mask = Path(triangle).contains_points(sample_points, radius=1e-9)
            if np.any(mask):
                medians[tri_idx] = float(np.median(slope_values[mask]))
            else:
                p0 = self.all_points_3d[simplex[0]]
                p1 = self.all_points_3d[simplex[1]]
                p2 = self.all_points_3d[simplex[2]]
                normal = np.cross(p1 - p0, p2 - p0)
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normal = normal / norm
                    if normal[2] < 0:
                        normal *= -1
                    medians[tri_idx] = float(np.degrees(np.arctan(np.sqrt(normal[0]**2 + normal[1]**2) / (normal[2] + 1e-12))))
                else:
                    medians[tri_idx] = 0.0

        return medians

    def fit(self, npy_file_path, snapshot_dir='comparacions_original', snapshot_interval=5):
  
        self._load_and_sample_grid(npy_file_path)
        
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
        
        if self.control_mode == 'ERROR':
            max_error_threshold = (self.target_error_percentage / 100.0) * self.elevation_range
            #print(f"MODO: Límit per Error. Objectiu: {max_error_threshold:.2f} unitats")
        elif self.control_mode == 'POINT_COUNT':
            max_error_threshold = 0.0 # Error 0, mai s'assolirà
            max_points_threshold = self.target_point_count
            #print(f"MODO: Límit per Punts. Objectiu: {max_points_threshold} vèrtexs.")
        else:
            raise ValueError("control_mode ha de ser 'ERROR' o 'POINT_COUNT'")

        corner_indices = [0, self.cols - 1, (self.rows - 1) * self.cols, self.rows * self.cols - 1]
        S_indices = list(set(corner_indices)) # Punts al TIN (índexs)
        P_indices = list(set(range(self.num_total_points)) - set(S_indices)) # Punts candidats
        
        
        iteration = 1
        while True:
            if not P_indices:
                print("\nProcés completat. S'han afegit tots els punts disponibles.")
                break

            current_S_points_2d = self.all_points_3d[S_indices, :2]
            current_S_points_z = self.all_points_3d[S_indices, 2]

            interpolator = LinearNDInterpolator(current_S_points_2d, current_S_points_z)
            points_to_check_2d = self.all_points_3d[P_indices, :2]
            interpolated_z = interpolator(points_to_check_2d)
            
            actual_z = self.all_points_3d[P_indices, 2]
            errors = np.nan_to_num(np.abs(actual_z - interpolated_z))
            
            max_err_local_index = np.argmax(errors)
            max_err = errors[max_err_local_index]
            self.final_error = max_err
            
            point_to_add_global_index = P_indices[max_err_local_index]
            
            print(f"Iter. {iteration}: Punts TIN = {len(S_indices)}, Error Màx. Actual = {max_err:.6f}")

            if max_err <= max_error_threshold:
                print(f"\nProcés completat (MODO ERROR). Error final: {max_err:.2f}")
                break
            if self.control_mode == 'POINT_COUNT' and len(S_indices) >= max_points_threshold:
                print(f"\nProcés completat (MODO PUNTS). Punts assolits: {len(S_indices)}")
                break
            
            # Continuar: Afegir el pitjor punt al TIN
            S_indices.append(point_to_add_global_index)
            P_indices.pop(max_err_local_index)
            
            # Generar snapshot si cal
            if snapshot_dir and iteration % snapshot_interval == 0:
                self._save_snapshot(snapshot_dir, iteration, S_indices)
            
            iteration += 1


        self.final_points_3d = self.all_points_3d[S_indices]
        self.tin = Delaunay(self.final_points_3d[:, :2])
        #print(f"TIN final generat amb {len(self.final_points_3d)} vèrtexs.")

    def fit_with_error_snapshots(self, npy_file_path, snapshot_dir='snapshots_error_original', snapshot_interval=5):

        self._load_and_sample_grid(npy_file_path)
        
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Carregar grid original per visualització
        h_full = np.load(npy_file_path)
        
        max_points_threshold = self.target_point_count
        
        corner_indices = [0, self.cols - 1, (self.rows - 1) * self.cols, self.rows * self.cols - 1]
        S_indices = list(set(corner_indices))
        P_indices = list(set(range(self.num_total_points)) - set(S_indices))
        
        iteration = 0
        spacing = self.step * self.pixel_size
        
        while len(S_indices) < max_points_threshold and P_indices:
            iteration += 1
            
            current_S_points_2d = self.all_points_3d[S_indices, :2]
            current_S_points_z = self.all_points_3d[S_indices, 2]
            interpolator = LinearNDInterpolator(current_S_points_2d, current_S_points_z)
            
            points_to_check_2d = self.all_points_3d[P_indices, :2]
            interpolated_z = interpolator(points_to_check_2d)
            
            actual_z = self.all_points_3d[P_indices, 2]
            errors = np.nan_to_num(np.abs(actual_z - interpolated_z))
            
            max_err_local_index = np.argmax(errors)
            max_err = errors[max_err_local_index]
            point_to_add_global_index = P_indices[max_err_local_index]
            
            if iteration % 10 == 0:
                print(f"  Iter {iteration}: Error màx = {max_err:.2f}m")
            
            # Generar snapshot d'error
            if iteration % snapshot_interval == 0:
                self._save_error_snapshot(snapshot_dir, iteration, S_indices, P_indices, 
                                         errors, max_err_local_index, spacing)
            
            S_indices.append(point_to_add_global_index)
            P_indices.pop(max_err_local_index)
        
        self.final_points_3d = self.all_points_3d[S_indices]
        self.tin = Delaunay(self.final_points_3d[:, :2])
        print(f"✓ {iteration} snapshots generats a {snapshot_dir}/")
    
    def _save_error_snapshot(self, snapshot_dir, iteration, S_indices, P_indices, 
                            errors, max_err_idx, spacing):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Crear mapa d'error (mostrejat cada 5 píxels)
        sample_step = 5
        sample_rows = self.rows // sample_step
        sample_cols = self.cols // sample_step
        error_grid = np.full((sample_rows, sample_cols), np.nan)
        
        for local_idx, global_idx in enumerate(P_indices):
            r = global_idx // self.cols
            c = global_idx % self.cols
            sr = r // sample_step
            sc = c // sample_step
            if 0 <= sr < sample_rows and 0 <= sc < sample_cols:
                if np.isnan(error_grid[sr, sc]):
                    error_grid[sr, sc] = errors[local_idx]
                else:
                    error_grid[sr, sc] = max(error_grid[sr, sc], errors[local_idx])
        
        # Mostrar error com a heatmap
        im = ax.imshow(error_grid, extent=[0, self.cols*spacing, 0, self.rows*spacing], 
                       origin='lower', cmap='hot_r', vmin=0, vmax=400, 
                       interpolation='nearest', alpha=0.9)
        
        # Dibuixar TIN actual
        current_points = self.all_points_3d[S_indices]
        tin = Delaunay(current_points[:, :2])
        ax.triplot(tin.points[:,0], tin.points[:,1], tin.simplices, 
                   color='cyan', linewidth=0.5, alpha=0.6)
        
        # Marcar punts TIN
        ax.scatter(tin.points[:,0], tin.points[:,1], c='white', s=15, 
                   edgecolors='black', linewidths=0.5, zorder=5, alpha=0.8)
        
        # Marcar últim punt afegit
        if len(S_indices) > 0:
            last_pt = current_points[-1]
            ax.scatter(last_pt[0], last_pt[1], c='lime', s=100, 
                       edgecolors='white', linewidths=2, zorder=10, marker='*')
        
        # Marcar punt amb màxim error
        if max_err_idx < len(P_indices):
            next_pt_idx = P_indices[max_err_idx]
            next_pt = self.all_points_3d[next_pt_idx]
            ax.scatter(next_pt[0], next_pt[1], c='red', s=100, 
                       edgecolors='yellow', linewidths=2, zorder=10, marker='X')
        
        plt.colorbar(im, ax=ax, label='Error alçada (m)')
        ax.set_title(f"original.py - ERROR D'ALÇADA | Iter {iteration} | Punts: {len(S_indices)}", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(0, self.cols*spacing)
        ax.set_ylim(0, self.rows*spacing)
        ax.set_aspect('equal', adjustable='box')
        
        filename = os.path.join(snapshot_dir, f'frame_{iteration:04d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

    def _save_snapshot(self, snapshot_dir, iteration, S_indices, vmin=2083, vmax=2902):
        """Guarda un snapshot del TIN actual amb escala fixa"""
        current_points = self.all_points_3d[S_indices]
        temp_tin = Delaunay(current_points[:, :2])
        median_slopes = self._triangle_median_slopes(temp_tin)
        if len(median_slopes) == 0:
            return
        slope_vmin = float(np.min(median_slopes))
        slope_vmax = float(np.max(median_slopes))
        if np.isclose(slope_vmin, slope_vmax):
            slope_vmin -= 1e-9
            slope_vmax += 1e-9

        fig, ax = plt.subplots(figsize=(10, 8))

        pts = temp_tin.points
        spacing = self.step * self.pixel_size
        width = self.cols * spacing
        height = self.rows * spacing

        # Rotate coordinates to match pendent7.py visualization
        pts_rot = np.column_stack((pts[:, 1], width - pts[:, 0]))
        polys = [pts_rot[simplex] for simplex in temp_tin.simplices]
        norm = Normalize(vmin=slope_vmin, vmax=slope_vmax)
        cmap = plt.get_cmap('coolwarm')

        poly_collection = PolyCollection(polys, cmap=cmap, norm=norm, edgecolors='black', linewidths=0.3)
        poly_collection.set_array(median_slopes)

        ax.add_collection(poly_collection)
        ax.autoscale_view()

        # Dibuixar edges amb coordenades rotades
        ax.triplot(pts_rot[:, 0], pts_rot[:, 1], temp_tin.simplices, 'k-', linewidth=0.3, alpha=0.5)

        # Últim punt afegit en vermell (rotat)
        if len(S_indices) > 0:
            last_pt = current_points[-1]
            last_x_rot = last_pt[1]
            last_y_rot = width - last_pt[0]
            ax.scatter(last_x_rot, last_y_rot, color='red', s=80,
                       edgecolors='white', zorder=10, linewidths=2)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Mediana del pendent (graus)')
        ax.set_title(f'original.py - Iter {iteration} | Punts: {len(S_indices)} | Mediana pendent',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(0, height)
        ax.set_ylim(0, width)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()

        filename = os.path.join(snapshot_dir, f'frame_{iteration:04d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

    def plot(self):
        if self.tin is None: 
            return

        median_slopes = self._triangle_median_slopes(self.tin)
        if len(median_slopes) == 0:
            print("No hi ha dades de pendent per dibuixar")
            return

        slope_vmin = float(np.min(median_slopes))
        slope_vmax = float(np.max(median_slopes))
        if np.isclose(slope_vmin, slope_vmax):
            slope_vmin -= 1e-9
            slope_vmax += 1e-9

        fig, ax = plt.subplots(figsize=(10, 8))
        pts = self.tin.points
        spacing = self.step * self.pixel_size
        width = self.cols * spacing
        height = self.rows * spacing

        # Rotate coordinates to match pendent7.py visualization
        pts_rot = np.column_stack((pts[:, 1], width - pts[:, 0]))
        polys = [pts_rot[simplex] for simplex in self.tin.simplices]
        norm = Normalize(vmin=slope_vmin, vmax=slope_vmax)
        cmap = plt.get_cmap('coolwarm')

        poly_collection = PolyCollection(polys, cmap=cmap, norm=norm, edgecolors='black', linewidths=0.2)
        poly_collection.set_array(median_slopes)

        ax.add_collection(poly_collection)
        ax.autoscale_view()
        ax.triplot(pts_rot[:, 0], pts_rot[:, 1], self.tin.simplices, 'k-', linewidth=0.2)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Mediana del pendent (graus)')
        ax.set_title(f"TIN Final - Mediana pendent (Punts: {len(self.tin.points)})")
        ax.set_xlim(0, height)
        ax.set_ylim(0, width)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        plt.show()
    


if __name__ == "__main__":
    

    # tin_builder_points = GridToTinConverter(
    #     step=20,  # Un 'step' més baix dona més punts per escollir
    #     pixel_size=2.0,
    #     control_mode='POINT_COUNT',
    #     target_point_count=700  # Volem un TIN amb 700 punts
    # )
    # tin_builder_points.fit('bassiero.npy')
    # tin_builder_points.plot()

    tin_builder_error = GridToTinConverter(
        step=1,
        pixel_size=2.0,
        control_mode='POINT_COUNT',
        target_point_count=1000  # Volem un TIN amb 2000 punts
    )

    t0 = time.perf_counter()
    tin_builder_error.fit('bassiero.npy', snapshot_dir='snapshots_comparacio_oscar_original', snapshot_interval=10)
    t1 = time.perf_counter()
    print(f"Temps total (carrega + refinament): {t1 - t0:.2f} s")
    tin_builder_error.plot()
