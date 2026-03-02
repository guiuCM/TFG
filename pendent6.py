import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial import Delaunay
import time
import os  #Per crear carpetes

class GridToTinIncremental:
    
    def __init__(self, step=25, pixel_size=2.0, target_point_count=500):
        
        self.step = step
        self.pixel_size = pixel_size
        self.target_point_count = target_point_count
        
        self.h_grid = None
        self.slope_grid = None
        self.rows = 0
        self.cols = 0
        
        # TIN
        self.tin = None
        self.tin_z_values = []
        
    def _load_data(self, npy_file_path):
        try:
            full_h = np.load(npy_file_path)
        except FileNotFoundError:
            print("Error: Fitxer no trobat.")
            return False

        self.h_grid = full_h[::self.step, ::self.step] # Submostreig
        self.rows, self.cols = self.h_grid.shape
        
        # Càlcul de gradients (derivades parcials)
        spacing = self.step * self.pixel_size 
        dz_dy, dz_dx = np.gradient(self.h_grid, spacing, spacing)
        
        # Càlcul de normals grid
        nx = -dz_dx
        ny = -dz_dy
        nz = 1
        
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        self.normal_grid = np.dstack((nx/norm, ny/norm, nz/norm))
        
        return True
    
    def _calculate_angular_error(self, candidate_indices):
        
        # normals reals
        rows, cols = np.divmod(candidate_indices, self.cols)
        real_normals = self.normal_grid[rows, cols]
        
        # normals TIN
        x = cols * self.step * self.pixel_size
        y = rows * self.step * self.pixel_size
        coords = np.column_stack((x, y))
        
        simplex_ids = self.tin.find_simplex(coords)
        
        # Agafem els índexs dels vèrtexs dels triangles
        tris_indices = self.tin.simplices[simplex_ids]
        tris_xy = self.tin.points[tris_indices]
        tris_z = np.array(self.tin_z_values)[tris_indices]
        
        # Agafem els 3 vertexs de cada triangle
        p0 = np.column_stack((tris_xy[:, 0, 0], tris_xy[:, 0, 1], tris_z[:, 0]))
        p1 = np.column_stack((tris_xy[:, 1, 0], tris_xy[:, 1, 1], tris_z[:, 1]))
        p2 = np.column_stack((tris_xy[:, 2, 0], tris_xy[:, 2, 1], tris_z[:, 2]))
        
        # Calculem la Normal del Triangle
        u = p1 - p0
        v = p2 - p0
        tin_normals = np.cross(u, v)
        
        # Normalitzem els vectors del TIN
        norms = np.linalg.norm(tin_normals, axis=1, keepdims=True)
        tin_normals = tin_normals / norms 
        #podria donar error de divisió per zero?
        #tin_normals = tin_normals / np.maximum(norms, 1e-10)
        
        # Assegurem que la Z sempre apunti amunt
        flip_vec = tin_normals[:, 2] < 0
        tin_normals[flip_vec] *= -1
        
        # CÀLCUL DE L'ERROR ANGULAR
        dot_product = np.einsum('ij,ij->i', real_normals, tin_normals) # Producte escalar
        
        # Limitar dot_product a [-1, 1] per evitar errors numèrics
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angles_rad = np.arccos(dot_product)
        angles_deg = np.degrees(angles_rad)
        
        return angles_deg
    
    def _calculate_triangle_areas(self):

        #Fórmula Shoelace: Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|

        points = self.tin.points
        simplices = self.tin.simplices
        
        # Coordenades dels 3 vèrtexs per a cada triangle: [N_triangles, 3, 2]
        tri_coords = points[simplices]
        
        x = tri_coords[:, :, 0]
        y = tri_coords[:, :, 1]
        
        area = 0.5 * np.abs(
            x[:,0]*(y[:,1] - y[:,2]) + 
            x[:,1]*(y[:,2] - y[:,0]) + 
            x[:,2]*(y[:,0] - y[:,1])
        )
        return area
    
    def _calculate_weighted_angular_error(self, candidate_indices):

        # Error angular pur
        angular_errors = self._calculate_angular_error(candidate_indices)
        
        #Trobar a quin triangle pertany cada candidat
        rows, cols = np.divmod(candidate_indices, self.cols)
        x = cols * self.step * self.pixel_size
        y = rows * self.step * self.pixel_size
        coords = np.column_stack((x, y))
        
        simplex_ids = self.tin.find_simplex(coords)
        
        # Calcular àrees de tots els triangles
        all_areas = self._calculate_triangle_areas()
        
        #Assignar àrea corresponent a cada candidat
        candidate_areas = np.zeros(len(candidate_indices))
        valid_mask = simplex_ids != -1
        candidate_areas[valid_mask] = all_areas[simplex_ids[valid_mask]]
        
        #Score = angular_error * sqrt(triangle_area)
        weighted_score = angular_errors * np.sqrt(candidate_areas)
        
        return weighted_score

    def _get_coords_from_index(self, idx_flat):
        # Converteix índex pla a coordenades X,Y i alçada Z
        r, c = divmod(idx_flat, self.cols)
        x = c * self.step * self.pixel_size
        y = r * self.step * self.pixel_size
        z = self.h_grid[r, c]
        return np.array([x, y]), z

    # Calcula les normals de tots els triangles del TIN en vectors unitaris
    def _triangle_normals(self):
        tris = self.tin.simplices
        pts = self.tin.points
        zvals = np.array(self.tin_z_values)
        p0 = np.column_stack((pts[tris[:, 0], 0], pts[tris[:, 0], 1], zvals[tris[:, 0]]))
        p1 = np.column_stack((pts[tris[:, 1], 0], pts[tris[:, 1], 1], zvals[tris[:, 1]]))
        p2 = np.column_stack((pts[tris[:, 2], 0], pts[tris[:, 2], 1], zvals[tris[:, 2]]))
        u = p1 - p0
        v = p2 - p0
        n = np.cross(u, v)
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / norms
        flip = n[:, 2] < 0
        n[flip] *= -1
        return n


    def _save_snapshot(self, iteration, last_point_xy, folder, vmin=2083, vmax=2902):

        if len(self.tin.simplices) == 0:
            return
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Mapcolor amb elevació - ESCALA FIXA
        tpc = ax.tripcolor(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 
                           np.array(self.tin_z_values), cmap='terrain', shading='flat',
                           vmin=vmin, vmax=vmax)
        
        ax.triplot(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 
                   'k-', linewidth=0.3, alpha=0.5)
        
        # EMFATITZAR ÚLTIM PUNT
        ax.scatter(last_point_xy[0], last_point_xy[1], color='red', s=80, edgecolors='white', zorder=10, linewidths=2)
        
        plt.colorbar(tpc, label='Elevació (m)')
        plt.title(f"pendent6.py - Iter {iteration} | Punts: {len(self.tin.points)}", fontweight='bold')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 3000)
        ax.set_aspect('equal', adjustable='box')
        
        # Guardar fitxer
        filename = os.path.join(folder, f"frame_{iteration:04d}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)


    def _save_snapshot_slope(self, iteration, last_point_xy, folder):

        if len(self.tin.simplices) == 0:
            return
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Calcular pendents
        tri_normals = self._triangle_normals()
        nx = tri_normals[:, 0]
        ny = tri_normals[:, 1]
        nz = tri_normals[:, 2]
        slopes = np.degrees(np.arctan(np.sqrt(nx**2 + ny**2) / nz))
        
        # print(f"Pendent: min={slopes.min():.4f}, max={slopes.max():.4f}, mean={slopes.mean():.4f}")
        # for tri_idx, simplex in enumerate(self.tin.simplices):
        #     z_vals = [self.tin_z_values[simplex[0]], self.tin_z_values[simplex[1]], self.tin_z_values[simplex[2]]]
        #     print(f"  Triangle {tri_idx}: z = {z_vals}, pendent = {slopes[tri_idx]:.2f} graus")   
        #     x_vals = self.tin.points[simplex, 0]
        #     y_vals = self.tin.points[simplex, 1]
        #     print(f"    Vèrtexs: ({x_vals[0]:.2f}, {y_vals[0]:.2f}), ({x_vals[1]:.2f}, {y_vals[1]:.2f}), ({x_vals[2]:.2f}, {y_vals[2]:.2f})")

        #PolyCollection
        pts = self.tin.points
        polys = []
        for simplex in self.tin.simplices:
            triangle = pts[simplex]
            polys.append(triangle)
        
        norm = Normalize(vmin=slopes.min(), vmax=slopes.max())
        cmap = plt.get_cmap('viridis')
        
        poly_collection = PolyCollection(polys, cmap=cmap, norm=norm, edgecolors='black', linewidths=0.5)
        poly_collection.set_array(slopes)
    
        ax.add_collection(poly_collection)
        ax.autoscale_view()
        
        # Colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Pendent (graus)')
        
        # EMFATITZAR ÚLTIM PUNT
        ax.scatter(last_point_xy[0], last_point_xy[1], color='red', s=100, edgecolors='white', zorder=10, label='Nou Punt')
        
        plt.title(f"Iteració {iteration} | Punts: {len(self.tin.points)} | Pendent")
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.legend()
        
        # Guardar fitxer
        filename = os.path.join(folder, f"frame_slope_{iteration:04d}.png")
        plt.savefig(filename, dpi=100)
        plt.close(fig)

    def fit(self, npy_file_path, snapshot_dir=None, snapshot_interval=100):
        if not self._load_data(npy_file_path): return None, None
        
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            print(f"Les imatges es guardaran a: {snapshot_dir}/")
        
        total_points = self.rows * self.cols
        
        # Inicialització: només les 4 cantonades
        corner_indices = [0, self.cols-1, (self.rows-1)*self.cols, total_points-1]
        candidate_indices = list(set(range(total_points)) - set(corner_indices))

        # Obtenir els 4 punts de les cantonades
        initial_points_xy = []
        self.tin_z_values = []
        for idx in corner_indices:
            xy, z = self._get_coords_from_index(idx)
            initial_points_xy.append(xy)
            self.tin_z_values.append(z)

        # Calcular z residual per trobar el punt de màxim error
        corner_rows, corner_cols = np.divmod(corner_indices, self.cols)
        corner_z = self.h_grid[corner_rows, corner_cols]
        
        A = np.c_[corner_cols, corner_rows, np.ones(len(corner_indices))]
        coef, _, _, _ = np.linalg.lstsq(A, corner_z, rcond=None)
        a, b, c = coef
        
        rows_grid, cols_grid = np.indices((self.rows, self.cols))
        z_pred = a * cols_grid + b * rows_grid + c
        residual = np.abs(self.h_grid - z_pred).ravel()
        residual[corner_indices] = -np.inf
        
        max_err_idx = int(np.argmax(residual))
        new_xy, new_z = self._get_coords_from_index(max_err_idx)
        
        # Afegir el punt de màxim residual als punts inicials
        initial_points_xy.append(new_xy)
        self.tin_z_values.append(new_z)
        candidate_indices.remove(max_err_idx)
        
        # Crear Delaunay amb 5 punts
        self.tin = Delaunay(np.array(initial_points_xy), incremental=True)
        
        # print(f"Iteració 0: punts totals = {len(self.tin.points)}")
                
        iteration = 0
        new_xy = None
        while len(self.tin.points) < self.target_point_count and len(candidate_indices) > 0:
            iteration += 1
            
            weighted_errors = self._calculate_weighted_angular_error(candidate_indices)
            
            worst_local_idx = np.argmax(weighted_errors)
            worst_global_idx = candidate_indices[worst_local_idx]
            max_weighted_error = weighted_errors[worst_local_idx]
            
            if iteration % 10 == 0:
                pure_angular = self._calculate_angular_error(candidate_indices)
                max_pure_error = pure_angular[worst_local_idx]
                print(f"Iteració {iteration}: weighted_error = {max_weighted_error:.2f}, "
                      f"angular_error = {max_pure_error:.2f}°, punts = {len(self.tin.points)}")
                
            new_xy, new_z = self._get_coords_from_index(worst_global_idx)
            
            self.tin.add_points([new_xy])
            self.tin_z_values.append(new_z)
            candidate_indices.pop(worst_local_idx)
            
            if snapshot_dir and (iteration % snapshot_interval == 0):
                self._save_snapshot(iteration, new_xy, snapshot_dir)
            
        # self._filter_erosion(max_len=self.step * self.pixel_size * 15)
        
        # Guardar l'última imatge per si iteration % snapshot_interval != 0
        if snapshot_dir and new_xy is not None:
            self._save_snapshot(iteration, new_xy, snapshot_dir)
        
        # VERIFICACIÓ: Comprovar si els punts i les Z estan sincronitzats
        print(f"\n[DEBUG] Verificant sincronització punts-Z:")
        print(f"  Nombre de punts XY al TIN: {len(self.tin.points)}")
        print(f"  Nombre de valors Z: {len(self.tin_z_values)}")
        if len(self.tin.points) != len(self.tin_z_values):
            print("  ⚠️ ALERTA: Desincronització detectada!")
            
        return self.tin.points, self.tin.simplices

    def fit_with_error_snapshots(self, npy_file_path, snapshot_dir='snapshots_error_pendent', snapshot_interval=5):
        
        if not self._load_data(npy_file_path): 
            return None, None
        
        os.makedirs(snapshot_dir, exist_ok=True)
        
        total_points = self.rows * self.cols
        corner_indices = [0, self.cols-1, (self.rows-1)*self.cols, total_points-1]
        candidate_indices = list(set(range(total_points)) - set(corner_indices))

        initial_points_xy = []
        self.tin_z_values = []
        for idx in corner_indices:
            xy, z = self._get_coords_from_index(idx)
            initial_points_xy.append(xy)
            self.tin_z_values.append(z)

        corner_rows, corner_cols = np.divmod(corner_indices, self.cols)
        corner_z = self.h_grid[corner_rows, corner_cols]
        
        A = np.c_[corner_cols, corner_rows, np.ones(len(corner_indices))]
        coef, _, _, _ = np.linalg.lstsq(A, corner_z, rcond=None)
        a, b, c = coef
        
        rows_grid, cols_grid = np.indices((self.rows, self.cols))
        z_pred = a * cols_grid + b * rows_grid + c
        residual = np.abs(self.h_grid - z_pred).ravel()
        residual[corner_indices] = -np.inf
        
        max_err_idx = int(np.argmax(residual))
        new_xy, new_z = self._get_coords_from_index(max_err_idx)
        initial_points_xy.append(new_xy)
        self.tin_z_values.append(new_z)
        candidate_indices.remove(max_err_idx)
        
        self.tin = Delaunay(np.array(initial_points_xy), incremental=True)
        
        iteration = 0
        spacing = self.step * self.pixel_size
        
        while len(self.tin.points) < self.target_point_count and len(candidate_indices) > 0:
            iteration += 1
            
            # Usar error ponderat per àrea
            weighted_errors = self._calculate_weighted_angular_error(candidate_indices)
            
            worst_local_idx = np.argmax(weighted_errors)
            worst_global_idx = candidate_indices[worst_local_idx]
            
            # Calcular error angular pur per visualització
            angular_errors = self._calculate_angular_error(candidate_indices)
            max_error = angular_errors[worst_local_idx]
            
            if iteration % 10 == 0:
                print(f"  Iter {iteration}: Error màx = {max_error:.2f}°")
            
            # Generar snapshot d'error (usar errors angulars purs per visualització)
            if iteration % snapshot_interval == 0:
                self._save_error_snapshot(snapshot_dir, iteration, candidate_indices, 
                                         angular_errors, worst_local_idx, spacing)
            
            new_xy, new_z = self._get_coords_from_index(worst_global_idx)
            self.tin.add_points([new_xy])
            self.tin_z_values.append(new_z)
            candidate_indices.pop(worst_local_idx)
        
        print(f"✓ {iteration} snapshots generats a {snapshot_dir}/")
        return self.tin.points, self.tin.simplices
    
    def _save_error_snapshot(self, snapshot_dir, iteration, candidate_indices, 
                            errors, max_err_idx, spacing):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Crear mapa d'error angular (mostrejat cada 5 píxels)
        sample_step = 5
        sample_rows = self.rows // sample_step
        sample_cols = self.cols // sample_step
        error_grid = np.full((sample_rows, sample_cols), np.nan)
        
        for local_idx, global_idx in enumerate(candidate_indices):
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
                       origin='lower', cmap='hot_r', vmin=0, vmax=180, 
                       interpolation='nearest', alpha=0.9)
        
        # Dibuixar TIN actual
        ax.triplot(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 
                   color='cyan', linewidth=0.5, alpha=0.6)
        
        # Marcar punts TIN
        ax.scatter(self.tin.points[:,0], self.tin.points[:,1], c='white', s=15, 
                   edgecolors='black', linewidths=0.5, zorder=5, alpha=0.8)
        
        # Marcar últim punt afegit
        if len(self.tin.points) > 0:
            last_pt = self.tin.points[-1]
            ax.scatter(last_pt[0], last_pt[1], c='lime', s=100, 
                       edgecolors='white', linewidths=2, zorder=10, marker='*')
        
        # Marcar punt amb màxim error
        if max_err_idx < len(candidate_indices):
            next_pt_idx = candidate_indices[max_err_idx]
            r = next_pt_idx // self.cols
            c = next_pt_idx % self.cols
            next_pt = np.array([c * spacing, r * spacing])
            ax.scatter(next_pt[0], next_pt[1], c='red', s=100, 
                       edgecolors='yellow', linewidths=2, zorder=10, marker='X')
        
        plt.colorbar(im, ax=ax, label='Error angular (graus)')
        ax.set_title(f"pendent6.py - ERROR ANGULAR | Iter {iteration} | Punts: {len(self.tin.points)}", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(0, self.cols*spacing)
        ax.set_ylim(0, self.rows*spacing)
        ax.set_aspect('equal', adjustable='box')
        
        filename = os.path.join(snapshot_dir, f'frame_{iteration:04d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()


    #Va dir en Rodrigo que millor no, per problemes de borrar massa triangles
    def _filter_erosion(self, max_len):
        keep_mask = np.ones(len(self.tin.simplices), dtype=bool)
        points = self.tin.points
        for i, simplex in enumerate(self.tin.simplices):
            p0, p1, p2 = points[simplex]
            if (np.linalg.norm(p0-p1) > max_len or 
                np.linalg.norm(p1-p2) > max_len or 
                np.linalg.norm(p2-p0) > max_len):
                keep_mask[i] = False
        self.tin.simplices = self.tin.simplices[keep_mask]

    def plot(self):
        if self.tin is None or len(self.tin.simplices) == 0: 
            print("No hi ha triangles per dibuixar")
            return
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        tpc = ax.tripcolor(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 
                           np.array(self.tin_z_values), cmap='coolwarm', shading='flat')
        ax.triplot(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 'k-', linewidth=0.2)
        plt.colorbar(tpc, label='Elevació (m)')
        plt.title(f"TIN Final (Punts: {len(self.tin.points)})")
        plt.axis('equal')
        plt.show()
    
    def plot2(self):
        if self.tin is None or len(self.tin.simplices) == 0:
            print("No hi ha triangles per dibuixar")
            return
        
        # Calcular pendent per cada triangle a partir de les normals
        tri_normals = self._triangle_normals()
        # Pendent = arctan(sqrt(nx^2 + ny^2) / nz) en graus
        slopes = np.degrees(np.arctan(np.sqrt(tri_normals[:, 0]**2 + tri_normals[:, 1]**2) / (tri_normals[:, 2] + 1e-12)))
        
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        tpc = ax.tripcolor(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 
                           slopes, cmap='viridis', shading='flat')
        ax.triplot(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 'k-', linewidth=0.2)
        plt.colorbar(tpc, label='Pendent (graus)')
        plt.title(f"TIN Final - Pendent (Punts: {len(self.tin.points)})")
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    converter = GridToTinIncremental(step=1, pixel_size=2.0, target_point_count=500)
    
    t0 = time.perf_counter()
    verts, triangles = converter.fit('bassiero.npy', snapshot_dir='snapshots_tin_pendent_video', snapshot_interval=1)
    t1 = time.perf_counter()
    print(f"Temps total: {t1 - t0:.2f} segons")
 
    converter.plot()
    converter.plot2()  # Mostra també el pendent