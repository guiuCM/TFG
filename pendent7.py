import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial import Delaunay
import time
import os

# MODES DE SELECCIÓ
# 'point'       : pendent6 original – tria el punt candidat amb major error ponderat (error * sqrt(àrea))
# 'triangle'    : pendent7 nou      – tria el triangle amb major (mean_error * sqrt(àrea)),
#                i dins d'ell escull el punt candidat amb major error angular pur
# 'mean_normal' : nou              – per a cada triangle calcula la normal mitjana de tots els
#                candidats interiors, mesura l'error angular TIN↔normal_mitjana,
#                tria el triangle amb score màxim i insereix el candidat més proper al centroide

class GridToTinIncremental:

    def __init__(self, step=25, pixel_size=2.0, target_point_count=500, mode='triangle'):
        assert mode in ('point', 'triangle', 'mean_normal'), \
            "mode ha de se 'point', 'triangle' o 'mean_normal'"
        self.mode = mode

        self.step = step
        self.pixel_size = pixel_size
        self.target_point_count = target_point_count

        self.h_grid = None
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

        self.h_grid = full_h[::self.step, ::self.step]
        self.rows, self.cols = self.h_grid.shape

        spacing = self.step * self.pixel_size
        dz_dy, dz_dx = np.gradient(self.h_grid, spacing, spacing)

        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(nx)

        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        self.normal_grid = np.dstack((nx / norm, ny / norm, nz / norm))

        return True

    def _get_coords_from_index(self, idx_flat):
        r, c = divmod(idx_flat, self.cols)
        x = c * self.step * self.pixel_size
        y = r * self.step * self.pixel_size
        z = self.h_grid[r, c]
        return np.array([x, y]), z

    def _calculate_triangle_areas(self):
        points = self.tin.points
        simplices = self.tin.simplices
        tri_coords = points[simplices]
        x = tri_coords[:, :, 0]
        y = tri_coords[:, :, 1]
        area = 0.5 * np.abs(
            x[:, 0] * (y[:, 1] - y[:, 2]) +
            x[:, 1] * (y[:, 2] - y[:, 0]) +
            x[:, 2] * (y[:, 0] - y[:, 1])
        )
        return area

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
        n = n / np.maximum(norms, 1e-10)
        flip = n[:, 2] < 0
        n[flip] *= -1
        return n

    def _find_simplex_for_candidates(self, candidate_indices):
        rows, cols = np.divmod(candidate_indices, self.cols)
        x = cols * self.step * self.pixel_size
        y = rows * self.step * self.pixel_size
        coords = np.column_stack((x, y))
        simplex_ids = self.tin.find_simplex(coords)
        return coords, simplex_ids

    def _calculate_angular_error(self, candidate_indices):
        rows, cols = np.divmod(candidate_indices, self.cols)
        real_normals = self.normal_grid[rows, cols]

        coords, simplex_ids = self._find_simplex_for_candidates(candidate_indices)

        tris_indices = self.tin.simplices[simplex_ids]
        tris_xy = self.tin.points[tris_indices]
        tris_z = np.array(self.tin_z_values)[tris_indices]

        p0 = np.column_stack((tris_xy[:, 0, 0], tris_xy[:, 0, 1], tris_z[:, 0]))
        p1 = np.column_stack((tris_xy[:, 1, 0], tris_xy[:, 1, 1], tris_z[:, 1]))
        p2 = np.column_stack((tris_xy[:, 2, 0], tris_xy[:, 2, 1], tris_z[:, 2]))

        u = p1 - p0
        v = p2 - p0
        tin_normals = np.cross(u, v)

        norms = np.linalg.norm(tin_normals, axis=1, keepdims=True)
        tin_normals = tin_normals / np.maximum(norms, 1e-10)

        flip = tin_normals[:, 2] < 0
        tin_normals[flip] *= -1

        dot = np.einsum('ij,ij->i', real_normals, tin_normals)
        dot = np.clip(dot, -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    # MODE 'point' (pendent 6)
    def _calculate_weighted_angular_error(self, candidate_indices):
        """Score = angular_error * sqrt(àrea_triangle) — mode 'point'."""
        angular_errors = self._calculate_angular_error(candidate_indices)

        _, simplex_ids = self._find_simplex_for_candidates(candidate_indices)
        all_areas = self._calculate_triangle_areas()

        candidate_areas = np.zeros(len(candidate_indices))
        valid = simplex_ids != -1
        candidate_areas[valid] = all_areas[simplex_ids[valid]]

        return angular_errors * np.sqrt(candidate_areas)

    def _select_worst_point(self, candidate_indices):
        """Mode 'point': retorna (local_idx, global_idx) del pitjor candidat."""
        weighted = self._calculate_weighted_angular_error(candidate_indices)
        worst_local = int(np.argmax(weighted))
        return worst_local, candidate_indices[worst_local]

    # MODE 'triangle'
    def _select_worst_by_triangle(self, candidate_indices):
        """
        1. Calcula error angular pur per a cada candidat.
        2. Agrupa per triangle; score de cada triangle = mean(errors) * sqrt(àrea).
        3. Tria el triangle amb score màxim.
        4. Dins del triangle, retorna el candidat amb el major error angular.

        Retorna (local_idx, global_idx).
        """
        candidate_indices_arr = np.asarray(candidate_indices)
        angular_errors = self._calculate_angular_error(candidate_indices_arr)

        _, simplex_ids = self._find_simplex_for_candidates(candidate_indices_arr)
        all_areas = self._calculate_triangle_areas()

        # Candidats que pertanyen a algun triangle
        valid_mask = simplex_ids != -1

        if not np.any(valid_mask):
            # Fallback: cap punt dins de cap triangle (no hauria de passar)
            worst_local = int(np.argmax(angular_errors))
            return worst_local, int(candidate_indices_arr[worst_local])

        unique_tris = np.unique(simplex_ids[valid_mask])

        best_tri = None
        best_score = -np.inf

        for tri_id in unique_tris:
            mask = simplex_ids == tri_id
            mean_err = np.mean(angular_errors[mask])
            area = all_areas[tri_id]
            score = mean_err * np.sqrt(area)
            if score > best_score:
                best_score = score
                best_tri = tri_id

        # Dins del triangle guanyador, el pitjor punt
        tri_mask = simplex_ids == best_tri
        local_indices_in_tri = np.where(tri_mask)[0]  # índexs locals dins candidate_indices
        best_in_tri = local_indices_in_tri[int(np.argmax(angular_errors[tri_mask]))]

        return int(best_in_tri), int(candidate_indices_arr[best_in_tri])

    # MODE 'mean_normal' : normal mitjana del triangle + punt central
    def _select_worst_by_mean_normal(self, candidate_indices):
        """
        1. Agrupa els candidats per triangle.
        2. Per a cada triangle, calcula la normal mitjana de totes les normals
           de la quadrícula dels candidats que hi cauen.
        3. Calcula l'error angular entre la normal del TIN i la normal mitjana.
        4. Score = error_angular * sqrt(àrea).
        5. Tria el triangle amb score màxim.
        6. Dins del triangle, retorna el candidat més proper al centroide.

        Retorna (local_idx, global_idx).
        """
        candidate_indices_arr = np.asarray(candidate_indices)

        rows, cols = np.divmod(candidate_indices_arr, self.cols)
        x = cols * self.step * self.pixel_size
        y = rows * self.step * self.pixel_size
        coords = np.column_stack((x, y))

        simplex_ids = self.tin.find_simplex(coords)
        valid_mask = simplex_ids != -1

        # Fallback: si cap punt és dins el TIN
        if not np.any(valid_mask):
            angular_errors = self._calculate_angular_error(candidate_indices_arr)
            worst_local = int(np.argmax(angular_errors))
            return worst_local, int(candidate_indices_arr[worst_local])

        all_areas = self._calculate_triangle_areas()
        tin_normals = self._triangle_normals()  # (n_tris, 3)

        unique_tris = np.unique(simplex_ids[valid_mask])

        best_tri = None
        best_score = -np.inf
        best_mean_normal = None

        for tri_id in unique_tris:
            mask = simplex_ids == tri_id
            # Normals de la quadrícula dels candidats dins el triangle
            r_in, c_in = rows[mask], cols[mask]
            grid_normals_in = self.normal_grid[r_in, c_in]  # (k, 3)
            mean_n = grid_normals_in.mean(axis=0)
            norm_len = np.linalg.norm(mean_n)
            if norm_len < 1e-10:
                continue
            mean_n /= norm_len

            tin_n = tin_normals[tri_id]  # (3,)
            dot = float(np.clip(np.dot(tin_n, mean_n), -1.0, 1.0))
            angle_err = np.degrees(np.arccos(dot))

            score = angle_err * np.sqrt(all_areas[tri_id])
            if score > best_score:
                best_score = score
                best_tri = tri_id
                best_mean_normal = mean_n

        if best_tri is None:
            angular_errors = self._calculate_angular_error(candidate_indices_arr)
            worst_local = int(np.argmax(angular_errors))
            return worst_local, int(candidate_indices_arr[worst_local])

        # Centroide del triangle guanyador
        tri_verts_xy = self.tin.points[self.tin.simplices[best_tri]]  # (3, 2)
        centroid = tri_verts_xy.mean(axis=0)  # (2,)

        # Candidats dins el triangle guanyador
        tri_mask = simplex_ids == best_tri
        local_indices_in_tri = np.where(tri_mask)[0]
        coords_in_tri = coords[tri_mask]  # (k, 2)

        # Punt candidat més proper al centroide
        dists = np.linalg.norm(coords_in_tri - centroid, axis=1)
        closest_local_in_tri = int(np.argmin(dists))
        best_local = local_indices_in_tri[closest_local_in_tri]

        return int(best_local), int(candidate_indices_arr[best_local])

    # ------------------------------------------------------------------
    # Dispatcher: tria la funció de selecció segons el mode
    # ------------------------------------------------------------------

    def _select_next(self, candidate_indices):
        if self.mode == 'point':
            return self._select_worst_point(candidate_indices)
        elif self.mode == 'triangle':
            return self._select_worst_by_triangle(candidate_indices)
        else:  # 'mean_normal'
            return self._select_worst_by_mean_normal(candidate_indices)

    # ------------------------------------------------------------------
    # Inicialització comuna (cantonades + punt de màxim residu)
    # ------------------------------------------------------------------

    def _initialize_tin(self):
        """Crea el TIN inicial amb les 4 cantonades i el punt de màxim residu."""
        total_points = self.rows * self.cols
        corner_indices = [0, self.cols - 1, (self.rows - 1) * self.cols, total_points - 1]
        candidate_indices = list(set(range(total_points)) - set(corner_indices))

        initial_points_xy = []
        self.tin_z_values = []
        for idx in corner_indices:
            xy, z = self._get_coords_from_index(idx)
            initial_points_xy.append(xy)
            self.tin_z_values.append(z)

        # Punt de màxim residu respecte el pla de les cantonades
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
        return candidate_indices

    # ------------------------------------------------------------------
    # Mètode principal: fit
    # ------------------------------------------------------------------

    def fit_with_error_snapshots(self, npy_file_path,
                                 snapshot_dir='snapshots_error_pendent',
                                 snapshot_interval=5):
        """Igual que fit() però guarda snapshots de l'error angular a cada interval."""
        if not self._load_data(npy_file_path):
            return None, None

        os.makedirs(snapshot_dir, exist_ok=True)
        print(f"Les imatges d'error es guardaran a: {snapshot_dir}/  [mode={self.mode}]")

        candidate_indices = self._initialize_tin()

        iteration = 0
        new_xy = None

        while len(self.tin.points) < self.target_point_count and len(candidate_indices) > 0:
            iteration += 1

            worst_local, worst_global = self._select_next(candidate_indices)

            if iteration % 10 == 0:
                angular = self._calculate_angular_error(candidate_indices)
                max_err = angular[worst_local]
                print(f"[{self.mode}] Iter {iteration}: error_angular_màx = {max_err:.2f}°, "
                      f"punts = {len(self.tin.points)}")

            new_xy, new_z = self._get_coords_from_index(worst_global)

            if iteration % snapshot_interval == 0:
                self._save_angular_error_snapshot(
                    snapshot_dir, iteration, candidate_indices, worst_local, self.tin.points[-1])

            self.tin.add_points([new_xy])
            self.tin_z_values.append(new_z)
            candidate_indices.pop(worst_local)

        # Snapshot final
        if new_xy is not None:
            self._save_angular_error_snapshot(
            snapshot_dir, iteration, candidate_indices, -1, self.tin.points[-1])

        print(f"\n[DEBUG] Punts XY: {len(self.tin.points)} | Valors Z: {len(self.tin_z_values)}")
        if len(self.tin.points) != len(self.tin_z_values):
            print("  ⚠️ ALERTA: Desincronització detectada!")

        print(f"✓ Snapshots d'error guardats a {snapshot_dir}/")
        return self.tin.points, self.tin.simplices

    def _save_angular_error_snapshot(self, snapshot_dir, iteration,
                                     candidate_indices, worst_local, last_point_xy):
        if len(self.tin.simplices) == 0 or len(candidate_indices) == 0:
            return

        candidate_arr = np.asarray(candidate_indices)
        angular_errors = self._calculate_angular_error(candidate_arr)
        
        spacing = self.step * self.pixel_size
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Crear mapa d'error angular (mostrejat cada 5 píxels) - igual que pendent6
        sample_step = 5
        sample_rows = self.rows // sample_step
        sample_cols = self.cols // sample_step
        error_grid = np.full((sample_rows, sample_cols), np.nan)
        
        for local_idx, global_idx in enumerate(candidate_arr):
            r = global_idx // self.cols
            c = global_idx % self.cols
            sr = r // sample_step
            sc = c // sample_step
            if 0 <= sr < sample_rows and 0 <= sc < sample_cols:
                if np.isnan(error_grid[sr, sc]):
                    error_grid[sr, sc] = angular_errors[local_idx]
                else:
                    error_grid[sr, sc] = max(error_grid[sr, sc], angular_errors[local_idx])
        
        # Mostrar error com a heatmap
        im = ax.imshow(error_grid, extent=[0, self.cols*spacing, 0, self.rows*spacing], 
                       origin='lower', cmap='hot_r', vmin=0, vmax=180, 
                       interpolation='nearest', alpha=0.9)
        
        # Dibuixar TIN actual
        ax.triplot(self.tin.points[:, 0], self.tin.points[:, 1], self.tin.simplices,
                   color='cyan', linewidth=0.5, alpha=0.6)
        
        # Marcar punts TIN
        ax.scatter(self.tin.points[:, 0], self.tin.points[:, 1],
                   c='white', s=15, edgecolors='black', linewidths=0.5, zorder=5, alpha=0.8)
        
        # Marcar últim punt afegit
        if len(self.tin.points) > 0:
            ax.scatter(last_point_xy[0], last_point_xy[1],
                       c='lime', s=100, edgecolors='white', linewidths=2, zorder=10, marker='*')
        
        # Marcar punt amb màxim error
        if 0 <= worst_local < len(candidate_indices):
            nr, nc = divmod(candidate_indices[worst_local], self.cols)
            ax.scatter(nc * spacing, nr * spacing,
                       c='red', s=100, edgecolors='yellow', linewidths=2, zorder=10, marker='X')

        plt.colorbar(im, ax=ax, label='Error angular (graus)')
        ax.set_title(
            f"pendent7.py [{self.mode}] - ERROR ANGULAR | Iter {iteration} | Punts: {len(self.tin.points)}",
            fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 3000)
        ax.set_aspect('equal', adjustable='box')

        filename = os.path.join(snapshot_dir, f"frame_{iteration:04d}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _save_snapshot(self, iteration, last_point_xy, folder, vmin=2083, vmax=2902):
        if len(self.tin.simplices) == 0:
            return
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()

        tpc = ax.tripcolor(self.tin.points[:, 0], self.tin.points[:, 1], self.tin.simplices,
                           np.array(self.tin_z_values), cmap='terrain', shading='flat',
                           vmin=vmin, vmax=vmax)
        ax.triplot(self.tin.points[:, 0], self.tin.points[:, 1], self.tin.simplices,
                   'k-', linewidth=0.3, alpha=0.5)
        ax.scatter(last_point_xy[0], last_point_xy[1], color='red', s=80,
                   edgecolors='white', zorder=10, linewidths=2)

        plt.colorbar(tpc, label='Elevació (m)')
        plt.title(f"pendent7.py [{self.mode}] - Iter {iteration} | Punts: {len(self.tin.points)}",
                  fontweight='bold')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 3000)
        ax.set_aspect('equal', adjustable='box')

        filename = os.path.join(folder, f"frame_{iteration:04d}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _save_snapshot_slope(self, iteration, last_point_xy, folder):
        if len(self.tin.simplices) == 0:
            return
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()

        tri_normals = self._triangle_normals()
        nx = tri_normals[:, 0]
        ny = tri_normals[:, 1]
        nz = tri_normals[:, 2]
        slopes = np.degrees(np.arctan(np.sqrt(nx**2 + ny**2) / nz))

        pts = self.tin.points
        polys = [pts[s] for s in self.tin.simplices]
        norm = Normalize(vmin=slopes.min(), vmax=slopes.max())
        cmap = plt.get_cmap('viridis')
        pc = PolyCollection(polys, cmap=cmap, norm=norm, edgecolors='black', linewidths=0.5)
        pc.set_array(slopes)
        ax.add_collection(pc)
        ax.autoscale_view()

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Pendent (graus)')

        ax.scatter(last_point_xy[0], last_point_xy[1], color='red', s=100,
                   edgecolors='white', zorder=10, label='Nou Punt')
        plt.title(f"Iteració {iteration} | Punts: {len(self.tin.points)} | Pendent [{self.mode}]")
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.legend()

        filename = os.path.join(folder, f"frame_slope_{iteration:04d}.png")
        plt.savefig(filename, dpi=100)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Visualització final
    # ------------------------------------------------------------------

    def plot(self):
        if self.tin is None or len(self.tin.simplices) == 0:
            print("No hi ha triangles per dibuixar")
            return
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        tpc = ax.tripcolor(self.tin.points[:, 0], self.tin.points[:, 1], self.tin.simplices,
                           np.array(self.tin_z_values), cmap='coolwarm', shading='flat')
        ax.triplot(self.tin.points[:, 0], self.tin.points[:, 1], self.tin.simplices,
                   'k-', linewidth=0.2)
        plt.colorbar(tpc, label='Elevació (m)')
        plt.title(f"TIN Final [{self.mode}] (Punts: {len(self.tin.points)})")
        plt.axis('equal')
        plt.show()

    def plot2(self):
        if self.tin is None or len(self.tin.simplices) == 0:
            print("No hi ha triangles per dibuixar")
            return
        tri_normals = self._triangle_normals()
        slopes = np.degrees(np.arctan(
            np.sqrt(tri_normals[:, 0]**2 + tri_normals[:, 1]**2) /
            (tri_normals[:, 2] + 1e-12)
        ))
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        tpc = ax.tripcolor(self.tin.points[:, 0], self.tin.points[:, 1], self.tin.simplices,
                           slopes, cmap='viridis', shading='flat')
        ax.triplot(self.tin.points[:, 0], self.tin.points[:, 1], self.tin.simplices,
                   'k-', linewidth=0.2)
        plt.colorbar(tpc, label='Pendent (graus)')
        plt.title(f"TIN Final - Pendent [{self.mode}] (Punts: {len(self.tin.points)})")
        plt.axis('equal')
        plt.show()


# ======================================================================
if __name__ == "__main__":

    # --- Mode mean_normal (NOU) ---
    converter = GridToTinIncremental(
        step=1, pixel_size=2.0, target_point_count=500,
        mode='triangle'
    )

    t0 = time.perf_counter()
    verts, triangles = converter.fit(
        'bassiero.npy',
        snapshot_dir='snapshots_tin_triangle',
        snapshot_interval=1
    )
    t1 = time.perf_counter()
    print(f"Temps total [{converter.mode}]: {t1 - t0:.2f} segons")

    converter.plot()
    converter.plot2()
