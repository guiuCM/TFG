import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import time
import os  # Necessari per crear carpetes

class GridToTinIncremental:
    
    def __init__(self, step=25, pixel_size=2.0, origin=(0.0, 0.0), target_point_count=500):
        
        self.step = step
        self.pixel_size = pixel_size
        self.origin = origin
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
        
        # CONSTRUCCIÓ DEL VECTOR NORMAL REAL 
        # La normal d'una superfície z=f(x,y) és (-dz/dx, -dz/dy, 1)
        # 1. Creem els components
        nx = -dz_dx
        ny = -dz_dy
        nz = 1
        
        # 2. Calculem la magnitud per normalitzar (vector unitari)
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        
        # 3. Guardem la matriu de normals (Rows, Cols, 3)
        # Això crea una matriu on cada cel·la té un vector [nx, ny, nz] unitari
        self.normal_grid = np.dstack((nx/norm, ny/norm, nz/norm))
        
        return True
    
    def _calculate_angular_error(self, candidate_indices):
        """
        Calcula la diferència angular (en graus) entre la normal del Grid
        i la normal del TIN en els punts candidats.
        """
        
        # --- A. OBTENIR NORMALS REALS (GRID) ---
        # Convertim índexs plans a (fila, columna)
        rows, cols = np.divmod(candidate_indices, self.cols)
        # Extraiem els vectors normals pre-calculats: shape (N, 3)
        real_normals = self.normal_grid[rows, cols]
        
        
        # --- B. OBTENIR NORMALS DEL TIN ---
        # 1. Localitzem en quin triangle cau cada punt
        x = cols * self.step * self.pixel_size + self.origin[0]
        y = rows * self.step * self.pixel_size + self.origin[1]
        coords = np.column_stack((x, y))
        
        simplex_ids = self.tin.find_simplex(coords)
        
        # Filtrem punts fora del TIN (si n'hi ha)
        valid_mask = simplex_ids != -1
        if not np.any(valid_mask):
            return np.zeros(len(candidate_indices))

        # 2. Recuperem els vèrtexs 3D dels triangles afectats
        # Necessitem coordenades 3D reals per fer el producte vectorial (Cross Product)
        
        # Índexs dels 3 vèrtexs de cada triangle on cauen els punts
        tris_indices = self.tin.simplices[simplex_ids[valid_mask]] # shape (K, 3)
        
        # Recuperem X, Y dels vèrtexs
        tris_xy = self.tin.points[tris_indices] # shape (K, 3, 2)
        # Recuperem Z dels vèrtexs (de la nostra llista tin_z_values)
        tris_z = np.array(self.tin_z_values)[tris_indices] # shape (K, 3)
        
        # Construïm els punts 3D: P0, P1, P2
        # P0 és el primer vèrtex de cada triangle, etc.
        p0 = np.column_stack((tris_xy[:, 0, 0], tris_xy[:, 0, 1], tris_z[:, 0]))
        p1 = np.column_stack((tris_xy[:, 1, 0], tris_xy[:, 1, 1], tris_z[:, 1]))
        p2 = np.column_stack((tris_xy[:, 2, 0], tris_xy[:, 2, 1], tris_z[:, 2]))
        
        # 3. Calculem la Normal del Triangle (Producte Vectorial)
        # Vector U = P1 - P0
        # Vector V = P2 - P0
        # Normal = U x V
        u = p1 - p0
        v = p2 - p0
        tin_normals_k = np.cross(u, v)
        
        # Normalitzem els vectors del TIN
        norms = np.linalg.norm(tin_normals_k, axis=1, keepdims=True)
        # Per terreny pla, el producte vectorial pot ser quasi zero
        # Assignem una normal per defecte (0, 0, 1) per triangles plans
        zero_norm_mask = norms.flatten() < 1e-8
        tin_normals_k = tin_normals_k / (norms + 1e-10) # Evitar divisió per zero
        tin_normals_k[zero_norm_mask] = [0, 0, 1]  # Normal per defecte (apuntant amunt)
        
        # Assegurem que la Z sempre apunti amunt (per coherència amb el Grid)
        # Si Z és negativa, girem el vector
        flip_vec = tin_normals_k[:, 2] < 0
        tin_normals_k[flip_vec] *= -1
        
        
        # --- C. CÀLCUL DE L'ERROR (PRODUCTE ESCALAR) ---
        # Error = arccos( N_real · N_tin )
        
        # Només calculem error pels punts vàlids
        real_normals_valid = real_normals[valid_mask]
        
        # Producte escalar (Dot product) fila per fila
        # einsum('ij,ij->i') fa el producte escalar de vectors corresponents
        dot_product = np.einsum('ij,ij->i', real_normals_valid, tin_normals_k)
        
        # CLIPPING: Per errors de precisió flotant, el dot product pot ser 1.00000001
        # això faria petar l'arccos. Ho limitem a [-1, 1]
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Angle en radians
        angles_rad = np.arccos(dot_product)
        
        # Convertim a graus (opcional, però més llegible per humans)
        angles_deg = np.degrees(angles_rad)
        
        # Preparem array de sortida complet
        final_errors = np.zeros(len(candidate_indices))
        final_errors[valid_mask] = angles_deg
        
        return final_errors

    def _get_coords_from_index(self, idx_flat):
        # Converteix índex pla a coordenades X,Y i alçada Z
        r, c = divmod(idx_flat, self.cols)
        x = c * self.step * self.pixel_size + self.origin[0]
        y = r * self.step * self.pixel_size + self.origin[1]
        z = self.h_grid[r, c]
        return np.array([x, y]), z

    # Càlcul del pendent del TIN als índexs donats (ja no s'usa)
    def _interpolate_tin_slope_at_indices(self, candidate_indices):
        # Convertim índexs a coordenades X,Y
        rows, cols = np.divmod(candidate_indices, self.cols)
        x = cols * self.step * self.pixel_size + self.origin[0]
        y = rows * self.step * self.pixel_size + self.origin[1]
        coords = np.column_stack((x, y))
        
        simplex_ids = self.tin.find_simplex(coords) # Identifica en quin triangle està comprès cada punt
        epsilon = 1e-5
        
        # Convertim la llista Z a array per velocitat
        current_z_array = np.array(self.tin_z_values)

        #Li passo les coordenades i l'index del triangle on està
        def get_z_batch(points_xy, simps):
            #Calcula l'alçada Z usant coordenades baricèntriques
            valid = simps != -1
            
            # Inicialitza l'array de resultats amb NaN
            z_final = np.full(len(points_xy), np.nan)
            
            # Si no hi ha punts vàlids, retornem directament (tot els restants NaN)
            if not np.any(valid): 
                return z_final
            
            p_val = points_xy[valid] # Els punts on volem saber l'alçada (X, Y)
            s_val = simps[valid]# Els triangles on cauen aquests punts
            
            #La matriu ve en forma (N, 2, 2) correcta.
            b = self.tin.transform[s_val, :2]  # Matriu de transformació per a cada triangle
            c = self.tin.transform[s_val, 2] # Origin dels triangles (X,Y)
            
            # Multiplicació de matrius per obtenir Alpha i Beta
            # 'ijk,ik->ij' vol dir: (N,2,2) * (N,2) -> (N,2)
            w = np.einsum('ijk,ik->ij', b, p_val - c)
            
            # Calculem Gamma (1 - alpha - beta)
            w = np.c_[w, 1 - w.sum(axis=1)]
            
            # Busquem els índexs dels vèrtexs dels triangles
            verts_indices = self.tin.simplices[s_val]
            
            # Busquem les Z corresponents
            z_tri = current_z_array[verts_indices]
            
            # Interpolació final: suma ponderada (Pesos * Alçades)
            z_final[valid] = np.einsum('ij,ij->i', w, z_tri)
            
            return z_final

        # Alçada al punt P
        z0 = get_z_batch(coords, simplex_ids)
        
        # Alçada una mica a la dreta (P + dx)
        coords_x = coords + [epsilon, 0]
        z_x = get_z_batch(coords_x, self.tin.find_simplex(coords_x))
        
        # Alçada una mica amunt (P + dy)
        coords_y = coords + [0, epsilon]
        z_y = get_z_batch(coords_y, self.tin.find_simplex(coords_y))
        
        # Càlcul del pendent
        slope_x = (z_x - z0) / epsilon
        slope_y = (z_y - z0) / epsilon
        
        return np.sqrt(slope_x**2 + slope_y**2)


    def _save_snapshot(self, iteration, last_point_xy, folder):
        #Guarda una imatge de l'estat actual emfatitzant l'últim punt.
        # No guardar si no hi ha triangles
        if len(self.tin.simplices) == 0:
            return
            
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # 1. Mapa de calor (shading='flat' mostra els triangles clars)
        tpc = ax.tripcolor(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 
                           np.array(self.tin_z_values), cmap='coolwarm', shading='flat')
        
        # 2. Malla (línies fines)
        ax.triplot(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 
                   'k-', linewidth=0.3, alpha=0.5)
        
        # 3. EMFATITZAR ÚLTIM PUNT (Punt vermell gros)
        ax.scatter(last_point_xy[0], last_point_xy[1], color='red', s=100, edgecolors='white', zorder=10, label='Nou Punt')
        
        plt.colorbar(tpc, label='Elevació (m)')
        plt.title(f"Iteració {iteration} | Punts: {len(self.tin.points)}")
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.legend()
        
        # Guardar fitxer
        filename = os.path.join(folder, f"frame_{iteration:04d}.png")
        plt.savefig(filename, dpi=100)
        plt.close(fig) # Important: tancar per alliberar memòria

    def fit(self, npy_file_path, snapshot_dir=None, snapshot_interval=100):
        if not self._load_data(npy_file_path): return None, None
        
        # Preparar carpeta de snapshots
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            print(f"Les imatges es guardaran a: {snapshot_dir}/")
        
        total_points = self.rows * self.cols
        
        # Inicialització (Cantonades + Centre)
        center_idx = (self.rows // 2) * self.cols + (self.cols // 2)
        corner_indices = [0, self.cols-1, (self.rows-1)*self.cols, total_points-1]


        initial_indices = list(set(corner_indices + [center_idx]))    
        candidate_indices = list(set(range(total_points)) - set(corner_indices) - {center_idx})

        initial_points_xy = []
        self.tin_z_values = []
        
        for idx in initial_indices:
            xy, z = self._get_coords_from_index(idx)
            initial_points_xy.append(xy)
            self.tin_z_values.append(z)
            
        self.tin = Delaunay(np.array(initial_points_xy), incremental=True)
                
        iteration = 0
        new_xy = None
        while len(self.tin.points) < self.target_point_count and len(candidate_indices) > 0:
            iteration += 1
            
            # Càlculs d'error (igual que abans)
            cand_arr = np.array(candidate_indices)
            r, c = np.divmod(cand_arr, self.cols)
            # real_slopes = self.slope_grid[r, c]
            # tin_slopes = self._interpolate_tin_slope_at_indices(cand_arr)
            # errors = np.nan_to_num(np.abs(real_slopes - tin_slopes))
            
            # Calculem directament l'error angular (en graus)
            errors = self._calculate_angular_error(candidate_indices)
            
            worst_local_idx = np.argmax(errors)
            worst_global_idx = candidate_indices[worst_local_idx]
            max_error = errors[worst_local_idx]
            
            if max_error <= 0.001: break
                
            new_xy, new_z = self._get_coords_from_index(worst_global_idx)
            
            self.tin.add_points([new_xy])
            self.tin_z_values.append(new_z)
            candidate_indices.pop(worst_local_idx)
            
            if snapshot_dir and (iteration % snapshot_interval == 0):
                print(f"iteració {iteration}...")
                self._save_snapshot(iteration, new_xy, snapshot_dir)
            
        # Comentat temporalment per terreny pla
        # self._filter_erosion(max_len=self.step * self.pixel_size * 15)
        
        # Guardar l'última imatge per si iteration % snapshot_interval != 0
        if snapshot_dir and new_xy is not None:
            self._save_snapshot(iteration, new_xy, snapshot_dir)
            
        return self.tin.points, self.tin.simplices


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

if __name__ == "__main__":
    converter = GridToTinIncremental(step=20, pixel_size=2.0, target_point_count=2000)
    
    t0 = time.perf_counter()
    verts, triangles = converter.fit('bassiero.npy', snapshot_dir='snapshots_tin2', snapshot_interval=100)
    t1 = time.perf_counter()
    print(f"Temps total: {t1 - t0:.2f} segons")
 
    converter.plot()