import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import time


class GridToTinConverter:
    
    def __init__(self, step=25, control_mode='POINT_COUNT', 
                 target_error_percentage=0.5, target_point_count=500):
        
        self.step = step
        self.control_mode = control_mode
        self.target_error_percentage = target_error_percentage
        self.target_point_count = target_point_count
        self.safeguard_max_points = 5000 
        self.all_points_3d = None
        self.num_total_points = 0
        self.elevation_range = 0
        self.rows = 0
        self.cols = 0
        
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
        
        h_sampled = h[::self.step, ::self.step]
        self.rows, self.cols = h_sampled.shape
        x = np.linspace(0, h.shape[1], self.cols)
        y = np.linspace(0, h.shape[0], self.rows)
        xx, yy = np.meshgrid(x, y)
        
        self.all_points_3d = np.vstack([xx.ravel(), yy.ravel(), h_sampled.ravel()]).T
        self.num_total_points = self.all_points_3d.shape[0]
        #print(f"Grid submostrejat a {self.num_total_points} punts (step={self.step}).")

    def fit(self, npy_file_path):
  
        self._load_and_sample_grid(npy_file_path)
        
        if self.control_mode == 'ERROR':
            max_error_threshold = (self.target_error_percentage / 100.0) * self.elevation_range
            max_points_threshold = self.safeguard_max_points
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
            
            if iteration % 10 == 0 or iteration == 1:
                print(f"  Iter. {iteration}: Punts TIN = {len(S_indices)}, Error Màx. Actual = {max_err:.2f}")

            if max_err <= max_error_threshold:
                print(f"\nProcés completat (MODO ERROR). Error final: {max_err:.2f}")
                break
            if len(S_indices) >= max_points_threshold:
                print(f"\nProcés completat (MODO PUNTS). Punts assolits: {len(S_indices)}")
                break
            if not P_indices:
                print("\nProcés completat. S'han afegit tots els punts.")
                break
            
            # Continuar: Afegir el pitjor punt al TIN
            S_indices.append(point_to_add_global_index)
            P_indices.pop(max_err_local_index)
            iteration += 1


        self.final_points_3d = self.all_points_3d[S_indices]
        self.tin = Delaunay(self.final_points_3d[:, :2])
        #print(f"TIN final generat amb {len(self.final_points_3d)} vèrtexs.")

    def plot(self):
        if self.tin is None: return
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        # Extreure Z del TIN
        z_values = self.final_points_3d[:, 2]
        tpc = ax.tripcolor(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 
                           z_values, cmap='coolwarm', shading='flat')
        ax.triplot(self.tin.points[:,0], self.tin.points[:,1], self.tin.simplices, 'k-', linewidth=0.2)
        plt.colorbar(tpc, label='Elevació (m)')
        plt.title(f"TIN Final (Punts: {len(self.tin.points)})")
        plt.axis('equal')
        plt.show()
    


if __name__ == "__main__":
    

    # tin_builder_points = GridToTinConverter(
    #     step=20,  # Un 'step' més baix dona més punts per escollir
    #     control_mode='POINT_COUNT',
    #     target_point_count=700  # Volem un TIN amb 700 punts
    # )
    # tin_builder_points.fit('bassiero.npy')
    # tin_builder_points.plot()

    tin_builder_error = GridToTinConverter(
        step=20,
        control_mode='POINT_COUNT',
        target_point_count=2000  # Volem un TIN amb 2000 punts
    )

    t0 = time.perf_counter()
    tin_builder_error.fit('bassiero.npy')
    t1 = time.perf_counter()
    print(f"Temps total (carrega + refinament): {t1 - t0:.2f} s")
    tin_builder_error.plot()
