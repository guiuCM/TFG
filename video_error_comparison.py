
import os
import argparse
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

from original import GridToTinConverter
from pendent6 import GridToTinIncremental as GridToTinIncrementalP6
from pendent7 import GridToTinIncremental as GridToTinIncrementalP7


def _default_metric_ranges():
    return {
        'angular': (0.0, 180.0),
        'height': (0.0, 400.0),
    }


def _compute_normal_grid(h_grid, spacing):
    dz_dy, dz_dx = np.gradient(h_grid, spacing, spacing)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(nx)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    return np.dstack((nx / norm, ny / norm, nz / norm))


def _compute_angular_error_from_tin(candidate_indices, cols, spacing, normal_grid, tin_points, tin_simplices, tin_z_values):
    candidate_indices = np.asarray(candidate_indices)
    rows, cols_idx = np.divmod(candidate_indices, cols)
    real_normals = normal_grid[rows, cols_idx]

    x = cols_idx * spacing
    y = rows * spacing
    coords = np.column_stack((x, y))

    tin = Delaunay(tin_points)
    simplex_ids = tin.find_simplex(coords)

    errors = np.zeros(len(candidate_indices))
    valid = simplex_ids != -1
    if not np.any(valid):
        return errors

    tris_indices = tin_simplices[simplex_ids[valid]]
    tris_xy = tin_points[tris_indices]
    z_arr = np.asarray(tin_z_values)
    tris_z = z_arr[tris_indices]

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

    dot = np.einsum('ij,ij->i', real_normals[valid], tin_normals)
    dot = np.clip(dot, -1.0, 1.0)
    errors[valid] = np.degrees(np.arccos(dot))
    return errors


def _compute_height_error_from_tin(conv, candidate_indices):
    candidate_indices = np.asarray(candidate_indices)
    rows, cols = np.divmod(candidate_indices, conv.cols)
    x = cols * conv.step * conv.pixel_size
    y = rows * conv.step * conv.pixel_size
    coords = np.column_stack((x, y))

    interpolator = LinearNDInterpolator(conv.tin.points, np.asarray(conv.tin_z_values))
    interpolated_z = interpolator(coords)
    actual_z = conv.h_grid[rows, cols]
    return np.nan_to_num(np.abs(actual_z - interpolated_z))


def _save_error_snapshot(snapshot_dir, iteration, rows, cols, spacing,
                         candidate_indices, errors, max_err_idx,
                         tin_points, tin_simplices, last_point_xy,
                         title_prefix, metric_label, vmin, vmax):
    fig, ax = plt.subplots(figsize=(12, 10))

    sample_step = 5
    sample_rows = rows // sample_step
    sample_cols = cols // sample_step
    error_grid = np.full((sample_rows, sample_cols), np.nan)

    for local_idx, global_idx in enumerate(candidate_indices):
        r = global_idx // cols
        c = global_idx % cols
        sr = r // sample_step
        sc = c // sample_step
        if 0 <= sr < sample_rows and 0 <= sc < sample_cols:
            if np.isnan(error_grid[sr, sc]):
                error_grid[sr, sc] = errors[local_idx]
            else:
                error_grid[sr, sc] = max(error_grid[sr, sc], errors[local_idx])

    im = ax.imshow(
        error_grid,
        extent=[0, cols * spacing, 0, rows * spacing],
        origin='lower',
        cmap='hot_r',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        alpha=0.9,
    )

    ax.triplot(tin_points[:, 0], tin_points[:, 1], tin_simplices,
               color='cyan', linewidth=0.5, alpha=0.6)

    ax.scatter(tin_points[:, 0], tin_points[:, 1], c='white', s=15,
               edgecolors='black', linewidths=0.5, zorder=5, alpha=0.8)

    if last_point_xy is not None:
        ax.scatter(last_point_xy[0], last_point_xy[1], c='lime', s=100,
                   edgecolors='white', linewidths=2, zorder=10, marker='*')

    if 0 <= max_err_idx < len(candidate_indices):
        next_pt_idx = candidate_indices[max_err_idx]
        r = next_pt_idx // cols
        c = next_pt_idx % cols
        next_pt = np.array([c * spacing, r * spacing])
        ax.scatter(next_pt[0], next_pt[1], c='red', s=100,
                   edgecolors='yellow', linewidths=2, zorder=10, marker='X')

    plt.colorbar(im, ax=ax, label=metric_label)
    ax.set_title(
        f"{title_prefix} | Iter {iteration} | Punts: {len(tin_points)}",
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(0, cols * spacing)
    ax.set_ylim(0, rows * spacing)
    ax.set_aspect('equal', adjustable='box')

    filename = os.path.join(snapshot_dir, f'frame_{iteration:04d}.png')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)


def generate_original_error_snapshots(npy_file, target_points, metric, output_dir,
                                      snapshot_interval, vmin, vmax, log_interval=10):
    print(f"Generant snapshots de {metric} per original.py...")
    os.makedirs(output_dir, exist_ok=True)

    converter = GridToTinConverter(
        step=1,
        pixel_size=2.0,
        control_mode='POINT_COUNT',
        target_point_count=target_points,
    )
    converter._load_and_sample_grid(npy_file)

    spacing = converter.step * converter.pixel_size
    h_sampled = converter.all_points_3d[:, 2].reshape(converter.rows, converter.cols)
    normal_grid = _compute_normal_grid(h_sampled, spacing)

    corner_indices = [0, converter.cols - 1, (converter.rows - 1) * converter.cols, converter.rows * converter.cols - 1]
    s_indices = list(set(corner_indices))
    p_indices = list(set(range(converter.num_total_points)) - set(s_indices))

    iteration = 0
    while len(s_indices) < target_points and p_indices:
        iteration += 1

        current_points = converter.all_points_3d[s_indices]
        tin = Delaunay(current_points[:, :2])

        interpolator = LinearNDInterpolator(current_points[:, :2], current_points[:, 2])
        points_to_check_2d = converter.all_points_3d[p_indices, :2]
        interpolated_z = interpolator(points_to_check_2d)
        actual_z = converter.all_points_3d[p_indices, 2]
        height_errors = np.nan_to_num(np.abs(actual_z - interpolated_z))

        # Selecció original: sempre per error d'alçada
        worst_for_selection = int(np.argmax(height_errors))
        point_to_add_global_index = p_indices[worst_for_selection]

        if metric == 'height':
            vis_errors = height_errors
            metric_label = 'Error alçada (m)'
            title_prefix = "original.py - ERROR D'ALÇADA"
        else:
            vis_errors = _compute_angular_error_from_tin(
                p_indices,
                converter.cols,
                spacing,
                normal_grid,
                tin.points,
                tin.simplices,
                current_points[:, 2],
            )
            metric_label = 'Error angular (graus)'
            title_prefix = 'original.py - ERROR ANGULAR'

        worst_for_visual = int(np.argmax(vis_errors))

        if log_interval > 0 and iteration % log_interval == 0:
            unit = 'm' if metric == 'height' else 'graus'
            print(
                f"  [original][{metric}] Iter {iteration}: "
                f"error_max={vis_errors[worst_for_visual]:.2f} {unit}"
            )

        if iteration % snapshot_interval == 0:
            _save_error_snapshot(
                output_dir,
                iteration,
                converter.rows,
                converter.cols,
                spacing,
                p_indices,
                vis_errors,
                worst_for_visual,
                tin.points,
                tin.simplices,
                current_points[-1] if len(current_points) > 0 else None,
                title_prefix,
                metric_label,
                vmin,
                vmax,
            )

        s_indices.append(point_to_add_global_index)
        p_indices.pop(worst_for_selection)

    print(f"✓ {iteration} snapshots generats a {output_dir}/")
    return output_dir


def _initialize_pendent6(conv):
    total_points = conv.rows * conv.cols
    corner_indices = [0, conv.cols - 1, (conv.rows - 1) * conv.cols, total_points - 1]
    candidate_indices = list(set(range(total_points)) - set(corner_indices))

    initial_points_xy = []
    conv.tin_z_values = []
    for idx in corner_indices:
        xy, z = conv._get_coords_from_index(idx)
        initial_points_xy.append(xy)
        conv.tin_z_values.append(z)

    corner_rows, corner_cols = np.divmod(corner_indices, conv.cols)
    corner_z = conv.h_grid[corner_rows, corner_cols]

    a_mat = np.c_[corner_cols, corner_rows, np.ones(len(corner_indices))]
    coef, _, _, _ = np.linalg.lstsq(a_mat, corner_z, rcond=None)
    a, b, c = coef

    rows_grid, cols_grid = np.indices((conv.rows, conv.cols))
    z_pred = a * cols_grid + b * rows_grid + c
    residual = np.abs(conv.h_grid - z_pred).ravel()
    residual[corner_indices] = -np.inf

    max_err_idx = int(np.argmax(residual))
    new_xy, new_z = conv._get_coords_from_index(max_err_idx)
    initial_points_xy.append(new_xy)
    conv.tin_z_values.append(new_z)
    candidate_indices.remove(max_err_idx)

    conv.tin = Delaunay(np.array(initial_points_xy), incremental=True)
    return candidate_indices


def generate_pendent_error_snapshots(npy_file, target_points, algorithm, p7_mode,
                                     metric, output_dir, snapshot_interval, vmin, vmax,
                                     log_interval=10):
    print(f"Generant snapshots de {metric} per {algorithm}.py...")
    os.makedirs(output_dir, exist_ok=True)

    if algorithm == 'pendent6':
        conv = GridToTinIncrementalP6(step=1, pixel_size=2.0, target_point_count=target_points)
    else:
        conv = GridToTinIncrementalP7(
            step=1,
            pixel_size=2.0,
            target_point_count=target_points,
            mode=p7_mode,
        )

    if not conv._load_data(npy_file):
        return output_dir

    if algorithm == 'pendent6':
        candidate_indices = _initialize_pendent6(conv)
    else:
        candidate_indices = conv._initialize_tin()

    spacing = conv.step * conv.pixel_size
    iteration = 0

    while len(conv.tin.points) < conv.target_point_count and len(candidate_indices) > 0:
        iteration += 1

        # Selecció original del mètode pendent: sempre basada en error angular/ponderat
        if algorithm == 'pendent6':
            weighted_errors = conv._calculate_weighted_angular_error(candidate_indices)
            worst_local = int(np.argmax(weighted_errors))
        else:
            worst_local, _ = conv._select_next(candidate_indices)

        worst_global = candidate_indices[worst_local]

        if metric == 'angular':
            vis_errors = conv._calculate_angular_error(np.asarray(candidate_indices))
            metric_label = 'Error angular (graus)'
            if algorithm == 'pendent7':
                title_prefix = f"pendent7.py [{conv.mode}] - ERROR ANGULAR"
            else:
                title_prefix = 'pendent6.py - ERROR ANGULAR'
        else:
            vis_errors = _compute_height_error_from_tin(conv, candidate_indices)
            metric_label = 'Error alçada (m)'
            if algorithm == 'pendent7':
                title_prefix = f"pendent7.py [{conv.mode}] - ERROR D'ALÇADA"
            else:
                title_prefix = "pendent6.py - ERROR D'ALÇADA"

        worst_for_visual = int(np.argmax(vis_errors))

        if log_interval > 0 and iteration % log_interval == 0:
            unit = 'm' if metric == 'height' else 'graus'
            mode_suffix = f"[{conv.mode}]" if algorithm == 'pendent7' else ''
            print(
                f"  [{algorithm}]{mode_suffix}[{metric}] Iter {iteration}: "
                f"error_max={vis_errors[worst_for_visual]:.2f} {unit}"
            )

        if iteration % snapshot_interval == 0:
            _save_error_snapshot(
                output_dir,
                iteration,
                conv.rows,
                conv.cols,
                spacing,
                candidate_indices,
                vis_errors,
                worst_for_visual,
                conv.tin.points,
                conv.tin.simplices,
                conv.tin.points[-1] if len(conv.tin.points) > 0 else None,
                title_prefix,
                metric_label,
                vmin,
                vmax,
            )

        new_xy, new_z = conv._get_coords_from_index(worst_global)
        conv.tin.add_points([new_xy])
        conv.tin_z_values.append(new_z)
        candidate_indices.pop(worst_local)

    print(f"✓ {iteration} snapshots generats a {output_dir}/")
    return output_dir


def create_side_by_side_video(dir_original, dir_pendent, output_file='video_error_comparison_6.1.mp4'):

    print(f"\nCombinant snapshots en vídeo side-by-side...")
    
    snapshots_o = sorted([f for f in os.listdir(dir_original) if f.endswith('.png')])
    snapshots_p = sorted([f for f in os.listdir(dir_pendent) if f.endswith('.png')])
    
    n_frames = min(len(snapshots_o), len(snapshots_p))
    print(f"  Frames a combinar: {n_frames}")
    
    # Crear frames combinats
    combined_dir = 'temp_combined_error_frames'
    os.makedirs(combined_dir, exist_ok=True)
    
    for i in range(n_frames):
        img_o = Image.open(os.path.join(dir_original, snapshots_o[i]))
        img_p = Image.open(os.path.join(dir_pendent, snapshots_p[i]))
        
        # Calcular dimensions (han de ser divisibles per 2 per H.264)
        width = img_o.width + img_p.width
        height = max(img_o.height, img_p.height)
        
        # Ajustar a dimensions parelles
        if width % 2 != 0:
            width += 1
        if height % 2 != 0:
            height += 1
        
        combined = Image.new('RGB', (width, height), (255, 255, 255))
        combined.paste(img_p, (0, 0))
        combined.paste(img_o, (img_p.width, 0))
        
        combined.save(os.path.join(combined_dir, f'combined_{i:04d}.png'))
        
        if (i + 1) % 50 == 0:
            print(f"    Frame {i+1}/{n_frames}")
    
    # Crear MP4
    print(f"  Generant MP4...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', '10',
        '-i', os.path.join(combined_dir, 'combined_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        output_file
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"✓ Vídeo guardat: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error amb ffmpeg:")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
    except FileNotFoundError as e:
        print(f"✗ ffmpeg no trobat: {e}")
    
    # Netejar
    for f in os.listdir(combined_dir):
        os.remove(os.path.join(combined_dir, f))
    os.rmdir(combined_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Genera 2 vídeos: comparació d\'error angular i d\'error d\'alçada.'
    )
    parser.add_argument('--npy', default='bassiero.npy', help='Fitxer .npy d\'entrada')
    parser.add_argument('--target-points', type=int, default=2000, help='Nombre de punts objectiu')
    parser.add_argument('--algorithm', choices=['pendent6', 'pendent7'], default='pendent7',
                        help='Algoritme incremental per comparar amb original.py')
    parser.add_argument('--p7-mode', choices=['point', 'triangle', 'mean_normal'], default='mean_normal',
                        help='Mode de pendent7 (només s\'usa amb --algorithm pendent7)')
    parser.add_argument('--snapshot-interval', type=int, default=5,
                        help='Cada quantes iteracions es guarda un frame')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Cada quantes iteracions es mostra error per terminal (0 = desactivar)')
    parser.add_argument('--vmin-angular', type=float, default=None, help='Vmin de la barra angular')
    parser.add_argument('--vmax-angular', type=float, default=None, help='Vmax de la barra angular')
    parser.add_argument('--vmin-height', type=float, default=None, help='Vmin de la barra d\'alçada')
    parser.add_argument('--vmax-height', type=float, default=None, help='Vmax de la barra d\'alçada')
    parser.add_argument('--output-prefix', default='video_error_comparison',
                        help='Prefix dels fitxers de vídeo')

    args = parser.parse_args()

    ranges = _default_metric_ranges()
    angular_vmin = ranges['angular'][0] if args.vmin_angular is None else args.vmin_angular
    angular_vmax = ranges['angular'][1] if args.vmax_angular is None else args.vmax_angular
    height_vmin = ranges['height'][0] if args.vmin_height is None else args.vmin_height
    height_vmax = ranges['height'][1] if args.vmax_height is None else args.vmax_height

    out_dir_o_ang = 'snapshots_error_original_angular'
    #out_dir_o_hgt = 'snapshots_error_original_height'

    if args.algorithm == 'pendent7':
        pend_suffix = f"{args.algorithm}_{args.p7_mode}"
    else:
        pend_suffix = args.algorithm

    #out_dir_p_ang = f"snapshots_error_{pend_suffix}_angular"
    out_dir_p_hgt = f"snapshots_error_{pend_suffix}_height"

    dir_original_angular = 'snapshots_error_original_angular'  # Placeholder per no generar aquest vídeo ara
    # dir_original_angular = generate_original_error_snapshots(
    #     args.npy,
    #     args.target_points,
    #     metric='angular',
    #     output_dir=out_dir_o_ang,
    #     snapshot_interval=args.snapshot_interval,
    #     vmin=angular_vmin,
    #     vmax=angular_vmax,
    #     log_interval=args.log_interval,
    # )

    dir_pendent_angular = generate_pendent_error_snapshots(
        args.npy,
        args.target_points,
        algorithm=args.algorithm,
        p7_mode=args.p7_mode,
        metric='angular',
        output_dir='snapshots_error_pendent_p7.2',
        snapshot_interval=args.snapshot_interval,
        vmin=angular_vmin,
        vmax=angular_vmax,
        log_interval=args.log_interval,
        )
    #dir_pendent_angular = 'snapshots_error_pendent_p7.1'  # Placeholder per no generar aquest vídeo ara

    angular_video = "video_error_comparison_error_angular_7.2.mp4"
    create_side_by_side_video(dir_original_angular, dir_pendent_angular, angular_video)

    # dir_original_height = generate_original_error_snapshots(
    #     args.npy,
    #     args.target_points,
    #     metric='height',
    #     output_dir=out_dir_o_hgt,
    #     snapshot_interval=args.snapshot_interval,
    #     vmin=height_vmin,
    #     vmax=height_vmax,
    #     log_interval=args.log_interval,
    # )

    dir_original_height = 'snapshots_error_original_height'  # Placeholder per no generar aquest vídeo ara

    # dir_pendent_height = generate_pendent_error_snapshots(
    #     args.npy,
    #     args.target_points,
    #     algorithm = 'pendent7',
    #     p7_mode=args.p7_mode,
    #     metric='height',
    #     output_dir="snapshots_error_pendent7.2_height",
    #     snapshot_interval=args.snapshot_interval,
    #     vmin=height_vmin,
    #     vmax=height_vmax,
    #     log_interval=args.log_interval,
    # )

    #height_video = "video_error_comparison_error_height_7.2.mp4"
    #create_side_by_side_video(dir_original_height, dir_pendent_height, height_video)

    print()
    print('✓ PROCÉS COMPLETAT')
    print(f"  Vídeo angular: {angular_video}")
    #print(f"  Vídeo alçada: {height_video}")
    print(f"  Escala angular: vmin={angular_vmin}, vmax={angular_vmax}")
    print(f"  Escala alçada: vmin={height_vmin}, vmax={height_vmax}")


if __name__ == '__main__':
    main()
