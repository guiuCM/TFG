#!/usr/bin/env python3
"""
Script per generar vídeos comparatius de la triangulació
entre pendent6.py (angular error) i original.py (height error)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pendent6 import GridToTinIncremental
from original import GridToTinConverter
import os

def create_video_pendent6(npy_file, target_points=500, output_file='video_pendent6.mp4'):
    """
    Genera vídeo de la triangulació amb pendent6.py
    """
    print(f"Generant vídeo per pendent6.py amb {target_points} punts...")
    
    # Configurar algoritme amb snapshots
    snapshot_dir = 'temp_snapshots_pendent6'
    os.makedirs(snapshot_dir, exist_ok=True)
    
    conv = GridToTinIncremental(step=1, pixel_size=2.0, target_point_count=target_points)
    conv.fit(npy_file, snapshot_dir=snapshot_dir, snapshot_interval=10)
    
    # Llegir snapshots
    snapshot_files = sorted([f for f in os.listdir(snapshot_dir) if f.endswith('.png')])
    
    if not snapshot_files:
        print("Error: No s'han generat snapshots!")
        return
    
    # Crear vídeo amb matplotlib animation
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame_num):
        ax.clear()
        img_path = os.path.join(snapshot_dir, snapshot_files[frame_num])
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'pendent6.py - Frame {frame_num + 1}/{len(snapshot_files)}', 
                     fontsize=16, fontweight='bold')
        return [ax]
    
    anim = FuncAnimation(fig, update, frames=len(snapshot_files), 
                        interval=200, blit=True)
    
    # Guardar com vídeo
    writer = FFMpegWriter(fps=5, bitrate=2000)
    anim.save(output_file, writer=writer)
    plt.close()
    
    print(f"✓ Vídeo guardat: {output_file}")
    
    # Netejar snapshots temporals
    for f in snapshot_files:
        os.remove(os.path.join(snapshot_dir, f))
    os.rmdir(snapshot_dir)


def create_video_original(npy_file, target_points=500, output_file='video_original.mp4'):
    """
    Genera vídeo de la triangulació amb original.py
    """
    print(f"Generant vídeo per original.py amb {target_points} punts...")
    
    # Configurar algoritme amb snapshots
    snapshot_dir = 'temp_snapshots_original'
    os.makedirs(snapshot_dir, exist_ok=True)
    
    conv = GridToTinConverter(step=1, pixel_size=2.0, target_point_count=target_points)
    conv.fit(npy_file, snapshot_dir=snapshot_dir, snapshot_interval=10)
    
    # Llegir snapshots
    snapshot_files = sorted([f for f in os.listdir(snapshot_dir) if f.endswith('.png')])
    
    if not snapshot_files:
        print("Error: No s'han generat snapshots!")
        return
    
    # Crear vídeo amb matplotlib animation
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame_num):
        ax.clear()
        img_path = os.path.join(snapshot_dir, snapshot_files[frame_num])
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'original.py - Frame {frame_num + 1}/{len(snapshot_files)}', 
                     fontsize=16, fontweight='bold')
        return [ax]
    
    anim = FuncAnimation(fig, update, frames=len(snapshot_files), 
                        interval=200, blit=True)
    
    # Guardar com vídeo
    writer = FFMpegWriter(fps=5, bitrate=2000)
    anim.save(output_file, writer=writer)
    plt.close()
    
    print(f"✓ Vídeo guardat: {output_file}")
    
    # Netejar snapshots temporals
    for f in snapshot_files:
        os.remove(os.path.join(snapshot_dir, f))
    os.rmdir(snapshot_dir)


def create_side_by_side_video(npy_file, target_points=500, output_file='video_comparison.mp4'):
    """
    Genera vídeo amb ambdós algoritmes side-by-side
    """
    print(f"Generant vídeo comparatiu amb {target_points} punts...")
    
    # Generar snapshots per ambdós
    snapshot_dir_pendent = 'temp_snapshots_pendent6_comp'
    snapshot_dir_original = 'temp_snapshots_original_comp'
    
    os.makedirs(snapshot_dir_pendent, exist_ok=True)
    os.makedirs(snapshot_dir_original, exist_ok=True)
    
    print("  Executant pendent6.py...")
    conv_p = GridToTinIncremental(step=1, pixel_size=2.0, target_point_count=target_points)
    conv_p.fit(npy_file, snapshot_dir=snapshot_dir_pendent, snapshot_interval=10)
    
    print("  Executant original.py...")
    conv_o = GridToTinConverter(step=1, pixel_size=2.0, target_point_count=target_points)
    conv_o.fit(npy_file, snapshot_dir=snapshot_dir_original, snapshot_interval=10)
    
    # Llegir snapshots
    snapshots_p = sorted([f for f in os.listdir(snapshot_dir_pendent) if f.endswith('.png')])
    snapshots_o = sorted([f for f in os.listdir(snapshot_dir_original) if f.endswith('.png')])
    
    # Utilitzar el mínim nombre de frames
    n_frames = min(len(snapshots_p), len(snapshots_o))
    
    # Crear vídeo comparatiu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Comparació: Angular Error vs Height Error ({target_points} punts)', 
                 fontsize=18, fontweight='bold')
    
    def update(frame_num):
        ax1.clear()
        ax2.clear()
        
        # Pendent6
        img_p = plt.imread(os.path.join(snapshot_dir_pendent, snapshots_p[frame_num]))
        ax1.imshow(img_p)
        ax1.axis('off')
        ax1.set_title('pendent6.py (Angular Error)', fontsize=14, fontweight='bold')
        
        # Original
        img_o = plt.imread(os.path.join(snapshot_dir_original, snapshots_o[frame_num]))
        ax2.imshow(img_o)
        ax2.axis('off')
        ax2.set_title('original.py (Height Error)', fontsize=14, fontweight='bold')
        
        return [ax1, ax2]
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=True)
    
    # Guardar
    writer = FFMpegWriter(fps=5, bitrate=3000)
    anim.save(output_file, writer=writer)
    plt.close()
    
    print(f"✓ Vídeo comparatiu guardat: {output_file}")
    
    # Netejar
    for f in snapshots_p:
        os.remove(os.path.join(snapshot_dir_pendent, f))
    for f in snapshots_o:
        os.remove(os.path.join(snapshot_dir_original, f))
    os.rmdir(snapshot_dir_pendent)
    os.rmdir(snapshot_dir_original)


if __name__ == '__main__':
    NPY_FILE = 'bassiero.npy'
    TARGET_POINTS = 500
    
    print("="*70)
    print("GENERACIÓ DE VÍDEOS COMPARATIUS")
    print("="*70)
    print()
    
    # Opció 1: Vídeos individuals
    # create_video_pendent6(NPY_FILE, TARGET_POINTS, 'video_pendent6_500pts.mp4')
    # create_video_original(NPY_FILE, TARGET_POINTS, 'video_original_500pts.mp4')
    
    # Opció 2: Vídeo side-by-side (recomanat)
    create_side_by_side_video(NPY_FILE, TARGET_POINTS, 'video_comparison_500pts.mp4')
    
    print()
    print("="*70)
    print("✓ PROCÉS COMPLETAT")
    print("="*70)
