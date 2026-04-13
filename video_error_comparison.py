
import os
from PIL import Image
import subprocess

from original import GridToTinConverter
from pendent7 import GridToTinIncremental


def generate_error_video_original(npy_file, target_points=500, output_dir='snapshots_error_original'):

    print(f"Generant snapshots d'error per original.py amb {target_points} punts...")
    
    converter = GridToTinConverter(step=1, pixel_size=2.0, 
                                   control_mode='POINT_COUNT',
                                   target_point_count=target_points)
    converter.fit_with_error_snapshots(npy_file, snapshot_dir=output_dir, snapshot_interval=5)
    
    return output_dir


def generate_error_video_pendent(npy_file, target_points=500, output_dir='snapshots_error_pendent_p7.1'):

    print(f"Generant snapshots d'error per pendent7.py amb {target_points} punts...")
    
    conv = GridToTinIncremental(step=1, pixel_size=2.0, target_point_count=target_points)
    conv.fit_with_error_snapshots(npy_file, snapshot_dir=output_dir, snapshot_interval=5)
    
    return output_dir


def create_side_by_side_video(dir_original, dir_pendent, output_file='video_error_comparison.mp4'):

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
        combined.paste(img_p, (0, 0))  # pendent7 a l'esquerra
        combined.paste(img_o, (img_p.width, 0))  # original a la dreta
        
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
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
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


if __name__ == '__main__':
    NPY_FILE = 'bassiero.npy'
    TARGET_POINTS = 2000
    OUTPUT_DIR = 'snapshots_error_pendent_p7.1'
    OUTPUT_VIDEO = 'video_error_comparison_p7.1.mp4'
    
    # Generar snapshots per ambdós algoritmes
    #dir_original = generate_error_video_original(NPY_FILE, TARGET_POINTS)
    dir_original = 'snapshots_error_original'
    
    dir_pendent = generate_error_video_pendent(NPY_FILE, TARGET_POINTS, output_dir=OUTPUT_DIR)
    #dir_pendent = 'snapshots_error_pendent_p6.2'
    
    # Combinar en vídeo
    create_side_by_side_video(dir_original, dir_pendent, OUTPUT_VIDEO)
    
    print()
    print("✓ PROCÉS COMPLETAT")
