import rawpy
import imageio
import cv2
import numpy as np
from pathlib import Path
import subprocess

def estimate_focus_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def get_xmp_path(image_path):
    return image_path.with_suffix('.xmp')

def write_xmp_tag(xmp_path, tags):
    args = ['exiftool', '-overwrite_original']
    for k, v in tags.items():
        args.append(f'-XMP:{k}={v}')
    args.append(str(xmp_path))
    subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_arw_file(arw_path):
    print(f'📷 Processing: {arw_path.name}')
    with rawpy.imread(str(arw_path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
    focus_score = estimate_focus_score(rgb)
    print(f'🔎 Focus score: {focus_score:.2f}')

    xmp_path = get_xmp_path(arw_path)
    write_xmp_tag(xmp_path, {'FocusScore': f'{focus_score:.2f}'})

def main(folder_path):
    arw_files = Path(folder_path).rglob("*.ARW")
    for arw_file in arw_files:
        process_arw_file(arw_file)

if __name__ == "__main__":
    folder = input("Введите путь к папке с .ARW файлами: ")
    main(folder)
