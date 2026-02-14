"""
Example: Simulate 2% ultra-low-dose from a PNG image and save result as PNG.
"""
from pathlib import Path
from sim_functions import apply_ld_sinogram_from_png


def main():
    """
    Read a high-dose PNG image, simulate 2% low-dose, and save as PNG.
    
    Modify the input/output paths as needed.
    """
    # Input directory containing high-dose PNGs (batch)
    png_dir = Path(r"C:/Users/pytorch/Desktop/Dataset/dataset/train/gt/")
    out_dir = Path(r"C:/Users/pytorch/Desktop/Dataset/dataset/simulated/2percent/tradition")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simulate 2% dose. The PNG-based function assumes a reference exposure of 125 mAs,
    # so 2% corresponds to mas_des = 125 * 0.02 = 2.5
    mas_des = 2.5

    png_files = sorted([p for p in png_dir.iterdir() if p.suffix.lower() == '.png'])
    if not png_files:
        print(f"No PNG files found in {png_dir}. Update `png_dir` to point to your images.")
        return

    print(f"Found {len(png_files)} PNG files in {png_dir}. Processing...")
    for src in png_files:
        dst = out_dir / src.name
        print(f"Processing {src.name} -> {dst.name}")
        try:
            apply_ld_sinogram_from_png(str(src), str(dst), mas_des=mas_des, print_logs=False)
        except Exception as e:
            print(f"Failed processing {src.name}: {e}")

    print(f"✓ All done. Low-dose PNGs saved to: {out_dir}")


if __name__ == '__main__':
    main()
