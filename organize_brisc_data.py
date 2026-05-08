#!/usr/bin/env python3
"""
Organize BRISC2025 dataset by anatomical orientation.

Creates brisc2025divided folder with same structure as brisc2025,
but organizes images into AX (axial), CO (coronal), SA (sagittal) subfolders
based on the image filename suffixes (ax_t1, co_t1, sa_t1).
"""

import os
import shutil
from pathlib import Path


def get_orientation(filename):
    """Extract orientation from filename (ax, co, or sa)."""
    if "_ax_t1" in filename:
        return "AX"
    elif "_co_t1" in filename:
        return "CO"
    elif "_sa_t1" in filename:
        return "SA"
    else:
        return None


def create_directory_structure(source_root, dest_root):
    """Create the destination directory structure mirroring source."""
    for root, dirs, files in os.walk(source_root):
        rel_path = os.path.relpath(root, source_root)
        dest_path = os.path.join(dest_root, rel_path)
        os.makedirs(dest_path, exist_ok=True)


def organize_classification_images(source_root, dest_root):
    """Organize classification task images by orientation."""
    class_path = os.path.join(source_root, "classification_task")
    if not os.path.exists(class_path):
        print(f"Classification folder not found: {class_path}")
        return

    for split in ["train", "test"]:
        split_path = os.path.join(class_path, split)
        dest_split_path = os.path.join(dest_root, "classification_task", split)

        if not os.path.exists(split_path):
            continue

        # Get all tumor types
        tumor_types = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]

        for tumor_type in tumor_types:
            tumor_src = os.path.join(split_path, tumor_type)
            tumor_dest = os.path.join(dest_split_path, tumor_type)

            # Create orientation folders
            for orientation in ["AX", "CO", "SA"]:
                orient_dir = os.path.join(tumor_dest, orientation)
                os.makedirs(orient_dir, exist_ok=True)

            # Copy and organize images
            for filename in os.listdir(tumor_src):
                file_path = os.path.join(tumor_src, filename)
                if not os.path.isfile(file_path):
                    continue

                orientation = get_orientation(filename)
                if orientation:
                    dest_file = os.path.join(tumor_dest, orientation, filename)
                    shutil.copy2(file_path, dest_file)
                    print(f"Copied: {filename} -> {orientation}/")


def organize_segmentation_images(source_root, dest_root):
    """Organize segmentation task images and masks by orientation."""
    seg_path = os.path.join(source_root, "segmentation_task")
    if not os.path.exists(seg_path):
        print(f"Segmentation folder not found: {seg_path}")
        return

    for split in ["train", "test"]:
        split_path = os.path.join(seg_path, split)
        dest_split_path = os.path.join(dest_root, "segmentation_task", split)

        if not os.path.exists(split_path):
            continue

        # Organize images
        images_src = os.path.join(split_path, "images")
        if os.path.exists(images_src):
            images_dest = os.path.join(dest_split_path, "images")

            # Create orientation folders
            for orientation in ["AX", "CO", "SA"]:
                orient_dir = os.path.join(images_dest, orientation)
                os.makedirs(orient_dir, exist_ok=True)

            # Copy and organize images
            for filename in os.listdir(images_src):
                file_path = os.path.join(images_src, filename)
                if not os.path.isfile(file_path):
                    continue

                orientation = get_orientation(filename)
                if orientation:
                    dest_file = os.path.join(images_dest, orientation, filename)
                    shutil.copy2(file_path, dest_file)
                    print(f"Copied image: {filename} -> {orientation}/")

        # Organize masks
        masks_src = os.path.join(split_path, "masks")
        if os.path.exists(masks_src):
            masks_dest = os.path.join(dest_split_path, "masks")

            # Create orientation folders
            for orientation in ["AX", "CO", "SA"]:
                orient_dir = os.path.join(masks_dest, orientation)
                os.makedirs(orient_dir, exist_ok=True)

            # Copy and organize masks
            for filename in os.listdir(masks_src):
                file_path = os.path.join(masks_src, filename)
                if not os.path.isfile(file_path):
                    continue

                orientation = get_orientation(filename)
                if orientation:
                    dest_file = os.path.join(masks_dest, orientation, filename)
                    shutil.copy2(file_path, dest_file)
                    print(f"Copied mask: {filename} -> {orientation}/")


def main():
    """Main function to orchestrate the data organization."""
    # Define paths
    data_dir = Path(__file__).parent / "data"
    source_root = str(data_dir / "brisc2025")
    dest_root = str(data_dir / "brisc2025divided")

    # Validate source
    if not os.path.exists(source_root):
        print(f"Error: Source directory not found: {source_root}")
        return

    # Create destination root
    os.makedirs(dest_root, exist_ok=True)
    print(f"Creating organized dataset in: {dest_root}\n")

    # Copy metadata files
    for filename in ["README.md", "manifest.csv", "manifest.json"]:
        src = os.path.join(source_root, filename)
        dest = os.path.join(dest_root, filename)
        if os.path.exists(src):
            shutil.copy2(src, dest)
            print(f"Copied metadata: {filename}")

    print("\n--- Organizing Classification Task ---")
    organize_classification_images(source_root, dest_root)

    print("\n--- Organizing Segmentation Task ---")
    organize_segmentation_images(source_root, dest_root)

    print(f"\n[SUCCESS] Dataset organization complete!")
    print(f"Output folder: {dest_root}")


if __name__ == "__main__":
    main()
