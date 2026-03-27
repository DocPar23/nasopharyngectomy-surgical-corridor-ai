"""
Visualization module for NPC surgical planning heatmaps.
Generates publication-quality images with distance overlays.
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom, distance_transform_edt
import os
from typing import Optional, List


def generate_surgical_heatmap(
    ct_path: str,
    seg_path: str,
    output_dir: str,
    slice_number: int,
    case_id: str = "Case",
    show_measurements: bool = True,
    dpi: int = 300
) -> str:
    """
    Generate surgical planning heatmap for a specific slice.
    
    Args:
        ct_path: Path to CT scan (NIFTI)
        seg_path: Path to segmentation (NIFTI)
        output_dir: Directory to save output
        slice_number: Axial slice number to visualize
        case_id: Case identifier for labeling
        show_measurements: Display distance measurements on image
        dpi: Resolution for output image
    
    Returns:
        Path to generated heatmap image
    """
    # Load data
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    
    # Ensure matching dimensions
    if ct_data.shape != seg_data.shape:
        zoom_factors = [ct_data.shape[i] / seg_data.shape[i] for i in range(3)]
        seg_data = zoom(seg_data, zoom_factors, order=0)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract masks
    tumor_mask = (seg_data == 1).astype(np.uint8)
    ica_r_mask = (seg_data == 2).astype(np.uint8)
    ica_l_mask = (seg_data == 3).astype(np.uint8)
    ica_mask = ica_r_mask | ica_l_mask
    pterygoid_mask = ((seg_data >= 6) & (seg_data <= 9)).astype(np.uint8)
    
    # Calculate distance map
    if np.sum(tumor_mask) == 0:
        raise ValueError("No tumor found in segmentation")
    
    spacing = ct_img.header.get_zooms()
    dist_map = distance_transform_edt(1 - tumor_mask, sampling=spacing)
    
    # Validate slice number
    if slice_number >= ct_data.shape[2] or slice_number < 0:
        raise ValueError(f"Slice {slice_number} out of range (0-{ct_data.shape[2]-1})")
    
    # Extract slice data
    ct_slice = ct_data[:, :, slice_number]
    dist_slice = dist_map[:, :, slice_number]
    tumor_slice = tumor_mask[:, :, slice_number]
    ica_slice = ica_mask[:, :, slice_number]
    ica_r_slice = ica_r_mask[:, :, slice_number]
    ica_l_slice = ica_l_mask[:, :, slice_number]
    pter_slice = pterygoid_mask[:, :, slice_number]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Custom colormap (red=danger, green=safe)
    colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'cyan', 'blue']
    cmap = LinearSegmentedColormap.from_list('surgical_risk', colors, N=256)
    
    # Panel 1: Original CT
    axes[0].imshow(np.rot90(ct_slice), cmap='gray', vmin=-150, vmax=350)
    axes[0].set_title('CT Scan', color='white', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Anatomical Structures
    axes[1].imshow(np.rot90(ct_slice), cmap='gray', vmin=-150, vmax=350)
    
    if np.any(tumor_slice):
        axes[1].contour(np.rot90(tumor_slice), colors='white', linewidths=2.5, levels=[0.5])
    if np.any(ica_slice):
        axes[1].contour(np.rot90(ica_slice), colors='cyan', linewidths=2.5, levels=[0.5])
    if np.any(pter_slice):
        axes[1].contour(np.rot90(pter_slice), colors='magenta', linewidths=1.5, levels=[0.5], alpha=0.7)
    
    axes[1].set_title('Anatomical Structures', color='white', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    # Legend for structures
    legend_elements = [
        plt.Line2D([0], [0], color='white', linewidth=2, label='Tumor'),
        plt.Line2D([0], [0], color='cyan', linewidth=2, label='ICA'),
        plt.Line2D([0], [0], color='magenta', linewidth=1.5, label='Pterygoids')
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10,
                   facecolor='black', edgecolor='white', labelcolor='white')
    
    # Panel 3: Distance Heatmap
    axes[2].imshow(np.rot90(ct_slice), cmap='gray', vmin=-150, vmax=350, alpha=1.0)
    
    # Apply heatmap (only where distance < 20mm and within anatomy)
    anatomy_mask = ct_slice > -500
    dist_mask = dist_slice <= 20
    final_mask = anatomy_mask & dist_mask
    
    masked_dist = np.ma.masked_where(~final_mask, dist_slice)
    im = axes[2].imshow(np.rot90(masked_dist), cmap=cmap, alpha=0.65, vmin=0, vmax=20)
    
    # Overlay contours
    if np.any(tumor_slice):
        axes[2].contour(np.rot90(tumor_slice), colors='white', linewidths=2, levels=[0.5])
    if np.any(ica_slice):
        axes[2].contour(np.rot90(ica_slice), colors='yellow', linewidths=2, levels=[0.5])
    
    # Display measurements
    if show_measurements:
        y_pos = 30
        if np.any(ica_r_slice):
            min_dist_r = dist_slice[ica_r_slice > 0].min()
            risk_r = "⚠ HIGH RISK" if min_dist_r < 5 else "⚠ CAUTION" if min_dist_r < 10 else "✓ SAFE"
            axes[2].text(15, y_pos, f"ICA Right: {min_dist_r:.1f}mm {risk_r}",
                        color='yellow', fontsize=11, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.8, pad=5))
            y_pos += 35
        
        if np.any(ica_l_slice):
            min_dist_l = dist_slice[ica_l_slice > 0].min()
            risk_l = "⚠ HIGH RISK" if min_dist_l < 5 else "⚠ CAUTION" if min_dist_l < 10 else "✓ SAFE"
            axes[2].text(15, y_pos, f"ICA Left: {min_dist_l:.1f}mm {risk_l}",
                        color='yellow', fontsize=11, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.8, pad=5))
    
    axes[2].set_title(f'Surgical Risk Heatmap - Slice {slice_number}',
                     color='white', fontsize=16, fontweight='bold')
    axes[2].axis('off')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Distance from Tumor (mm)', color='white', fontsize=12, fontweight='bold')
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=10)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontweight='bold')
    
    # Risk zone legend
    risk_legend = (
        "RISK ZONES:\n"
        "━━━━━━━━━━━━━━\n"
        "🔴 Red: 0-5mm (HIGH RISK)\n"
        "🟡 Yellow: 5-10mm (CAUTION)\n"
        "🟢 Green: >10mm (SAFE)"
    )
    fig.text(0.98, 0.02, risk_legend, color='white', fontsize=11,
            fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, pad=10),
            verticalalignment='bottom', horizontalalignment='right',
            family='monospace')
    
    # Title
    fig.suptitle(f'NPC Surgical Planning - {case_id}',
                color='white', fontsize=18, fontweight='bold', y=0.98)
    
    fig.patch.set_facecolor('black')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save
    output_path = os.path.join(output_dir, f'{case_id}_slice_{slice_number:03d}_heatmap.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='black')
    plt.close()
    
    return output_path


def generate_multi_slice_montage(
    ct_path: str,
    seg_path: str,
    output_dir: str,
    slice_range: tuple,
    case_id: str = "Case",
    rows: int = 3,
    cols: int = 4
) -> str:
    """
    Generate montage of multiple slices for comprehensive view.
    
    Args:
        ct_path: Path to CT scan
        seg_path: Path to segmentation
        output_dir: Output directory
        slice_range: (start_slice, end_slice) tuple
        case_id: Case identifier
        rows: Number of rows in montage
        cols: Number of columns in montage
    
    Returns:
        Path to montage image
    """
    # Load data
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    
    # Matching dimensions
    if ct_data.shape != seg_data.shape:
        zoom_factors = [ct_data.shape[i] / seg_data.shape[i] for i in range(3)]
        seg_data = zoom(seg_data, zoom_factors, order=0)
    
    # Extract masks
    tumor_mask = (seg_data == 1).astype(np.uint8)
    ica_mask = ((seg_data == 2) | (seg_data == 3)).astype(np.uint8)
    
    # Distance map
    spacing = ct_img.header.get_zooms()
    dist_map = distance_transform_edt(1 - tumor_mask, sampling=spacing)
    
    # Select slices
    start, end = slice_range
    total_panels = rows * cols
    slice_indices = np.linspace(start, end, total_panels, dtype=int)
    
    # Colormap
    colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'cyan', 'blue']
    cmap = LinearSegmentedColormap.from_list('surgical_risk', colors, N=256)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    
    for idx, slice_num in enumerate(slice_indices):
        if slice_num >= ct_data.shape[2]:
            axes[idx].axis('off')
            continue
        
        ct_slice = ct_data[:, :, slice_num]
        dist_slice = dist_map[:, :, slice_num]
        tumor_slice = tumor_mask[:, :, slice_num]
        ica_slice = ica_mask[:, :, slice_num]
        
        # Display
        axes[idx].imshow(np.rot90(ct_slice), cmap='gray', vmin=-150, vmax=350)
        
        # Heatmap
        anatomy_mask = ct_slice > -500
        dist_mask = dist_slice <= 20
        final_mask = anatomy_mask & dist_mask
        masked_dist = np.ma.masked_where(~final_mask, dist_slice)
        axes[idx].imshow(np.rot90(masked_dist), cmap=cmap, alpha=0.5, vmin=0, vmax=20)
        
        # Contours
        if np.any(tumor_slice):
            axes[idx].contour(np.rot90(tumor_slice), colors='white', linewidths=1, levels=[0.5])
        if np.any(ica_slice):
            axes[idx].contour(np.rot90(ica_slice), colors='cyan', linewidths=1, levels=[0.5])
        
        # Label
        axes[idx].text(10, 20, f'Slice {slice_num}', color='white', fontsize=10,
                      fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))
        axes[idx].axis('off')
    
    fig.suptitle(f'NPC Surgical Planning Montage - {case_id}',
                color='white', fontsize=16, fontweight='bold')
    fig.patch.set_facecolor('black')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_id}_montage.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python visualization.py <ct_path> <seg_path> <output_dir> <slice_number>")
        sys.exit(1)
    
    ct_path = sys.argv[1]
    seg_path = sys.argv[2]
    output_dir = sys.argv[3]
    slice_num = int(sys.argv[4])
    
    print(f"Generating surgical heatmap for slice {slice_num}...")
    output = generate_surgical_heatmap(
        ct_path=ct_path,
        seg_path=seg_path,
        output_dir=output_dir,
        slice_number=slice_num
    )
    
    print(f"✓ Heatmap saved: {output}")
