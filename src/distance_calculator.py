"""
Distance calculation module for NPC surgical planning.
Computes Euclidean distances from tumor to anatomical landmarks.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from typing import Dict, Tuple

# Landmark label mapping
LANDMARK_LABELS = {
    1: "Tumor",
    2: "ICA_right",
    3: "ICA_left",
    4: "Torus_tubarius_right",
    5: "Torus_tubarius_left",
    6: "Medial_pterygoid_right",
    7: "Medial_pterygoid_left",
    8: "Lateral_pterygoid_right",
    9: "Lateral_pterygoid_left",
    10: "Posterior_choana",
    11: "Posterior_septum"
}

# Clinical risk thresholds (mm)
HIGH_RISK_THRESHOLD = 5.0
CAUTION_THRESHOLD = 10.0


def calculate_landmark_distances(
    ct_path: str,
    seg_path: str,
    save_distance_map: bool = True,
    output_path: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate distances from tumor to all anatomical landmarks.
    
    Args:
        ct_path: Path to CT scan (NIFTI format)
        seg_path: Path to segmentation mask (NIFTI format)
        save_distance_map: Whether to save 3D distance map
        output_path: Path to save distance map (if save_distance_map=True)
    
    Returns:
        Dictionary containing distance metrics for each landmark
    """
    # Load data
    ct_img = nib.load(ct_path)
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    
    # Get voxel spacing
    spacing = ct_img.header.get_zooms()
    
    # Extract tumor mask
    tumor_mask = (seg_data == 1).astype(np.uint8)
    
    if np.sum(tumor_mask) == 0:
        raise ValueError("No tumor found in segmentation (label=1)")
    
    # Compute distance transform from tumor
    distance_map = distance_transform_edt(1 - tumor_mask, sampling=spacing)
    
    # Save distance map if requested
    if save_distance_map and output_path:
        dist_nii = nib.Nifti1Image(distance_map, ct_img.affine, ct_img.header)
        nib.save(dist_nii, output_path)
    
    # Calculate distances for each landmark
    results = {}
    
    for label_id, landmark_name in LANDMARK_LABELS.items():
        if label_id == 1:  # Skip tumor itself
            continue
        
        landmark_mask = (seg_data == label_id)
        
        if not np.any(landmark_mask):
            print(f"Warning: {landmark_name} not found in segmentation")
            continue
        
        # Get distances at landmark locations
        distances_at_landmark = distance_map[landmark_mask]
        
        # Calculate metrics
        min_dist = float(distances_at_landmark.min())
        mean_dist = float(distances_at_landmark.mean())
        median_dist = float(np.median(distances_at_landmark))
        max_dist = float(distances_at_landmark.max())
        
        # Calculate percentage within risk zones
        pct_high_risk = float((distances_at_landmark < HIGH_RISK_THRESHOLD).sum() / 
                             len(distances_at_landmark) * 100)
        pct_caution = float(((distances_at_landmark >= HIGH_RISK_THRESHOLD) & 
                            (distances_at_landmark < CAUTION_THRESHOLD)).sum() / 
                           len(distances_at_landmark) * 100)
        
        # Risk classification
        if min_dist < HIGH_RISK_THRESHOLD:
            risk_level = "HIGH RISK"
        elif min_dist < CAUTION_THRESHOLD:
            risk_level = "CAUTION"
        else:
            risk_level = "SAFE"
        
        results[landmark_name] = {
            'min_distance_mm': round(min_dist, 2),
            'mean_distance_mm': round(mean_dist, 2),
            'median_distance_mm': round(median_dist, 2),
            'max_distance_mm': round(max_dist, 2),
            'pct_within_5mm': round(pct_high_risk, 1),
            'pct_within_5_10mm': round(pct_caution, 1),
            'risk_level': risk_level,
            'voxel_count': int(np.sum(landmark_mask))
        }
    
    return results


def generate_risk_zones(
    distance_map: np.ndarray,
    thresholds: Tuple[float, float, float] = (5.0, 10.0, 15.0)
) -> np.ndarray:
    """
    Convert continuous distance map to categorical risk zones.
    
    Args:
        distance_map: 3D array of distances (mm)
        thresholds: (high_risk, caution, safe) thresholds in mm
    
    Returns:
        3D array with risk zone labels:
            0 = Background
            1 = High Risk (<5mm)
            2 = Caution (5-10mm)
            3 = Safe (10-15mm)
            4 = Very Safe (>15mm)
    """
    risk_zones = np.zeros_like(distance_map, dtype=np.uint8)
    
    high_risk, caution, safe = thresholds
    
    risk_zones[distance_map < high_risk] = 1
    risk_zones[(distance_map >= high_risk) & (distance_map < caution)] = 2
    risk_zones[(distance_map >= caution) & (distance_map < safe)] = 3
    risk_zones[distance_map >= safe] = 4
    
    return risk_zones


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python distance_calculator.py <ct_path> <seg_path>")
        sys.exit(1)
    
    ct_path = sys.argv[1]
    seg_path = sys.argv[2]
    
    print("Calculating distances from tumor to anatomical landmarks...")
    results = calculate_landmark_distances(
        ct_path=ct_path,
        seg_path=seg_path,
        save_distance_map=True,
        output_path="distance_map.nii.gz"
    )
    
    print("\n" + "="*70)
    print("DISTANCE ANALYSIS RESULTS")
    print("="*70)
    
    for landmark, metrics in results.items():
        print(f"\n{landmark}:")
        print(f"  Min Distance: {metrics['min_distance_mm']} mm")
        print(f"  Mean Distance: {metrics['mean_distance_mm']} mm")
        print(f"  Risk Level: {metrics['risk_level']}")
        print(f"  % Within 5mm: {metrics['pct_within_5mm']}%")
