"""
Example usage of NPC Surgical Planning Tool
"""

from src.visualization import generate_surgical_heatmap, generate_multi_slice_montage
from src.distance_calculator import calculate_landmark_distances
from src.report_generator import generate_clinical_report

# Example paths (update these to your actual files)
CT_PATH = "path/to/patient_ct.nii.gz"
SEG_PATH = "path/to/patient_seg.nii.gz"
OUTPUT_DIR = "path/to/output/"
CASE_ID = "Patient_001"

# Example 1: Generate single-slice heatmap
print("Generating surgical heatmap...")
heatmap_path = generate_surgical_heatmap(
    ct_path=CT_PATH,
    seg_path=SEG_PATH,
    output_dir=OUTPUT_DIR,
    slice_number=60,  # Nasopharynx level
    case_id=CASE_ID
)
print(f"✓ Heatmap saved: {heatmap_path}")

# Example 2: Generate multi-slice montage
print("\nGenerating multi-slice montage...")
montage_path = generate_multi_slice_montage(
    ct_path=CT_PATH,
    seg_path=SEG_PATH,
    output_dir=OUTPUT_DIR,
    slice_range=(55, 70),  # Slices through nasopharynx region
    case_id=CASE_ID,
    rows=3,
    cols=4
)
print(f"✓ Montage saved: {montage_path}")

# Example 3: Calculate distances
print("\nCalculating distances...")
distances = calculate_landmark_distances(
    ct_path=CT_PATH,
    seg_path=SEG_PATH,
    save_distance_map=True,
    output_path=f"{OUTPUT_DIR}/{CASE_ID}_distance_map.nii.gz"
)

print("\nDistance Summary:")
for landmark, metrics in distances.items():
    print(f"{landmark}: {metrics['min_distance_mm']:.2f}mm ({metrics['risk_level']})")

# Example 4: Generate complete clinical report
print("\nGenerating clinical report...")
report_path = generate_clinical_report(
    ct_path=CT_PATH,
    seg_path=SEG_PATH,
    case_id=CASE_ID,
    output_path=f"{OUTPUT_DIR}/{CASE_ID}_report.json",
    patient_info={
        'patient_id': 'MH001234567',
        'diagnosis': 'Nasopharyngeal Carcinoma',
        'age': '52',
        'sex': 'Male'
    }
)
print(f"✓ Report saved: {report_path}")
