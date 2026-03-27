"""
Clinical report generation module for NPC surgical planning.
Creates structured reports with distance tables and risk assessments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import json
from datetime import datetime
from src.distance_calculator import calculate_landmark_distances, LANDMARK_LABELS


def generate_distance_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Generate formatted distance table (similar to Table 1 in study).
    
    Args:
        results: Distance calculation results from calculate_landmark_distances()
    
    Returns:
        Pandas DataFrame with formatted results
    """
    data = []
    
    for landmark, metrics in results.items():
        data.append({
            'Anatomical Landmark': landmark.replace('_', ' ').title(),
            'Min Distance (mm)': f"{metrics['min_distance_mm']:.2f}",
            'Mean Distance (mm)': f"{metrics['mean_distance_mm']:.2f}",
            'Median Distance (mm)': f"{metrics['median_distance_mm']:.2f}",
            'Risk Level': metrics['risk_level'],
            '% Within 5mm': f"{metrics['pct_within_5mm']:.1f}%"
        })
    
    df = pd.DataFrame(data)
    return df


def generate_risk_summary(results: Dict[str, Dict[str, float]]) -> Dict[str, any]:
    """
    Generate surgical risk summary.
    
    Args:
        results: Distance calculation results
    
    Returns:
        Dictionary with risk summary
    """
    high_risk_structures = []
    caution_structures = []
    safe_structures = []
    
    for landmark, metrics in results.items():
        if metrics['risk_level'] == 'HIGH RISK':
            high_risk_structures.append(landmark)
        elif metrics['risk_level'] == 'CAUTION':
            caution_structures.append(landmark)
        else:
            safe_structures.append(landmark)
    
    # Get ICA distances (most critical)
    ica_right_dist = results.get('ICA_right', {}).get('min_distance_mm', None)
    ica_left_dist = results.get('ICA_left', {}).get('min_distance_mm', None)
    
    # Overall risk assessment
    if len(high_risk_structures) > 0:
        overall_risk = "HIGH RISK"
        surgical_recommendation = "Extreme caution required. Consider alternative treatment modalities."
    elif len(caution_structures) > 2:
        overall_risk = "MODERATE RISK"
        surgical_recommendation = "Surgical resection feasible with experienced team. Careful margin planning essential."
    else:
        overall_risk = "LOW RISK"
        surgical_recommendation = "Surgical resection favorable. Standard precautions apply."
    
    return {
        'overall_risk': overall_risk,
        'high_risk_structures': high_risk_structures,
        'caution_structures': caution_structures,
        'safe_structures': safe_structures,
        'ica_right_distance': ica_right_dist,
        'ica_left_distance': ica_left_dist,
        'surgical_recommendation': surgical_recommendation,
        'total_structures_analyzed': len(results)
    }


def generate_clinical_report(
    ct_path: str,
    seg_path: str,
    case_id: str,
    output_path: str,
    patient_info: Dict[str, str] = None
) -> str:
    """
    Generate complete clinical report.
    
    Args:
        ct_path: Path to CT scan
        seg_path: Path to segmentation
        case_id: Case identifier
        output_path: Path to save report (JSON format)
        patient_info: Optional patient information dictionary
    
    Returns:
        Path to generated report
    """
    # Calculate distances
    print("Calculating distances...")
    distance_results = calculate_landmark_distances(
        ct_path=ct_path,
        seg_path=seg_path,
        save_distance_map=True,
        output_path=output_path.replace('.json', '_distance_map.nii.gz')
    )
    
    # Generate tables and summaries
    distance_table = generate_distance_table(distance_results)
    risk_summary = generate_risk_summary(distance_results)
    
    # Compile report
    report = {
        'report_metadata': {
            'case_id': case_id,
            'generation_date': datetime.now().isoformat(),
            'ct_scan': ct_path,
            'segmentation': seg_path,
            'software_version': '1.0.0'
        },
        'patient_info': patient_info or {},
        'distance_measurements': distance_results,
        'risk_summary': risk_summary,
        'distance_table_csv': distance_table.to_csv(index=False)
    }
    
    # Save JSON report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save CSV table
    csv_path = output_path.replace('.json', '_distances.csv')
    distance_table.to_csv(csv_path, index=False)
    
    # Print summary to console
    print("\n" + "="*80)
    print(f"SURGICAL PLANNING REPORT - {case_id}")
    print("="*80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOVERALL RISK: {risk_summary['overall_risk']}")
    print(f"\nSURGICAL RECOMMENDATION:")
    print(f"  {risk_summary['surgical_recommendation']}")
    
    if risk_summary['high_risk_structures']:
        print(f"\n⚠ HIGH RISK STRUCTURES ({len(risk_summary['high_risk_structures'])}):")
        for structure in risk_summary['high_risk_structures']:
            dist = distance_results[structure]['min_distance_mm']
            print(f"  - {structure}: {dist:.2f}mm")
    
    if risk_summary['caution_structures']:
        print(f"\n⚠ CAUTION STRUCTURES ({len(risk_summary['caution_structures'])}):")
        for structure in risk_summary['caution_structures']:
            dist = distance_results[structure]['min_distance_mm']
            print(f"  - {structure}: {dist:.2f}mm")
    
    print(f"\n✓ SAFE STRUCTURES ({len(risk_summary['safe_structures'])}):")
    for structure in risk_summary['safe_structures']:
        dist = distance_results[structure]['min_distance_mm']
        print(f"  - {structure}: {dist:.2f}mm")
    
    print("\n" + "="*80)
    print(f"\nDetailed report saved to: {output_path}")
    print(f"Distance table saved to: {csv_path}")
    print("="*80 + "\n")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python report_generator.py <ct_path> <seg_path> <case_id>")
        sys.exit(1)
    
    ct_path = sys.argv[1]
    seg_path = sys.argv[2]
    case_id = sys.argv[3]
    
    output_path = f"{case_id}_surgical_report.json"
    
    generate_clinical_report(
        ct_path=ct_path,
        seg_path=seg_path,
        case_id=case_id,
        output_path=output_path
    )
