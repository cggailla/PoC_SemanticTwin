"""
Validate semantic anchor clusters from settings.yaml.

This script runs the ClusterValidator on specified dimensions
and generates a quality report.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from modules.cluster_validator import ClusterValidator, ClusterValidatorConfig


def load_dimensions(settings_path: Path) -> dict:
    """Load dimension configurations from settings.yaml.
    
    Args:
        settings_path: Path to settings.yaml.
    
    Returns:
        Dictionary of dimension configs.
    """
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    
    return settings["probes"]["vector_probe"]["params"]["dimensions"]


def main(dimension_name: str | None = None) -> int:
    """Validate anchor clusters.
    
    Args:
        dimension_name: Specific dimension to validate, or None for all.
    
    Returns:
        Exit code.
    """
    base_path = Path(__file__).parent
    settings_path = base_path / "config" / "settings.yaml"
    output_dir = base_path / "output" / "validations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dimensions
    all_dimensions = load_dimensions(settings_path)
    
    if dimension_name:
        if dimension_name not in all_dimensions:
            print(f"ERROR: Dimension '{dimension_name}' not found.")
            print(f"Available: {list(all_dimensions.keys())}")
            return 1
        dimensions = {dimension_name: all_dimensions[dimension_name]}
    else:
        dimensions = all_dimensions
    
    print(f"📋 Validating {len(dimensions)} dimension(s)...")
    print()
    
    # Initialize validator
    validator = ClusterValidator()
    config = ClusterValidatorConfig()
    
    reports = []
    
    for dim_name, dim_config in dimensions.items():
        print(f"🔍 Validating: {dim_name}")
        
        report = validator.validate_dimension(
            dimension_name=dim_name,
            anchor_a=dim_config["anchor_a"],
            anchor_b=dim_config["anchor_b"],
            config=config,
        )
        reports.append(report)
        
        # Print summary
        print(f"   {report.to_emoji_status()}")
        print(f"   Legacy: density={report.metrics['legacy'].density:.2f}, variance={report.metrics['legacy'].variance:.3f}")
        print(f"   Strategy: density={report.metrics['strategy'].density:.2f}, variance={report.metrics['strategy'].variance:.3f}")
        print(f"   Separation: {report.axis_separation:.3f}")
        
        if report.outliers_detected:
            print(f"   ⚠️  Outliers: {len(report.outliers_detected)}")
            for outlier in report.outliers_detected[:3]:
                print(f"      - '{outlier.word}' ({outlier.cluster}, z={outlier.z_score})")
        
        print()
    
    # Save cache
    validator.save_cache()
    
    # Save report
    output_data = {
        "overall_status": (
            "BROKEN" if any(r.status == "BROKEN" for r in reports) else
            "WARNING" if any(r.status == "WARNING" for r in reports) else
            "VALID"
        ),
        "dimensions": [r.model_dump() for r in reports],
    }
    
    output_file = output_dir / f"cluster_validation_{dimension_name or 'all'}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print("=" * 60)
    print(f"Overall Status: {output_data['overall_status']}")
    print(f"Report saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    dim = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(dim))
