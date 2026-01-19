"""
Generate visualization report from the latest audit JSON.

This script loads the most recent audit result and generates interactive
HTML visualizations.
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.reporting import AuditVisualizer


def find_latest_audit(output_dir: Path) -> Path | None:
    """Find the most recent audit JSON file.
    
    Args:
        output_dir: Directory containing audit files.
    
    Returns:
        Path to latest audit file, or None if not found.
    """
    json_files = list(output_dir.glob("audit_*.json"))
    if not json_files:
        return None
    return max(json_files, key=lambda p: p.stat().st_mtime)


def main() -> int:
    """Generate visualization report.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    output_dir = Path(__file__).parent / "output"
    
    # Find latest audit
    latest_audit = find_latest_audit(output_dir)
    if latest_audit is None:
        print("ERROR: No audit files found in output directory")
        return 1
    
    print(f"Loading audit: {latest_audit.name}")
    
    # Generate visualizations
    visualizer = AuditVisualizer(output_dir=output_dir / "visuals")
    
    radar_path, bars_path, combined_path = visualizer.generate_report(latest_audit)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION REPORT GENERATED")
    print("=" * 60)
    print(f"\n📊 Radar Chart: {radar_path}")
    print(f"📊 Dimension Bars: {bars_path}")
    print(f"📊 Combined Report: {combined_path}")
    print(f"\nOpen the combined report in your browser:")
    print(f"  {combined_path.absolute()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
