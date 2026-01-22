"""
Semantic Twin Engine - Entry Point

This is the main entry point for running the Semantic Twin Engine.
It initializes the configuration, registers probes, and executes
the audit pipeline with visualization.
"""

import json
import logging
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

# Add the package to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config_loader import load_settings
from core.orchestrator import Orchestrator, probe_registry

# Register available probes
from modules.vector_probe import VectorProbe
from modules.logit_probe import LogitProbe
from modules.cluster_validator import ClusterValidator
from modules.reporting import AuditReportVisualizer


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def register_probes() -> None:
    """Register all available probe modules with the registry.
    
    Add new probes here as they are implemented.
    """
    probe_registry.register("vector_probe", VectorProbe)
    probe_registry.register("logit_probe", LogitProbe)
    probe_registry.register("cluster_validator", ClusterValidator)
    
    logging.info(
        "Registered probes: %s", 
        ", ".join(probe_registry.list_probes())
    )


def generate_visualizations(
    report_path: Path,
    settings,
    logger: logging.Logger,
) -> Path | None:
    """Generate visual dashboard from audit report.
    
    Args:
        report_path: Path to the audit JSON file.
        settings: Application settings.
        logger: Logger instance.
    
    Returns:
        Path to generated HTML dashboard, or None on failure.
    """
    try:
        # Load audit report
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        
        # Check if vector_probe ran successfully
        vector_probe_data = report_data.get("probes", {}).get("vector_probe", {})
        if not vector_probe_data.get("success"):
            logger.warning("Vector probe did not succeed, skipping visualizations")
            return None
        
        # Load dimension configs
        dimensions_config = settings.probes.get("vector_probe", {}).params.get("dimensions", {})
        
        # Generate report
        visualizer = AuditReportVisualizer()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        entity_slug = settings.entity.name.lower().replace(" ", "_")
        output_path = Path("output/visuals") / f"{entity_slug}_{timestamp}_report.html"
        
        result_path = visualizer.generate_report(
            audit_data=vector_probe_data.get("data", {}),
            dimensions_config=dimensions_config,
            entity_name=settings.entity.name,
            output_path=output_path,
            comparators=settings.anchors.entities,
        )
        
        # Save embedding cache
        visualizer.save_cache()
        
        return result_path
        
    except Exception as e:
        logger.error("Failed to generate visualizations: %s", e)
        return None


def main() -> int:
    """Main entry point for the Semantic Twin Engine.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        settings = load_settings()
        
        # Register probes
        register_probes()
        
        # Create and run orchestrator
        logger.info("Initializing orchestrator...")
        orchestrator = Orchestrator(settings)
        
        report, output_path = orchestrator.run_and_save()
        
        logger.info("=" * 60)
        logger.info("AUDIT COMPLETE")
        logger.info("=" * 60)
        logger.info("Entity: %s", report.entity_name)
        logger.info("Probes executed: %d", report.metadata["probes_executed"])
        logger.info("Succeeded: %d", report.metadata["probes_succeeded"])
        logger.info("Failed: %d", report.metadata["probes_failed"])
        logger.info("Report saved to: %s", output_path)
        
        # Generate visualizations
        logger.info("Generating visual report...")
        visual_path = generate_visualizations(
            report_path=Path(output_path),
            settings=settings,
            logger=logger,
        )
        
        if visual_path:
            logger.info("Visual report saved to: %s", visual_path)
            webbrowser.open(str(visual_path))
        
        return 0
        
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
