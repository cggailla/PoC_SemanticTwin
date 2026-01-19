"""
Data Manager for the Semantic Twin Engine.

Handles I/O operations and result aggregation from probe executions.
All data persistence logic is centralized here to maintain separation
of concerns.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.config_loader import Settings
from core.exceptions import DataManagerError
from core.interfaces import ProbeResult

logger = logging.getLogger(__name__)


class AuditReport:
    """Aggregated audit report containing all probe results.
    
    This class collects results from all executed probes and provides
    methods for serialization and persistence.
    
    Attributes:
        entity_name: Name of the analyzed entity.
        timestamp: When the audit was executed.
        results: Dictionary mapping probe names to their results.
        metadata: Execution metadata (timing, versions, etc.).
    """
    
    def __init__(self, entity_name: str) -> None:
        """Initialize the audit report.
        
        Args:
            entity_name: Name of the entity being audited.
        """
        self.entity_name = entity_name
        self.timestamp = datetime.now(timezone.utc)
        self.results: dict[str, ProbeResult] = {}
        self.metadata: dict[str, Any] = {
            "start_time": self.timestamp.isoformat(),
            "end_time": None,
            "probes_executed": 0,
            "probes_succeeded": 0,
            "probes_failed": 0,
        }
    
    def add_result(self, result: ProbeResult) -> None:
        """Add a probe result to the report.
        
        Args:
            result: The probe result to add.
        """
        self.results[result.probe_name] = result
        self.metadata["probes_executed"] += 1
        if result.success:
            self.metadata["probes_succeeded"] += 1
        else:
            self.metadata["probes_failed"] += 1
    
    def finalize(self) -> None:
        """Mark the report as complete and record end time."""
        self.metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self, include_metadata: bool = True) -> dict[str, Any]:
        """Convert the report to a dictionary.
        
        Args:
            include_metadata: Whether to include execution metadata.
        
        Returns:
            Dictionary representation of the report.
        """
        output: dict[str, Any] = {
            "entity": self.entity_name,
            "timestamp": self.timestamp.isoformat(),
            "probes": {
                name: result.model_dump()
                for name, result in self.results.items()
            },
        }
        
        if include_metadata:
            output["metadata"] = self.metadata
        
        return output


class DataManager:
    """Manages data I/O for the Semantic Twin Engine.
    
    This class centralizes all file operations including:
    - Loading input data
    - Saving audit reports
    - Handling output formats (JSON, YAML)
    
    Attributes:
        settings: The application settings.
        output_dir: Path to the output directory.
    """
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the data manager.
        
        Args:
            settings: The application settings.
        """
        self.settings = settings
        self.output_dir = Path(settings.output.directory)
        self._ensure_output_directory()
    
    def _ensure_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Output directory ensured: %s", self.output_dir)
        except OSError as e:
            raise DataManagerError(
                f"Failed to create output directory: {self.output_dir}",
                details={"error": str(e)}
            ) from e
    
    def create_report(self) -> AuditReport:
        """Create a new audit report for the configured entity.
        
        Returns:
            A new AuditReport instance.
        """
        return AuditReport(entity_name=self.settings.entity.name)
    
    def save_report(self, report: AuditReport) -> Path:
        """Save the audit report to a file.
        
        The filename is generated based on entity name and timestamp.
        The format is determined by settings.output.format.
        
        Args:
            report: The audit report to save.
        
        Returns:
            Path to the saved report file.
        
        Raises:
            DataManagerError: If the report cannot be saved.
        """
        report.finalize()
        
        # Generate filename
        timestamp_str = report.timestamp.strftime("%Y%m%d_%H%M%S")
        entity_slug = report.entity_name.lower().replace(" ", "_")
        extension = self.settings.output.format
        filename = f"audit_{entity_slug}_{timestamp_str}.{extension}"
        output_path = self.output_dir / filename
        
        # Serialize and save
        data = report.to_dict(
            include_metadata=self.settings.output.include_metadata
        )
        
        try:
            with output_path.open("w", encoding="utf-8") as f:
                if self.settings.output.format == "json":
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    yaml.safe_dump(data, f, default_flow_style=False)
            
            logger.info("Report saved to: %s", output_path)
            return output_path
            
        except (OSError, TypeError) as e:
            raise DataManagerError(
                f"Failed to save report: {output_path}",
                details={"error": str(e)}
            ) from e
    
    def load_json(self, path: Path) -> dict[str, Any]:
        """Load data from a JSON file.
        
        Args:
            path: Path to the JSON file.
        
        Returns:
            Parsed JSON data.
        
        Raises:
            DataManagerError: If the file cannot be loaded.
        """
        if not path.exists():
            raise DataManagerError(
                f"File not found: {path}",
                details={"path": str(path)}
            )
        
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise DataManagerError(
                f"Invalid JSON format: {path}",
                details={"path": str(path), "error": str(e)}
            ) from e
