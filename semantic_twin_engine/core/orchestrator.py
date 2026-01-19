"""
Orchestrator for the Semantic Twin Engine.

The Orchestrator is the main pipeline runner that:
1. Loads configuration
2. Instantiates registered probes
3. Executes each probe with the proper context
4. Aggregates results via DataManager
"""

import logging
from typing import Type

from core.config_loader import Settings
from core.data_manager import AuditReport, DataManager
from core.embedding_store import get_store
from core.exceptions import ProbeExecutionError
from core.interfaces import ProbeContext, ProbeInterface, ProbeResult

logger = logging.getLogger(__name__)


class ProbeRegistry:
    """Registry for managing available probe modules.
    
    The registry maintains a mapping of probe names to their classes.
    Probes must be registered before they can be used by the Orchestrator.
    
    Example:
        registry = ProbeRegistry()
        registry.register("vector_probe", VectorProbe)
        probe_class = registry.get("vector_probe")
    """
    
    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._probes: dict[str, Type[ProbeInterface]] = {}
    
    def register(self, name: str, probe_class: Type[ProbeInterface]) -> None:
        """Register a probe class with the given name.
        
        Args:
            name: Unique identifier for the probe.
            probe_class: The probe class (must implement ProbeInterface).
        
        Raises:
            ValueError: If a probe with this name is already registered.
        """
        if name in self._probes:
            raise ValueError(f"Probe already registered: {name}")
        
        if not issubclass(probe_class, ProbeInterface):
            raise TypeError(
                f"Probe must implement ProbeInterface: {probe_class.__name__}"
            )
        
        self._probes[name] = probe_class
        logger.debug("Registered probe: %s -> %s", name, probe_class.__name__)
    
    def get(self, name: str) -> Type[ProbeInterface] | None:
        """Get a registered probe class by name.
        
        Args:
            name: The probe identifier.
        
        Returns:
            The probe class, or None if not found.
        """
        return self._probes.get(name)
    
    def list_probes(self) -> list[str]:
        """List all registered probe names.
        
        Returns:
            List of registered probe identifiers.
        """
        return list(self._probes.keys())


# Global registry instance
probe_registry = ProbeRegistry()


class Orchestrator:
    """Main pipeline runner for the Semantic Twin Engine.
    
    The Orchestrator coordinates the execution of all enabled probes,
    handling errors gracefully to allow partial execution.
    
    Attributes:
        settings: The application settings.
        data_manager: The data manager for I/O operations.
        registry: The probe registry.
    """
    
    def __init__(
        self,
        settings: Settings,
        registry: ProbeRegistry | None = None
    ) -> None:
        """Initialize the orchestrator.
        
        Args:
            settings: The application settings.
            registry: Optional custom probe registry. Uses global if None.
        """
        self.settings = settings
        self.data_manager = DataManager(settings)
        self.registry = registry or probe_registry
    
    def _build_context(self, probe_name: str) -> ProbeContext:
        """Build the context object for a probe execution.
        
        Args:
            probe_name: Name of the probe to build context for.
        
        Returns:
            ProbeContext with entity data and probe-specific config.
        """
        probe_config = self.settings.probes.get(probe_name)
        
        return ProbeContext(
            entity_name=self.settings.entity.name,
            legacy_identity=self.settings.entity.legacy_identity,
            strategic_target=self.settings.entity.strategic_target,
            anchors=(
                self.settings.anchors.entities + 
                self.settings.anchors.concepts
            ),
            probe_config=probe_config.params if probe_config else {},
        )
    
    def _execute_probe(
        self,
        probe_name: str,
        probe_class: Type[ProbeInterface]
    ) -> ProbeResult:
        """Execute a single probe with error handling.
        
        Args:
            probe_name: Name of the probe.
            probe_class: The probe class to instantiate and run.
        
        Returns:
            ProbeResult from the probe execution.
        """
        context = self._build_context(probe_name)
        
        try:
            probe = probe_class()
            
            # Validate before running
            if not probe.validate(context):
                logger.warning("Probe validation failed: %s", probe_name)
                return ProbeResult(
                    probe_name=probe_name,
                    success=False,
                    data={},
                    errors=["Probe validation failed"],
                )
            
            logger.info("Executing probe: %s", probe_name)
            result = probe.run(context)
            logger.info("Probe completed: %s (success=%s)", probe_name, result.success)
            return result
            
        except ProbeExecutionError as e:
            logger.error("Probe execution error: %s - %s", probe_name, e)
            return ProbeResult(
                probe_name=probe_name,
                success=False,
                data={},
                errors=[str(e)],
            )
        except Exception as e:
            logger.exception("Unexpected error in probe: %s", probe_name)
            return ProbeResult(
                probe_name=probe_name,
                success=False,
                data={},
                errors=[f"Unexpected error: {str(e)}"],
            )
    
    def run(self) -> AuditReport:
        """Execute all enabled probes and aggregate results.
        
        This is the main entry point for running the audit pipeline.
        Each probe is executed independently - a failure in one probe
        does not prevent others from running.
        
        Returns:
            AuditReport containing all probe results.
        """
        logger.info(
            "Starting audit for entity: %s", 
            self.settings.entity.name
        )
        
        report = self.data_manager.create_report()
        
        # Get enabled probes from configuration
        for probe_name, probe_config in self.settings.probes.items():
            if not probe_config.enabled:
                logger.info("Skipping disabled probe: %s", probe_name)
                continue
            
            # Get probe class from registry
            probe_class = self.registry.get(probe_name)
            if probe_class is None:
                logger.warning(
                    "Probe not found in registry: %s (skipping)", 
                    probe_name
                )
                continue
            
            # Execute and add result
            result = self._execute_probe(probe_name, probe_class)
            report.add_result(result)
        
        logger.info(
            "Audit complete: %d probes executed (%d succeeded, %d failed)",
            report.metadata["probes_executed"],
            report.metadata["probes_succeeded"],
            report.metadata["probes_failed"],
        )
        
        # Save embedding cache
        store = get_store()
        store.save()
        logger.info("Embedding cache saved (%d embeddings, %d centroids)", 
                    store.get_stats()["embeddings_count"],
                    store.get_stats()["centroids_count"])
        
        return report
    
    def run_and_save(self) -> tuple[AuditReport, str]:
        """Execute all probes and save the report.
        
        Convenience method that runs the audit and persists the results.
        
        Returns:
            Tuple of (AuditReport, path to saved file).
        """
        report = self.run()
        output_path = self.data_manager.save_report(report)
        return report, str(output_path)
