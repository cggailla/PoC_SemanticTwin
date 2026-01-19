"""
Abstract Base Classes defining the contracts for the Semantic Twin Engine.

This module defines the interfaces that all pluggable modules must implement.
The Orchestrator relies on these contracts to run probes without knowing
their internal logic.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ProbeContext(BaseModel):
    """Context object passed to probes containing all necessary data.
    
    This immutable context is created by the Orchestrator and passed to each
    probe's run() method. It provides a standardized way to access entity
    information and configuration.
    
    Attributes:
        entity_name: The name of the corporate entity being analyzed.
        legacy_identity: Keywords/phrases representing the current perception.
        strategic_target: Keywords/phrases representing the desired perception.
        anchors: Reference points for comparison (e.g., competitor names).
        probe_config: Probe-specific configuration from settings.yaml.
    """
    
    entity_name: str
    legacy_identity: list[str]
    strategic_target: list[str]
    anchors: list[str]
    probe_config: dict[str, Any]
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Make the context immutable


class ProbeResult(BaseModel):
    """Standardized result object returned by all probes.
    
    Every probe must return a ProbeResult object to ensure consistent
    data aggregation by the DataManager.
    
    Attributes:
        probe_name: Identifier of the probe that generated this result.
        success: Whether the probe executed successfully.
        data: The actual analysis results (structure varies by probe).
        errors: List of error messages if any occurred.
        metadata: Additional information about the execution (timing, versions).
    """
    
    probe_name: str
    success: bool
    data: dict[str, Any]
    errors: list[str] = []
    metadata: dict[str, Any] = {}


class ProbeInterface(ABC):
    """Abstract interface that all probes must implement.
    
    This is the contract between the Orchestrator and the pluggable modules.
    The Orchestrator only knows about this interface - it calls .run(context)
    without any knowledge of the probe's internal logic.
    
    Example:
        class VectorProbe(ProbeInterface):
            @property
            def name(self) -> str:
                return "vector_probe"
            
            def run(self, context: ProbeContext) -> ProbeResult:
                # Implementation here
                return ProbeResult(...)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this probe.
        
        Returns:
            A string identifier used for logging and result aggregation.
        """
        pass
    
    @abstractmethod
    def run(self, context: ProbeContext) -> ProbeResult:
        """Execute the probe analysis.
        
        This method contains the main business logic of the probe.
        It receives the context with all necessary data and must return
        a standardized ProbeResult.
        
        Args:
            context: The probe context containing entity data and configuration.
        
        Returns:
            A ProbeResult object containing the analysis results.
        
        Raises:
            ProbeExecutionError: If the probe fails during execution.
        """
        pass
    
    def validate(self, context: ProbeContext) -> bool:
        """Validate that the probe can run with the given context.
        
        Override this method to add pre-execution validation.
        Default implementation returns True.
        
        Args:
            context: The probe context to validate.
        
        Returns:
            True if the probe can run, False otherwise.
        """
        return True
