"""
Custom exceptions for the Semantic Twin Engine.

All exceptions inherit from SemanticTwinError to enable unified error handling
across the application.
"""


class SemanticTwinError(Exception):
    """Base exception for all Semantic Twin Engine errors.
    
    All custom exceptions in the application should inherit from this class
    to enable unified error handling and logging.
    
    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional error context.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional error context.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(SemanticTwinError):
    """Raised when configuration loading or validation fails.
    
    Examples:
        - Missing required configuration file
        - Invalid YAML syntax
        - Missing required configuration keys
        - Invalid configuration values
    """

    pass


class ProbeExecutionError(SemanticTwinError):
    """Raised when a probe fails during execution.
    
    This exception should be caught by the Orchestrator to enable
    graceful degradation - allowing other probes to continue running.
    
    Attributes:
        probe_name: Name of the probe that failed.
    """

    def __init__(
        self, 
        message: str, 
        probe_name: str, 
        details: dict | None = None
    ) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error description.
            probe_name: Name of the probe that failed.
            details: Optional dictionary with additional error context.
        """
        self.probe_name = probe_name
        super().__init__(message, details)

    def __str__(self) -> str:
        """Return string representation of the exception."""
        base = f"[{self.probe_name}] {self.message}"
        if self.details:
            return f"{base} | Details: {self.details}"
        return base


class APIError(SemanticTwinError):
    """Raised when an external API call fails.
    
    This exception wraps errors from external services (e.g., OpenAI)
    to provide consistent error handling across the application.
    
    Attributes:
        service: Name of the external service that failed.
        status_code: HTTP status code if applicable.
    """

    def __init__(
        self,
        message: str,
        service: str,
        status_code: int | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error description.
            service: Name of the external service that failed.
            status_code: HTTP status code if applicable.
            details: Optional dictionary with additional error context.
        """
        self.service = service
        self.status_code = status_code
        super().__init__(message, details)

    def __str__(self) -> str:
        """Return string representation of the exception."""
        status = f" (HTTP {self.status_code})" if self.status_code else ""
        base = f"[{self.service}]{status} {self.message}"
        if self.details:
            return f"{base} | Details: {self.details}"
        return base


class DataManagerError(SemanticTwinError):
    """Raised when data I/O operations fail.
    
    Examples:
        - Failed to read input file
        - Failed to write output file
        - Invalid data format
    """

    pass
