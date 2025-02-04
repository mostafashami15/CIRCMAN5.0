from typing import Optional, Dict, Any


class ManufacturingError(Exception):
    """Base class for manufacturing-related errors"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ValidationError(ManufacturingError):
    """Error raised for data validation issues"""

    def __init__(self, message: str, invalid_data: Optional[Dict[Any, Any]] = None):
        super().__init__(message, error_code="VAL_ERR")
        self.invalid_data = invalid_data


class ProcessError(ManufacturingError):
    """Error raised for manufacturing process issues"""

    def __init__(self, message: str, process_name: Optional[str] = None):
        super().__init__(message, error_code="PROC_ERR")
        self.process_name = process_name


class DataError(ManufacturingError):
    """Error raised for data handling issues"""

    def __init__(self, message: str, data_source: Optional[str] = None):
        super().__init__(message, error_code="DATA_ERR")
        self.data_source = data_source


class ResourceError(ManufacturingError):
    """Error raised for resource-related issues"""

    def __init__(self, message: str, resource_type: Optional[str] = None):
        super().__init__(message, error_code="RES_ERR")
        self.resource_type = resource_type
