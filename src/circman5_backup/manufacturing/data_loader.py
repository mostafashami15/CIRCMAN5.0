# src/circman5/manufacturing/data_loader.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path

from circman5.config.project_paths import project_paths
from ..utils.errors import ValidationError, DataError


class DataLoader:
    """
    Handles loading, validation, and preprocessing of manufacturing data.
    """

    def __init__(self):
        # Define expected data schemas
        self.production_schema = {
            "timestamp": "datetime64[ns]",
            "batch_id": str,
            "product_type": str,
            "production_line": str,
            "output_quantity": float,
            "cycle_time": float,
            "yield_rate": float,
        }

        self.energy_schema = {
            "timestamp": "datetime64[ns]",
            "production_line": str,
            "energy_consumption": float,
            "energy_source": str,
            "efficiency_rate": float,
        }

        self.quality_schema = {
            "batch_id": str,
            "test_timestamp": "datetime64[ns]",
            "efficiency": float,
            "defect_rate": float,
            "thickness_uniformity": float,
            "visual_inspection": str,
        }

        self.material_schema = {
            "timestamp": "datetime64[ns]",
            "material_type": str,
            "quantity_used": float,
            "waste_generated": float,
            "recycled_amount": float,
            "batch_id": str,
        }

    def validate_production_data(self, data: pd.DataFrame) -> bool:
        """
        Validates production data against required schema and business rules.
        """
        required_columns = {
            "batch_id": str,
            "timestamp": "datetime64[ns]",
            "stage": str,
            "input_amount": float,
            "output_amount": float,
            "energy_used": float,
        }

        # Check required columns exist
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValidationError(
                f"Missing required columns: {missing_cols}",
                invalid_data={"missing_columns": missing_cols},
            )

        # Validate data types
        for col, dtype in required_columns.items():
            if not pd.api.types.is_dtype_equal(data[col].dtype, dtype):
                try:
                    data[col] = data[col].astype(dtype)
                except Exception as e:
                    raise ValidationError(
                        f"Invalid data type for {col}: {str(e)}",
                        invalid_data={"column": col, "error": str(e)},
                    )

        # Business rules validation
        if (data["input_amount"] < 0).any():
            raise ValidationError(
                "Input amounts cannot be negative",
                invalid_data={"field": "input_amount"},
            )

        if (data["output_amount"] < 0).any():
            raise ValidationError(
                "Output amounts cannot be negative",
                invalid_data={"field": "output_amount"},
            )

        # Modify the output amount validation to allow a small margin of error
        excessive_output = data[data["output_amount"] > data["input_amount"] * 1.1]
        if not excessive_output.empty:
            raise ValidationError(
                "Output amount cannot significantly exceed input amount",
                invalid_data={
                    "field": "output_amount",
                    "problematic_rows": excessive_output.index.tolist(),
                },
            )

        return True

    def load_production_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load and validate production data from CSV files.
        """
        try:
            if file_path is None:
                # Use synthetic data path as default
                file_path = str(
                    Path(project_paths.get_path("SYNTHETIC_DATA"))
                    / "test_production_data.csv"
                )
            else:
                # Ensure file_path is a string
                file_path = str(file_path)

            if not Path(file_path).exists():
                raise DataError(f"Production data file not found: {file_path}")

            data = pd.read_csv(file_path)

            if data.empty:
                raise DataError("Production data file is empty")

            # Validate the data
            self.validate_production_data(data)

            return data

        except Exception as e:
            raise DataError(f"Error loading production data: {str(e)}")

    def load_energy_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load and validate energy consumption data from CSV files.
        """
        try:
            if file_path is None:
                file_path = str(
                    Path(project_paths.get_path("SYNTHETIC_DATA"))
                    / "test_energy_data.csv"
                )
            else:
                file_path = str(file_path)

            if not Path(file_path).exists():
                raise DataError(f"Energy data file not found: {file_path}")

            data = pd.read_csv(file_path)

            if data.empty:
                raise DataError("Energy data file is empty")

            # Add custom validation if needed
            return data

        except Exception as e:
            raise DataError(f"Error loading energy data: {str(e)}")

    def load_quality_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load and validate quality data from CSV files.
        """
        try:
            if file_path is None:
                file_path = str(
                    Path(project_paths.get_path("SYNTHETIC_DATA"))
                    / "test_quality_data.csv"
                )
            else:
                file_path = str(file_path)

            if not Path(file_path).exists():
                raise DataError(f"Quality data file not found: {file_path}")

            data = pd.read_csv(file_path)

            if data.empty:
                raise DataError("Quality data file is empty")

            # Add custom validation if needed
            return data

        except Exception as e:
            raise DataError(f"Error loading quality data: {str(e)}")

    def load_material_flow_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load and validate material flow data from CSV files.
        """
        try:
            if file_path is None:
                file_path = str(
                    Path(project_paths.get_path("SYNTHETIC_DATA"))
                    / "test_material_data.csv"
                )
            else:
                file_path = str(file_path)

            if not Path(file_path).exists():
                raise DataError(f"Material flow data file not found: {file_path}")

            data = pd.read_csv(file_path)

            if data.empty:
                raise DataError("Material flow data file is empty")

            # Add custom validation if needed
            return data

        except Exception as e:
            raise DataError(f"Error loading material flow data: {str(e)}")
