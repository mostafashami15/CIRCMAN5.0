"""Manufacturing data loading and validation module."""

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from pathlib import Path
from typing import Dict, Optional, Union
from circman5.utils.logging_config import setup_logger
from circman5.utils.errors import ValidationError, DataError
from circman5.utils.results_manager import results_manager
from .schemas import (
    PRODUCTION_SCHEMA,
    ENERGY_SCHEMA,
    QUALITY_SCHEMA,
    MATERIAL_SCHEMA,
    LCA_MATERIAL_SCHEMA,
    LCA_ENERGY_SCHEMA,
    LCA_PROCESS_SCHEMA,
)


class ManufacturingDataLoader:
    def __init__(self):
        self.logger = setup_logger("manufacturing_data_loader")

        # Use schemas from the schemas module
        self.production_schema = PRODUCTION_SCHEMA
        self.energy_schema = ENERGY_SCHEMA
        self.quality_schema = QUALITY_SCHEMA
        self.material_schema = MATERIAL_SCHEMA
        self.lca_material_schema = LCA_MATERIAL_SCHEMA
        self.lca_energy_schema = LCA_ENERGY_SCHEMA
        self.lca_process_schema = LCA_PROCESS_SCHEMA

    def load_process_data(self, filepath: str) -> pd.DataFrame:
        """Load process data from CSV file."""
        try:
            data = pd.read_csv(filepath)
            # Add any necessary validation here
            return data
        except Exception as e:
            self.logger.error(f"Error loading process data: {str(e)}")
            raise DataError(f"Failed to load process data: {str(e)}")

    def load_production_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Load and validate production data."""
        if file_path is None:
            file_path = (
                results_manager.get_path("SYNTHETIC_DATA") / "test_production_data.csv"
            )

        return self._load_and_validate_csv(
            file_path, self.production_schema, "production"
        )

    def load_energy_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Load and validate energy consumption data."""
        if file_path is None:
            file_path = (
                results_manager.get_path("SYNTHETIC_DATA") / "test_energy_data.csv"
            )

        return self._load_and_validate_csv(file_path, self.energy_schema, "energy")

    def load_quality_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Load and validate quality metrics data."""
        if file_path is None:
            file_path = (
                results_manager.get_path("SYNTHETIC_DATA") / "test_quality_data.csv"
            )

        return self._load_and_validate_csv(file_path, self.quality_schema, "quality")

    def load_material_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Load and validate material flow data."""
        if file_path is None:
            file_path = (
                results_manager.get_path("SYNTHETIC_DATA") / "test_material_data.csv"
            )

        return self._load_and_validate_csv(file_path, self.material_schema, "material")

    def load_lca_data(
        self,
        material_data_path: Optional[str] = None,
        energy_data_path: Optional[str] = None,
        process_data_path: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load all LCA-related data files."""
        lca_data = {
            "material_flow": pd.DataFrame(),
            "energy_consumption": pd.DataFrame(),
            "process_data": pd.DataFrame(),
        }

        try:
            if material_data_path:
                lca_data["material_flow"] = self._load_and_validate_csv(
                    material_data_path, self.lca_material_schema, "lca_material"
                )
            if energy_data_path:
                lca_data["energy_consumption"] = self._load_and_validate_csv(
                    energy_data_path, self.lca_energy_schema, "lca_energy"
                )
            if process_data_path:
                lca_data["process_data"] = self._load_and_validate_csv(
                    process_data_path, self.lca_process_schema, "lca_process"
                )

            # Save loaded data to input_data directory
            for data_type, df in lca_data.items():
                if not df.empty:
                    csv_path = f"lca_{data_type}.csv"
                    df.to_csv(csv_path, index=False)
                    results_manager.save_file(csv_path, "input_data")
                    Path(csv_path).unlink()  # Clean up temporary file

            self.logger.info("Successfully loaded LCA data")
            return lca_data

        except Exception as e:
            self.logger.error(f"Error loading LCA data: {str(e)}")
            raise

    def validate_production_data(self, data: pd.DataFrame) -> bool:
        """Validate production data against schema and business rules."""
        if data.empty:
            raise ValidationError("Empty production data provided")

        # Check required columns and types
        for col, dtype in self.production_schema.items():
            if col not in data.columns:
                raise ValidationError(f"Missing required column: {col}")
            try:
                if dtype == "datetime64[ns]":
                    data[col] = pd.to_datetime(data[col])
                elif dtype in (str, float):
                    if dtype == str:
                        data[col] = data[col].astype("string")
                    else:
                        data[col] = data[col].astype("float64")
                else:
                    data[col] = data[col].astype(dtype)
            except Exception as e:
                raise ValidationError(f"Invalid data type for {col}: {str(e)}")

        # Validate business rules
        if (data["input_amount"] < 0).any():
            raise ValidationError("Input amounts cannot be negative")

        if (data["output_amount"] < 0).any():
            raise ValidationError("Output amounts cannot be negative")

        # Check output doesn't exceed input (with margin)
        excessive_output = data[data["output_amount"] > data["input_amount"] * 1.1]
        if not excessive_output.empty:
            raise ValidationError(
                "Output amount cannot significantly exceed input amount",
                invalid_data={"problematic_rows": excessive_output.index.tolist()},
            )

        return True

    def _load_and_validate_csv(
        self, file_path: Union[str, Path], schema: Dict, data_type: str
    ) -> pd.DataFrame:
        """Helper method to load and validate CSV files."""
        try:
            file_path = str(file_path)  # Ensure string path
            self.logger.info(f"Loading {data_type} data from {file_path}")

            if not Path(file_path).exists():
                raise DataError(f"{data_type} data file not found: {file_path}")

            data = pd.read_csv(file_path)
            if data.empty:
                raise DataError(f"{data_type} data file is empty")

            # Validate schema
            missing_cols = [col for col in schema if col not in data.columns]
            if missing_cols:
                raise ValidationError(
                    f"Missing required columns in {data_type} data: {missing_cols}"
                )

            # Convert data types
            for col, dtype in schema.items():
                try:
                    if dtype == "datetime64[ns]":
                        data[col] = pd.to_datetime(data[col])
                    elif dtype in (str, float):
                        if dtype == str:
                            data[col] = data[col].astype("string")
                        else:
                            data[col] = data[col].astype("float64")
                    else:
                        data[col] = data[col].astype(dtype)
                except Exception as e:
                    raise ValidationError(
                        f"Invalid data type for {col} in {data_type} data: {str(e)}"
                    )

            # Save copy of validated data to input_data directory
            csv_path = f"{data_type}_data.csv"
            data.to_csv(csv_path, index=False)
            results_manager.save_file(csv_path, "input_data")
            Path(csv_path).unlink()  # Clean up temporary file

            self.logger.info(f"Successfully loaded {len(data)} {data_type} records")
            return data

        except Exception as e:
            self.logger.error(f"Error loading {data_type} data: {str(e)}")
            raise
