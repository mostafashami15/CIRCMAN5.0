# src/circman5/manufacturing/digital_twin/simulation/process_models.py
"""
Process Models for CIRCMAN5.0 Digital Twin.

This module implements physics-based models of PV manufacturing processes
for use in digital twin simulations.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import math

from ....utils.logging_config import setup_logger


class PVManufacturingProcessModel:
    """Base class for PV manufacturing process models."""

    def __init__(self):
        """Initialize the process model."""
        self.logger = setup_logger("process_model")

    def simulate_step(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate one time step of the process.

        Args:
            current_state: Current state of the process

        Returns:
            Dict[str, Any]: Updated state after simulation step
        """
        raise NotImplementedError("Subclasses must implement this method")


class SiliconPurificationModel(PVManufacturingProcessModel):
    """Model for silicon purification process."""

    def simulate_step(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate silicon purification process step.

        Args:
            current_state: Current state of the process

        Returns:
            Dict[str, Any]: Updated state after simulation step
        """
        # Create a copy of the current state
        next_state = current_state.copy()

        # Extract relevant parameters
        input_amount = next_state.get("input_amount", 0.0)
        temperature = next_state.get("temperature", 0.0)
        energy = next_state.get("energy_consumption", 0.0)

        # Physics-based model for silicon purification
        # Based on simplified Czochralski process model
        base_yield = 0.9  # Base yield rate

        # Temperature effect (optimal range: 1400-1500°C)
        temperature_factor = 1.0
        if temperature > 0:  # Only apply if temperature is provided
            optimal_temp = 1450  # Optimal temperature in °C
            temp_sensitivity = 0.001  # Sensitivity factor
            temperature_factor = 1.0 - temp_sensitivity * abs(
                temperature - optimal_temp
            )
            temperature_factor = max(0.8, min(1.0, temperature_factor))

        # Energy efficiency effect
        energy_factor = 1.0
        if energy > 0 and input_amount > 0:  # Only apply if energy data is provided
            energy_per_unit = energy / input_amount
            optimal_energy = 50.0  # Optimal energy per unit
            energy_sensitivity = 0.005
            energy_factor = 1.0 - energy_sensitivity * abs(
                energy_per_unit - optimal_energy
            )
            energy_factor = max(0.85, min(1.0, energy_factor))

        # Calculate output
        actual_yield = base_yield * temperature_factor * energy_factor
        output_amount = input_amount * actual_yield

        # Update state
        next_state["output_amount"] = output_amount
        next_state["yield_rate"] = actual_yield * 100  # As percentage

        # Calculate waste
        next_state["waste_generated"] = input_amount - output_amount

        # Calculate quality metrics
        purity = 99.0 + temperature_factor * energy_factor  # Base purity plus factors
        next_state["purity"] = min(99.999, purity)  # Maximum practical purity

        return next_state


class WaferProductionModel(PVManufacturingProcessModel):
    """Model for wafer production process."""

    def simulate_step(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate wafer production process step.

        Args:
            current_state: Current state of the process

        Returns:
            Dict[str, Any]: Updated state after simulation step
        """
        # Create a copy of the current state
        next_state = current_state.copy()

        # Extract relevant parameters
        input_amount = next_state.get("input_amount", 0.0)
        cutting_speed = next_state.get("cutting_speed", 1.0)
        wire_tension = next_state.get("wire_tension", 1.0)

        # Physics-based model for wafer cutting
        base_yield = 0.95  # Base yield rate

        # Cutting speed effect (optimal value depends on material)
        speed_factor = 1.0
        optimal_speed = 10.0  # Arbitrary optimal value
        speed_sensitivity = 0.02
        speed_factor = 1.0 - speed_sensitivity * abs(cutting_speed - optimal_speed)
        speed_factor = max(0.85, min(1.0, speed_factor))

        # Wire tension effect
        tension_factor = 1.0
        optimal_tension = 25.0  # Arbitrary optimal value
        tension_sensitivity = 0.015
        tension_factor = 1.0 - tension_sensitivity * abs(wire_tension - optimal_tension)
        tension_factor = max(0.9, min(1.0, tension_factor))

        # Calculate output with random variation
        random_factor = np.random.normal(1.0, 0.01)  # Small random variation
        actual_yield = base_yield * speed_factor * tension_factor * random_factor
        output_amount = input_amount * actual_yield

        # Update state
        next_state["output_amount"] = output_amount
        next_state["yield_rate"] = actual_yield * 100  # As percentage

        # Calculate thickness uniformity
        uniformity_base = 95.0
        uniformity = uniformity_base + (speed_factor + tension_factor - 1.0) * 5.0
        next_state["thickness_uniformity"] = min(99.9, uniformity)

        # Calculate waste
        next_state["waste_generated"] = input_amount - output_amount
        next_state["kerf_loss"] = (
            input_amount * 0.15 * (2.0 - speed_factor)
        )  # Wire saw kerf loss

        return next_state
