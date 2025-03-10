# Digital Twin AI Integration API

## 1. Introduction

This API reference document describes the integration between the CIRCMAN5.0 Digital Twin system and its AI/ML optimization components. The AI Integration API enables parameter extraction for AI processing, optimization of manufacturing parameters, and application of AI-driven results back to the digital twin.

## 2. AIIntegration Class

The central class for integrating AI capabilities with the Digital Twin.

```python
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration
```

### 2.1 Constructor

```python
AIIntegration(
    digital_twin: DigitalTwin,
    model: Optional[ManufacturingModel] = None,
    optimizer: Optional[ProcessOptimizer] = None
)
```

**Parameters:**
- `digital_twin` (DigitalTwin): Digital Twin instance to integrate with.
- `model` (ManufacturingModel, optional): Optional ManufacturingModel instance.
- `optimizer` (ProcessOptimizer, optional): Optional ProcessOptimizer instance.

**Notes:**
- If `model` is not provided, a default ManufacturingModel will be created.
- If `optimizer` is not provided, a default ProcessOptimizer will be created using the provided or default model.

### 2.2 Parameter Management Methods

#### 2.2.1 extract_parameters_from_state()

```python
extract_parameters_from_state(
    state: Optional[Dict[str, Any]] = None
) -> Dict[str, float]
```

Extracts relevant parameters from digital twin state for AI optimization.

**Parameters:**
- `state` (Dict[str, Any], optional): Optional state dictionary (uses current state if None).

**Returns:**
- `Dict[str, float]`: Parameters dictionary for optimization.

**Example:**
```python
# Get AI integration instance
ai_integration = AIIntegration(digital_twin)

# Extract parameters from current state
parameters = ai_integration.extract_parameters_from_state()
print("Extracted parameters:", parameters)
```

#### 2.2.2 _convert_parameters_to_state()

```python
_convert_parameters_to_state(
    parameters: Dict[str, float]
) -> Dict[str, Any]
```

Converts AI-optimized parameters back to digital twin state format.

**Parameters:**
- `parameters` (Dict[str, float]): AI parameter dictionary.

**Returns:**
- `Dict[str, Any]`: State update dictionary ready for digital twin.

**Note:**
- This is an internal method primarily used by `apply_optimized_parameters()`.

### 2.3 Analysis and Optimization Methods

#### 2.3.1 predict_outcomes()

```python
predict_outcomes(
    parameters: Optional[Dict[str, float]] = None
) -> Dict[str, Any]
```

Predicts manufacturing outcomes for the current or provided parameters.

**Parameters:**
- `parameters` (Dict[str, float], optional): Optional parameters dictionary.

**Returns:**
- `Dict[str, Any]`: Prediction results including:
  - `predicted_output`: Predicted production output
  - `predicted_quality`: Predicted quality score
  - `predicted_energy`: Predicted energy consumption
  - `confidence_score`: Confidence level of prediction

**Example:**
```python
# Get AI integration instance
ai_integration = AIIntegration(digital_twin)

# Predict outcomes for specific parameters
custom_params = {
    "temperature": 23.5,
    "cycle_time": 30.0,
    "pressure": 5.2
}
prediction = ai_integration.predict_outcomes(custom_params)
print(f"Predicted output: {prediction['predicted_output']}")
print(f"Confidence: {prediction['confidence_score']}")
```

#### 2.3.2 optimize_parameters()

```python
optimize_parameters(
    current_params: Optional[Dict[str, float]] = None,
    constraints: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None
) -> Dict[str, float]
```

Optimizes process parameters for the digital twin.

**Parameters:**
- `current_params` (Dict[str, float], optional): Optional current parameters (extracted from current state if None).
- `constraints` (Dict[str, Union[float, Tuple[float, float]]], optional): Optional parameter constraints.

**Returns:**
- `Dict[str, float]`: Optimized parameters.

**Notes:**
- Constraints can be specified as either:
  - Single values: `{"temperature": 23.0}` (exact value constraint)
  - Tuples: `{"temperature": (20.0, 25.0)}` (min/max range)

**Example:**
```python
# Get AI integration instance
ai_integration = AIIntegration(digital_twin)

# Set optimization constraints
constraints = {
    "temperature": (20.0, 25.0),  # Min/max range
    "cycle_time": (25.0, 40.0),
    "pressure": 5.0  # Exact value
}

# Optimize parameters
optimized = ai_integration.optimize_parameters(constraints=constraints)
print("Optimized parameters:", optimized)
```

#### 2.3.3 apply_optimized_parameters()

```python
apply_optimized_parameters(
    optimized_params: Dict[str, float],
    simulation_steps: int = 10
) -> bool
```

Applies optimized parameters to the digital twin through simulation validation.

**Parameters:**
- `optimized_params` (Dict[str, float]): Optimized parameters to apply.
- `simulation_steps` (int): Number of simulation steps to run for validation.

**Returns:**
- `bool`: True if parameters were applied successfully, False otherwise.

**Example:**
```python
# Get AI integration instance
ai_integration = AIIntegration(digital_twin)

# Optimize parameters
optimized = ai_integration.optimize_parameters()

# Apply the optimized parameters
success = ai_integration.apply_optimized_parameters(optimized, simulation_steps=20)
if success:
    print("Parameters applied successfully")
else:
    print("Failed to apply parameters")
```

### 2.4 Reporting and History Methods

#### 2.4.1 _record_optimization()

```python
_record_optimization(
    original_params: Dict[str, float],
    optimized_params: Dict[str, float]
) -> None
```

Records an optimization run to the optimization history.

**Parameters:**
- `original_params` (Dict[str, float]): Original parameters before optimization.
- `optimized_params` (Dict[str, float]): Optimized parameters.

**Note:**
- This is an internal method automatically called by `optimize_parameters()`.

#### 2.4.2 _save_optimization_history()

```python
_save_optimization_history() -> bool
```

Saves the optimization history to a file.

**Returns:**
- `bool`: True if save was successful, False otherwise.

**Note:**
- This is an internal method automatically called after optimization.

#### 2.4.3 generate_optimization_report()

```python
generate_optimization_report() -> Dict[str, Any]
```

Generates a comprehensive optimization report based on history.

**Returns:**
- `Dict[str, Any]`: Report data including:
  - `average_improvement`: Average improvement percentage
  - `latest_optimization`: Details of the most recent optimization
  - `parameter_trends`: Trends in parameter changes
  - `improvement_trends`: Trends in improvement metrics

**Example:**
```python
# Get AI integration instance
ai_integration = AIIntegration(digital_twin)

# Generate and process optimization report
report = ai_integration.generate_optimization_report()
print(f"Average improvement: {report['average_improvement']:.2f}%")
print(f"Parameter trends: {report['parameter_trends']}")
```

### 2.5 Model Training Methods

#### 2.5.1 train_model_from_digital_twin()

```python
train_model_from_digital_twin(
    history_limit: Optional[int] = None
) -> bool
```

Trains the AI model using data from digital twin history.

**Parameters:**
- `history_limit` (int, optional): Optional limit on number of historical states to use.

**Returns:**
- `bool`: True if training was successful, False otherwise.

**Example:**
```python
# Get AI integration instance
ai_integration = AIIntegration(digital_twin)

# Train model with last 100 historical states
success = ai_integration.train_model_from_digital_twin(history_limit=100)
if success:
    print("Model training successful")
else:
    print("Model training failed")
```

#### 2.5.2 evaluate_model_performance()

```python
evaluate_model_performance() -> Dict[str, float]
```

Evaluates the current AI model performance using test data.

**Returns:**
- `Dict[str, float]`: Performance metrics including:
  - `mse`: Mean Squared Error
  - `mae`: Mean Absolute Error
  - `r2`: R-squared score
  - `accuracy`: Overall accuracy score

**Example:**
```python
# Get AI integration instance
ai_integration = AIIntegration(digital_twin)

# Evaluate model performance
metrics = ai_integration.evaluate_model_performance()
print(f"Model R2 score: {metrics['r2']:.4f}")
print(f"Model accuracy: {metrics['accuracy']:.2f}%")
```

## 3. ManufacturingModel Class

The AI model for manufacturing optimization.

```python
from circman5.manufacturing.optimization.model import ManufacturingModel
```

### 3.1 Constructor

```python
ManufacturingModel(config=None)
```

**Parameters:**
- `config` (Dict[str, Any], optional): Model configuration options.

### 3.2 Core Model Methods

#### 3.2.1 train_optimization_model()

```python
train_optimization_model(
    production_data: pd.DataFrame,
    quality_data: pd.DataFrame
) -> Dict[str, float]
```

Trains the manufacturing optimization model.

**Parameters:**
- `production_data` (pd.DataFrame): Production data for training.
- `quality_data` (pd.DataFrame): Quality data for training.

**Returns:**
- `Dict[str, float]`: Training metrics.

#### 3.2.2 predict_batch_outcomes()

```python
predict_batch_outcomes(
    parameters: Dict[str, float]
) -> Dict[str, Any]
```

Predicts manufacturing outcomes for given parameters.

**Parameters:**
- `parameters` (Dict[str, float]): Manufacturing parameters.

**Returns:**
- `Dict[str, Any]`: Prediction results including confidence scores.

#### 3.2.3 save_model()

```python
save_model(
    file_path: Optional[Union[str, Path]] = None
) -> bool
```

Saves the trained model to file.

**Parameters:**
- `file_path` (Union[str, Path], optional): Path to save model. Uses results_manager if None.

**Returns:**
- `bool`: True if save was successful, False otherwise.

#### 3.2.4 load_model()

```python
load_model(
    file_path: Union[str, Path]
) -> bool
```

Loads a trained model from file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to model file.

**Returns:**
- `bool`: True if load was successful, False otherwise.

## 4. ProcessOptimizer Class

Optimizer for manufacturing process parameters.

```python
from circman5.manufacturing.optimization.optimizer import ProcessOptimizer
```

### 4.1 Constructor

```python
ProcessOptimizer(model, config=None)
```

**Parameters:**
- `model` (ManufacturingModel): Manufacturing model for optimization.
- `config` (Dict[str, Any], optional): Optimizer configuration.

### 4.2 Optimization Methods

#### 4.2.1 optimize_process_parameters()

```python
optimize_process_parameters(
    current_params: Dict[str, float],
    constraints: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None
) -> Dict[str, float]
```

Optimizes process parameters.

**Parameters:**
- `current_params` (Dict[str, float]): Current manufacturing parameters.
- `constraints` (Dict[str, Union[float, Tuple[float, float]]], optional): Parameter constraints.

**Returns:**
- `Dict[str, float]`: Optimized parameters.

#### 4.2.2 optimize_for_efficiency()

```python
optimize_for_efficiency(
    current_params: Dict[str, float],
    min_quality: float = 0.9,
    constraints: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None
) -> Dict[str, float]
```

Optimizes parameters for energy efficiency.

**Parameters:**
- `current_params` (Dict[str, float]): Current manufacturing parameters.
- `min_quality` (float): Minimum acceptable quality level.
- `constraints` (Dict[str, Union[float, Tuple[float, float]]], optional): Parameter constraints.

**Returns:**
- `Dict[str, float]`: Optimized parameters.

#### 4.2.3 optimize_for_quality()

```python
optimize_for_quality(
    current_params: Dict[str, float],
    max_energy: Optional[float] = None,
    constraints: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None
) -> Dict[str, float]
```

Optimizes parameters for production quality.

**Parameters:**
- `current_params` (Dict[str, float]): Current manufacturing parameters.
- `max_energy` (float, optional): Maximum acceptable energy consumption.
- `constraints` (Dict[str, Union[float, Tuple[float, float]]], optional): Parameter constraints.

**Returns:**
- `Dict[str, float]`: Optimized parameters.

#### 4.2.4 multi_objective_optimization()

```python
multi_objective_optimization(
    current_params: Dict[str, float],
    objectives: List[str],
    weights: Optional[List[float]] = None,
    constraints: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None
) -> Dict[str, float]
```

Performs multi-objective optimization.

**Parameters:**
- `current_params` (Dict[str, float]): Current manufacturing parameters.
- `objectives` (List[str]): List of optimization objectives.
- `weights` (List[float], optional): Weights for each objective.
- `constraints` (Dict[str, Union[float, Tuple[float, float]]], optional): Parameter constraints.

**Returns:**
- `Dict[str, float]`: Optimized parameters.

## 5. Usage Examples

### 5.1 Basic Parameter Optimization

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration

# Initialize digital twin
twin = DigitalTwin()
twin.initialize()

# Create AI integration
ai = AIIntegration(twin)

# Extract current parameters
current_params = ai.extract_parameters_from_state()
print("Current parameters:", current_params)

# Set optimization constraints
constraints = {
    "temperature": (21.0, 24.0),
    "cycle_time": (25.0, 35.0),
    "energy_used": (50.0, 120.0)
}

# Run optimization
optimized_params = ai.optimize_parameters(
    current_params=current_params,
    constraints=constraints
)
print("Optimized parameters:", optimized_params)

# Apply optimized parameters
success = ai.apply_optimized_parameters(optimized_params)
if success:
    print("Parameters applied successfully")
else:
    print("Failed to apply parameters")
```

### 5.2 Multi-Objective Optimization

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration
from circman5.manufacturing.optimization.optimizer import ProcessOptimizer
from circman5.manufacturing.optimization.model import ManufacturingModel

# Initialize digital twin
twin = DigitalTwin()
twin.initialize()

# Create model and optimizer
model = ManufacturingModel()
optimizer = ProcessOptimizer(model)

# Create AI integration with custom optimizer
ai = AIIntegration(twin, optimizer=optimizer)

# Extract current parameters
current_params = ai.extract_parameters_from_state()

# Define multi-objective optimization
objectives = ["energy_efficiency", "quality", "production_rate"]
weights = [0.5, 0.3, 0.2]  # Prioritize energy efficiency

# Run multi-objective optimization
optimized_params = optimizer.multi_objective_optimization(
    current_params=current_params,
    objectives=objectives,
    weights=weights
)
print("Multi-objective optimized parameters:", optimized_params)

# Apply optimized parameters
ai.apply_optimized_parameters(optimized_params)
```

### 5.3 Model Training and Evaluation

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration

# Initialize digital twin
twin = DigitalTwin()
twin.initialize()

# Create AI integration
ai = AIIntegration(twin)

# Train model using digital twin history
training_success = ai.train_model_from_digital_twin(history_limit=200)
if training_success:
    print("Model training successful")

    # Evaluate model performance
    metrics = ai.evaluate_model_performance()
    print(f"Model accuracy: {metrics['accuracy']:.2f}%")
    print(f"Mean absolute error: {metrics['mae']:.4f}")
    print(f"R-squared score: {metrics['r2']:.4f}")

    # Save trained model
    ai.model.save_model("trained_manufacturing_model.pkl")
else:
    print("Model training failed")
```

### 5.4 Generating Optimization Reports

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration

# Initialize digital twin
twin = DigitalTwin()
twin.initialize()

# Create AI integration
ai = AIIntegration(twin)

# Run several optimizations
for _ in range(5):
    # Modify some state parameters to simulate different conditions
    twin.update({
        "production_line": {
            "temperature": 22.0 + _ * 0.5,
            "production_rate": 8.0 - _ * 0.2
        }
    })

    # Run optimization
    ai.optimize_parameters()

# Generate optimization report
report = ai.generate_optimization_report()

# Process report data
print(f"Average improvement: {report['average_improvement']:.2f}%")
print("Parameter trends:")
for param, trend in report['parameter_trends'].items():
    print(f"  {param}: {trend['direction']} ({trend['magnitude']:.2f}%)")

print("\nLatest optimization:")
latest = report['latest_optimization']
print(f"  Date: {latest['timestamp']}")
print(f"  Improvement: {latest['improvement']:.2f}%")
print(f"  Parameters: {latest['parameters']}")
```

## 6. Error Handling

The AI Integration API implements comprehensive error handling:

1. All public methods catch exceptions and provide meaningful error messages
2. Methods return explicit success/failure status
3. Training and optimization failures are logged and reported
4. Parameter validation ensures values are within acceptable ranges

Example error handling:

```python
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration

# Create AI integration
ai = AIIntegration(digital_twin)

# Set intentionally invalid constraints
invalid_constraints = {
    "temperature": (30.0, 20.0)  # Min > Max, which is invalid
}

try:
    # This will raise a ValueError due to invalid constraints
    optimized = ai.optimize_parameters(constraints=invalid_constraints)
except ValueError as e:
    print(f"Optimization error: {str(e)}")

    # Try with valid constraints instead
    valid_constraints = {
        "temperature": (20.0, 30.0)  # Correct order
    }
    try:
        optimized = ai.optimize_parameters(constraints=valid_constraints)
        print("Optimization succeeded with valid constraints")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
```

## 7. Thread Safety

The AI Integration API is designed to be thread-safe:

1. Model training can run concurrently with other operations
2. Parameter extraction and conversion are protected from race conditions
3. Optimization history uses thread-safe access patterns
4. Integration components respect the thread safety of the Digital Twin core

## 8. Configuration

The AI Integration API is configurable through the constants service:

```json
{
    "AI_INTEGRATION": {
        "DEFAULT_PARAMETERS": {
            "input_amount": 100.0,
            "energy_used": 50.0,
            "cycle_time": 30.0,
            "efficiency": 0.9,
            "defect_rate": 0.05,
            "thickness_uniformity": 95.0
        },
        "PARAMETER_MAPPING": {
            "production_rate": "output_amount",
            "energy_consumption": "energy_used",
            "temperature": "temperature",
            "cycle_time": "cycle_time"
        },
        "OPTIMIZATION_CONSTRAINTS": {
            "energy_used": [10.0, 100.0],
            "cycle_time": [20.0, 60.0],
            "defect_rate": [0.01, 0.1]
        }
    }
}
```

This configuration can be accessed through the constants service:

```python
from circman5.adapters.services.constants_service import ConstantsService

# Get configuration
constants = ConstantsService()
ai_config = constants.get_digital_twin_config().get("AI_INTEGRATION", {})

# Access specific sections
parameter_mapping = ai_config.get("PARAMETER_MAPPING", {})
default_constraints = ai_config.get("OPTIMIZATION_CONSTRAINTS", {})
```

## 9. Optimization Algorithms

The ProcessOptimizer implements several optimization algorithms:

1. **Gradient Descent**: For continuous parameter optimization
2. **Bayesian Optimization**: For expensive objective functions
3. **Genetic Algorithms**: For discrete parameter optimization
4. **Simulated Annealing**: For global optimization problems

The algorithm selection is automatic based on problem characteristics, but can be specified in the configuration:

```python
# Create optimizer with specific algorithm
config = {"algorithm": "bayesian", "n_iterations": 100}
optimizer = ProcessOptimizer(model, config=config)
```

## 10. Integration with Machine Learning Frameworks

The ManufacturingModel can integrate with various ML frameworks:

1. **scikit-learn**: Default implementation for regression models
2. **TensorFlow/Keras**: For deep learning models (if available)
3. **PyTorch**: For custom neural network models (if available)

Framework selection is determined by model configuration:

```python
# Create model with specific framework
config = {"framework": "tensorflow", "model_type": "neural_network"}
model = ManufacturingModel(config=config)
```

## 11. Extending the AI Integration

The AI Integration API is designed to be extensible:

1. Custom models can be created by extending ManufacturingModel
2. Custom optimizers can be created by extending ProcessOptimizer
3. Additional optimization objectives can be defined in configuration

Example of extending with a custom model:

```python
from circman5.manufacturing.optimization.model import ManufacturingModel

class CustomManufacturingModel(ManufacturingModel):
    def __init__(self, config=None):
        super().__init__(config)
        # Custom initialization

    def predict_batch_outcomes(self, parameters):
        # Custom prediction logic
        base_prediction = super().predict_batch_outcomes(parameters)
        # Enhance with custom predictions
        base_prediction["custom_metric"] = self._calculate_custom_metric(parameters)
        return base_prediction

    def _calculate_custom_metric(self, parameters):
        # Custom calculation
        return some_calculation(parameters)
```

Using the custom model:

```python
# Create custom model
custom_model = CustomManufacturingModel()

# Create AI integration with custom model
ai = AIIntegration(twin, model=custom_model)
```
