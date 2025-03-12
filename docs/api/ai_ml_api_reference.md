# AI/ML Enhancement API Reference

## Overview

This document details the API for the AI/ML Enhancement components of CIRCMAN5.0, providing reference information for developers integrating or extending these capabilities. These components enable advanced prediction, online learning, and uncertainty quantification for PV manufacturing optimization.

## Table of Contents

1. [Advanced Models](#advanced-models)
   - [DeepLearningModel](#deeplearningmodel)
   - [EnsembleModel](#ensemblemodel)
2. [Online Learning](#online-learning)
   - [AdaptiveModel](#adaptivemodel)
   - [RealTimeTrainer](#realtimetrainer)
3. [Validation Framework](#validation-framework)
   - [CrossValidator](#crossvalidator)
   - [UncertaintyQuantifier](#uncertaintyquantifier)
4. [Digital Twin Integration](#digital-twin-integration)

## Advanced Models

### DeepLearningModel

The `DeepLearningModel` provides neural network-based modeling capabilities for manufacturing optimization.

#### Constructor

```python
def __init__(self, model_type: str = None)
```

- **Parameters**:
  - `model_type`: Optional model type (lstm, mlp). If None, loads from configuration.

#### Methods

```python
def train(self, X_train: Union[np.ndarray, pd.DataFrame],
          y_train: Union[np.ndarray, pd.DataFrame],
          X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
          y_val: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Dict[str, Any]
```
- **Description**: Train the deep learning model.
- **Parameters**:
  - `X_train`: Training features
  - `y_train`: Training targets
  - `X_val`: Optional validation features
  - `y_val`: Optional validation targets
- **Returns**: Dictionary with training metrics and history

```python
def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray
```
- **Description**: Generate predictions using the trained model.
- **Parameters**:
  - `X`: Input features
- **Returns**: Predictions as numpy array

```python
def evaluate(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.DataFrame]) -> Dict[str, float]
```
- **Description**: Evaluate model performance.
- **Parameters**:
  - `X`: Test features
  - `y`: Test targets
- **Returns**: Dictionary of evaluation metrics

```python
def save_model(self, file_path: Optional[Union[str, Path]] = None) -> Path
```
- **Description**: Save the trained model.
- **Parameters**:
  - `file_path`: Optional file path to save the model. If None, saves to default location.
- **Returns**: Path where model was saved

```python
def load_model(self, file_path: Union[str, Path]) -> None
```
- **Description**: Load a saved model.
- **Parameters**:
  - `file_path`: Path to the saved model

### EnsembleModel

The `EnsembleModel` implements ensemble-based models for manufacturing optimization.

#### Constructor

```python
def __init__(self)
```

#### Methods

```python
def train(self, X_train: Union[np.ndarray, pd.DataFrame],
          y_train: Union[np.ndarray, pd.DataFrame],
          X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
          y_val: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Dict[str, Any]
```
- **Description**: Train the ensemble model.
- **Parameters**:
  - `X_train`: Training features
  - `y_train`: Training targets
  - `X_val`: Optional validation features
  - `y_val`: Optional validation targets
- **Returns**: Training results dictionary

```python
def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray
```
- **Description**: Generate predictions using the trained ensemble.
- **Parameters**:
  - `X`: Input features
- **Returns**: Predictions as numpy array

```python
def evaluate(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.DataFrame]) -> Dict[str, float]
```
- **Description**: Evaluate ensemble model performance.
- **Parameters**:
  - `X`: Test features
  - `y`: Test targets
- **Returns**: Dictionary of evaluation metrics including base model metrics

```python
def save_model(self, file_path: Optional[Union[str, Path]] = None) -> Path
```
- **Description**: Save the trained ensemble model.
- **Parameters**:
  - `file_path`: Optional file path to save the model. If None, saves to default location.
- **Returns**: Path where model was saved

```python
def load_model(self, file_path: Union[str, Path]) -> None
```
- **Description**: Load a saved ensemble model.
- **Parameters**:
  - `file_path`: Path to the saved model

## Online Learning

### AdaptiveModel

The `AdaptiveModel` implements a learning model that adapts to new data over time.

#### Constructor

```python
def __init__(self, base_model_type: str = "ensemble")
```
- **Parameters**:
  - `base_model_type`: Type of base model to use (ensemble or deep_learning)

#### Methods

```python
def add_data_point(self, X: Union[np.ndarray, pd.DataFrame],
                   y: Union[np.ndarray, pd.DataFrame],
                   weight: float = 1.0) -> bool
```
- **Description**: Add a new data point to the buffer and potentially update the model.
- **Parameters**:
  - `X`: Feature vector (single sample)
  - `y`: Target value
  - `weight`: Importance weight for this sample
- **Returns**: True if model was updated, False otherwise

```python
def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray
```
- **Description**: Generate predictions using the adaptive model.
- **Parameters**:
  - `X`: Input features
- **Returns**: Predictions as numpy array

```python
def evaluate(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.DataFrame]) -> Dict[str, float]
```
- **Description**: Evaluate adaptive model performance.
- **Parameters**:
  - `X`: Test features
  - `y`: Test targets
- **Returns**: Dictionary of evaluation metrics with additional adaptive model metrics

### RealTimeTrainer

The `RealTimeTrainer` enables continuous model training from streaming data.

#### Constructor

```python
def __init__(self, data_source_callback: Optional[Callable] = None)
```
- **Parameters**:
  - `data_source_callback`: Optional callback function to retrieve new data

#### Methods

```python
def start(self, interval_seconds: int = 10) -> None
```
- **Description**: Start the real-time training loop.
- **Parameters**:
  - `interval_seconds`: Seconds between training iterations

```python
def stop(self) -> None
```
- **Description**: Stop the training loop.

## Validation Framework

### CrossValidator

The `CrossValidator` implements enhanced cross-validation for manufacturing optimization models.

#### Constructor

```python
def __init__(self)
```

#### Methods

```python
def validate(self, model: Any, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.DataFrame],
             groups: Optional[np.ndarray] = None) -> Dict[str, Any]
```
- **Description**: Perform cross-validation on the provided model.
- **Parameters**:
  - `model`: Model instance with fit and predict methods
  - `X`: Feature data
  - `y`: Target data
  - `groups`: Optional group labels for grouped cross-validation
- **Returns**: Cross-validation results dictionary

### UncertaintyQuantifier

The `UncertaintyQuantifier` enables uncertainty estimation for model predictions.

#### Constructor

```python
def __init__(self)
```

#### Methods

```python
def quantify_uncertainty(self, model: Any,
                         X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]
```
- **Description**: Quantify prediction uncertainty for the given model and inputs.
- **Parameters**:
  - `model`: Model instance with prediction capability
  - `X`: Input features
- **Returns**: Dictionary of uncertainty metrics for each prediction

```python
def calibrate(self, model: Any,
              X_cal: Union[np.ndarray, pd.DataFrame],
              y_cal: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]
```
- **Description**: Calibrate uncertainty estimation using validation data.
- **Parameters**:
  - `model`: Model instance
  - `X_cal`: Calibration features
  - `y_cal`: Calibration targets
- **Returns**: Calibration parameters dictionary

## Digital Twin Integration

The AI/ML components integrate with the Digital Twin through the AIIntegration class, with the following extended methods:

```python
def create_advanced_model(self, model_type: str, name: str) -> Any
```
- **Description**: Create a new advanced model for optimization.
- **Parameters**:
  - `model_type`: Type of model ('deep_learning' or 'ensemble')
  - `name`: Unique identifier for the model
- **Returns**: Created model instance

```python
def create_adaptive_model(self, name: str, base_model_type: str = "ensemble") -> Any
```
- **Description**: Create a new adaptive model for online learning.
- **Parameters**:
  - `name`: Unique identifier for the model
  - `base_model_type`: Type of base model ('deep_learning' or 'ensemble')
- **Returns**: Created adaptive model instance

```python
def start_real_time_learning(self, interval_seconds: int = 10) -> None
```
- **Description**: Start real-time model training using Digital Twin data.
- **Parameters**:
  - `interval_seconds`: Seconds between training iterations

```python
def stop_real_time_learning(self) -> None
```
- **Description**: Stop real-time model training.

```python
def validate_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]
```
- **Description**: Validate a model using cross-validation.
- **Parameters**:
  - `model_name`: Name of the model to validate
  - `X`: Validation features
  - `y`: Validation targets
- **Returns**: Dictionary of validation results

```python
def quantify_prediction_uncertainty(self, model_name: str, X: np.ndarray) -> Dict[str, np.ndarray]
```
- **Description**: Quantify uncertainty in model predictions.
- **Parameters**:
  - `model_name`: Name of the model
  - `X`: Input features
- **Returns**: Dictionary of uncertainty metrics
