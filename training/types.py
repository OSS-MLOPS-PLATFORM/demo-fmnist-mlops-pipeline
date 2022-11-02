"""Custom typings
"""
from typing import Any, Mapping

RawData = Any  # FIXME: appropriate typing for the data before preprocessing
PreprocessedData = Any  # FIXME: appropriate typing for the data after preprocessing
TrainData = Any  # FIXME: appropriate typing for the data used in model training
TestData = Any  # FIXME: appropriate typing for the data used in model evaluation
Model = Any  # FIXME: use the appropriate typing for the model
EvaluationMetrics = Any  # FIXME: use the appropriate typing for model metrics
EvaluationResult = Any  # FIXME: use the appropriate typing for the evaluation result
ModelParameters = Mapping[str, Any]  # typing for parameters used in model building
TrainingParameters = Mapping[str, Any]  # typing for parameters used in model training
