import pandera.typing as papd

from .example import ExampleInputsSchema as InputsSchema
from .example import ExampleTargetsSchema as TargetsSchema
from .example import ExampleOutputsSchema as OutputsSchema
from .example import ExampleFeatureImportancesSchema as FeatureImportancesSchema
from .example import ExampleSHAPValuesSchema as SHAPValuesSchema

Inputs = papd.DataFrame[InputsSchema]
Targets = papd.DataFrame[TargetsSchema]
Outputs = papd.DataFrame[OutputsSchema]
SHAPValues = papd.DataFrame[SHAPValuesSchema]
FeatureImportances = papd.DataFrame[FeatureImportancesSchema]

__all__ = [
    "Inputs",
    "Targets",
    "Outputs",
    "SHAPValues",
    "FeatureImportances",
]
