import pandera.typing as papd

from .CDM_schema import InputsSchema
from .CDM_schema import ExampleTargetsSchema as TargetsSchema
from .CDM_schema import ExampleOutputsSchema as OutputsSchema
from .CDM_schema import ExampleFeatureImportancesSchema as FeatureImportancesSchema
from .CDM_schema import ExampleSHAPValuesSchema as SHAPValuesSchema

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
