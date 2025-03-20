import pandera.typing as papd
import pandera as pa
import pandera.typing.common as padt

from .._base import Schema


class ExampleInputsSchema(Schema):
    """Schema for the project inputs.

    Example:
        instant: papd.Index[padt.UInt32] = pa.Field(ge=0)
        dteday: papd.Series[padt.DateTime] = pa.Field()
        season: papd.Series[padt.UInt8] = pa.Field(isin=[1, 2, 3, 4])
        yr: papd.Series[padt.UInt8] = pa.Field(ge=0, le=1)
        mnth: papd.Series[padt.UInt8] = pa.Field(ge=1, le=12)
        hr: papd.Series[padt.UInt8] = pa.Field(ge=0, le=23)
        holiday: papd.Series[padt.Bool] = pa.Field()
        weekday: papd.Series[padt.UInt8] = pa.Field(ge=0, le=6)
        workingday: papd.Series[padt.Bool] = pa.Field()
        weathersit: papd.Series[padt.UInt8] = pa.Field(ge=1, le=4)
        temp: papd.Series[padt.Float16] = pa.Field(ge=0, le=1)
        atemp: papd.Series[padt.Float16] = pa.Field(ge=0, le=1)
        hum: papd.Series[padt.Float16] = pa.Field(ge=0, le=1)
        windspeed: papd.Series[padt.Float16] = pa.Field(ge=0, le=1)
        casual: papd.Series[padt.UInt32] = pa.Field(ge=0)
        registered: papd.Series[padt.UInt32] = pa.Field(ge=0)
    """

    index: papd.Index[padt.UInt32] = pa.Field(ge=0)
    A: papd.Series[padt.Float16] = pa.Field(ge=0)
    B: papd.Series[padt.UInt32] = pa.Field(ge=0)
    C: papd.Series[padt.String] = pa.Field()
    D: papd.Series[padt.UInt32] = pa.Field(ge=0)


class ExampleTargetsSchema(Schema):
    """Schema for the project target.

    Example:
        instant: papd.Index[padt.UInt32] = pa.Field(ge=0)
        cnt: papd.Series[padt.UInt32] = pa.Field(ge=0)
    """

    index: papd.Index[padt.UInt32] = pa.Field(ge=0)
    target: papd.Series[padt.UInt32] = pa.Field(ge=0)


class ExampleOutputsSchema(Schema):
    """Schema for the project output

    Example:
        instant: papd.Index[padt.UInt32] = pa.Field(ge=0)
        prediction: papd.Series[padt.UInt32] = pa.Field(ge=0)

    """

    index: papd.Index[padt.UInt32] = pa.Field(ge=0)
    prediction: papd.Series[padt.UInt32] = pa.Field(ge=0)


class ExampleSHAPValuesSchema(Schema):
    """Schema for the project shap values."""

    class Config:
        """Default configurations this schema.

        Parameters:
            dtype (str): dataframe default data type.
            strict (bool): ensure the data type is correct.
        """

        dtype: str = "float32"
        strict: bool = False


class ExampleFeatureImportancesSchema(Schema):
    """Schema for the project feature importances."""

    feature: papd.Series[padt.String] = pa.Field()
    importance: papd.Series[padt.Float32] = pa.Field()
