import typing as T
import pandas as pd
from ._base import Reader, Lineage
from ._base import Writer
import os
import requests
import zipfile
import io
import loguru
import mlflow.data.pandas_dataset as lineage

non_column_names = [
    "index",
    "Config",
    "__annotations__",
    "__checks__",
    "__class__",
    "__class_getitem__",
    "__config__",
    "__delattr__",
    "__dict__",
    "__dir__",
    "__doc__",
    "__eq__",
    "__extras__",
    "__fields__",
    "__format__",
    "__ge__",
    "__get_pydantic_core_schema__",
    "__get_pydantic_json_schema__",
    "__get_validators__",
    "__getattribute__",
    "__getstate__",
    "__gt__",
    "__hash__",
    "__init__",
    "__init_subclass__",
    "__le__",
    "__lt__",
    "__modify_schema__",
    "__module__",
    "__ne__",
    "__new__",
    "__orig_bases__",
    "__parameters__",
    "__parsers__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__root_checks__",
    "__root_parsers__",
    "__schema__",
    "__setattr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "__weakref__",
    "_build_columns_index",
    "_collect_check_infos",
    "_collect_config_and_extras",
    "_collect_fields",
    "_collect_parser_infos",
    "_extract_checks",
    "_extract_config_options_and_extras",
    "_extract_df_checks",
    "_extract_df_parsers",
    "_extract_parsers",
    "_get_model_attrs",
    "_regex_filter",
    "build_schema_",
    "check",
    "example",
    "get_metadata",
    "pydantic_validate",
    "strategy",
    "to_json_schema",
    "to_schema",
    "to_yaml",
    "validate",
]


class CSVReader(Reader):
    """Read a dataframe dataset in csv format."""

    KIND: T.Literal["CSVReader"] = "CSVReader"
    path: str
    limit: int | None = None
    number_of_events: int | None = None
    date_tca: str | None = None
    remove_outliers: bool | None = True
    url: str = "https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/train_data.zip"

    def read(self, columns_to_keep) -> pd.DataFrame:
        if os.path.exists(self.path):
            data = pd.read_csv(self.path)
        else:
            loguru.logger.info(
                f"File not found at {self.path}. Downloading from {self.url}."
            )
            data = self._download_data()

        if self.limit is not None:
            data = data.head(self.limit)

        columns_to_keep = [
            col for col in columns_to_keep if col not in non_column_names
        ]
        data = data[columns_to_keep]
        data = data.dropna()
        if self.remove_outliers:
            data = self._remove_outliers(data)
        return data

    def lineage(
        self,
        name: str,
        data: pd.DataFrame,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        # TODO: Confirm the lineage function output
        return lineage.from_pandas(
            data, name=name, targets=targets, predictions=predictions
        )

    def _download_data(self) -> pd.DataFrame:
        """Download the dataset from the url.

        Returns:
            pd.DataFrame: dataframe representation.
        """

        response = requests.get(self.url)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(os.path.dirname(self.path))
            return pd.read_csv(self.path)
        else:
            raise FileNotFoundError(f"Unable to download the file from {self.url}")

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the dataset.

        Args:
            data (pd.DataFrame): dataframe representation.

        Returns:
            pd.DataFrame: dataframe representation without outliers.
        """
        conditions = [
            data["t_sigma_r"] <= 20,
            data["c_sigma_r"] <= 1000,
            data["t_sigma_t"] <= 2000,
            data["c_sigma_t"] <= 100000,
            data["t_sigma_n"] <= 10,
            data["c_sigma_n"] <= 450,
        ]
        for condition in conditions:
            data = data[condition]
        return data


class CSVWriter(Writer):
    """CSV writer for a dataset.

    Parameters:
        path (str): path to write the dataset.
    """

    KIND: T.Literal["CSVWriter"] = "CSVWriter"

    def write(self, data: pd.DataFrame) -> None:
        """Write a dataframe to a dataset.

        Args:
            data (pd.DataFrame): dataframe representation.
        """
        data.to_csv(self.path, index=False)
