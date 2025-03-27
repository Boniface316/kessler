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
from datetime import datetime, timedelta

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
    CCDS_CDM_VERS: T.Optional[float] = 1.0
    CREATION_DATE: T.Optional[str] = None
    ORIGINATOR: T.Optional[str] = "ESA"
    MESSAGE_FOR: T.Optional[str] = "ESA"
    MESSAGE_ID: T.Optional[int] = 1
    OBJECT_1: T.Optional[str] = "Target"
    OBJECT_2: T.Optional[str] = "Chaser"
    url: str = "https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/train_data.zip"

    def read(self, columns_to_keep) -> pd.DataFrame:
        if os.path.exists(self.path):
            data = pd.read_csv(self.path)
        else:
            loguru.logger.info(f"File not found at {self.path}. Downloading from {self.url}.")
            data = self._download_data()

        if self.limit is not None:
            data = data.head(self.limit)

        columns_to_keep = [col for col in columns_to_keep if col not in non_column_names]
        missing_columns = [col for col in columns_to_keep if col not in data.columns]
        if missing_columns:
            loguru.logger.warning(
                f"The following columns are not in the dataframe: {missing_columns}"
            )
        columns_to_keep = [col for col in columns_to_keep if col in data.columns]
        data = data[columns_to_keep]
        data = data.dropna()
        if self.remove_outliers:
            data = self._remove_outliers(data)

        data = data.sample(frac=1, axis=1).reset_index(drop=True)
        data_grouped_by_event_id = data.groupby("event_id")
        loguru.logger.warning(f"Number of events: {len(data_grouped_by_event_id)}")

        if self.date_tca is None:
            date_tca = datetime.now()
            loguru.logger.warning(f"Using current time as TCA: {date_tca}")

        if self.number_of_events is None:
            number_of_events = min(self.number_of_events, len(data_grouped_by_event_id))
            loguru.logger.warning(f"Using all events: {number_of_events}")

        data = self.create_event_dataset(data_grouped_by_event_id, number_of_events, date_tca)

        return data

    def lineage(
        self,
        name: str,
        data: pd.DataFrame,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        # TODO: Confirm the lineage function output
        return lineage.from_pandas(data, name=name, targets=targets, predictions=predictions)

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

    def create_event_dataset(self, data_grouped_by_event_id, number_of_events, date_tca):
        for i, (event_id, event_data) in enumerate(data_grouped_by_event_id):
            if i > number_of_events:
                break
            cdms = self.event_data_to_cdms(event_data, date_tca)

    def event_data_to_cdms(self, event_data, date_tca):
        cdms = []

        for _, single_cdm_data in event_data:
            cdm = self.single_event_to_cdm(single_cdm_data, date_tca)

    def single_event_to_cdm(self, single_cdm_data: pd.DataFrame, date_tca):
        time_to_tca = single_cdm_data["time_to_tca"]
        creation_date = date_tca - timedelta(days=time_to_tca)
        header = {
            "CCSDS_CDM_VERS": self.CCDS_CDM_VERS,
            "CREATION_DATE": creation_date.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "ORIGINATOR": self.ORIGINATOR,
            "MESSAGE_FOR": self.MESSAGE_FOR,
            "MESSAGE_ID": self.MESSAGE_ID,
        }

        date_creation = date_tca - timedelta(days=time_to_tca)

        relative_metadata = {
            "TCA": date_tca.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "MISS_DISTANCE": single_cdm_data.get("miss_distance", None),
            "RELATIVE_SPEED": single_cdm_data.get("relative_speed", None),
            "RELATIVE_POSITION_R": single_cdm_data.get("relative_position_r", None),
            "RELATIVE_POSITION_T": single_cdm_data.get("relative_position_t", None),
            "RELATIVE_POSITION_N": single_cdm_data.get("relative_position_n", None),
            "RELATIVE_VELOCITY_R": single_cdm_data.get("relative_velocity_r", None),
            "RELATIVE_VELOCITY_T": single_cdm_data.get("relative_velocity_t", None),
            "RELATIVE_VELOCITY_N": single_cdm_data.get("relative_velocity_n", None),
            "START_SCREEN_PERIOD": single_cdm_data.get("start_screen_period", None),
            "STOP_SCREEN_PERIOD": single_cdm_data.get("stop_screen_period", None),
            "SCREEN_VOLUME_FRAME": single_cdm_data.get("screen_volume_frame", None),
            "SCREEN_VOLUME_SHAPE": single_cdm_data.get("screen_volume_shape", None),
            "SCREEN_VOLUME_X": single_cdm_data.get("screen_volume_x", None),
            "SCREEN_VOLUME_Y": single_cdm_data.get("screen_volume_y", None),
            "SCREEN_VOLUME_Z": single_cdm_data.get("screen_volume_z", None),
            "SCREEN_ENTRY_TIME": single_cdm_data.get("screen_entry_time", None),
            "SCREEN_EXIT_TIME": single_cdm_data.get("screen_exit_time", None),
            "COLLISION_PROBABILITY": single_cdm_data.get("collision_probability", None),
            "COLLISION_PROBABILITY_METHOD": single_cdm_data.get(
                "collision_probability_method", None
            ),
        }

        target_metadata = {
            "OBJECT": self.OBJECT_1,
            "OBJECT_DESIGNATOR": single_cdm_data.get("object_designator", None),
            "CATALOG_NAME": single_cdm_data.get("catalog_name", None),
            "OBJECT_NAME": single_cdm_data.get("object_name", None),
            "INTERNATIONAL_DESIGNATOR": single_cdm_data.get("international_designator", None),
            "OBJECT_TYPE": single_cdm_data.get("object_type", None),
            "OPERATOR_CONTACT_POSITION": single_cdm_data.get("operator_contact_position", None),
            "OPERATOR_ORGANIZATION": single_cdm_data.get("operator_organization", None),
            "OPERATOR_PHONE": single_cdm_data.get("operator_phone", None),
            "OPERATOR_EMAIL": single_cdm_data.get("operator_email", None),
            "EPHEMERIS_NAME": single_cdm_data.get("ephemeris_name", None),
            "COVARIANCE_METHOD": single_cdm_data.get("covariance_method", None),
            "MANEUVERABLE": single_cdm_data.get("maneuverable", None),
            "ORBIT_CENTER": single_cdm_data.get("orbit_center", None),
            "REF_FRAME": single_cdm_data.get("ref_frame", None),
            "GRAVITY_MODEL": single_cdm_data.get("gravity_model", None),
            "ATMOSPHERIC_MODEL": single_cdm_data.get("atmospheric_model", None),
            "N_BODY_PERTURBATIONS": single_cdm_data.get("n_body_perturbations", None),
            "SOLAR_RAD_PRESSURE": single_cdm_data.get("solar_rad_pressure", None),
            "EARTH_TIDES": single_cdm_data.get("earth_tides", None),
            "INTRACK_THRUST": single_cdm_data.get("intrack_thrust", None),
        }

        target_time_lastob_start = single_cdm_data.get("t_time_lastob_start", None)
        target_time_lastob_start = date_creation - timedelta(days=target_time_lastob_start)
        target_time_lastob_end = single_cdm_data.get("t_time_lastob_end", None)
        target_time_lastob_end = date_creation - timedelta(days=target_time_lastob_end)

        # TODO: Find from the original script the values of the empty fields
        # TODO: complete get_data_od
        # TODO: Compare the values from the original script


        target_data_od = {
            "RECOMMENDED_OD_SPAN": single_cdm_data.get("t_recommended_od_span", None),
            "ACTUAL_OD_SPAN": single_cdm_data.get("t_actual_od_span", None),
            "OBS_AVAILABLE": single_cdm_data.get("t_obs_available", None),
            "OBS_USED": single_cdm_data.get("t_obs_used", None),
            "TRACKS_AVAILABLE",
            "TRACKS_USED",
            "RESIDUALS_ACCEPTED": single_cdm_data.get("t_residuals_accepted", None),
            "WEIGHTED_RMS": single_cdm_data.get("t_weighted_rms", None),
            "AREA_PC",
            "AREA_DRG",
            "AREA_SRP",
            "MASS",
            "CD_AREA_OVER_MASS",
            "CR_AREA_OVER_MASS",
            "THRUST_ACCELERATION",
            "SEDR": single_cdm_data.get("t_sedr", None),
            "TIME_LASTOB_START": target_time_lastob_start.strftime("%Y-%m-%dT%H:%M:%S.%f")
            "TIME_LASTOB_END": target_time_lastob_end.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        }

        target_data_state = {"X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"}

        target_data_covariance = self.get_covariance_data("t", single_cdm_data)

    def get_data_od(self, column_prefix, single_cdm_data):
        pass

    def get_covariance_data(self, column_prefix, single_cdm_data):
        return {
            f"{column_prefix}_CR_R": single_cdm_data[f"{column_prefix}_sigma_r"] ** 2.0,
            f"{column_prefix}_CT_R": (
                single_cdm_data[f"{column_prefix}_ct_r"]
                * single_cdm_data[f"{column_prefix}_sigma_r"]
                * single_cdm_data[f"{column_prefix}_sigma_t"]
            ),
            f"{column_prefix}_CT_T": single_cdm_data[f"{column_prefix}_sigma_t"] ** 2.0,
            f"{column_prefix}_CN_R": (
                single_cdm_data[f"{column_prefix}_cn_r"]
                * single_cdm_data[f"{column_prefix}_sigma_n"]
                * single_cdm_data[f"{column_prefix}_sigma_r"]
            ),
            f"{column_prefix}_CN_T": (
                single_cdm_data[f"{column_prefix}_cn_t"]
                * single_cdm_data[f"{column_prefix}_sigma_n"]
                * single_cdm_data[f"{column_prefix}_sigma_t"]
            ),
            f"{column_prefix}_CN_N": single_cdm_data[f"{column_prefix}_sigma_n"] ** 2.0,
            f"{column_prefix}_CRDOT_R": (
                single_cdm_data[f"{column_prefix}_crdot_r"]
                * single_cdm_data[f"{column_prefix}_sigma_rdot"]
                * single_cdm_data[f"{column_prefix}_sigma_r"]
            ),
            f"{column_prefix}_CRDOT_T": (
                single_cdm_data[f"{column_prefix}_crdot_t"]
                * single_cdm_data[f"{column_prefix}_sigma_rdot"]
                * single_cdm_data[f"{column_prefix}_sigma_t"]
            ),
            f"{column_prefix}_CRDOT_N": (
                single_cdm_data[f"{column_prefix}_crdot_n"]
                * single_cdm_data[f"{column_prefix}_sigma_rdot"]
                * single_cdm_data[f"{column_prefix}_sigma_n"]
            ),
            f"{column_prefix}_CRDOT_RDOT": single_cdm_data[f"{column_prefix}_sigma_rdot"] ** 2.0,
            f"{column_prefix}_CTDOT_R": (
                single_cdm_data[f"{column_prefix}_ctdot_r"]
                * single_cdm_data[f"{column_prefix}_sigma_tdot"]
                * single_cdm_data[f"{column_prefix}_sigma_r"]
            ),
            f"{column_prefix}_CTDOT_T": (
                single_cdm_data[f"{column_prefix}_ctdot_t"]
                * single_cdm_data[f"{column_prefix}_sigma_tdot"]
                * single_cdm_data[f"{column_prefix}_sigma_t"]
            ),
            f"{column_prefix}_CTDOT_N": (
                single_cdm_data[f"{column_prefix}_ctdot_n"]
                * single_cdm_data[f"{column_prefix}_sigma_tdot"]
                * single_cdm_data[f"{column_prefix}_sigma_n"]
            ),
            f"{column_prefix}_CTDOT_RDOT": (
                single_cdm_data[f"{column_prefix}_ctdot_rdot"]
                * single_cdm_data[f"{column_prefix}_sigma_tdot"]
                * single_cdm_data[f"{column_prefix}_sigma_rdot"]
            ),
            f"{column_prefix}_CTDOT_TDOT": single_cdm_data[f"{column_prefix}_sigma_tdot"] ** 2.0,
            f"{column_prefix}_CNDOT_R": (
                single_cdm_data[f"{column_prefix}_cndot_r"]
                * single_cdm_data[f"{column_prefix}_sigma_ndot"]
                * single_cdm_data[f"{column_prefix}_sigma_r"]
            ),
            f"{column_prefix}_CNDOT_T": (
                single_cdm_data[f"{column_prefix}_cndot_t"]
                * single_cdm_data[f"{column_prefix}_sigma_ndot"]
                * single_cdm_data[f"{column_prefix}_sigma_t"]
            ),
            f"{column_prefix}_CNDOT_N": (
                single_cdm_data[f"{column_prefix}_cndot_n"]
                * single_cdm_data[f"{column_prefix}_sigma_ndot"]
                * single_cdm_data[f"{column_prefix}_sigma_n"]
            ),
            f"{column_prefix}_CNDOT_RDOT": (
                single_cdm_data[f"{column_prefix}_cndot_rdot"]
                * single_cdm_data[f"{column_prefix}_sigma_ndot"]
                * single_cdm_data[f"{column_prefix}_sigma_rdot"]
            ),
            f"{column_prefix}_CNDOT_TDOT": (
                single_cdm_data[f"{column_prefix}_cndot_tdot"]
                * single_cdm_data[f"{column_prefix}_sigma_ndot"]
                * single_cdm_data[f"{column_prefix}_sigma_tdot"]
            ),
            f"{column_prefix}_CNDOT_NDOT": single_cdm_data[f"{column_prefix}_sigma_ndot"] ** 2.0,
        }


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
