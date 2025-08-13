import typing as T
import pandas as pd
from ._base import Reader, Lineage
import os
import requests
import zipfile
import io
import loguru
import mlflow.data.pandas_dataset as lineage
from datetime import datetime
from typing import List
from typing import Iterable, Tuple
from loguru import logger
from typing import Any
from typing import Callable
from datetime import timedelta
from typing import Dict
from typing import Optional
from typing import Union


# TODO: use _ as a prefix to eliminate non-column names
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
    number_of_events: int | None = None
    date_tca: str | None = None
    remove_outliers: bool | None = True
    CCDS_CDM_VERS: T.Optional[float] = 1.0
    CREATION_DATE: T.Optional[str] = None
    ORIGINATOR: T.Optional[str] = "ESA"
    MESSAGE_FOR: T.Optional[str] = "ESA"
    MESSAGE_ID: T.Optional[int] = 1
    OBJECT_1: T.Optional[str] = "target"
    OBJECT_2: T.Optional[str] = "chaser"
    url: str = "https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/train_data.zip"


def read(self, columns_to_keep: List[str]) -> pd.DataFrame:
    """
    Reads the dataset from a local path or downloads it if not found,
    performs preprocessing steps including column filtering, NaN removal,
    optional outlier removal, shuffling, and grouping by event ID.

    Args:
        columns_to_keep (List[str]): A list of column names to retain in the dataset.

    Returns:
        pd.DataFrame: The processed event dataset.

    Processing steps:
    - Reads CSV data from `self.path` if it exists, otherwise downloads via `self._download_data`.
    - Filters out any columns listed in `non_column_names`.
    - Warns if any of the desired columns are missing.
    - Drops rows with NaN values.
    - Optionally removes outliers if `self.remove_outliers` is True.
    - Shuffles the columns randomly.
    - Groups the data by the "event_id" column.
    - Uses `self.date_tca` or current time as TCA (Time of Central Activity).
    - Uses `self.number_of_events` if specified, otherwise includes all events.
    - Calls `self.create_event_dataset()` to generate the final processed data.

    Logs:
        - Missing file warnings
        - Missing columns
        - Number of events detected
        - Whether current date is used as TCA
        - Number of events used in final dataset

    Returns:
        pd.DataFrame: Final preprocessed event dataset.
    """
    if os.path.exists(self.path):
        data = pd.read_csv(self.path)
    else:
        loguru.logger.info(f"File not found at {self.path}. Downloading from {self.url}.")
        data = self._download_data()

    columns_to_keep = [col for col in columns_to_keep if col not in non_column_names]
    missing_columns = [col for col in columns_to_keep if col not in data.columns]
    if missing_columns:
        loguru.logger.warning(f"The following columns are not in the dataframe: {missing_columns}")
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
    else:
        date_tca = self.date_tca

    if self.number_of_events is None:
        number_of_events = len(data_grouped_by_event_id)
    else:
        number_of_events = self.number_of_events

    loguru.logger.warning(f"Using all events: {number_of_events}")

    data = self.create_event_dataset(data_grouped_by_event_id, number_of_events, date_tca)

    return data


def lineage(
    self,
    name: str,
    data: pd.DataFrame,
    targets: [str] = None,
    predictions: [str] = None,
) -> Lineage:
    """
    Creates a Lineage object from a pandas DataFrame, with optional target and prediction columns.

    Args:
        name (str): A name identifier for the dataset lineage.
        data (pd.DataFrame): The DataFrame containing the dataset to trace.
        targets (Optional[str], optional): The name of the target column, if applicable. Defaults to None.
        predictions (Optional[str], optional): The name of the prediction column, if applicable. Defaults to None.

    Returns:
        Lineage: A Lineage object constructed from the provided DataFrame.

    Notes:
        - This function delegates to `lineage.from_pandas()`.
        - It is assumed that `lineage` is an imported module or object containing a `from_pandas()` method.

    TODO:
        - Confirm the expected structure and content of the Lineage object.
        - Validate compatibility of column names passed as `targets` and `predictions`.
    """
    return lineage.from_pandas(data, name=name, targets=targets, predictions=predictions)


def _download_data(self) -> pd.DataFrame:
    """
    Downloads a zipped dataset from the specified URL and extracts it to the local file path.

    This method assumes that:
    - The URL (`self.url`) points to a ZIP archive.
    - The ZIP archive contains a file that should be extracted to the directory of `self.path`.
    - After extraction, the data is read from `self.path` using `pandas.read_csv`.

    Args:
        self (object): An instance of the class containing attributes:
            - url (str): The URL pointing to the ZIP archive to download.
            - path (str): The local file path where the extracted file will be saved and read from.

    Returns:
        pd.DataFrame: The dataset loaded into a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file cannot be downloaded (non-200 status code).
        pd.errors.EmptyDataError: If the extracted file is empty or unreadable.
        zipfile.BadZipFile: If the downloaded content is not a valid ZIP archive.
    """
    response = requests.get(self.url)

    if response.status_code == 200:
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(os.path.dirname(self.path))
            return pd.read_csv(self.path)
        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(
                f"The downloaded content from {self.url} is not a valid ZIP file."
            )
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"Extracted file at {self.path} is empty or unreadable.")
    else:
        raise FileNotFoundError(
            f"Unable to download the file from {self.url} (status code: {response.status_code})"
        )


def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from the dataset based on predefined thresholds for specific columns.

    This method filters the input DataFrame by applying the following conditions:
    - t_sigma_r <= 20
    - c_sigma_r <= 1000
    - t_sigma_t <= 2000
    - c_sigma_t <= 100000
    - t_sigma_n <= 10
    - c_sigma_n <= 450

    Args:
        data (pd.DataFrame): The input DataFrame containing columns:
            't_sigma_r', 'c_sigma_r', 't_sigma_t', 'c_sigma_t', 't_sigma_n', 'c_sigma_n'.

    Returns:
        pd.DataFrame: A new DataFrame with rows containing outliers removed based on the above thresholds.
    """
    condition = (
        (data["t_sigma_r"] <= 20)
        & (data["c_sigma_r"] <= 1000)
        & (data["t_sigma_t"] <= 2000)
        & (data["c_sigma_t"] <= 100000)
        & (data["t_sigma_n"] <= 10)
        & (data["c_sigma_n"] <= 450)
    )
    return data[condition].copy()


def create_event_dataset(
    self,
    data_grouped_by_event_id: Iterable[Tuple[str, pd.DataFrame]],
    number_of_events: int,
    date_tca: pd.Timestamp,
) -> "EventDataset":
    """
    Create an EventDataset by processing a limited number of events grouped by event ID.

    Each event group is converted into a list of CDMs, which are then wrapped into an Event.
    All such Events are compiled into an EventDataset.

    Args:
        data_grouped_by_event_id (Iterable[Tuple[str, pd.DataFrame]]):
            An iterable of tuples, where each tuple contains an event ID (as string or int)
            and its corresponding DataFrame.
        number_of_events (int):
            The maximum number of events to process.
        date_tca (pd.Timestamp):
            The reference date used to compute the event creation date.

    Returns:
        EventDataset:
            A dataset containing processed Event objects based on input data.
    """
    events = []
    for i, (event_id, event_data) in enumerate(data_grouped_by_event_id):
        if i >= number_of_events:
            break
        logger.warning(f"Processing event {i + 1} with event_id {event_id}")

        first_date_of_event = self._get_creation_date(event_data.time_to_tca.iloc[0], date_tca)

        cdms_per_event_id = self.event_data_to_cdms(
            event_data, date_tca, event_id, first_date_of_event
        )

        events.append(Event(cdms=cdms_per_event_id))

    return EventDataset(events=events)


def event_data_to_cdms(
    self,
    event_data: pd.DataFrame,
    date_tca: pd.Timestamp,
    event_id: str,
    first_date_of_event: datetime,
) -> List[Any]:
    """
    Convert all CDM rows for a single event into a list of CDM representations.

    Each row of `event_data` represents a CDM, which is converted using
    `self.single_event_to_cdm`.

    Args:
        event_data (pd.DataFrame):
            A DataFrame where each row corresponds to one CDM of the event.
        date_tca (pd.Timestamp):
            The reference Time of Closest Approach (TCA) for the event.
        event_id (str):
            The unique identifier for the event.
        first_date_of_event (datetime):
            The creation date for the event, typically derived from the time to TCA.

    Returns:
        List[Any]:
            A list of CDM representations (custom format returned by `single_event_to_cdm`).
    """
    single_row_cdm = []

    for _, single_cdm_data in event_data.iterrows():
        cdm = self.single_event_to_cdm(single_cdm_data, date_tca, event_id, first_date_of_event)
        single_row_cdm.append(cdm)

    return single_row_cdm


def single_event_to_cdm(
    self,
    single_cdm_data: pd.DataFrame,
    date_tca: datetime,
    event_id: str,
    first_date_of_event: datetime,
    _from_date_str_to_days: Callable[[str, datetime], float] = _from_date_str_to_days,
) -> "CDM":
    """
    Converts a single CDM (Conjunction Data Message) event row into a structured CDM object.

    Args:
        single_cdm_data : pd.DataFrame
            A DataFrame containing CDM data for a single conjunction event.
        date_tca : datetime
            Time of Closest Approach (TCA) for the event.
        event_id : str
            Unique identifier for the conjunction event.
        first_date_of_event : datetime
            Reference date for converting timestamps into relative days.
        _from_date_str_to_days : Callable[[str, datetime], float], optional
            Function to convert datetime string into float days since `first_date_of_event`.

    Returns:
        CDM
            A structured CDM object containing metadata, orbit data, and relative state data
            for the target and chaser objects involved in the event.
    """
    creation_date = self._get_creation_date(single_cdm_data["time_to_tca"], date_tca)

    creation_date_str = creation_date.strftime("%Y-%m-%dT%H:%M:%S.%f")
    TCA = date_tca.strftime("%Y-%m-%dT%H:%M:%S.%f")

    creation_date_in_days = _from_date_str_to_days(creation_date_str, first_date_of_event)
    TCA_days = _from_date_str_to_days(TCA, first_date_of_event)
    DAYS_TO_TCA = TCA_days - creation_date_in_days

    values_extra = {
        "CREATION_DATE_IN_DAYS": creation_date_in_days,
        "TCA_IN_DAYS": TCA_days,
        "DAYS_TO_TCA": DAYS_TO_TCA,
    }

    header = {
        "CCSDS_CDM_VERS": self.CCDS_CDM_VERS,
        "EVENT_ID": event_id,
        "CREATION_DATE": creation_date_str,
        "ORIGINATOR": self.ORIGINATOR,
        "MESSAGE_FOR": self.MESSAGE_FOR,
        "MESSAGE_ID": self.MESSAGE_ID,
    }

    relative_metadata = {
        "TCA": TCA,
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
        "COLLISION_PROBABILITY_METHOD": single_cdm_data.get("collision_probability_method", None),
    }

    target_metadata = self.get_object_metadata(self.OBJECT_1, single_cdm_data)
    target_data_od = self.get_data_od("t", single_cdm_data, creation_date)
    target_data_covariance = self.get_covariance_data("t", single_cdm_data)
    target_data_state = self.get_state("t", single_cdm_data)

    chaser_metadata = self.get_object_metadata(self.OBJECT_2, single_cdm_data)
    chaser_data_od = self.get_data_od("c", single_cdm_data, creation_date)
    chaser_data_covariance = self.get_covariance_data("c", single_cdm_data)
    chaser_data_state = self.get_state("c", single_cdm_data)

    return CDM(
        header=header,
        relative_metadata=relative_metadata,
        values_extra=values_extra,
        target_metadata=target_metadata,
        target_data_od=target_data_od,
        target_data_state=target_data_state,
        target_data_covariance=target_data_covariance,
        chaser_metadata=chaser_metadata,
        chaser_data_od=chaser_data_od,
        chaser_data_state=chaser_data_state,
        chaser_data_covariance=chaser_data_covariance,
    )


def get_data_od(
    self, column_prefix: str, single_cdm_data: Dict[str, Any], creation_date: datetime
) -> Dict[str, Any]:
    """
    Extracts orbit determination (OD) related metadata for a specified object
    (target or chaser) involved in a conjunction event.

    Args:
        column_prefix : str
            Prefix used to identify the object (e.g., 't' for target, 'c' for chaser).
        single_cdm_data : Dict[str, Any]
            Dictionary containing CDM data fields for a single conjunction event.
        creation_date : datetime
            The creation date of the CDM, used as a reference to compute
            absolute timestamps from relative day values.

    Returns:
        Dict[str, Any]
            A dictionary containing OD-related metadata fields including observation
            spans, mass properties, area ratios, and observation timestamps.
    """
    time_lastob_start = single_cdm_data.get(f"{column_prefix}_time_lastob_start", None)
    time_lastob_start = creation_date - timedelta(days=time_lastob_start)

    time_lastob_end = single_cdm_data.get(f"{column_prefix}_time_lastob_end", None)
    time_lastob_end = creation_date - timedelta(days=time_lastob_end)

    return {
        "RECOMMENDED_OD_SPAN": single_cdm_data.get(f"{column_prefix}_recommended_od_span", None),
        "ACTUAL_OD_SPAN": single_cdm_data.get(f"{column_prefix}_actual_od_span", None),
        "OBS_AVAILABLE": single_cdm_data.get(f"{column_prefix}_obs_available", None),
        "OBS_USED": single_cdm_data.get(f"{column_prefix}_obs_used", None),
        "TRACKS_AVAILABLE": single_cdm_data.get(f"{column_prefix}_tracks_available", None),
        "TRACKS_USED": single_cdm_data.get(f"{column_prefix}_tracks_used", None),
        "RESIDUALS_ACCEPTED": single_cdm_data.get(f"{column_prefix}_residuals_accepted", None),
        "WEIGHTED_RMS": single_cdm_data.get(f"{column_prefix}_weighted_rms", None),
        "AREA_PC": single_cdm_data.get(f"{column_prefix}_area_pc", None),
        "AREA_DRG": single_cdm_data.get(f"{column_prefix}_area_drg", None),
        "AREA_SRP": single_cdm_data.get(f"{column_prefix}_area_srp", None),
        "MASS": single_cdm_data.get(f"{column_prefix}_mass", None),
        "CD_AREA_OVER_MASS": single_cdm_data.get(f"{column_prefix}_cd_area_over_mass", None),
        "CR_AREA_OVER_MASS": single_cdm_data.get(f"{column_prefix}_cr_area_over_mass", None),
        "THRUST_ACCELERATION": single_cdm_data.get(f"{column_prefix}_thrust_acceleration", None),
        "SEDR": single_cdm_data.get(f"{column_prefix}_sedr", None),
        "TIME_LASTOB_START": time_lastob_start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "TIME_LASTOB_END": time_lastob_end.strftime("%Y-%m-%dT%H:%M:%S.%f"),
    }


def get_covariance_data(
    self, column_prefix: str, single_cdm_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constructs the covariance matrix components for a specified object (target or chaser)
    involved in a conjunction event.

    The covariance data is derived from sigma values and correlation coefficients
    in the RTN (Radial, Transverse, Normal) frame and their time derivatives (e.g., R, R_dot, etc.).

    Args:
        column_prefix : str
            Prefix used to identify the object ('t' for target, 'c' for chaser).
        single_cdm_data : Dict[str, Any]
            Dictionary containing all relevant CDM fields for a single conjunction event.

    Returns:
        Dict[str, Any]
            A dictionary containing computed and raw covariance values including:
            Variances (sigma²)
            Covariances (correlation * sigma products)
            Optional force-related covariance terms (e.g., drag, SRP, thrust)
    """
    return {
        # Position covariance terms
        "CR_R": single_cdm_data[f"{column_prefix}_sigma_r"] ** 2.0,
        "CT_R": (
            single_cdm_data[f"{column_prefix}_ct_r"]
            * single_cdm_data[f"{column_prefix}_sigma_r"]
            * single_cdm_data[f"{column_prefix}_sigma_t"]
        ),
        "CT_T": single_cdm_data[f"{column_prefix}_sigma_t"] ** 2.0,
        "CN_R": (
            single_cdm_data[f"{column_prefix}_cn_r"]
            * single_cdm_data[f"{column_prefix}_sigma_n"]
            * single_cdm_data[f"{column_prefix}_sigma_r"]
        ),
        "CN_T": (
            single_cdm_data[f"{column_prefix}_cn_t"]
            * single_cdm_data[f"{column_prefix}_sigma_n"]
            * single_cdm_data[f"{column_prefix}_sigma_t"]
        ),
        "CN_N": single_cdm_data[f"{column_prefix}_sigma_n"] ** 2.0,
        # Velocity covariance terms
        "CRDOT_R": (
            single_cdm_data[f"{column_prefix}_crdot_r"]
            * single_cdm_data[f"{column_prefix}_sigma_rdot"]
            * single_cdm_data[f"{column_prefix}_sigma_r"]
        ),
        "CRDOT_T": (
            single_cdm_data[f"{column_prefix}_crdot_t"]
            * single_cdm_data[f"{column_prefix}_sigma_rdot"]
            * single_cdm_data[f"{column_prefix}_sigma_t"]
        ),
        "CRDOT_N": (
            single_cdm_data[f"{column_prefix}_crdot_n"]
            * single_cdm_data[f"{column_prefix}_sigma_rdot"]
            * single_cdm_data[f"{column_prefix}_sigma_n"]
        ),
        "CRDOT_RDOT": single_cdm_data[f"{column_prefix}_sigma_rdot"] ** 2.0,
        "CTDOT_R": (
            single_cdm_data[f"{column_prefix}_ctdot_r"]
            * single_cdm_data[f"{column_prefix}_sigma_tdot"]
            * single_cdm_data[f"{column_prefix}_sigma_r"]
        ),
        "CTDOT_T": (
            single_cdm_data[f"{column_prefix}_ctdot_t"]
            * single_cdm_data[f"{column_prefix}_sigma_tdot"]
            * single_cdm_data[f"{column_prefix}_sigma_t"]
        ),
        "CTDOT_N": (
            single_cdm_data[f"{column_prefix}_ctdot_n"]
            * single_cdm_data[f"{column_prefix}_sigma_tdot"]
            * single_cdm_data[f"{column_prefix}_sigma_n"]
        ),
        "CTDOT_RDOT": (
            single_cdm_data[f"{column_prefix}_ctdot_rdot"]
            * single_cdm_data[f"{column_prefix}_sigma_tdot"]
            * single_cdm_data[f"{column_prefix}_sigma_rdot"]
        ),
        "CTDOT_TDOT": single_cdm_data[f"{column_prefix}_sigma_tdot"] ** 2.0,
        "CNDOT_R": (
            single_cdm_data[f"{column_prefix}_cndot_r"]
            * single_cdm_data[f"{column_prefix}_sigma_ndot"]
            * single_cdm_data[f"{column_prefix}_sigma_r"]
        ),
        "CNDOT_T": (
            single_cdm_data[f"{column_prefix}_cndot_t"]
            * single_cdm_data[f"{column_prefix}_sigma_ndot"]
            * single_cdm_data[f"{column_prefix}_sigma_t"]
        ),
        "CNDOT_N": (
            single_cdm_data[f"{column_prefix}_cndot_n"]
            * single_cdm_data[f"{column_prefix}_sigma_ndot"]
            * single_cdm_data[f"{column_prefix}_sigma_n"]
        ),
        "CNDOT_RDOT": (
            single_cdm_data[f"{column_prefix}_cndot_rdot"]
            * single_cdm_data[f"{column_prefix}_sigma_ndot"]
            * single_cdm_data[f"{column_prefix}_sigma_rdot"]
        ),
        "CNDOT_TDOT": (
            single_cdm_data[f"{column_prefix}_cndot_tdot"]
            * single_cdm_data[f"{column_prefix}_sigma_ndot"]
            * single_cdm_data[f"{column_prefix}_sigma_tdot"]
        ),
        "CNDOT_NDOT": single_cdm_data[f"{column_prefix}_sigma_ndot"] ** 2.0,
        # Drag covariance (optional)
        "CDRG_R": single_cdm_data.get(f"{column_prefix}_cdrg_r", None),
        "CDRG_T": single_cdm_data.get(f"{column_prefix}_cdrg_t", None),
        "CDRG_N": single_cdm_data.get(f"{column_prefix}_cdrg_n", None),
        "CDRG_RDOT": single_cdm_data.get(f"{column_prefix}_cdrg_rdot", None),
        "CDRG_TDOT": single_cdm_data.get(f"{column_prefix}_cdrg_tdot", None),
        "CDRG_NDOT": single_cdm_data.get(f"{column_prefix}_cdrg_ndot", None),
        "CDRG_DRG": single_cdm_data.get(f"{column_prefix}_cdrg_drg", None),
        # SRP covariance (optional)
        "CSRP_R": single_cdm_data.get(f"{column_prefix}_csrp_r", None),
        "CSRP_T": single_cdm_data.get(f"{column_prefix}_csrp_t", None),
        "CSRP_N": single_cdm_data.get(f"{column_prefix}_csrp_n", None),
        "CSRP_RDOT": single_cdm_data.get(f"{column_prefix}_csrp_rdot", None),
        "CSRP_TDOT": single_cdm_data.get(f"{column_prefix}_csrp_tdot", None),
        "CSRP_NDOT": single_cdm_data.get(f"{column_prefix}_csrp_ndot", None),
        "CSRP_DRG": single_cdm_data.get(f"{column_prefix}_csrp_drg", None),
        "CSRP_SRP": single_cdm_data.get(f"{column_prefix}_csrp_srp", None),
        # Thrust covariance (optional)
        "CTHR_R": single_cdm_data.get(f"{column_prefix}_cthr_r", None),
        "CTHR_T": single_cdm_data.get(f"{column_prefix}_cthr_t", None),
        "CTHR_N": single_cdm_data.get(f"{column_prefix}_cthr_n", None),
        "CTHR_RDOT": single_cdm_data.get(f"{column_prefix}_cthr_rdot", None),
        "CTHR_TDOT": single_cdm_data.get(f"{column_prefix}_cthr_tdot", None),
        "CTHR_NDOT": single_cdm_data.get(f"{column_prefix}_cthr_ndot", None),
        "CTHR_DRG": single_cdm_data.get(f"{column_prefix}_cthr_drg", None),
        "CTHR_SRP": single_cdm_data.get(f"{column_prefix}_cthr_srp", None),
        "CTHR_THR": single_cdm_data.get(f"{column_prefix}_cthr_thr", None),
    }


def get_state(
    self, column_prefix: str, single_cdm_data: Dict[str, Any]
) -> Dict[str, Optional[float]]:
    """
    Extracts the Cartesian state vector components for the specified object.

    Args:
        column_prefix : str
            Prefix indicating the object ('t' for target, 'c' for chaser).
        single_cdm_data : Dict[str, Any]
            Dictionary containing all relevant CDM data fields.

    Returns:
        Dict[str, Optional[float]]
            A dictionary containing position and velocity components:
            X, Y, Z: Cartesian position coordinates
            X_DOT, Y_DOT, Z_DOT: Cartesian velocity components
            Values may be None if not present in the input data.
    """
    return {
        "X": single_cdm_data.get(f"{column_prefix}_x", None),
        "Y": single_cdm_data.get(f"{column_prefix}_y", None),
        "Z": single_cdm_data.get(f"{column_prefix}_z", None),
        "X_DOT": single_cdm_data.get(f"{column_prefix}_x_dot", None),
        "Y_DOT": single_cdm_data.get(f"{column_prefix}_y_dot", None),
        "Z_DOT": single_cdm_data.get(f"{column_prefix}_z_dot", None),
    }


def get_object_metadata(
    self, object_name: str, single_cdm_data: Dict[str, Any]
) -> Dict[str, Optional[Any]]:
    """
    Retrieves metadata information for a specified object ('target' or 'chaser') from the CDM data.

    Args:
        object_name : str
            The name of the object to retrieve metadata for. Expected values: 'target' or 'chaser' (case-insensitive).
        single_cdm_data : Dict[str, Any]
            Dictionary containing all relevant CDM data fields.

    Returns:
        Dict[str, Optional[Any]]
            A dictionary containing metadata attributes for the specified object.
            The keys include:
            OBJECT: The provided object_name
            OBJECT_DESIGNATOR, CATALOG_NAME, OBJECT_NAME, INTERNATIONAL_DESIGNATOR, OBJECT_TYPE,
              OPERATOR_CONTACT_POSITION, OPERATOR_ORGANIZATION, OPERATOR_PHONE, OPERATOR_EMAIL,
              EPHEMERIS_NAME, COVARIANCE_METHOD, MANEUVERABLE, ORBIT_CENTER, REF_FRAME,
              GRAVITY_MODEL, ATMOSPHERIC_MODEL, N_BODY_PERTURBATIONS, SOLAR_RAD_PRESSURE,
              EARTH_TIDES, INTRACK_THRUST
            Values may be None if the field is not present in `single_cdm_data`.

    Raises
        ValueError
            If the `object_name` is not 'target' or 'chaser' (case-insensitive).
    """
    if object_name.lower() == "target":
        prefix = "t"
    elif object_name.lower() == "chaser":
        prefix = "c"
    else:
        raise ValueError(f"Unknown object name: {object_name}")

    return {
        "OBJECT": object_name,
        "OBJECT_DESIGNATOR": single_cdm_data.get(f"{prefix}_object_designator", None),
        "CATALOG_NAME": single_cdm_data.get(f"{prefix}_catalog_name", None),
        "OBJECT_NAME": single_cdm_data.get(f"{prefix}_object_name", None),
        "INTERNATIONAL_DESIGNATOR": single_cdm_data.get(f"{prefix}_international_designator", None),
        "OBJECT_TYPE": single_cdm_data.get(f"{prefix}_object_type", None),
        "OPERATOR_CONTACT_POSITION": single_cdm_data.get(
            f"{prefix}_operator_contact_position", None
        ),
        "OPERATOR_ORGANIZATION": single_cdm_data.get(f"{prefix}_operator_organization", None),
        "OPERATOR_PHONE": single_cdm_data.get(f"{prefix}_operator_phone", None),
        "OPERATOR_EMAIL": single_cdm_data.get(f"{prefix}_operator_email", None),
        "EPHEMERIS_NAME": single_cdm_data.get(f"{prefix}_ephemeris_name", None),
        "COVARIANCE_METHOD": single_cdm_data.get(f"{prefix}_covariance_method", None),
        "MANEUVERABLE": single_cdm_data.get(f"{prefix}_maneuverable", None),
        "ORBIT_CENTER": single_cdm_data.get(f"{prefix}_orbit_center", None),
        "REF_FRAME": single_cdm_data.get(f"{prefix}_ref_frame", None),
        "GRAVITY_MODEL": single_cdm_data.get(f"{prefix}_gravity_model", None),
        "ATMOSPHERIC_MODEL": single_cdm_data.get(f"{prefix}_atmospheric_model", None),
        "N_BODY_PERTURBATIONS": single_cdm_data.get(f"{prefix}_n_body_perturbations", None),
        "SOLAR_RAD_PRESSURE": single_cdm_data.get(f"{prefix}_solar_rad_pressure", None),
        "EARTH_TIDES": single_cdm_data.get(f"{prefix}_earth_tides", None),
        "INTRACK_THRUST": single_cdm_data.get(f"{prefix}_intrack_thrust", None),
    }


def _get_creation_date(self, time_to_tca: Union[int, float], date_tca: datetime) -> datetime:
    """
    Calculates the creation date by subtracting the time to TCA (in days) from the TCA date.

    Args:
        time_to_tca : int or float
            The time to Time of Closest Approach (TCA) in days.
        date_tca : datetime
            The datetime of the Time of Closest Approach.

    Returns
        datetime
            The calculated creation date.
    """
    return date_tca - timedelta(days=time_to_tca)




class CSVWriter(Writer):
    """CSV writer for a dataset.

    Args:
        path (str): path to write the dataset.
    """

    KIND: T.Literal["CSVWriter"] = "CSVWriter"


def write(self, data: pd.DataFrame) -> None:
    self.path: str

    """
    Write a pandas DataFrame to a CSV file at the specified dataset path.

    Args:
        data : pd.DataFrame
            The DataFrame to write to the CSV file.

    Returns:
        None
    """
    data.to_csv(self.path, index=False)
