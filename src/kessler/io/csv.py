import io
import os
import typing as T
import zipfile
from datetime import datetime, timedelta

import loguru
import mlflow.data.pandas_dataset as lineage
import pandas as pd
import requests

from .__keys import (
    data_covariance_list,
    data_od_list,
    data_state_list,
    header_list,
    object_metadata_list,
    relative_metadata_list,
)
from ._base import Lineage, Reader, Writer
from ._CDM import CDM
from ._event import Event, EventDataset
from ._utils import _from_date_str_to_days
from .schemas import InputsSchema


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
    non_column_names: list = [
        "index",
        "Config",
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

    def read(self, columns_to_keep=None) -> pd.DataFrame:
        if os.path.exists(self.path):
            data = pd.read_csv(self.path)
        else:
            loguru.logger.info(f"File not found at {self.path}. Downloading from {self.url}.")
            data = self._download_data()

        if columns_to_keep is None:
            columns_to_keep = dir(InputsSchema)

        columns_to_keep = [col for col in columns_to_keep if col not in self.non_column_names]
        columns_to_keep = [col for col in columns_to_keep if not col.startswith("_")]
        missing_columns = [col for col in columns_to_keep if col not in data.columns]
        if missing_columns:
            loguru.logger.warning(
                f"The following columns are not in the dataframe: {missing_columns}"
            )
        breakpoint()
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
            number_of_events = len(data_grouped_by_event_id)
        else:
            number_of_events = self.number_of_events

        loguru.logger.warning(f"Using all events: {number_of_events}")

        data = self._create_event_dataset(data_grouped_by_event_id, number_of_events, date_tca)

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

    def _create_event_dataset(self, data_grouped_by_event_id, number_of_events, date_tca):
        events = []
        for i, (event_id, event_data) in enumerate(data_grouped_by_event_id):
            if i > number_of_events - 1:
                break
            loguru.logger.warning(f"Processing event {i + 1} with event_id {event_id}")
            first_date_of_event = self._get_creation_date(event_data.time_to_tca.iloc[0], date_tca)
            cdms_per_event_id = self._event_data_to_cdms(
                event_data, date_tca, event_id, first_date_of_event
            )
            events.append(Event(cdms=cdms_per_event_id))

        return EventDataset(events=events)

    def _event_data_to_cdms(self, event_data, date_tca, event_id, first_date_of_event):
        single_row_cdm = []

        for _, single_cdm_data in event_data.iterrows():
            single_row_cdm.append(
                self._single_event_to_cdm(single_cdm_data, date_tca, event_id, first_date_of_event)
            )

        return single_row_cdm

    def _single_event_to_cdm(
        self,
        single_cdm_data: pd.DataFrame,
        date_tca,
        event_id,
        first_date_of_event,
        _from_date_str_to_days=_from_date_str_to_days,
    ):
        creation_date = self._get_creation_date(single_cdm_data["time_to_tca"], date_tca)

        creation_date_str = creation_date.strftime("%Y-%m-%dT%H:%M:%S.%f")
        TCA = date_tca.strftime("%Y-%m-%dT%H:%M:%S.%f")

        creation_date_in_days = _from_date_str_to_days(creation_date_str, first_date_of_event)
        TCA_days = _from_date_str_to_days(TCA, first_date_of_event)
        DAYS_TO_TCA = TCA_days - creation_date_in_days

        header = self._get_header(event_id, creation_date_str)
        values_extra = self._get_values_extra(creation_date_in_days, TCA_days, DAYS_TO_TCA)

        relative_metadata = self._get_relative_metadata(single_cdm_data, TCA)

        target_metadata = self._get_object_metadata(self.OBJECT_1, single_cdm_data)
        target_data_od = self._get_data_od("t", single_cdm_data, creation_date)
        target_data_covariance = self._get_covariance_data("t", single_cdm_data)
        target_data_state = self._get_state("t", single_cdm_data)

        chaser_metadata = self._get_object_metadata(self.OBJECT_2, single_cdm_data)
        chaser_data_od = self._get_data_od("c", single_cdm_data, creation_date)
        chaser_data_covariance = self._get_covariance_data("c", single_cdm_data)
        chaser_data_state = self._get_state("c", single_cdm_data)

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

    def _verify_keys(self, dict_with_keys, reference_keys):
        missing_header_keys = [k for k in dict_with_keys.keys() if k not in reference_keys]
        if missing_header_keys:
            raise ValueError(f"Header keys missing from header_list: {missing_header_keys}")

    def _get_header(self, event_id, creation_date_str):
        header = {
            "CCSDS_CDM_VERS": self.CCDS_CDM_VERS,
            "EVENT_ID": event_id,
            "CREATION_DATE": creation_date_str,
            "ORIGINATOR": self.ORIGINATOR,
            "MESSAGE_FOR": self.MESSAGE_FOR,
            "MESSAGE_ID": self.MESSAGE_ID,
        }
        self._verify_keys(header, header_list)
        return header

    def _get_values_extra(self, creation_date_in_days, TCA_days, DAYS_TO_TCA):
        return {
            "CREATION_DATE_IN_DAYS": creation_date_in_days,
            "TCA_IN_DAYS": TCA_days,
            "DAYS_TO_TCA": DAYS_TO_TCA,
        }

    def _get_relative_metadata(self, single_cdm_data, TCA):
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
            "COLLISION_PROBABILITY_METHOD": single_cdm_data.get(
                "collision_probability_method", None
            ),
        }

        self._verify_keys(relative_metadata, relative_metadata_list)
        return relative_metadata

    def _get_data_od(self, column_prefix, single_cdm_data, creation_date):
        time_lastob_start = single_cdm_data.get(f"{column_prefix}_time_lastob_start", None)
        time_lastob_start = creation_date - timedelta(days=time_lastob_start)
        time_lastob_end = single_cdm_data.get(f"{column_prefix}_time_lastob_end", None)
        time_lastob_end = creation_date - timedelta(days=time_lastob_end)
        data_od = {
            "RECOMMENDED_OD_SPAN": single_cdm_data.get(
                f"{column_prefix}_recommended_od_span", None
            ),
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
            "THRUST_ACCELERATION": single_cdm_data.get(
                f"{column_prefix}_thrust_acceleration", None
            ),
            "SEDR": single_cdm_data.get(f"{column_prefix}_sedr", None),
            "TIME_LASTOB_START": time_lastob_start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "TIME_LASTOB_END": time_lastob_end.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        }
        self._verify_keys(data_od, data_od_list)
        return data_od

    def _get_covariance_data(self, column_prefix, single_cdm_data):
        covariance_data = {
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
            "CDRG_R": single_cdm_data.get(f"{column_prefix}_cdrg_r", None),
            "CDRG_T": single_cdm_data.get(f"{column_prefix}_cdrg_t", None),
            "CDRG_N": single_cdm_data.get(f"{column_prefix}_cdrg_n", None),
            "CDRG_RDOT": single_cdm_data.get(f"{column_prefix}_cdrg_rdot", None),
            "CDRG_TDOT": single_cdm_data.get(f"{column_prefix}_cdrg_tdot", None),
            "CDRG_NDOT": single_cdm_data.get(f"{column_prefix}_cdrg_ndot", None),
            "CDRG_DRG": single_cdm_data.get(f"{column_prefix}_cdrg_drg", None),
            "CSRP_R": single_cdm_data.get(f"{column_prefix}_csrp_r", None),
            "CSRP_T": single_cdm_data.get(f"{column_prefix}_csrp_t", None),
            "CSRP_N": single_cdm_data.get(f"{column_prefix}_csrp_n", None),
            "CSRP_RDOT": single_cdm_data.get(f"{column_prefix}_csrp_rdot", None),
            "CSRP_TDOT": single_cdm_data.get(f"{column_prefix}_csrp_tdot", None),
            "CSRP_NDOT": single_cdm_data.get(f"{column_prefix}_csrp_ndot", None),
            "CSRP_DRG": single_cdm_data.get(f"{column_prefix}_csrp_drg", None),
            "CSRP_SRP": single_cdm_data.get(f"{column_prefix}_csrp_srp", None),
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
        self._verify_keys(covariance_data, data_covariance_list)
        return covariance_data

    def _get_state(self, column_prefix, single_cdm_data):
        data_state = {
            "X": single_cdm_data.get(f"{column_prefix}_x", None),
            "Y": single_cdm_data.get(f"{column_prefix}_y", None),
            "Z": single_cdm_data.get(f"{column_prefix}_z", None),
            "X_DOT": single_cdm_data.get(f"{column_prefix}_x_dot", None),
            "Y_DOT": single_cdm_data.get(f"{column_prefix}_y_dot", None),
            "Z_DOT": single_cdm_data.get(f"{column_prefix}_z_dot", None),
        }
        self._verify_keys(data_state, data_state_list)
        return data_state

    def _get_object_metadata(self, object_name, single_cdm_data):
        if object_name.lower() == "target":
            prefix = "t"
        elif object_name.lower() == "chaser":
            prefix = "c"
        else:
            raise ValueError(f"Unknown object name: {object_name}")
        object_metadata = {
            "OBJECT": object_name,
            "OBJECT_DESIGNATOR": single_cdm_data.get(f"{prefix}_object_designator", None),
            "CATALOG_NAME": single_cdm_data.get(f"{prefix}_catalog_name", None),
            "OBJECT_NAME": single_cdm_data.get(f"{prefix}_object_name", None),
            "INTERNATIONAL_DESIGNATOR": single_cdm_data.get(
                f"{prefix}_international_designator", None
            ),
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
        self._verify_keys(object_metadata, object_metadata_list)
        return object_metadata

    def _get_creation_date(self, time_to_tca, date_tca):
        return date_tca - timedelta(days=time_to_tca)


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
