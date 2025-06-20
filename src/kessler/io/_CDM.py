from pydantic import BaseModel
from copy import deepcopy
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import loguru
from .__keys import (
    header_obligatory,
    relative_metadata_obligatory,
    metadata_obligatory,
    data_od_obligatory,
    data_state_obligatory,
    data_covariance_obligatory,
    dict_keys,
    covariance_indices_dict,
)


class ConjuctionDataMessage(
    BaseModel,
    strict=False,
    frozen=False,
    extra="forbid",
):
    header: dict
    relative_metadata: dict
    values_extra: dict

    target_metadata: dict
    target_data_od: dict
    target_data_state: dict
    target_data_covariance: dict

    chaser_metadata: dict
    chaser_data_od: dict
    chaser_data_state: dict
    chaser_data_covariance: dict

    keys_header_obligatory: list = header_obligatory
    keys_relative_metadata_obligatory: list = relative_metadata_obligatory
    keys_metadata_obligatory: list = metadata_obligatory
    keys_data_od_obligatory: list = data_od_obligatory

    keys_data_state_obligatory: list = data_state_obligatory
    keys_data_covariance_obligatory: list = data_covariance_obligatory
    covariance_indices_dict: dict = covariance_indices_dict

    dict_keys: list = dict_keys

    def copy(self):
        return ConjuctionDataMessage(
            header=deepcopy(self.header),
            values_extra=deepcopy(self.values_extra),
            relative_metadata=deepcopy(self.relative_metadata),
            target_metadata=deepcopy(self.target_metadata),
            target_data_od=deepcopy(self.target_data_od),
            target_data_state=deepcopy(self.target_data_state),
            target_data_covariance=deepcopy(self.target_data_covariance),
            chaser_metadata=deepcopy(self.chaser_metadata),
            chaser_data_od=deepcopy(self.chaser_data_od),
            chaser_data_state=deepcopy(self.chaser_data_state),
            chaser_data_covariance=deepcopy(self.chaser_data_covariance),
        )

    def copy_from_other_cdm(self, other_cdm):
        self.header = deepcopy(other_cdm.header)
        self.values_extra = deepcopy(other_cdm.values_extra)
        self.relative_metadata = deepcopy(other_cdm.relative_metadata)
        self.target_metadata = deepcopy(other_cdm.target_metadata)
        self.target_data_od = deepcopy(other_cdm.target_data_od)
        self.target_data_state = deepcopy(other_cdm.target_data_state)
        self.target_data_covariance = deepcopy(other_cdm.target_data_covariance)
        self.chaser_metadata = deepcopy(other_cdm.chaser_metadata)
        self.chaser_data_od = deepcopy(other_cdm.chaser_data_od)
        self.chaser_data_state = deepcopy(other_cdm.chaser_data_state)
        self.chaser_data_covariance = deepcopy(other_cdm.chaser_data_covariance)

    def to_dict(
        self,
    ):
        data = {}

        for dict_key in self.dict_keys:
            v = getattr(self, dict_key)
            if dict_key.startswith("target"):
                prefix = "t_"
            elif dict_key.startswith("chaser"):
                prefix = "c_"
            else:
                prefix = ""
            for k2, v2 in v.items():
                data[prefix + k2] = v2

        return data

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict(), index=[0])

    @staticmethod
    def load(file_name):
        try:
            with open(file_name, "r") as file:
                data = yaml.safe_load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load file {file_name}: {e}")

        for section, content in data.items():
            if section == "Header":
                header = content
            elif section == "Metadata":
                relative_metadata = content
            elif section == "Target_metadata":
                target_metadata = content
            elif section == "Target_data_od":
                target_data_od = content
            elif section == "Target_data_state":
                target_data_state = content
            elif section == "Target_data_covariance":
                target_data_covariance = content
            elif section == "Chaser_metadata":
                chaser_metadata = content
            elif section == "Chaser_data_od":
                chaser_data_od = content
            elif section == "Chaser_data_state":
                chaser_data_state = content
            elif section == "Chaser_data_covariance":
                chaser_data_covariance = content
            else:
                extras = content

        return ConjuctionDataMessage(
            header=header,
            relative_metadata=relative_metadata,
            values_extra=extras,
            target_metadata=target_metadata,
            target_data_od=target_data_od,
            target_data_state=target_data_state,
            target_data_covariance=target_data_covariance,
            chaser_metadata=chaser_metadata,
            chaser_data_od=chaser_data_od,
            chaser_data_state=chaser_data_state,
            chaser_data_covariance=chaser_data_covariance,
        )

    def save(self, file_name):
        content = self._key_value_notation()
        data = {}
        current_section = None
        for line in content.splitlines():
            if line.endswith(":"):  # Section header
                current_section = line[:-1]
                data[current_section] = {}
            elif current_section and ": " in line:  # Key-value pair
                key, value = line.strip().split(": ", 1)
                # Convert "None" to null and numeric strings to numbers
                if value == "None":
                    value = None
                elif value.replace(".", "", 1).isdigit():
                    value = float(value) if "." in value else int(value)
                data[current_section][key] = value

        # Use PyYAML to save the dictionary as YAML
        with open(file_name, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)

    def set_header(self, key, value):
        if key == "CREATION_DATE":
            time_format = self.__get_ccsds_time_format(value)
            idx = time_format.find("DDD")
            if idx != -1:
                value = self.__doy_2_date(value, value[idx : idx + 3], value[:4], idx)
            try:
                datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError as e:
                raise RuntimeError(f"{key} ({value}) is not in the expected format.\n{str(e)}")
        self.header[key] = value

    def set_relative_metadata(self, key, value):
        if key in self.relative_metadata.keys():
            self.relative_metadata[key] = value
        else:
            raise ValueError("Invalid key ({}) for relative metadata".format(key))

    def set_object(self, object, key, value):
        self._object_validation(object)

        if object == self.target_metadata.get("OBJECT"):
            if key in self.target_metadata.keys():
                self.target_metadata[key] = value
            elif key in self.target_data_od.keys():
                self.target_data_od[key] = value
            elif key in self.target_data_state.keys():
                self.target_data_state[key] = value
            elif key in self.target_data_covariance.keys():
                self.target_data_covariance[key] = value
            else:
                raise KeyError(f"Key {key} not found in target data.")
        else:
            if key in self.chaser_metadata.keys():
                self.chaser_metadata[key] = value
            elif key in self.chaser_data_od.keys():
                self.chaser_data_od[key] = value
            elif key in self.chaser_data_state.keys():
                self.chaser_data_state[key] = value
            elif key in self.chaser_data_covariance.keys():
                self.chaser_data_covariance[key] = value
            else:
                raise KeyError(f"Key {key} not found in chaser data.")

    def set_state(self, object, state):
        for idx, key in enumerate(self.keys_data_state_obligatory):
            self.set_object(object, key, state[idx // 3, idx % 3])
        self._update_miss_distance()
        self._update_state_relative()

    def set_covariance(self, object, covariance_matrix):
        for value in self.keys_data_covariance_obligatory:
            i, j = self.covariance_indices_dict.get(value)
            if i is not None and j is not None:
                self.set_object(object, value, covariance_matrix[i, j])
            else:
                raise ValueError(f"Invalid covariance key: {value}")

    def get_object(self, object, key):
        self._object_validation(object)
        if object == self.target_metadata.get("OBJECT"):
            prefix = "t_"
        else:
            prefix = "c_"

        data = self.to_dict()
        return data.get(prefix + key, None)

    def get_relative_metadata(self, key):
        if key in self.relative_metadata.keys():
            return self.relative_metadata[key]
        else:
            raise KeyError(f"Key {key} not found in relative metadata.")

    def get_state(self, object):
        state = np.zeros([2, 3])
        for idx, key in enumerate(self.keys_data_state_obligatory):
            state[idx // 3, idx % 3] = self.get_object(object, key)
        return state

    def get_state_relative(self):
        relative_state = np.zeros([2, 3])
        relative_state[0, 0] = self.get_relative_metadata("RELATIVE_POSITION_R")
        relative_state[0, 1] = self.get_relative_metadata("RELATIVE_POSITION_T")
        relative_state[0, 2] = self.get_relative_metadata("RELATIVE_POSITION_N")
        relative_state[1, 0] = self.get_relative_metadata("RELATIVE_VELOCITY_R")
        relative_state[1, 1] = self.get_relative_metadata("RELATIVE_VELOCITY_T")
        relative_state[1, 2] = self.get_relative_metadata("RELATIVE_VELOCITY_N")
        return relative_state

    def get_covariance(self, object):
        covariance = np.zeros([6, 6])
        for value in self.keys_data_covariance_obligatory:
            i, j = self.covariance_indices_dict.get(value)
            if i is not None and j is not None:
                covariance[i, j] = self.get_object(object, value)
            else:
                raise ValueError(f"Invalid covariance key: {value}")
        covariance = covariance + covariance.T - np.diag(np.diag(covariance))
        return covariance

    def validate(self):
        header_missing_keys = self._validate_or_filter_obligatory_items(
            "",
            self.header,
            self.keys_header_obligatory,
            return_missing_keys=True,
        )
        loguru.logger.info(f"Header missing keys: {header_missing_keys}")
        metadata_missing_keys = self._validate_or_filter_obligatory_items(
            "",
            self.relative_metadata,
            self.keys_relative_metadata_obligatory,
            return_missing_keys=True,
        )
        loguru.logger.info(f"Metadata missing keys: {metadata_missing_keys}")
        target_metadata_missing_keys = self._validate_or_filter_obligatory_items(
            "",
            self.target_metadata,
            self.keys_metadata_obligatory,
            return_missing_keys=True,
        )
        loguru.logger.info(f"Target metadata missing keys: {target_metadata_missing_keys}")
        target_data_od_missing_keys = self._validate_or_filter_obligatory_items(
            "",
            self.target_data_od,
            self.keys_data_od_obligatory,
            return_missing_keys=True,
        )
        loguru.logger.info(f"Target data od missing keys: {target_data_od_missing_keys}")
        target_data_state_missing_keys = self._validate_or_filter_obligatory_items(
            "",
            self.target_data_state,
            self.keys_data_state_obligatory,
            return_missing_keys=True,
        )
        loguru.logger.info(f"Target data state missing keys: {target_data_state_missing_keys}")
        target_data_covariance_missing_keys = self._validate_or_filter_obligatory_items(
            "",
            self.target_data_covariance,
            self.keys_data_covariance_obligatory,
            return_missing_keys=True,
        )
        loguru.logger.info(
            f"Target data covariance missing keys: {target_data_covariance_missing_keys}"
        )
        chaser_metadata_missing_keys = self._validate_or_filter_obligatory_items(
            "",
            self.chaser_metadata,
            self.keys_metadata_obligatory,
            return_missing_keys=True,
        )
        loguru.logger.info(f"Chaser metadata missing keys: {chaser_metadata_missing_keys}")
        chaser_data_od_missing_keys = self._validate_or_filter_obligatory_items(
            "",
            self.chaser_data_od,
            self.keys_data_od_obligatory,
            return_missing_keys=True,
        )
        loguru.logger.info(f"Chaser data od missing keys: {chaser_data_od_missing_keys}")
        chaser_data_state_missing_keys = self._validate_or_filter_obligatory_items(
            "",
            self.chaser_data_state,
            self.keys_data_state_obligatory,
            return_missing_keys=True,
        )
        loguru.logger.info(f"Chaser data state missing keys: {chaser_data_state_missing_keys}")
        chaser_data_covariance_missing_keys = self._validate_or_filter_obligatory_items(
            "",
            self.chaser_data_covariance,
            self.keys_data_covariance_obligatory,
            return_missing_keys=True,
        )
        loguru.logger.info(
            f"Chaser data covariance missing keys: {chaser_data_covariance_missing_keys}"
        )

    def _object_validation(self, object):
        if object not in [
            self.target_metadata.get("OBJECT"),
            self.chaser_metadata.get("OBJECT"),
        ]:
            raise ValueError(f"Invalid object {object}. Make sure it matches with object metadata.")

    def _get_state_objects(self):
        state_object1 = self.get_state(self.target_metadata.get("OBJECT"))
        state_object2 = self.get_state(self.chaser_metadata.get("OBJECT"))
        for idx, state_object in enumerate([state_object1, state_object2], start=1):
            object_name = (
                self.target_metadata.get("OBJECT")
                if idx == 1
                else self.chaser_metadata.get("OBJECT")
            )
            if np.isnan(state_object.sum()):
                loguru.logger.warning(f"{object_name} has NaN values in its state.")
        return state_object1, state_object2

    def _update_miss_distance(self):
        state_object1, state_object2 = self._get_state_objects()
        miss_distance = np.linalg.norm(state_object1[0] - state_object2[0])
        self.set_relative_metadata("MISS_DISTANCE", miss_distance)

    def _update_state_relative(self):
        state_object1, state_object2 = self._get_state_objects()
        relative_state = self._relative_state_between_objects(state_object1, state_object2)
        self.set_relative_metadata("RELATIVE_POSITION_R", relative_state[0, 0])
        self.set_relative_metadata("RELATIVE_POSITION_T", relative_state[0, 1])
        self.set_relative_metadata("RELATIVE_POSITION_N", relative_state[0, 2])
        self.set_relative_metadata("RELATIVE_VELOCITY_R", relative_state[1, 0])
        self.set_relative_metadata("RELATIVE_VELOCITY_T", relative_state[1, 1])
        self.set_relative_metadata("RELATIVE_VELOCITY_N", relative_state[1, 2])
        self.set_relative_metadata("RELATIVE_SPEED", np.linalg.norm(relative_state[1]))

    def __uvw_matrix(self, r, v):
        u = r / np.linalg.norm(r)
        w = np.cross(r, v)
        w = w / np.linalg.norm(w)
        v = np.cross(w, u)
        return np.vstack((u, v, w))

    def _relative_state_between_objects(self, state_obj_1, state_obj_2):
        rot_matrix = self.__uvw_matrix(state_obj_1[0], state_obj_1[1])
        rel_position_xyz = state_obj_2[0] - state_obj_1[0]
        rel_velocity_xyz = state_obj_2[1] - state_obj_1[1]
        relative_state = np.zeros([2, 3])
        relative_state[0] = np.array(
            [
                np.dot(rot_matrix[0], rel_position_xyz),
                np.dot(rot_matrix[1], rel_position_xyz),
                np.dot(rot_matrix[2], rel_position_xyz),
            ]
        )
        relative_state[1] = np.array(
            [
                np.dot(rot_matrix[0], rel_velocity_xyz),
                np.dot(rot_matrix[1], rel_velocity_xyz),
                np.dot(rot_matrix[2], rel_velocity_xyz),
            ]
        )
        return relative_state

    def _validate_or_filter_obligatory_items(
        self,
        return_string,
        items_dict,
        keys_obligatory,
        show_all=False,
        return_missing_keys=False,
    ):
        """
        Filter the keys to only include the obligatory ones.
        """
        missing_keys = []
        for k, v in items_dict.items():
            if v is None:
                if show_all or k in keys_obligatory:
                    return_string += f" {k}: None\n"
                    missing_keys.append(k)
            else:
                return_string += f" {k}: {v}\n"

        if return_missing_keys:
            return_value = missing_keys
        else:
            return_value = return_string
        return return_value

    def _key_value_notation(self, show_all=False):
        ret = ""
        ret += "Header:\n"
        ret = self._validate_or_filter_obligatory_items(
            ret,
            self.header,
            self.keys_header_obligatory,
            show_all=show_all,
        )
        ret += "Metadata:\n"
        ret = self._validate_or_filter_obligatory_items(
            ret,
            self.relative_metadata,
            self.keys_relative_metadata_obligatory,
            show_all=show_all,
        )
        ret += "Target_metadata:\n"
        ret = self._validate_or_filter_obligatory_items(
            ret,
            self.target_metadata,
            self.keys_metadata_obligatory,
            show_all=show_all,
        )
        ret += "Target_data_od:\n"
        ret = self._validate_or_filter_obligatory_items(
            ret,
            self.target_data_od,
            self.keys_data_od_obligatory,
            show_all=show_all,
        )
        ret += "Target_data_state:\n"
        ret = self._validate_or_filter_obligatory_items(
            ret,
            self.target_data_state,
            self.keys_data_state_obligatory,
            show_all=show_all,
        )
        ret += "Target_data_covariance:\n"
        ret = self._validate_or_filter_obligatory_items(
            ret,
            self.target_data_covariance,
            self.keys_data_covariance_obligatory,
            show_all=show_all,
        )
        ret += "Chaser_metadata:\n"
        ret = self._validate_or_filter_obligatory_items(
            ret,
            self.chaser_metadata,
            self.keys_metadata_obligatory,
            show_all=show_all,
        )
        ret += "Chaser_data_od:\n"
        ret = self._validate_or_filter_obligatory_items(
            ret,
            self.chaser_data_od,
            self.keys_data_od_obligatory,
            show_all=show_all,
        )
        ret += "Chaser_data_state:\n"
        ret = self._validate_or_filter_obligatory_items(
            ret,
            self.chaser_data_state,
            self.keys_data_state_obligatory,
            show_all=show_all,
        )
        ret += "Chaser_data_covariance:\n"
        ret = self._validate_or_filter_obligatory_items(
            ret,
            self.chaser_data_covariance,
            self.keys_data_covariance_obligatory,
            show_all=show_all,
        )

        ret += "Values_extra:\n"
        for k, v in self.values_extra.items():
            if v is None:
                ret += f" {k}: None\n"
            else:
                ret += f" {k}: {v}\n"

        return ret

    def __doy_2_date(value, doy, year, idx):
        """
        Written by Andrew Ng, 18/03/2022,
        Based on source code @ https://github.com/nasa/CARA_Analysis_Tools/blob/master/two-dimension_Pc/Main/TransformationCode/TimeTransformations/DOY2Date.m
        Use the datetime python package.
        doy_2_date  - Converts Day of Year (DOY) date format to date format.

        Args:
            - value(``str``): Original date time string with day of year format "YYYY-DDDTHH:MM:SS.ff"
            - doy  (``str``): The day of year in the DOY format.
            - year (``str``): The year.
            - idx  (``int``): Index of the start of the original "value" string at which characters 'DDD' are found.
        Returns:
            -value (``str``): Transformed date in traditional date format. i.e.: "YYYY-mm-ddTHH:MM:SS.ff"

        """
        # Calculate datetime format
        date_num = datetime.datetime(int(year), 1, 1) + timedelta(int(doy) - 1)

        # Split datetime object into a date list
        date_vec = [
            date_num.year,
            date_num.month,
            date_num.day,
            date_num.hour,
            date_num.minute,
        ]
        # Extract final date string. Use zfill() to pad year, month and day fields with zeroes if not filling up sufficient spaces.
        value = (
            str(date_vec[0]).zfill(4)
            + "-"
            + str(date_vec[1]).zfill(2)
            + "-"
            + str(date_vec[2]).zfill(2)
            + "T"
            + value[idx + 4 : -1]
        )
        return value

    def __get_ccsds_time_format(self, time_string):
        """
        Determines the format of a CCSDS time string.

        The CCSDS time format is required to be of the general form:
        yyyy-[mm-dd|ddd]THH:MM:SS[.F*][Z]

        Args:
            time_string (str): Original time string stored in CDM.

        Returns:
            str: The format of the time string.

        Raises:
            RuntimeError: If the time string is invalid.
        """
        if time_string.count("T") != 1:
            raise RuntimeError(
                f"Invalid CCSDS time string: {time_string}. "
                "The string must contain exactly one 'T' separator."
            )

        date_time_split = time_string.split("T")
        date_part, time_part = date_time_split[0], date_time_split[1]

        if len(date_part) == 10:
            time_format = "yyyy-mm-ddTHH:MM:SS"
        elif len(date_part) == 8:
            time_format = "yyyy-DDDTHH:MM:SS"
        else:
            raise RuntimeError(
                f"Invalid CCSDS time string: {time_string}. "
                "Date format must be either yyyy-mm-dd or yyyy-DDD."
            )

        z_opt = time_part.endswith("Z")
        if z_opt:
            time_part = time_part[:-1]

        if "." in time_part:
            if time_part.count(".") > 1:
                raise RuntimeError(
                    f"Invalid CCSDS time string: {time_string}. "
                    "The string must contain at most one '.' for fractional seconds."
                )
            frac_length = len(time_part.split(".")[1])
            time_format += f".{'F' * frac_length}"

        if z_opt:
            time_format += "Z"

        return time_format

    def __eq__(self, other):
        if isinstance(other, CDM):
            return hash(self) == hash(other)
        return False

    def __hash__(self):
        return hash(self._key_value_notation(show_all=True))

    def __repr__(self):
        return self._key_value_notation()

    def __getitem__(self, key):
        return self.to_dict().get(key, None)

    def __setitem__(self, key, value: str | dict):
        if isinstance(value, dict):
            object = value.get("object", None)
            key = value.get("key", None)
            value = value.get("value", None)

            self.set_object(object, key, value)

        else:
            if key in self.header.keys():
                self.header[key] = value
            elif key in self.relative_metadata.keys():
                self.relative_metadata[key] = value
            else:
                raise KeyError(f"Key {key} not found in header or relative_metadata.")


CDM = ConjuctionDataMessage
