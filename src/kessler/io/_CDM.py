from pydantic import BaseModel
from copy import deepcopy
import pandas as pd
import yaml


class ConjuctionDataMessage(BaseModel, strict=False, frozen=False, extra="forbid"):
    header: dict
    relative_metadata: dict

    target_metadata: dict
    target_data_od: dict
    target_data_state: dict
    target_data_covariance: dict

    chaser_metadata: dict
    chaser_data_od: dict
    chaser_data_state: dict
    chaser_data_covariance: dict

    keys_header_obligatory: list = [
        "CCSDS_CDM_VERS",
        "CREATION_DATE",
        "ORIGINATOR",
        "MESSAGE_ID",
    ]
    keys_relative_metadata_obligatory: list = [
        "TCA",
        "MISS_DISTANCE",
    ]
    keys_metadata_obligatory: list = [
        "OBJECT",
        "OBJECT_DESIGNATOR",
        "CATALOG_NAME",
        "OBJECT_NAME",
        "INTERNATIONAL_DESIGNATOR",
        "EPHEMERIS_NAME",
        "COVARIANCE_METHOD",
        "MANEUVERABLE",
        "REF_FRAME",
    ]
    keys_data_od_obligatory: list = [
        "TIME_LASTOB_START",
        "TIME_LASTOB_END",
        "RECOMMENDED_OD_SPAN",
        "ACTUAL_OD_SPAN",
        "OBS_AVAILABLE",
        "OBS_USED",
        "RESIDUALS_ACCEPTED",
        "WEIGHTED_RMS",
        "SEDR",
    ]

    keys_data_state_obligatory: list = ["X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"]
    keys_data_covariance_obligatory: list = [
        "CR_R",
        "CT_R",
        "CT_T",
        "CN_R",
        "CN_T",
        "CN_N",
        "CRDOT_R",
        "CRDOT_T",
        "CRDOT_N",
        "CRDOT_RDOT",
        "CTDOT_R",
        "CTDOT_T",
        "CTDOT_N",
        "CTDOT_RDOT",
        "CTDOT_TDOT",
        "CNDOT_R",
        "CNDOT_T",
        "CNDOT_N",
        "CNDOT_RDOT",
        "CNDOT_TDOT",
        "CNDOT_NDOT",
    ]

    def copy(self):
        return ConjuctionDataMessage(
            header=deepcopy(self.header),
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
        self.relative_metadata = deepcopy(other_cdm.relative_metadata)
        self.target_metadata = deepcopy(other_cdm.target_metadata)
        self.target_data_od = deepcopy(other_cdm.target_data_od)
        self.target_data_state = deepcopy(other_cdm.target_data_state)
        self.target_data_covariance = deepcopy(other_cdm.target_data_covariance)
        self.chaser_metadata = deepcopy(other_cdm.chaser_metadata)
        self.chaser_data_od = deepcopy(other_cdm.chaser_data_od)
        self.chaser_data_state = deepcopy(other_cdm.chaser_data_state)
        self.chaser_data_covariance = deepcopy(other_cdm.chaser_data_covariance)

    def to_dict(self):
        data = {}

        dict_names = [
            "header",
            "relative_metadata",
            "target_metadata",
            "target_data_od",
            "target_data_state",
            "target_data_covariance",
            "chaser_metadata",
            "chaser_data_od",
            "chaser_data_state",
            "chaser_data_covariance",
        ]

        for dict_name in dict_names:
            v = getattr(self, dict_name)
            if dict_name.startswith("target"):
                prefix = "t_"
            elif dict_name.startswith("chaser"):
                prefix = "c_"
            else:
                prefix = ""
            for k2, v2 in v.items():
                data[prefix + k2] = v2

        return data

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict(), index=[0])

    def load(self, file_name):
        # Load the YAML file
        with open(file_name, "r") as file:
            data = yaml.safe_load(file)

        # Convert the loaded data to the appropriate format
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

        return ConjuctionDataMessage(
            header=header,
            relative_metadata=relative_metadata,
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
        content = self.key_value_notation()
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

    def __hash__(self):
        pass

    def __eq__(self, value):
        pass

    def set_header(self, key, value):
        pass

    def set_relative_metadata(self, key, value):
        pass

    def set_object(self, key, value):
        pass

    def get_object(self, object_id, key):
        pass

    def get_relative_metadata(self, key):
        pass

    def set_state(self, object_id, state):
        pass

    def _update_miss_distance(self):
        pass

    def _update_state_relative(self):
        pass

    def get_state_relative(self):
        pass

    def get_state(self, object_id):
        pass

    def get_covariance(self, object_id):
        pass

    def set_covariance(self, object_id, covariance):
        pass

    def validate(self):
        pass

    def _filter_obligatory_items(
        self, return_string, items_dict, keys_obligatory, show_all=False
    ):
        """
        Filter the keys to only include the obligatory ones.
        """
        for k, v in items_dict.items():
            if v is None:
                if show_all or k in keys_obligatory:
                    return_string += f" {k}: None\n"
            else:
                return_string += f" {k}: {v}\n"
        return return_string

    def key_value_notation(self):
        ret = ""
        ret += "Header:\n"
        ret = self._filter_obligatory_items(
            ret,
            self.header,
            self.keys_header_obligatory,
            show_all=False,
        )
        ret += "Metadata:\n"
        ret = self._filter_obligatory_items(
            ret,
            self.relative_metadata,
            self.keys_relative_metadata_obligatory,
            show_all=False,
        )
        ret += "Target_metadata:\n"
        ret = self._filter_obligatory_items(
            ret,
            self.target_metadata,
            self.keys_metadata_obligatory,
            show_all=False,
        )
        ret += "Target_data_od:\n"
        ret = self._filter_obligatory_items(
            ret,
            self.target_data_od,
            self.keys_data_od_obligatory,
            show_all=False,
        )
        ret += "Target_data_state:\n"
        ret = self._filter_obligatory_items(
            ret,
            self.target_data_state,
            self.keys_data_state_obligatory,
            show_all=False,
        )
        ret += "Target_data_covariance:\n"
        ret = self._filter_obligatory_items(
            ret,
            self.target_data_covariance,
            self.keys_data_covariance_obligatory,
            show_all=False,
        )
        ret += "Chaser_metadata:\n"
        ret = self._filter_obligatory_items(
            ret,
            self.chaser_metadata,
            self.keys_metadata_obligatory,
            show_all=False,
        )
        ret += "Chaser_data_od:\n"
        ret = self._filter_obligatory_items(
            ret,
            self.chaser_data_od,
            self.keys_data_od_obligatory,
            show_all=False,
        )
        ret += "Chaser_data_state:\n"
        ret = self._filter_obligatory_items(
            ret,
            self.chaser_data_state,
            self.keys_data_state_obligatory,
            show_all=False,
        )
        ret += "Chaser_data_covariance:\n"
        ret = self._filter_obligatory_items(
            ret,
            self.chaser_data_covariance,
            self.keys_data_covariance_obligatory,
            show_all=False,
        )

        return ret

    def __repr__(self):
        return self.key_value_notation()

    def __getitem__(self, key):
        return self.to_dict().get(key, None)

    def __setitem__(self, key, value: str | dict):
        if isinstance(value, dict):
            object = value.get("object", None)
            key = value.get("key", None)
            value = value.get("value", None)

            if object == "target":
                if key in self.target_metadata.keys():
                    self.target_metadata[key] = value
                elif key in self.target_data_od.keys():
                    self.target_data_od[key] = value
                elif key in self.target_data_state.keys():
                    self.target_data_state[key] = value
                elif key in self.target_data_covariance.keys():
                    self.target_data_covariance[key] = value
            elif object == "chaser":
                if key in self.chaser_metadata.keys():
                    self.chaser_metadata[key] = value
                elif key in self.chaser_data_od.keys():
                    self.chaser_data_od[key] = value
                elif key in self.chaser_data_state.keys():
                    self.chaser_data_state[key] = value
                elif key in self.chaser_data_covariance.keys():
                    self.chaser_data_covariance[key] = value
            else:
                raise KeyError(f"Key {key} not found in target or chaser.")

        else:
            if key in self.header.keys():
                self.header[key] = value
            elif key in self.relative_metadata.keys():
                self.relative_metadata[key] = value
            else:
                raise KeyError(f"Key {key} not found in header or relative_metadata.")


CDM = ConjuctionDataMessage
