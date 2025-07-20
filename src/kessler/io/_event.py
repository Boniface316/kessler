import copy
import os
import re
from glob import glob

import loguru
import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from .__keys import data_covariance, data_od, data_state, header, metadata, relative_metadata
from ._CDM import CDM
from ._utils import _add_days_to_date_str, _from_date_str_to_days


class Event(BaseModel, arbitrary_types_allowed=True):
    cdms: list
    data_frame: pd.DataFrame | None = None

    @field_validator("cdms")
    def validate_cdms(cls, cdms):
        if all(isinstance(c, str) for c in cdms):
            file_names = sorted(cdms)

            cdms = [CDM.load(file_name=file_name) for file_name in file_names]
        elif all(isinstance(c, CDM) for c in cdms):
            pass

        else:
            raise ValueError(
                "cdms must be a list of CDM objects or list of string with file names."
            )

        return cdms

    def add(self, cdm, return_result=False):
        if isinstance(cdm, CDM):
            self.cdms.append(cdm)
        elif isinstance(cdm, list):
            for c in cdm:
                self.add(c)
        # TODO: include update_cdm_extras
        else:
            raise ValueError("cdm must be a CDM object or a list of CDM objects.")

    def copy(self):
        return Event(cdms=copy.deepcopy(self.cdms))

    def to_dataframe(self):
        if self.data_frame is None:
            self.data_frame = [cdm.to_dataframe() for cdm in self.cdms]
            self.data_frame = pd.concat(self.data_frame, ignore_index=True)
        else:
            loguru.logger.warning("DataFrame already exists, returning the existing one.")

        return self.data_frame

    def __repr__(self):
        return "Event(CDMs: {})".format(len(self.cdms))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Event(cdms=self.cdms[index])
        else:
            return self.cdms[index]

    def __len__(self):
        return len(self.cdms)


class EventDataset(BaseModel):
    events: list | str

    @field_validator("events")
    def validate_events(cls, events):
        if isinstance(events, str):
            file_names = os.listdir(events)

            loguru.logger.info(f"Loading CDMs from frolder {events}")
            file_names = sorted(glob(os.path.join(events, "*" + ".yaml")))
            regex = r"(.*)_([0-9]+.yaml)"
            matches = re.finditer(regex, "\n".join(file_names))

            event_prefixes = []
            for m in matches:
                m = m.groups()[0]
                event_prefixes.append(m)
            event_prefixes = sorted(set(event_prefixes))

            event_file_names = []
            for event_prefix in event_prefixes:
                event_file_names.append(
                    list(filter(lambda f: f.startswith(event_prefix), file_names))
                )

            events = [Event(cdms=f) for f in event_file_names]

        elif isinstance(events, list):
            pass
        else:
            raise ValueError("events must be a list of Event objects or a string of the folder.")
        return events

    @staticmethod
    def from_pandas(
        df,
        groups_events_by="EVENT_ID",
        header=header,
        relative_metadata=relative_metadata,
        object_metadata=metadata,
        data_state=data_state,
        data_od=data_od,
        data_covariance=data_covariance,
        from_date_str_to_days=_from_date_str_to_days,
    ):
        loguru.logger.info(f"Dataframe with {len(df)} rows and {len(df.columns)} columns")
        column_names_after_dropping = list(df.columns)
        df_events = df.groupby(groups_events_by)
        events = []

        for _, event_data in df_events:
            cdms = []
            for _, single_cdm in event_data.iterrows():
                header_dict = {
                    header: single_cdm[header]
                    for header in header
                    if header in column_names_after_dropping
                }

                relative_metadata_dict = {
                    relative_metadata: single_cdm[relative_metadata]
                    for relative_metadata in relative_metadata
                    if relative_metadata in column_names_after_dropping
                }
                target_metadata_dict = {
                    target_metadata: single_cdm["t_" + target_metadata]
                    for target_metadata in object_metadata
                    if "t_" + target_metadata in column_names_after_dropping
                }
                target_data_od_dict = {
                    target_data_od: single_cdm["t_" + target_data_od]
                    for target_data_od in data_od
                    if "t_" + target_data_od in column_names_after_dropping
                }
                target_data_state_dict = {
                    target_data_state: single_cdm["t_" + target_data_state]
                    for target_data_state in data_state
                    if "t_" + target_data_state in column_names_after_dropping
                }
                target_data_covariance_dict = {
                    target_data_covariance: single_cdm["t_" + target_data_covariance]
                    for target_data_covariance in data_covariance
                    if "t_" + target_data_covariance in column_names_after_dropping
                }
                chaser_metadata_dict = {
                    chaser_metadata: single_cdm["c_" + chaser_metadata]
                    for chaser_metadata in object_metadata
                    if "c_" + chaser_metadata in column_names_after_dropping
                }
                chaser_data_od_dict = {
                    chaser_data_od: single_cdm["c_" + chaser_data_od]
                    for chaser_data_od in data_od
                    if "c_" + chaser_data_od in column_names_after_dropping
                }
                chaser_data_state_dict = {
                    chaser_data_state: single_cdm["c_" + chaser_data_state]
                    for chaser_data_state in data_state
                    if "c_" + chaser_data_state in column_names_after_dropping
                }
                chaser_data_covariance_dict = {
                    chaser_data_covariance: single_cdm["c_" + chaser_data_covariance]
                    for chaser_data_covariance in data_covariance
                    if "c_" + chaser_data_covariance in column_names_after_dropping
                }

                values_extra = {
                    "CREATION_DATE_IN_DAYS": single_cdm["CREATION_DATE"],
                    "TCA_IN_DAYS": single_cdm["TCA_IN_DAYS"],
                    "DAYS_TO_TCA": single_cdm["DAYS_TO_TCA"],
                }

                single_cdm = CDM(
                    header=header_dict,
                    relative_metadata=relative_metadata_dict,
                    values_extra=values_extra,
                    target_metadata=target_metadata_dict,
                    target_data_od=target_data_od_dict,
                    target_data_state=target_data_state_dict,
                    target_data_covariance=target_data_covariance_dict,
                    chaser_metadata=chaser_metadata_dict,
                    chaser_data_od=chaser_data_od_dict,
                    chaser_data_state=chaser_data_state_dict,
                    chaser_data_covariance=chaser_data_covariance_dict,
                )
                cdms.append(single_cdm)
            events.append(Event(cdms=cdms))
        return EventDataset(events=events)

    def to_dataframe(self):
        event_dataframes = []
        for event in self.events:
            event_dataframes.append(event.to_dataframe())
        return pd.concat(event_dataframes, ignore_index=True)

    def dates(self, _add_days_to_date_str=_add_days_to_date_str):
        print("CDM| CREATION_DATE (mean)       | Days (mean, std)  | Days to TCA (mean, std)")
        for i in range(self.event_lengths_max):
            creation_date_days = []
            days_to_tca = []
            for event in self.events:
                if i < len(event):
                    creation_date_days.append(event[i].values_extra.get("CREATION_DATE_IN_DAYS"))
                    days_to_tca.append(event[i].values_extra.get("DAYS_TO_TCA"))

            creation_date_days = np.array(creation_date_days)
            creation_date_days_mean, creation_date_days_stddev = (
                creation_date_days.mean(),
                creation_date_days.std(),
            )

            days_to_tca = np.array(days_to_tca)
            days_to_tca_mean, days_to_tca_stddev = days_to_tca.mean(), days_to_tca.std()

            first_creation_date = self.events[0].cdms[0].header["CREATION_DATE"]
            creation_date_days_mean_str = _add_days_to_date_str(
                first_creation_date, creation_date_days_mean
            )
            print(
                "{:02d} | {} | {:.6f} {:.6f} | {:.6f} {:.6f}".format(
                    i + 1,
                    creation_date_days_mean_str,
                    creation_date_days_mean,
                    creation_date_days_stddev,
                    days_to_tca_mean,
                    days_to_tca_stddev,
                )
            )

    @property
    def event_lengths(self):
        return list(map(len, self.events))

    @property
    def event_lengths_min(self):
        return min(self.event_lengths)

    @property
    def event_lengths_max(self):
        return max(self.event_lengths)

    @property
    def event_lengths_mean(self):
        return np.array(self.event_lengths).mean()

    @property
    def event_lengths_stddev(self):
        return np.array(self.event_lengths).std()

    def common_features(self, only_numeric=False):
        df = self.to_dataframe()
        df = df.dropna(axis=1)
        if only_numeric:
            df = df.select_dtypes(include=["int", "float64", "float32"])
        features = list(df.columns)
        if "DAYS_TO_TCA" in features:
            features.remove("DAYS_TO_TCA")
        return features

    def get_CDMs(self):
        cdms = []
        for event in self.events:
            for cdm in event.cdms:
                cdms.append(cdm)
        return cdms

    def filter(self, filter_function):
        events = []
        for event in self.events:
            if filter_function(event):
                events.append(event)
        return EventDataset(events=events)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return EventDataset(events=self.events[index])
        else:
            return self.events[index]

    def __len__(self):
        return len(self.events)

    def __repr__(self):
        if len(self.events) == 0:
            return "EventDataset()"
        else:
            event_lengths = [len(i.cdms) for i in self.events]
            event_lengths_min = min(event_lengths)
            event_lengths_max = max(event_lengths)
            event_lengths_mean = sum(event_lengths) / len(event_lengths)

            return f"EventDataset(Events: {len(self.events)}, number of CDMs per event: {event_lengths_max} (max), {event_lengths_min} (min), {event_lengths_mean:.2f} (mean))"
