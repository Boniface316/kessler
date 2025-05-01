from pydantic import BaseModel, field_validator
from ._CDM import CDM
import copy
import pandas as pd
import os
import loguru
import re
from glob import glob


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

        return self.data_frame

    def __repr__(self):
        return "Event(CDMs: {})".format(len(self.cdms))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Event(cdms=self._cdms[index])
        else:
            return self._cdms[index]

    def __len__(self):
        return len(self._cdms)


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
            raise ValueError(
                "events must be a list of Event objects or a string of the folder."
            )
        return events

    @staticmethod
    def from_pandas(self, df, groups_events_by="event_id"):
        loguru.logger(f"Dataframe with {len(df)} rows and {len(df.columns)} columns")
        df = df.dropna(axis=1)
        column_names_after_dropping = list(df.columns)
        df_events = df.groupby(groups_events_by)
        events = []
        for event_id, event_data in df_events:
            for _, single_cdm in event_data.iterrows():
                breakpoint()

    def to_dataframe(self):
        event_dataframes = []
        for event in self.events:
            event_dataframes.extend(event.to_dataframe())
        return pd.concat(event_dataframes, ignore_index=True)

    def dates(self):
        pass

    @property
    def event_lengths(self):
        pass

    @property
    def event_lenths_min(self):
        pass

    @property
    def event_lengths_max(self):
        pass

    @property
    def event_lengths_mean(self):
        pass

    @property
    def event_lengths_stddev(self):
        pass

    def common_features(self, only_numeric=False):
        pass

    def get_CDMs(self):
        pass

    def filter(self, filter_function):
        pass

    def __getitem__(self, index):
        if isinstance(index, slice):
            return EventDataset(events=self.events[index])
        else:
            return self.events[index]

    def __len__(self):
        pass

    def __repr__(self):
        if len(self.events) == 0:
            return "EventDataset()"
        else:
            event_lengths = [len(i.cdms) for i in self.events]
            event_lengths_min = min(event_lengths)
            event_lengths_max = max(event_lengths)
            event_lengths_mean = sum(event_lengths) / len(event_lengths)
            return "EventDataset(Events:{}, number of CDMs per event: {} (min), {} (max), {:.2f} (mean))".format(
                len(self.events),
                event_lengths_max,
                event_lengths_min,
                event_lengths_mean,
            )
