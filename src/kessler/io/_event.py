from pydantic import BaseModel, field_validator
import os
from ._CDM import CDM


class Event(BaseModel):
    cdms: list | str

    @field_validator("cdms")
    def validate_cdms(cls, cdms):
        if isinstance(cdms, str):
            if os.path.exists(cdms):
                files = os.listdir(cdms)
                file_names = [os.path.join(cdms, file) for file in files if file.endswith(".yaml")]

                cdms = [CDM.load(file_name=file_name) for file_name in file_names]
            else:
                raise ValueError(f"File {cdms} does not exist.")
        else:
            if all(isinstance(c, CDM) for c in cdms):
                pass
            else:
                raise ValueError(
                    "cdms must be a list of CDM objects or a string representing a file path."
                )

        return cdms

    def __repr__(self):
        return "Event(CDMs: {})".format(len(self.cdms))


class EventDataset(BaseModel):
    events: list

    def __repr__(self):
        if len(self.events) == 0:
            return "EventDataset()"
        else:
            event_lengths = [len(i.cdms) for i in self.events]
            event_lengths_min = min(event_lengths)
            event_lengths_max = max(event_lengths)
            event_lengths_mean = sum(event_lengths) / len(event_lengths)
            return "EventDataset(Events:{}, number of CDMs per event: {} (min), {} (max), {:.2f} (mean))".format(
                len(self.events), event_lengths_min, event_lengths_max, event_lengths_mean
            )


CNDOT_NDOT: 0.14935299999999999
