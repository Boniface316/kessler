from pydantic import BaseModel


class Event(BaseModel):
    cdms: list

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
