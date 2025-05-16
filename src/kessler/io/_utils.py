import functools
from datetime import datetime, timedelta


@functools.lru_cache(maxsize=None)
def _from_date_str_to_days(
    cdm_date,
    date0="2020-05-22T21:41:31.975",
    date_format="%Y-%m-%dT%H:%M:%S.%f",
):
    cdm_date = datetime.strptime(cdm_date, date_format)
    if isinstance(date0, str):
        date0 = datetime.strptime(date0, date_format)
    dd = cdm_date - date0
    days = dd.days
    days_fraction = (dd.seconds + dd.microseconds / 1e6) / (60 * 60 * 24)
    return days + days_fraction


def _add_days_to_date_str(date0, days):
    date0 = datetime.strptime(date0, "%Y-%m-%dT%H:%M:%S.%f")
    date = date0 + timedelta(days=days)
    return date.strftime("%Y-%m-%dT%H:%M:%S.%f")
