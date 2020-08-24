
from datetime import datetime, timedelta, date as Date
from astropy.time import Time
import functools as ftl


def nearest_midnight_date(t=None, switch_hour=9):
    """
    Get the date of local midnight time which is nearest the input time `t`
    (defaults to current time if not given).

    Parameters
    ----------
    t : astropy.time.Time or datetime.datetime or datetime.date or str, optional
        The default behaviour of this function changes depending on the time of
        day the function is called (this is convenient for helping to plan an
        observing schedule):
        if calling during early morning hours (presumably at telescope):
            time returned is current day local midnight 00:00:00
            (ie. a few hours in the past)
        if calling during afternoon hours:
            time returned is midnight of the next calendar day
            (ie. The next local midnight, a few hours in the future)
    switch_hour : int, optional
        The hour from midnight at which the swich from past to future occurs,
        by default 9. ie. If this function is called before 9am local time,
        the previous midnight time is returned, while the next midnight will be
        returned if the call takes place after 9am.

    Returns
    -------
    datetime.datetime
        [description]
    """
    if t is None:
        t = datetime.now()  # current local time
    elif isinstance(t, str):
        t = datetime.fromisoformat(t)
    elif isinstance(t, Time):
        t = t.datetime

    # 
    if isinstance(t, datetime):
        h = t.hour
    elif isinstance(t, Date):
        h = 0            # use evening of given DATE
    else:
        raise TypeError('Input time is of invalid type')

    day_inc = int(h >= switch_hour)  # int((now.hour - 12) > 12)
    midnight = datetime(t.year, t.month, t.day, 0, 0, 0)
    return midnight + timedelta(day_inc)


@ftl.lru_cache()
def get_midnight(date, longitude):
    """
    Get midnight time from a date. The time returned by this function will
    usually be the midnight time following the specified input date. This logic
    allows one to specify the night on which your observations will be starting
    as input.  If date is None, the current date will be used, which means the
    next upcoming local midnight will be returned as midnight time.

    Parameters
    ----------
    date : str or Date or Time or None
        Calander date. If None, current local date is used.
    longitude : [type]
        [description]

    Returns
    -------
    Date
        The input date resolved as a `datetime.Date` object
    Time
        Time of local midnight
    Time
        Sidereal time of local midnight
    """

    date = nearest_midnight_date(date)
    midnight = Time(date)
    mid_sid = midnight.sidereal_time('mean', longitude)
    return date, midnight, mid_sid
