"""
Miscellaneous utility functions
"""

from recipes.pprint import decimal_repr


def hms(t):
    """Convert time in seconds to hms tuple"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return h, m, s              # TODO: use named tuple ??


def fmt_hms(t, precision=None, sep='hms', short=None, unicode=False):
    """
    Convert time in seconds to sexagesimal representation

    Parameters
    ----------
    t : float
        time in seconds
    precision: int or None
        maximum precision to use. Will be ignored if a shorter numerical
        representation exists
    sep: str
        separator(s) to use for time representation
    short: bool or None
        will strip unnecessary parts from the repr if True.
        eg: '0h00m15.4s' becomes '15.4s'
    unicode: bool
        Unicode superscripts

    Returns
    -------
    formatted time str

    Examples
    --------
    >>> fmt_hms(1e4)
    '2h46m40s'
    >>> fmt_hms(1.333121112e2, 5)
    '2m13.31211s'
    >>> fmt_hms(1.333121112e2, 5, ':')
    '0:02:13.31211'
    >>> fmt_hms(1.333121112e2, 5, short=False)
    '0h02m13.31211s'
    """

    if len(sep) == 1:
        sep = (sep, sep, '')

    if short is None:
        # short representation only meaningful if time expressed in hms units
        short = (sep == 'hms')

    if unicode and (sep == 'hms'):
        sep = 'ʰᵐˢ'

    #
    sexa = hms(t)
    precision = (0, 0, precision)

    tstr = ''
    for i, (n, p, s) in enumerate(zip(sexa, precision, sep)):
        part = decimal_repr(n, p)

        # if this is the first non-zero part, skip zfill
        if len(tstr):
            zfill = 2
            len(part) - len(str(int(n)))
            part = part.zfill(zfill)

        # special treatment for last (meaningful) sexagesimal part
        last = (i == 2)
        if short and not (last or float(part) or len(tstr)):
            # if short format requested and final part has 0s only, omit
            continue
        tstr = (part + s)

    return tstr
