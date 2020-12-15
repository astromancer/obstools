from recipes import pprint


def fmt_ra(x):
    return pprint.hms(x * 3600 / 15, unicode=True, precision=1)


def fmt_dec(x):
    return pprint.hms(x * 3600, sep='°’”', precision=1)
