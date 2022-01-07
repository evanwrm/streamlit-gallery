import datetime as dt


def unix_timestamp(date: dt.datetime):
    """Converts datetime to unix timestamp"""
    epoch = dt.datetime.utcfromtimestamp(0)
    return (date - epoch).total_seconds()
