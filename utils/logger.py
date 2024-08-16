import logging
import pytz
from datetime import datetime
from logging import Formatter

class CustomFormatter(Formatter):
    def __init__(self, fmt=None, datefmt=None, timezone=None):
        super().__init__(fmt, datefmt)
        self.timezone = timezone

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.timezone)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat()
        return s

def setup_logging(log_file, level, include_host=False, timezone='Asia/Seoul'):
    tz = pytz.timezone(timezone)

    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = CustomFormatter(
            f'%(asctime)s | {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', timezone=tz)
    else:
        formatter = CustomFormatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', timezone=tz)

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
