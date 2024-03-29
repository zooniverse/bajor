import os
import logging
from logging.config import dictConfig

# logger configuration in dict format
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",

        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "BAJOR": {"handlers": ["default"], "level": os.environ.get('LOG_LEVEL', 'INFO')},
    },
}


# setup the logger
dictConfig(log_config)

log = logging.getLogger('BAJOR')
