import logging
import os
import threading
from contextlib import contextmanager
from logging import config

import structlog
from structlog import contextvars
from structlog.types import EventDict

APPLICATION_NAME = "cloud-agnostic-training"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json_formatter": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.processors.JSONRenderer(),
        },
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "json_formatter"}
    },
    "loggers": {APPLICATION_NAME: {"handlers": ["console"], "level": LOG_LEVEL}},
}


def _add_context(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """https://github.com/jrobichaud/django-structlog/issues/29#issuecomment-600991068"""  # noqa: E501
    try:
        frame, module_str = structlog._frames._find_first_app_frame_and_name(  # type: ignore  # noqa: E501
            additional_ignores=[__name__]
        )
        event_dict["module"] = module_str
        event_dict["file"] = frame.f_code.co_filename
        event_dict["lineno"] = frame.f_lineno
        event_dict["thread"] = threading.current_thread().name
        event_dict["application"] = APPLICATION_NAME
    except Exception:
        pass
    finally:
        return event_dict


config.dictConfig(_LOGGING)

structlog.configure(
    processors=[
        contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        _add_context,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    context_class=structlog.threadlocal.wrap_dict(dict),
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(APPLICATION_NAME)


@contextmanager
def logging_context(**kwargs):
    """
    Add keywords to logging context to, for example, trace a task.
    https://www.structlog.org/en/stable/contextvars.html
    """
    try:
        contextvars.bind_contextvars(**kwargs)
        yield
    finally:
        contextvars.unbind_contextvars(*kwargs.keys())
