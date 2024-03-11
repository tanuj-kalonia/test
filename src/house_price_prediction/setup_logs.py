import logging

formatter = logging.Formatter(
    "Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s",
)

def setup_logger(name,log_file, no_console_log, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    if not no_console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger