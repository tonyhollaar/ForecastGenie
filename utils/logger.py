import logging


def setup_logger():
    # Create a logger
    logger = logging.getLogger('my_logger')

    # Check if the logger has any handlers
    # If not, create a new handler and add it to the logger
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set the logging level

        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)  # Set the logging level for this handler

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add the formatter to the handler
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)

    return logger