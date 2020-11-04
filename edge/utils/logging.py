import logging

CONFIG = '[CONFIG]'


class ContentFilter(logging.Filter):

    def __init__(self, content, name="", log_if_match=True):
        super(ContentFilter, self).__init__(name)
        self.content = content
        self.log_if_match = log_if_match  # allow for exclusion behavior as well

    def filter(self, record):
        if record.msg:  # ignore empty logs
            if self.content in record.msg:  # note: you can filter on other `record` fields...
                return self.log_if_match
        return not self.log_if_match


class ConfigFilter(ContentFilter):
    def __init__(self, log_if_match=False):
        super(ConfigFilter, self).__init__(
            content=CONFIG, log_if_match=log_if_match
        )


def config_msg(msg):
    return f'{CONFIG} {msg}'


def setup_default_logging_configuration(log_path):
    training_handler = logging.FileHandler(log_path)
    training_handler.addFilter(ConfigFilter(log_if_match=False))
    training_handler.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(training_handler)
    root_logger.addHandler(stdout_handler)


def reset_default_logging_configuration():
    root_logger = logging.getLogger()
    list(map(root_logger.removeHandler, root_logger.handlers[:]))
    list(map(root_logger.removeFilter, root_logger.filters[:]))
