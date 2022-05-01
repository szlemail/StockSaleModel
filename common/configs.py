import configparser
import logging
import os


class Config(object):
    def __init__(self):
        conf = configparser.ConfigParser()
        env = os.environ.get("ALGO_ENV")
        print(f"env:{env}")
        if env is None or env == "dev":
            conf.read("conf/dev.ini")
        elif env == "pro":
            conf.read("conf/pro.ini")
        elif env == "win":
            conf.read("conf/win.int")
        else:
            raise EnvironmentError(f"{env} not support now!")
        self.conf = conf

    def get(self, section, option, default=None):
        if default is None:
            return self.conf.get(section=section, option=option, raw=True)
        else:
            try:
                return self.conf.get(section=section, option=option, raw=True)
            except Exception as e:
                logging.error(f" config not found: {type(e)}, {e}")
                return default


config = Config()
