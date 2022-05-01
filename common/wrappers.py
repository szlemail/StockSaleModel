import logging
import time
from functools import wraps
import traceback


def timing(func):
    """
    打印函数执行时间，以毫秒为单位  使用方式示例： @timing
    """

    @wraps(func)
    def wrapper_fun(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        cost = (time.time() - start_time) * 1000
        logging.info(f"{func.__name__} cost {cost} ms")
        return res

    return wrapper_fun


def exception(exception_types=(Exception,)):
    """
    对函数执行异常捕获并处理
    """
    def inner_exception(func):
        @wraps(func)
        def wrapper_fun(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
                return res
            except exception_types as e:
                logging.error(f"{func.__name__} {type(e)}, msg:{e}, trace:{traceback.format_exc(10, False)}")
        return wrapper_fun
    return inner_exception
