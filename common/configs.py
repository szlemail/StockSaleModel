class Configs(object):
    # windows path
    TIGER_PRIVATE_KEY_WINDOWS = "G:\\project\\stock\\tiger\\rsa\\rsa_private_key.pem"
    TUSHARE_TOKEN_PATH_WINDOWS = "G:\\project\\stock\\tushare\\token.txt"
    QQ_CODE_PATH_WINDOWS = "G:\\project\\stock\\qqmail\\qqcode.txt"

    # linux path
    TIGER_PRIVATE_KEY_LINUX = "/home/common/keys/tiger_rsa_private_key.pem"
    TUSHARE_TOKEN_PATH_LINUX = "/home/common/keys/tushare.config"
    QQ_CODE_PATH_LINUX = "/home/common/keys/qqcode.config"
    REDIS_PATH_LINUX = "/home/common/keys/redis.config"

    # 管理员邮件l
    ADMIN_EMAIL = "szlemail@tom.com;szlemail@qq.com"
    STOCK_INDEX_LIST = ['399001.SZ', '399006.SZ', '000001.SH']

    # 计算特征的进程个数
    POOL_PROCESS = 15
    GBM_JOB = POOL_PROCESS

    # 调试 控制参数
    DEBUG_MINI_DATA = False
    DEBUG_MINI_DATA_STOCKS = 100
