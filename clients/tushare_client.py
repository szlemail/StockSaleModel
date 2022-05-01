import tushare as ts
import pandas as pd
import numpy as np
import time

from common.old_configs import Configs


class TushareClient(object):
    """
    tushare 客户端，用于获取数据
    """
    token = ""
    pro = None
    FREQUENCY_LIMIT_RETRY_INTERVAL = 10
    FREQUENCY_LIMIT_MAX_RETRY = 10
    MAX_SYMBOLS_GET_BRIEF = 500

    def __init__(self):
        try:
            self.token = TushareClient.get_tushare_token(Configs.TUSHARE_TOKEN_PATH_WINDOWS)
        except:
            self.token = TushareClient.get_tushare_token(Configs.TUSHARE_TOKEN_PATH_LINUX)
        ts.set_token(self.token)
        self.pro = ts.pro_api()

    @classmethod
    def pro_api(cls):
        if cls.pro is None:
            try:
                token = TushareClient.get_tushare_token(Configs.TUSHARE_TOKEN_PATH_WINDOWS)
            except:
                token = TushareClient.get_tushare_token(Configs.TUSHARE_TOKEN_PATH_LINUX)
            ts.set_token(token)
            if cls.pro is None:
                cls.pro = ts.pro_api()
                print("pro api init")
        return cls.pro

    def stock_basic(self, is_hs="", list_status='L', fields='symbol,name'):
        """获取当前所有股票代码列表"""
        return self.pro.stock_basic(is_hs=is_hs, list_status=list_status, fields=fields)

    def get_trade_date(self, start_date, end_date):
        """
        获取上证指数开盘交易日
        """
        caldf = self.pro.query('trade_cal', start_date=start_date, end_date=end_date)
        df = caldf[caldf['is_open'] == 1][['cal_date']]
        df.columns = ['trade_date']
        df['trade_date'] = df['trade_date'].astype(np.int64)
        return df

    @classmethod
    def add_stock_market_fix(cls, symbol):
        """
        :param symbol: eg:600001
        :return: symbol with market after-fix, eg:600001.SH
        """
        if symbol is not None and "." not in symbol:
            market_fix = '.SH' if symbol[0] == '6' else '.SZ'
            return symbol + market_fix
        return symbol

    @classmethod
    def add_index_market_fix(cls, symbol):
        """
        :param symbol: eg:000001
        :return: symbol with market after-fix, eg:000001.SH
        """
        if symbol is not None and "." not in symbol:
            market_fix = '.SH' if symbol[0] == '0' else '.SZ'
            return symbol + market_fix
        return symbol

    @classmethod
    def time2date(cls, timestamp=time.time()):
        if timestamp > 9999999999:
            local_time = time.localtime(timestamp / 1000)
        else:
            local_time = time.localtime(timestamp)
        return time.strftime("%Y%m%d", local_time)

    @classmethod
    def long2short_date(cls, long_date):
        return "".join(long_date.split("-"))

    @classmethod
    def date2time(cls, date):
        return int(time.mktime(time.strptime(date, "%Y%m%d")) * 1000)

    @classmethod
    def get_tushare_token(cls, path):
        with open(path) as f:
            t = f.readline()
        return t

    @classmethod
    def stock_min_bar(cls, ts_code="", end_date="", start_date=""):
        start_time = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]} 09:00:00"
        end_time = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]} 15:00:00"
        """接口默认网络重试次数为3， 增加超频延迟重试，防止api超限"""
        for i in range(cls.FREQUENCY_LIMIT_MAX_RETRY):
            try:
                dfs = []
                while True:
                    df = ts.pro_bar(ts_code=ts_code, freq='30min', start_date=start_time, end_date=end_time)
                    dfs.append(df.dropna())
                    if len(df) > 1:
                        end_time = df.trade_time.values[-1]
                    else:
                        break
                ndf = pd.DataFrame()
                for df in dfs:
                    ndf = ndf.append(df)
                ndf = ndf.drop_duplicates('trade_time').sort_values('trade_time', ascending=False)
                return ts_code, ndf
            except OSError as e:
                print(f"tushare_client stock_min_bar get OSError {ts_code} retry {i}", e)
                time.sleep(cls.FREQUENCY_LIMIT_RETRY_INTERVAL)
            except Exception as e:
                if "权限的具体详情访问" in str(e):
                    time.sleep(cls.FREQUENCY_LIMIT_RETRY_INTERVAL)
                    print(f"tushare_client stock_min_bar get limit error {ts_code} retry {i}", e)
                else:
                    time.sleep(cls.FREQUENCY_LIMIT_RETRY_INTERVAL)
                    print(f"tushare_client stock_min_bar get other error {ts_code} retry {i}", e)
        print(f"tushare_client stock_min_bar retry failed {ts_code}!!!")
        return ts_code, None

    @classmethod
    def stock_bar(cls, ts_code="", end_date="", start_date="", adj='qfq'):
        """接口默认网络重试次数为3， 增加超频延迟重试，防止api超限"""
        for i in range(cls.FREQUENCY_LIMIT_MAX_RETRY):
            try:
                df_pre = ts.pro_bar(ts_code=ts_code, adj=adj, start_date=start_date, end_date=end_date)
                df_normal = ts.pro_bar(ts_code=ts_code, adj=None, start_date=start_date, end_date=end_date)
                df_limit = ts.pro_api().stk_limit(ts_code=ts_code, start_date=start_date, end_date=end_date)
                df = df_normal.merge(df_limit, on='trade_date')
                df['gs'] = np.array(df['close'] == df['up_limit']).astype(np.int)
                df['ds'] = np.array(df['close'] == df['down_limit']).astype(np.int)
                df['normal_close'] = df['close']
                result = df_pre.merge(df[['trade_date', 'normal_close', 'gs', 'ds']], on='trade_date')
                return ts_code, result
            except OSError as e:
                print(f"tushare_client pro_bar get OSError {ts_code} retry {i}", e)
                time.sleep(cls.FREQUENCY_LIMIT_RETRY_INTERVAL)
            except Exception as e:
                if "权限的具体详情访问" in str(e):
                    time.sleep(cls.FREQUENCY_LIMIT_RETRY_INTERVAL)
                    print(f"tushare_client stock_bar get limit error {ts_code} retry {i}", e)
                else:
                    time.sleep(cls.FREQUENCY_LIMIT_RETRY_INTERVAL)
                    print(f"tushare_client pro_bar get other error {ts_code} retry {i}", e)
        print(f"tushare_client pro_bar retry failed {ts_code}!!!")
        return ts_code, None

    @classmethod
    def index_bar(cls, ts_code="", end_date="", start_date="", limit=70):
        """接口默认网络重试次数为3， 增加超频延迟重试，防止api超限"""
        for i in range(cls.FREQUENCY_LIMIT_MAX_RETRY):
            try:
                df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date, asset='I')
                df['gs'] = 0
                df['ds'] = 0
                return ts_code, df
            except Exception as e:
                print(f"tushare_client index_bar get error {ts_code} retry {i}", e)
                time.sleep(cls.FREQUENCY_LIMIT_RETRY_INTERVAL)
        return ts_code, None

    @classmethod
    def index_min_bar(cls, ts_code="", end_date="", start_date="", limit=70):
        """接口默认网络重试次数为3， 增加超频延迟重试，防止api超限"""
        start_time = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]} 09:00:00"
        end_time = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]} 15:00:00"
        for i in range(cls.FREQUENCY_LIMIT_MAX_RETRY):
            try:
                dfs = []
                while True:
                    df = ts.pro_bar(ts_code=ts_code, start_date=start_time, freq='30min', end_date=end_time, asset='I')
                    dfs.append(df.dropna())
                    if len(df) > 1:
                        end_time = df.trade_time.values[-1]
                    else:
                        break
                ndf = pd.DataFrame()
                for df in dfs:
                    ndf = ndf.append(df)
                ndf = ndf.drop_duplicates('trade_time').sort_values('trade_time', ascending=False)
                ndf['trade_date'] = ndf['trade_time'].apply(lambda x: int("".join(x.split(" ")[0].split("-"))))
                ndf['pre_close'] = ndf['close'].shift(-1)
                return ts_code, ndf
            except Exception as e:
                print(f"tushare_client index_min_bar get error {ts_code} retry {i}", e)
                time.sleep(cls.FREQUENCY_LIMIT_RETRY_INTERVAL)
        return ts_code, None

    @classmethod
    def get_stock_realtime_brief(cls, symbol_list):
        for i in range(cls.FREQUENCY_LIMIT_MAX_RETRY):
            try:
                symbols = [s.split(".")[0] for s in symbol_list]
                df = ts.get_realtime_quotes(symbols)
                df = df[['open', 'pre_close', 'price', 'high', 'low', 'volume', 'date', 'time', 'code']]
                for column in ['open', 'pre_close', 'price', 'high', 'low', 'volume']:
                    df[column] = df[column].apply(float)
                df = df[df.volume > 0]
                abnormal_symbols = list(df[df.volume <= 0]['code'])
                if len(abnormal_symbols) > 0:
                    print("get_stock_realtime_brief abnormal_symbols: {abnormal_symbols}")
                return df
            except Exception as e:
                print(f"get_stock_realtime_brief retry {i}, {symbol_list}, e:{e}")
                time.sleep(1)
        print(f"get_stock_realtime_brief error, {symbol_list}")
        return None

    @classmethod
    def get_index_realtime_brief(cls, symbols):
        def transform_index_symbol(symbol):
            """
            tushare 的实时数据股票代码和历史数据不一致，比如上证指数，tushare需要用sh来获取，所以这里做个转换
            :param symbol: 指数代码
            :return: tushare实时接口支持的指数符号
            """
            if symbol == '000001.SH':
                symbol = 'sh'
            elif symbol == '399106.SZ':
                symbol = '399106'
            else:
                symbol = symbol.split(".")[0]
            return symbol

        new_symbols = [transform_index_symbol(s) for s in symbols]
        for i in range(cls.FREQUENCY_LIMIT_MAX_RETRY):
            try:
                df = ts.get_realtime_quotes(new_symbols)
                df = df[['open', 'pre_close', 'price', 'high', 'low', 'volume', 'date', 'time', 'code']]
                for column in ['open', 'pre_close', 'price', 'high', 'low', 'volume']:
                    df[column] = df[column].apply(float)
                return df
            except Exception as e:
                print(f"get_index_realtime_brief retry {i}, {symbols}, e:{e}")
                time.sleep(1)
        print(f"get_index_realtime_brief error, {symbols}")
        return None

    def limit_list(self, trade_date, limit_type='U', fields='ts_code,close,first_time,last_time'):
        return self.pro.limit_list(trade_date=trade_date, limit_type=limit_type, fields=fields)

    @classmethod
    def get_st_code(cls, date):
        df = ts.get_stock_basics(date)
        code_list = df[df['name'].apply(lambda x: "ST" in x)].index.values
        return [cls.add_stock_market_fix(c).replace("SH", "SS") for c in code_list]
