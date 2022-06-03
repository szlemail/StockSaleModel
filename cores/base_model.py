import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
from common.wrappers import timing, exception
import logging
from functools import partial
from abc import ABCMeta, abstractmethod
from common.configs import config


class BaseModel(object):
    """
    基礎模型，處理通用的數據預處理，模型訓練、驗證保存等邏輯，繼承的模型可以專注在建模上
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.seq_len = int(config.get("model", "seq_len", default='90'))
        self.embedding_size = int(config.get("model", "embedding_size", default='16'))
        self.batch_size = int(config.get("model", "batch_size", default='512'))
        self.debug = config.get("debug", "debug_enable", default='false') == 'true'
        self.debug_stock_count = int(config.get("debug", "debug_stock_count", default='100'))
        self.skip_stock_count = 0
        self.origin_data_path = config.get("data", "stock_origin_path", default='data/origin')
        self.origin_min_data_path = config.get("data", "stock_origin_min_path", default='data/origin_min')
        self.stock_index_list = ['399001.SZ', '399006.SZ', '000001.SH']
        self.price_bounds = self.get_discrete_bounds(5000, round_number=2)
        self.price_bounds.reverse()
        self.vol_bounds = self.get_discrete_bounds(5000000000, decay=0.995, round_number=0) + [-1]
        self.vol_bounds.reverse()
        self.time_bounds = [925, 955, 1025, 1055, 1125, 1155, 1325, 1355, 1425, 1455, 1525]
        self.stock_code_list = np.array([s.strip(".csv") for s in os.listdir(self.origin_data_path)])
        self.stock_code_mapping = dict(zip(self.stock_code_list, np.arange(1, len(self.stock_code_list) + 1)))

    @staticmethod
    def years_before(years):
        fmt = "%Y%m%d"
        days = years * 365 + years // 4
        return int((datetime.now() + timedelta(days=-days)).strftime(fmt))

    @staticmethod
    def get_market(ts_code):
        if ts_code[:1] == '3':
            return 4
        elif ts_code[:3] == '002':
            return 3
        elif ts_code[:3] == '0':
            return 2
        elif ts_code[:2] == '68':
            return 1
        else:
            return 0

    @staticmethod
    def get_discrete_bounds(max_value, min_value=0.1, round_number=2, decay=0.999):
        assert min_value > 0
        bounds = []
        value = max_value
        last = round(value, round_number)
        while value > min_value:
            value = value * decay
            bound = round(value, round_number)
            if bound != last:
                bounds.append(bound)
            last = round(value, round_number)
        return bounds

    @timing
    def load_data(self, years=10):
        stock_file_list = np.array(
            [s for s in os.listdir(self.origin_data_path) if s.strip(".csv") not in self.stock_index_list])
        stock_file_list = stock_file_list[
                          self.skip_stock_count:self.debug_stock_count] if self.debug else stock_file_list
        param = {
            "price_bounds": self.price_bounds,
            "vol_bounds": self.vol_bounds,
            "time_bounds": self.time_bounds,
            "stock_code_mapping": self.stock_code_mapping,
            "origin_data_path": self.origin_data_path,
            "years": years,
            "is_min": False
        }
        with Pool(17) as p:
            dfs = p.map(partial(self.__class__.read_csv, param=param), stock_file_list)
        return pd.concat(dfs, axis=0)

    @classmethod
    def read_csv(cls, filename, param):
        f32, i32, i64 = np.float32, np.int32, np.int64
        dytpes = {'open': f32, 'close': f32, 'high': f32, 'low': f32, 'pre_close': f32, 'vol': f32, 'amount': f32,
                  'trade_date': i32,
                  'ts_code': str}
        try:
            if '688350' in filename:
                logging.info(f"found{filename}")
            df = pd.read_csv(f"{param['origin_data_path']}/{filename}", dtype=dytpes, usecols=list(dytpes.keys()))
            df = cls.transform_day(df, param)
            return df
        except Exception as e:
            logging.error(f"read_csv:{filename} error {e}")
        return pd.DataFrame()

    @classmethod
    def transform_day(cls, df, param):
        df['day_vol'] = df.groupby(['trade_date'])['vol'].transform('sum')
        df = df[(df.trade_date > BaseModel.years_before(param['years'])) & (df.day_vol > 0)].copy()
        df = cls.transform_feature(df, param)
        return df

    @classmethod
    @abstractmethod
    def transform_feature(cls, df, param):
        logging.error(f"{cls} shouldn't be here")
        return df

    @timing
    def load_min_data(self, years):
        stock_file_list = np.array(
            [s for s in os.listdir(self.origin_min_data_path) if s.strip(".csv") not in self.stock_index_list])
        stock_file_list = stock_file_list[
                          self.skip_stock_count:self.debug_stock_count] if self.debug else stock_file_list
        param = {
            "price_bounds": self.price_bounds,
            "vol_bounds": self.vol_bounds,
            "time_bounds": self.time_bounds,
            "stock_code_mapping": self.stock_code_mapping,
            "origin_min_data_path": self.origin_min_data_path,
            "years": years,
            "is_min": True
        }
        with Pool(17) as p:
            dfs = p.map(partial(self.__class__.read_min_csv, param=param), stock_file_list)
        return pd.concat(dfs, axis=0)

    @classmethod
    def transform_min(cls, df, param):
        if 'trade_date' not in df.columns.values:
            df['trade_date'] = df['trade_time'].apply(lambda x: int("".join(x.split(" ")[0].split("-"))))
        df['day_vol'] = df.groupby(['trade_date'])['vol'].transform('sum')
        df = df[(df.trade_date > cls.years_before(param['years'])) & (df.day_vol > 0) & (df['open'] > 0)].copy()
        df['day_cummax'] = df.groupby(['trade_date'])['close'].transform('cummax')
        df['day_cummin'] = df.groupby(['trade_date'])['close'].transform('cummin')
        df['day_close'] = df.groupby(['trade_date'])['close'].transform('first')
        df['day_open'] = df.groupby(['trade_date'])['close'].transform('last')
        tdf = df[df['pre_close'] == 0]
        df['day_close_pre'] = df['day_close'].shift(-1)
        if len(tdf) > 0:
            df.loc[df['pre_close'] == 0, 'pre_close'] = df.loc[df['pre_close'] == 0, 'day_close_pre']
        df['day_pre_close'] = df.groupby(['trade_date'])['pre_close'].transform('last')
        df['day_min_close'] = df.groupby(['trade_date'])['close'].transform('min')
        df['day_max_close'] = df.groupby(['trade_date'])['close'].transform('max')
        df['sell'] = np.array(df['close'] >= df['day_cummax']).astype(np.int8)
        df['buy'] = np.array(df['close'] <= df['day_cummin']).astype(np.int8)
        df = cls.transform_feature(df, param)
        df['close_t'] = df.groupby(['trade_date'])['t'].transform('first')
        df['close_t'] = np.array(df['close_t'] == df['t']).astype(np.int8)
        return df

    @classmethod
    def read_min_csv(cls, filename, param):
        f32, i32, i64 = np.float32, np.int32, np.int64
        dytpes = {'open': f32, 'close': f32, 'high': f32, 'low': f32, 'vol': i64, 'amount': f32, 'trade_date': i32}
        try:
            if '000156' in filename:
                logging.info(filename)
            df = pd.read_csv(f"{param['origin_min_data_path']}/{filename}", dtype=dytpes)
            df = cls.transform_min(df, param)
            return df
        except Exception as e:
            logging.error(f"read_min_csv:{filename} error {e}")
        return pd.DataFrame()

    @abstractmethod
    def make_model(self):
        raise NotImplementedError("make_model method must implemented")

    @abstractmethod
    def feature_generator(self, df_min, df_day, seq_len, n_round, last_only=False, is_train=True):
        raise NotImplementedError("make_model method must implemented")

    def batch_feature_generator(self, tdf_min, tdf, last_only=False):
        features, labels = [], []
        while True:
            for f, l in self.feature_generator(tdf, self.seq_len, is_train=True):
                features.append(f)
                labels.append(l)
                if len(features) >= self.batch_size:
                    yield np.array(features), np.array(labels)
                    features, labels = [], []

    def get_steps(self, tdf):
        size = len(tdf)
        stock_count = len(np.unique(tdf.ts_code))
        print(size, stock_count, stock_count * (self.seq_len * 10 + 18), self.batch_size)
        return int((size - stock_count * (self.seq_len * 10 + 18)) / self.batch_size)
