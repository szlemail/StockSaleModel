import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from datetime import datetime, timedelta
from multiprocessing import Pool
from common.wrappers import timing, exception
import logging
from functools import partial


class BaseModel(object):
    """
    基礎模型，處理通用的數據預處理，模型訓練、驗證保存等邏輯，繼承的模型可以專注在建模上
    """

    def __init__(self, seq_len=90, debug=False, origin_data_path="data/origin", origin_min_data_path="data/origin_min"):
        self.seq_len = seq_len
        self.debug = debug
        self.debug_stock_count = 100
        self.origin_data_path = origin_data_path
        self.origin_min_data_path = origin_min_data_path
        self.stock_index_list = ['399001.SZ', '399006.SZ', '000001.SH']
        self.price_bounds = self.get_discrete_bounds(5000, round_number=2)
        self.price_bounds.reverse()
        self.vol_bounds = self.get_discrete_bounds(5000000000, decay=0.997, round_number=0)
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
    @exception
    def load_data(self, years=10):
        stock_file_list = np.array([s for s in os.listdir(self.origin_data_path)])
        stock_file_list = stock_file_list[self.debug_stock_count] if self.debug else stock_file_list
        dfs = [pd.read_csv(f"{self.origin_data_path}/{filename}") for filename in stock_file_list]
        df = pd.concat(dfs, axis=0)
        df = df[df.trade_date > self.years_before(years)].copy()
        return df

    @staticmethod
    def read_min_csv(filename, param):
        f32, i32, i64 = np.float32, np.int32, np.int64
        dytpes = {'open': f32, 'close': f32, 'high': f32, 'low': f32, 'vol': i64, 'amount': f32, 'trade_date': i32}
        try:
            df = pd.read_csv(f"data/origin_min/{filename}", dtype=dytpes)
            df['day_vol'] = df.groupby(['trade_date'])['vol'].transform('sum')
            df = df[(df.trade_date > BaseModel.years_before(param['years'])) & (df.day_vol > 0)].copy()
            df['day_cummax'] = df.groupby(['trade_date'])['close'].transform('cummax')
            df['sell'] = np.array(df['close'] >= df['day_cummax']).astype(np.int8)
            price_discrete_label = np.arange(1, len(param['price_bounds']))
            vol_discrete_label = np.arange(1, len(param['vol_bounds']))
            time_bounds_label = np.arange(1, len(param['time_bounds']))
            df['o'] = np.array(pd.cut(df['open'], param['price_bounds'], labels=price_discrete_label)).astype(np.int16)
            df['h'] = np.array(pd.cut(df['high'], param['price_bounds'], labels=price_discrete_label)).astype(np.int16)
            df['l'] = np.array(pd.cut(df['low'], param['price_bounds'], labels=price_discrete_label)).astype(np.int16)
            df['c'] = np.array(pd.cut(df['close'], param['price_bounds'], labels=price_discrete_label)).astype(np.int16)
            df['v'] = np.array(pd.cut(df['vol'], param['vol_bounds'], labels=vol_discrete_label)).astype(np.int16)
            df['t'] = np.array(pd.cut(df.trade_time.apply(lambda x: int("".join(x.split(" ")[1].split(":"))) / 100),
                                      param['time_bounds'], labels=time_bounds_label)).astype(np.int16)
            df['w'] = pd.to_datetime(df.trade_date.apply(lambda x: "%s" % x)).dt.dayofweek
            df['s'] = df.ts_code.map(param['stock_code_mapping'])
            df.fillna(0, inplace=True)
        except Exception as e:
            logging.error(f"read_min_csv:{filename} error {e}")
        return pd.DataFrame()

    @timing
    def load_min_data(self):
        stock_file_list = np.array(
            [s for s in os.listdir(self.origin_min_data_path) if s.strip(".csv") not in self.stock_index_list])

        self.seq_len = seq_len
        self.debug = debug
        self.debug_stock_count = 100
        self.origin_data_path = origin_data_path
        self.origin_min_data_path = origin_min_data_path
        self.stock_index_list = ['399001.SZ', '399006.SZ', '000001.SH']
        self.price_bounds = self.get_discrete_bounds(5000, round_number=2)
        self.price_bounds.reverse()
        self.vol_bounds = self.get_discrete_bounds(5000000000, decay=0.997, round_number=0)
        self.vol_bounds.reverse()
        self.time_bounds = [925, 955, 1025, 1055, 1125, 1155, 1325, 1355, 1425, 1455, 1525]
        self.stock_code_list = np.array([s.strip(".csv") for s in os.listdir(self.origin_data_path)])
        self.stock_code_mapping = dict(zip(self.stock_code_list, np.arange(1, len(self.stock_code_list) + 1)))

        param = {
            "price_bounds":self.price_bounds,
            "vol_bounds": self.vol_bounds,
            "time_bounds": self.time_bounds,
            "stock_code_mapping": self.stock_code_mapping,
            "price_bounds": self.price_bounds,
        }
        with Pool(17) as p:
            dfs = p.map(partial(BaseModel.read_min_csv, param=param), stock_file_list)
        return pd.concat(dfs, axis=0)

    def make_model(self):
        x = layers.Input(shape=(self.seq_len, 8))
        x1 = layers.Flatten()(x)
        h = layers.Dense(1024, activation='relu')(x1)
        h = layers.BatchNormalization()(h)
        h = layers.Dense(512, activation='relu')(h)
        o = layers.Dense(1, activation='sigmoid')(h)
        model = Model(inputs=x, outputs=o)
        model.compile(optimizer=optimizers.Adam(lr=3e-4), loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.AUC()])
        model.summary()
        return model
