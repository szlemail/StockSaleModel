import logging
import os
import time
from datetime import datetime, timedelta
from functools import wraps
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model
from keras.layers import Lambda
from keras.models import load_model

logging.basicConfig(level='INFO')


def timing(func):
    @wraps(func)
    def wrapper_fun(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        cost = (time.time() - start_time) * 1000
        logging.info(f"{func.__name__} cost {cost} ms")
        return res

    return wrapper_fun


def years_before(years):
    fmt = "%Y%m%d"
    days = years * 365 + years // 4
    return int((datetime.now() + timedelta(days=-days)).strftime(fmt))


@timing
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


STOCK_INDEX_LIST = ['399001.SZ', '399006.SZ', '000001.SH']
price_bounds = get_discrete_bounds(5000, round_number=2)
price_bounds.reverse()
vol_bounds = get_discrete_bounds(5000000000, decay=0.997, round_number=0)
vol_bounds.reverse()
time_bounds = [925, 955, 1025, 1055, 1125, 1155, 1325, 1355, 1425, 1455, 1525]
stock_code_list = np.array([s.strip(".csv") for s in os.listdir("data/origin")])
stock_code_mapping = dict(zip(stock_code_list, np.arange(1, len(stock_code_list) + 1)))


@timing
def load_data():
    stock_file_list = np.array([s for s in os.listdir("data/origin")])[:100]
    dfs = [pd.read_csv(f"data/origin/{filename}") for filename in stock_file_list]
    df = pd.concat(dfs, axis=0)
    df = df[df.trade_date > years_before(10)].copy()
    return df


def read_min_csv(filename):
    f32, i32, i64 = np.float32, np.int32, np.int64
    dytpes = {'open': f32, 'close': f32, 'high': f32, 'low': f32, 'vol': i64, 'amount': f32, 'trade_date': i32}
    try:
        df = pd.read_csv(f"data/origin_min/{filename}", dtype=dytpes)
        df['day_vol'] = df.groupby(['trade_date'])['vol'].transform('sum')
        df = df[(df.trade_date > years_before(10)) & (df.day_vol > 0)].copy()
        # df['day_high'] = df.groupby(['trade_date'])['high'].transform('max')
        # df['day_low'] = df.groupby(['trade_date'])['low'].transform('min')
        # df['day_close'] = df.groupby(['trade_date'])['close'].transform('first')
        # df['day_open'] = df.groupby(['trade_date'])['open'].transform('last')
        df['day_cummax'] = df.groupby(['trade_date'])['close'].transform('cummax')
        df['sell'] = np.array(df['close'] >= df['day_cummax']).astype(np.int8)
        index = 1
        df['w'] = pd.to_datetime(df.trade_date.apply(lambda x: "%s" % x)).dt.dayofweek
        df['w'] = df['w'].apply(lambda x: x + index).astype(np.int16)
        index = index + 7
        # month day
        df['md'] = pd.to_datetime(df.trade_date.apply(lambda x: "%s" % x)).dt.day
        df['md'] = df['md'].apply(lambda x: x + index).astype(np.int16)
        index = index + 31
        # month
        df['m'] = pd.to_datetime(df.trade_date.apply(lambda x: "%s" % x)).dt.month
        df['m'] = df['m'].apply(lambda x: x + index).astype(np.int16)
        index = index + 12
        price_labels = np.arange(index, index + len(price_bounds) - 1)
        df['o'] = np.array(
            pd.cut(df['open'], price_bounds, labels=price_labels)).astype(np.int16)
        df['h'] = np.array(
            pd.cut(df['high'], price_bounds, labels=price_labels)).astype(np.int16)
        df['l'] = np.array(
            pd.cut(df['low'], price_bounds, labels=price_labels)).astype(np.int16)
        df['c'] = np.array(
            pd.cut(df['close'], price_bounds, labels=price_labels)).astype(np.int16)
        index = index + len(price_bounds)
        vol_labels = np.arange(index, index + len(vol_bounds) - 1)
        df['v'] = np.array(pd.cut(df['vol'], vol_bounds, labels=vol_labels)).astype(np.int16)
        index = index + len(vol_bounds)
        df['t'] = np.array(
            pd.cut(df.trade_time.apply(lambda x: int("".join(x.split(" ")[1].split(":"))) / 100), time_bounds,
                   labels=np.arange(index, index + len(time_bounds) - 1))).astype(np.int16)
        index = index + len(time_bounds)
        df['s'] = df.ts_code.map(stock_code_mapping)
        df['s'] = index + df['s'].apply(lambda x: x + index)

        df.fillna(0, inplace=True)
    except Exception as e:
        logging.error(f"read_min_csv:{filename} error {e}")
        raise e
    return df


@timing
def load_min_data():
    stock_file_list = np.array([s for s in os.listdir("data/origin_min") if s.strip(".csv") not in STOCK_INDEX_LIST])[
                      :]
    with Pool(17) as p:
        dfs = p.map(read_min_csv, stock_file_list)
    return pd.concat(dfs, axis=0)


def feature_generator(idf, seq_len=90):
    """
    price: max:2589.0 min:0.6700000166893005
    vol: max:1946133833 min:1
    :param df:
    :return:
    """
    cols = "o,h,l,c,v,t,md,w,m,s,sell".split(",")
    feature_col = "o,h,l,c,v,t,md,w,m,s".split(",")
    label_col = "sell"
    for ts_code in np.unique(idf.ts_code):
        tdf = idf[idf.ts_code == ts_code].sort_values(by='trade_time')[cols]
        feature = tdf[feature_col].values
        label = tdf[label_col].values
        size = len(feature) - seq_len
        for start in range(size):
            yield feature[start:start + seq_len].tolist(), [label[start + seq_len - 1]]


def batch_feature_generator(tdf, seq_len=90, batch_size=128):
    features, labels = [], []
    while True:
        for f, l in feature_generator(tdf, seq_len):
            features.append(f)
            labels.append(l)
            if len(features) >= batch_size:
                yield np.array(features), np.array(labels)
                features, labels = [], []


def make_model(seq_len=90, embedding_size=16):
    def encoder(x):
        attention_layer = layers.MultiHeadAttention(num_heads=8, key_dim=4)
        attention_out = attention_layer(x, x)
        attention_out = layers.Dropout(0.2)(attention_out)
        out1 = (x + attention_out)
        print("out1.shape", out1.shape)
        ffn_out = layers.Dense(embedding_size, activation='relu')(out1)
        ffn_out = layers.Dense(embedding_size, activation='relu')(ffn_out)
        ffn_out = layers.Dropout(0.2)(ffn_out)

        out2 = (out1 + ffn_out)
        print("out2.shape", out2.shape)
        return out2

    layer_in = layers.Input(shape=(seq_len, 8))
    l1 = layers.Embedding(25000, embedding_size)(layer_in)
    l2 = Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2]))(l1)
    l3 = Lambda(lambda x: tf.reduce_sum(x, axis=-1))(l2)
    for i in range(5):
        l3 = encoder(l3)
    flat = layers.Flatten()(l3)
    h = layers.Dense(1024, activation='relu')(flat)
    h = layers.BatchNormalization()(h)
    h = layers.Dense(512, activation='relu')(h)
    o = layers.Dense(1, activation='sigmoid')(h)
    model = Model(inputs=layer_in, outputs=o)
    model.compile(optimizer=tf.optimizers.Adam(lr=1e-6), loss=tf.losses.BinaryCrossentropy(),
                  metrics=[tf.metrics.AUC()])
    model.summary()
    return model


def get_steps(tdf, batch_size, seq_len):
    size = len(tdf)
    stock_count = len(np.unique(tdf.ts_code))
    return int((size - stock_count * seq_len) / batch_size)


if __name__ == '__main__':
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 30)
    # 显示所有行
    pd.set_option('display.max_rows', 100)  # 最多显示10行

    SEQ_SIZE = 180
    BATCH_SIZE = 512
    model = make_model(seq_len=SEQ_SIZE)
    df = load_min_data()
    train = df[df.trade_date < 20180101].copy()
    val = df[(df.trade_date > 20180101) & (df.trade_date < 20190101)].copy()
    train_steps = get_steps(train, batch_size=BATCH_SIZE, seq_len=SEQ_SIZE)
    val_steps = get_steps(val, batch_size=BATCH_SIZE, seq_len=SEQ_SIZE)
    logging.info(f"train_steps:{train_steps}, val_steps:{val_steps}")
    model = load_model("model")
    model.fit(batch_feature_generator(train, seq_len=SEQ_SIZE, batch_size=BATCH_SIZE),
              steps_per_epoch=5,
              epochs=3,
              shuffle=True,
              validation_data=batch_feature_generator(val, seq_len=SEQ_SIZE, batch_size=BATCH_SIZE),
              validation_steps=val_steps,
              workers=14,
              use_multiprocessing=True)
    # model.save("model")


