import gc
import random
from datetime import datetime, timedelta
import tensorflow as tf
from keras import layers, Model, losses, optimizers
from keras.models import load_model

from cores.base_model import BaseModel
import numpy as np
import logging
from common.configs import config


class Transformer(BaseModel):
    """
    借鉴 transformer 的 ENCODER 方式进行模型训练
    """

    def __init__(self):
        super(Transformer, self).__init__()
        self.mask_prob = float(config.get("model", "mask_prob", default='0.05'))
        self.is_pre_train = False
        self.model, self.pre_model = None, None

    def make_model(self, seq_len, embedding_size):
        # base model
        def encoder(x):
            attention_layer = layers.MultiHeadAttention(num_heads=8, key_dim=4)
            attention_out = attention_layer(x, x)
            attention_out = layers.Dropout(0.2)(attention_out)
            out1 = (x + attention_out)
            ffn_out = layers.Dense(out1.shape[-1], activation='relu')(out1)
            ffn_out = layers.Dense(out1.shape[-1], activation='relu')(ffn_out)
            ffn_out = layers.Dropout(0.2)(ffn_out)
            out2 = (out1 + ffn_out)
            return out2

        layer_in = layers.Input(shape=(seq_len, 10))
        l1 = layers.Embedding(25000, embedding_size)(layer_in)
        l2 = layers.Flatten()(l1)
        l2 = layers.Reshape(target_shape=(seq_len, -1))(l2)
        for i in range(5):
            l2 = encoder(l2)
        middle_model = Model(inputs=layer_in, outputs=l2)
        middle_model.summary()

        # finetune 训练模型
        flat = layers.Flatten()(middle_model(middle_model.inputs))
        # h = layers.Dense(1024, activation='relu')(flat)
        # h = layers.BatchNormalization()(h)
        # h = layers.Dense(512, activation='relu')(h)
        """
        [[sell], [buy_sell_close], [buy_sell_open], [buy_safe], [buy_safe_l], [buy_gain_c2],
                                 [buy_gain_c5]]
        """
        sell_out = layers.Dense(1, activation='sigmoid', name='sell')(flat)
        buy_sell_close = layers.Dense(1, activation='sigmoid', name='bsc')(flat)
        buy_sell_open = layers.Dense(1, activation='sigmoid', name='bso')(flat)
        buy_safe = layers.Dense(1, activation='sigmoid', name='bs')(flat)
        buy_safe_l = layers.Dense(1, activation='sigmoid', name='bsl')(flat)
        buy_gain_c2 = layers.Dense(1, activation='sigmoid', name='bgc2')(flat)
        buy_gain_c5 = layers.Dense(1, activation='sigmoid', name='bgc5')(flat)
        model = Model(inputs=middle_model.inputs,
                      outputs=[sell_out, buy_sell_close, buy_sell_open, buy_safe, buy_safe_l, buy_gain_c2, buy_gain_c5])
        lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-5,
            first_decay_steps=50000,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.01,
            name=None
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss=losses.BinaryFocalCrossentropy(label_smoothing=0.01),
                      metrics=tf.metrics.AUC())
        model.summary()

        # pretrain 模型
        mask_size = int(seq_len * self.mask_prob)
        soft_max_dim = len(self.price_bounds) + 1
        vectors = middle_model(middle_model.inputs)
        position = layers.Input(shape=(mask_size, 1), dtype=tf.int32)
        gather_vector = layers.Lambda(lambda x: tf.gather_nd(x[0], x[1], batch_dims=1))([vectors, position])
        l_open = layers.Dense(1, activation='sigmoid')(gather_vector) * soft_max_dim
        l_high = layers.Dense(1, activation='sigmoid')(gather_vector) * soft_max_dim
        l_low = layers.Dense(1, activation='sigmoid')(gather_vector) * soft_max_dim
        l_close = layers.Dense(1, activation='sigmoid')(gather_vector) * soft_max_dim
        pre_model = Model(middle_model.inputs + [position], [l_open, l_high, l_low, l_close])
        lr_pre = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-4,
            first_decay_steps=50000,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.01,
            name=None
        )
        pre_model.compile(optimizer=tf.keras.optimizers.Adam(lr_pre), loss='mse',
                          metrics=['mse'])
        pre_model.summary()
        return model, pre_model

    def batch_feature_generator(self, tdf_min, tdf, last_only=False):
        if self.is_pre_train:
            f0, f1, l0, l1, l2, l3 = [], [], [], [], [], []
            while True:
                for f, l in self.feature_generator(tdf_min, tdf, self.seq_len):
                    f0.append(f[0])
                    f1.append(f[1])
                    l0.append(l[0])
                    l1.append(l[1])
                    l2.append(l[2])
                    l3.append(l[3])
                    if len(f0) >= self.batch_size:
                        yield [np.array(f0), np.array(f1)], [np.array(l0), np.array(l1), np.array(l2), np.array(l3)]
                        f0, f1, l0, l1, l2, l3 = [], [], [], [], [], []
        else:
            features, l0, l1, l2, l3, l4, l5, l6 = [], [], [], [], [], [], [], []
            while True:
                for f, l in self.feature_generator(tdf_min, tdf, self.seq_len, last_only):
                    features.append(f)
                    l0.append(l[0])
                    l1.append(l[1])
                    l2.append(l[2])
                    l3.append(l[3])
                    l4.append(l[4])
                    l5.append(l[5])
                    l6.append(l[6])
                    if len(features) >= self.batch_size:
                        yield np.array(features), [np.array(l0), np.array(l1), np.array(l2), np.array(l3), np.array(l4),
                                                   np.array(l5), np.array(l6)]
                        features, l0, l1, l2, l3, l4, l5, l6 = [], [], [], [], [], [], [], []

    def feature_generator(self, df_min, df, seq_len, last_only=False):
        """
        price: max:2589.0 min:0.6700000166893005
        vol: max:1946133833 min:1
        :param df: 子数据集
        :param seq_len 序列长度
        :return:
        """
        min_cols = "o,h,l,c,v,t,md,w,m,s,sell,buy,close_t,day_open,day_close,day_pre_close,day_min_close,day_max_close,trade_date".split(
            ",")
        cols = "o,h,l,c,v,t,md,w,m,s,trade_date".split(",")
        feature_col = "o,h,l,c,v,t,md,w,m,s".split(",")
        O, H, L, C, V, T, MD, W, M, S = range(len(feature_col))
        label_col = "sell,buy,close_t,day_open,day_close,day_pre_close,day_min_close,day_max_close".split(",")
        SELL, BUY, CLOSE_T, DAY_OPEN, DAY_CLOSE, DAY_PRE_CLOSE, DAY_MIN_CLOSE, DAY_MAX_CLOSE = range(len(label_col))
        for ts_code in np.unique(df_min.ts_code):
            tdf_min = df_min[df_min.ts_code == ts_code].sort_values(by='trade_time')[min_cols].copy()
            tdf_day = df[df.ts_code == ts_code].sort_values(by='trade_date')[cols].copy()
            if len(tdf_day) < seq_len:
                continue
            feature_min = tdf_min[feature_col].values
            date_min = tdf_min['trade_date'].values
            label = tdf_min[label_col].values
            size = len(feature_min) - seq_len * 10 - 18
            for start in range(seq_len * 9, size + seq_len * 9):
                if self.is_pre_train:
                    cur_feature = feature_min[start:start + seq_len].copy()
                    mask_size = int(self.mask_prob * self.seq_len)
                    soft_max_dim = len(self.price_bounds) + 1
                    positions = np.arange(self.seq_len)
                    random.shuffle(positions)
                    positions = np.array(sorted(positions[:mask_size]))
                    l_open, l_high, l_low, l_close = [], [], [], []
                    for p in positions:
                        l_open.append(cur_feature[p, 0])
                        l_high.append(cur_feature[p, 1])
                        l_low.append(cur_feature[p, 2])
                        l_close.append(cur_feature[p, 3])
                        if random.random() < 0.8:
                            cur_feature[p, :4] = [1, 1, 1, 1]
                        else:
                            if random.random() > 0.5:
                                cur_feature[p, :4] = np.random.randint([soft_max_dim] * 4)
                    yield [cur_feature.tolist(), positions.reshape(mask_size, 1).tolist()], [l_open, l_high, l_low,
                                                                                             l_close]
                else:
                    last_pos = start + seq_len - 1
                    cur_min_feature = feature_min[start:last_pos + 1].tolist()
                    first_date = date_min[start]
                    cur_day_feature = tdf_day[tdf_day.trade_date <= first_date][feature_col].values[-seq_len:].tolist()
                    sep = [[2] * (len(feature_col))]
                    cur_feature = cur_day_feature + sep + cur_min_feature
                    if len(cur_day_feature) != seq_len:
                        continue
                    sell = label[last_pos, SELL]
                    last_t = label[last_pos, CLOSE_T]
                    if last_only and last_t != 1:
                        continue
                    adj = label[last_pos, DAY_CLOSE] - label[last_pos + 9, DAY_PRE_CLOSE]
                    buy_sell_close = int(
                        label[last_pos, BUY] == 1 & (
                                    feature_min[last_pos, C] + 2 < label[last_pos + 9, DAY_CLOSE] - adj))
                    buy_sell_open = int(
                        label[last_pos, BUY] == 1 & (
                                    feature_min[last_pos, C] + 2 < label[last_pos + 9, DAY_OPEN] - adj))
                    buy_safe = int(
                        label[last_pos, BUY] == 1 & (
                                    feature_min[last_pos, C] + 2 < label[last_pos + 9, DAY_MIN_CLOSE] - adj))
                    buy_safe_l = int(
                        label[last_pos, BUY] == 1 & (
                                    feature_min[last_pos, C] - 8 < label[last_pos + 9, DAY_MIN_CLOSE] - adj))
                    buy_gain_c2 = int(
                        label[last_pos, BUY] == 1 & (
                                feature_min[last_pos, C] + 20 < label[last_pos + 9, DAY_MAX_CLOSE] - adj))
                    buy_gain_c5 = int(
                        label[last_pos, BUY] == 1 & (
                                feature_min[last_pos, C] + 50 < label[last_pos + 9, DAY_MAX_CLOSE] - adj))
                    cur_label = [[sell], [buy_sell_close], [buy_sell_open], [buy_safe], [buy_safe_l], [buy_gain_c2],
                                 [buy_gain_c5]]
                    yield cur_feature, cur_label
            gc.collect()

    def build(self):
        self.model, self.pre_model = self.make_model(seq_len=self.seq_len * 2 + 1, embedding_size=self.embedding_size)

    def pre_train(self, df, epochs, workers=8):
        self.is_pre_train = True
        train = df[df.trade_date < 20170101].copy()
        val = df[(df.trade_date > 20170101) & (df.trade_date < 20190101)].copy()
        train_steps = self.get_steps(train)
        val_steps = self.get_steps(val)
        logging.info(f"train_steps:{train_steps}, val_steps:{val_steps}")
        self.pre_model.fit(self.batch_feature_generator(train),
                           steps_per_epoch=train_steps,
                           epochs=epochs,
                           shuffle=True,
                           validation_data=self.batch_feature_generator(val),
                           validation_steps=val_steps,
                           workers=workers,
                           use_multiprocessing=True)
        model_name = f"pre_model_{datetime.now().strftime('%Y%m%d')}"
        self.pre_model.save(f"model/{model_name}")

    def train(self, df_min, df_day, epochs, workers=8):
        self.is_pre_train = False
        train_min = df_min[df_min.trade_date < 20170101].copy()
        train_day = df_day[df_day.trade_date < 20170101].copy()
        vals = []
        last = 20170101
        for i, s in enumerate([20170101, 20180101, 20190101, 20200101, 20210101, 20220101]):
            if i > 0:
                vals.append((df_min[(df_min.trade_date > last) & (df_min.trade_date < s)].copy(),
                             df_day[(df_day.trade_date > last) & (df_day.trade_date < s)].copy()
                             ))
                last = s
        train_steps = self.get_steps(train_min)
        val_steps = self.get_steps(vals[0][0])
        logging.info(f"train_steps:{train_steps}, val_steps:{val_steps}")

        self.model.fit(self.batch_feature_generator(train_min, train_day),
                       steps_per_epoch=train_steps,
                       epochs=epochs,
                       shuffle=True,
                       validation_data=self.batch_feature_generator(vals[0][0], vals[0][1]),
                       validation_steps=val_steps,
                       workers=workers,
                       use_multiprocessing=True)
        model_name = f"model_{datetime.now().strftime('%Y%m%d')}"
        self.model.save(f"model/{model_name}")
        for i in range(len(vals) - 1):
            print(f"i:{i}/{len(vals) - 2}")
            self.model.fit(self.batch_feature_generator(vals[i][0], vals[i][1]),
                           steps_per_epoch=val_steps,
                           epochs=epochs,
                           shuffle=True,
                           validation_data=self.batch_feature_generator(vals[i + 1][0], vals[i + 1][1]),
                           validation_steps=val_steps,
                           workers=workers,
                           use_multiprocessing=True)

    @staticmethod
    def predict(x):
        model = load_model("model/model")
        model.predict(x)
