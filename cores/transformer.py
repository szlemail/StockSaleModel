import gc
import random
from datetime import datetime, timedelta

import keras.losses
import tensorflow as tf
from keras import layers, Model, losses, optimizers, regularizers
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
        self.model, self.pre_model, middle_model = None, None, None
        self.model_seq_len = self.seq_len * 2 + 2
        self.feature_size = 8

    def make_pre_model(self):
        # base model
        def encoder(x):
            attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=6)
            attention_out = attention_layer(x, x)
            attention_out = layers.Dropout(0.2)(attention_out)
            out1 = (x + attention_out)
            ffn_out = layers.Dense(out1.shape[-1], activation='relu')(out1)
            ffn_out = layers.Dense(out1.shape[-1], activation='relu')(ffn_out)
            ffn_out = layers.Dropout(0.2)(ffn_out)
            out2 = out1 + ffn_out
            return out2

        layer_in = layers.Input(shape=(self.model_seq_len, self.feature_size))
        emb = layers.Embedding(20000, self.embedding_size)
        l1 = emb(layer_in)
        l2 = layers.Flatten()(l1)
        l2 = layers.Reshape(target_shape=(self.model_seq_len, -1))(l2)
        for i in range(3):
            l2 = encoder(l2)
        middle_model = Model(inputs=layer_in, outputs=l2, name='base_model')
        # middle_model.summary()

        # pretrain 模型
        mask_size = int(self.model_seq_len * self.mask_prob)
        soft_max_dim = len(self.price_bounds) + 4
        vectors = middle_model(middle_model.inputs)
        position = layers.Input(shape=(mask_size, 1), dtype=tf.int32)
        gather_vector = layers.Lambda(lambda x: tf.gather_nd(x[0], x[1], batch_dims=1))([vectors, position])
        open = layers.Dense(1, activation='sigmoid')(gather_vector)
        high = layers.Dense(1, activation='sigmoid')(gather_vector)
        low = layers.Dense(1, activation='sigmoid')(gather_vector)
        close = layers.Dense(1, activation='sigmoid')(gather_vector)
        pre_close = layers.Dense(1, activation='sigmoid')(gather_vector)
        vol = layers.Dense(1, activation='sigmoid')(gather_vector)
        pool = layers.Lambda(lambda x: x[:, 0, :])(vectors)
        is_next = layers.Dense(1, activation='sigmoid', name='next')(pool)
        stock = layers.Dense(len(self.stock_code_list), activation='softmax', name='stock')(pool)
        market = layers.Dense(5, activation='softmax', name='market')(pool)
        pre_model = Model(middle_model.inputs + [position],
                          [open, high, low, close, pre_close, vol, is_next, stock, market],
                          name='pre_model')
        lr_pre = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-6,
            first_decay_steps=50000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1,
            name=None
        )
        losses = ['mse'] * 6 + ['binary_focal_crossentropy'] + ['sparse_categorical_crossentropy'] * 2
        weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1]
        pre_model.compile(optimizer=tf.keras.optimizers.Adam(lr_pre), loss=losses, loss_weights=weights)
        # pre_model.summary()
        self.pre_model, self.middle_model = pre_model, middle_model

    def make_model(self):
        # fine tune 训练模型
        token_out = self.middle_model(self.middle_model.inputs)
        flat = layers.Flatten()(token_out)
        drop_out = layers.Dropout(0.5)(flat)
        sell_out = layers.Dense(1, activation='sigmoid', name='sell',
                                kernel_regularizer=regularizers.l1_l2(l1=0.1, l2=0.1))(drop_out)
        # buy_sell_close = layers.Dense(1, activation='sigmoid', name='bsc')(flat)
        # buy_sell_open = layers.Dense(1, activation='sigmoid', name='bso')(flat)
        # buy_safe = layers.Dense(1, activation='sigmoid', name='bs')(flat)
        # buy_safe_l = layers.Dense(1, activation='sigmoid', name='bsl')(flat)
        # buy_gain_c2 = layers.Dense(1, activation='sigmoid', name='bgc2')(flat)
        # buy_gain_c5 = layers.Dense(1, activation='sigmoid', name='bgc5')(flat)
        # outs = [sell_out, buy_sell_close, buy_sell_open, buy_safe, buy_safe_l, buy_gain_c2, buy_gain_c5]
        # outs = self.mmoe(input_v=flat, expert_dim=128, tower_dim=64, n_expert=7,
        #                  target_names=['sell', 'bsc', 'bso', 'bs', 'bsl', 'bgc2', 'bgc5'], activaton='sigmoid')
        model = Model(inputs=self.middle_model.inputs, outputs=sell_out, name='sale_model')
        self.model = model

    @staticmethod
    def mmoe(input_v, expert_dim, tower_dim, n_expert, target_names, activaton='sigmoid'):
        def gate(v_in, n_expert):
            v_out = layers.Dense(n_expert, activation='softmax')(v_in)
            return v_out

        def expert(v_in, dim=128):
            v_in = layers.Dropout(0.2)(v_in)
            v_h = layers.Dense(dim, activation=None)(v_in)
            v_h1 = layers.BatchNormalization()(layers.Dense(dim, activation='relu')(v_h))
            v_out = layers.Add()([v_h, layers.Dense(dim, activation='relu')(v_h1)])
            return v_out

        def gate_base(gates_in, expert_outs):
            return layers.Dot(axes=(1, 1))([gates_in, expert_outs])

        def tower(v_in, name, dim=64, activation=None):
            v_h = layers.Dense(dim, activation='relu')(v_in)
            v_out = layers.Dense(1, name=name, activation=activation)(v_h)
            return v_out

        experts = layers.Concatenate()([expert(input_v, dim=expert_dim) for _ in range(n_expert)])
        experts = layers.Reshape(target_shape=(n_expert, expert_dim))(experts)
        gates = [gate(input_v, n_expert) for _ in range(len(target_names))]
        gate_bases = [gate_base(g, experts) for g in gates]
        out = [tower(gb, target_names[i], dim=tower_dim, activation=activaton) for i, gb in enumerate(gate_bases)]
        return out

    def batch_feature_generator(self, tdf_min, tdf, last_only=False):
        if self.is_pre_train:
            f0, f1, l0, l1, l2, l3, l4, l5, l6, l7, l8 = [], [], [], [], [], [], [], [], [], [], []
            while True:
                for f, l in self.feature_generator(tdf_min, tdf, self.seq_len):
                    f0.append(f[0])
                    f1.append(f[1])
                    l0.append(l[0])
                    l1.append(l[1])
                    l2.append(l[2])
                    l3.append(l[3])
                    l4.append(l[4])
                    l5.append(l[5])
                    l6.append(l[6])
                    l7.append(l[7])
                    l8.append(l[8])
                    if len(f0) >= self.batch_size:
                        yield [np.array(f0), np.array(f1)], [np.array(l0), np.array(l1), np.array(l2), np.array(l3),
                                                             np.array(l4), np.array(l5), np.array(l6), np.array(l7),
                                                             np.array(l8)]
                        f0, f1, l0, l1, l2, l3, l4, l5, l6, l7, l8 = [], [], [], [], [], [], [], [], [], [], []
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
                        yield np.array(features), [np.array(l0)]
                        features, l0, l1, l2, l3, l4, l5, l6 = [], [], [], [], [], [], [], []

    def feature_generator(self, df_min, df_day, seq_len, last_only=False):
        """
        price: max:2589.0 min:0.6700000166893005
        vol: max:1946133833 min:1
        :param df_min: 分钟线子数据集
        :param df_day: 日线子数据集
        :param seq_len 序列长度
        :param last_only 是否只生成收盘时刻样本
        :return:
        """
        min_cols = "o,h,l,c,p,v,t,md,w,m,s,sell,buy,close_t,do,dc,dp,d_min,d_max,trade_date".split(
            ",")
        cols = "o,h,l,c,p,v,t,md,w,m,s,trade_date".split(",")
        for ts_code in np.unique(df_min.ts_code):
            tdf_min = df_min[df_min.ts_code == ts_code].sort_values(by='trade_time')[min_cols].copy()
            tdf_day = df_day[df_day.ts_code == ts_code].sort_values(by='trade_date')[cols].copy()
            if len(tdf_day) < seq_len:
                continue
            for cur_feature, cur_label in self.__feature_generator(tdf_min, tdf_day, seq_len, last_only):
                yield cur_feature, cur_label + [[self.stock_code_mapping.get(ts_code)], [self.get_market(ts_code)]]
            gc.collect()

    def make_pretrain_sample(self, cur_feature, is_next):
        cur_feature = np.array(cur_feature)
        mask_size = int(self.mask_prob * self.seq_len)
        soft_max_dim = len(self.price_bounds) + 4 + len(self.vol_bounds)
        positions = np.arange(1, self.seq_len)
        random.shuffle(positions)
        positions_d = np.array(sorted(positions[:mask_size]))
        random.shuffle(positions)
        positions_m = np.array(sorted(positions[mask_size:mask_size * 2])) + self.seq_len
        positions = positions_d.tolist()
        positions += positions_m.tolist()
        positions = np.array(positions)
        l_open, l_high, l_low, l_close, l_pre_close, l_val = [], [], [], [], [], []
        for p in positions:
            l_open.append(cur_feature[p, 0] * 1.0 / soft_max_dim)
            l_high.append(cur_feature[p, 1] * 1.0 / soft_max_dim)
            l_low.append(cur_feature[p, 2] * 1.0 / soft_max_dim)
            l_close.append(cur_feature[p, 3] * 1.0 / soft_max_dim)
            l_pre_close.append(cur_feature[p, 4] * 1.0 / soft_max_dim)
            l_val.append(cur_feature[p, 5] * 1.0 / soft_max_dim)
            if random.random() < 0.8:
                if random.random() > 0.5:
                    cur_feature[p, :4] = [1, 1, 1, 1]
            else:
                if random.random() > 0.5:
                    cur_feature[p, :4] = np.random.randint([soft_max_dim] * 4)
        return [cur_feature.tolist(), positions.reshape(mask_size * 2, 1).tolist()], [l_open, l_high, l_low, l_close,
                                                                                      l_pre_close, l_val, [is_next]]

    def __feature_generator(self, tdf_min, tdf_day, seq_len, last_only):
        # feature_col = "o,h,l,c,p,v,t,md,w,m,s".split(",")
        # O, H, L, C, P, V, T, MD, W, M, S = range(len(feature_col))
        feature_col = "o,h,l,c,p,v,t,w".split(",")
        O, H, L, C, P, V, T, W = range(len(feature_col))
        label_col = "sell,buy,close_t,do,dc,dp,d_min,d_max".split(",")
        SELL, BUY, CLOSE_T, DAY_OPEN, DAY_CLOSE, DAY_PRE_CLOSE, DAY_MIN_CLOSE, DAY_MAX_CLOSE = range(len(label_col))
        feature_min = tdf_min[feature_col].values
        date_min = tdf_min['trade_date'].values
        label = tdf_min[label_col].values
        size = len(feature_min) - seq_len * 10 - 18
        is_next = 0
        for start in range(seq_len * 9, size + seq_len * 9):
            last_pos = start + seq_len - 1
            cur_min_feature = feature_min[start:last_pos + 1].tolist()
            last_date = date_min[last_pos]
            if self.is_pre_train:
                random_shift = np.random.randint(1, 26)
                random_sign = np.random.randint(0, 2)
                random_shift = random_shift * 5 if random_sign == 1 else -random_shift * 5
                if random_shift + last_pos <= seq_len * 10 or random_shift + last_pos >= len(date_min) - 18:
                    continue
                if random.random() > 0.5:
                    last_date = date_min[last_pos]
                    is_next = 1
                else:
                    last_date = date_min[random_shift + last_pos]
                    is_next = 0

            cur_day_feature = tdf_day[tdf_day.trade_date < last_date][feature_col].values[-seq_len:]
            if len(cur_day_feature) != seq_len:
                continue
            # adj qfq ajust
            if cur_day_feature[-1, C] != label[last_pos - 9, DAY_CLOSE]:
                shift = label[last_pos - 9, DAY_CLOSE] - cur_day_feature[-1, C]
                for col in O, H, L, C, P:
                    cur_day_feature[:, col] = cur_day_feature[:, col] + shift
            sep = [[2] * (len(feature_col))]
            start = [[3] * (len(feature_col))]
            cur_feature = start + cur_day_feature.tolist() + sep + cur_min_feature
            last_t = label[last_pos, CLOSE_T]
            if last_only and last_t != 1:
                continue

            def price_shift_lt_next_day(price_shift, target_col):
                today_buy = label[last_pos, BUY] == 1
                buy_price = feature_min[last_pos, C] + price_shift
                adj = label[last_pos, DAY_CLOSE] - label[last_pos + 9, DAY_PRE_CLOSE]
                sell_price = label[last_pos + 9, target_col] - adj
                return int(today_buy & (buy_price < sell_price))

            sell = label[last_pos, SELL]
            buy_sell_close = price_shift_lt_next_day(2, DAY_CLOSE)
            buy_sell_open = price_shift_lt_next_day(2, DAY_OPEN)
            buy_safe = price_shift_lt_next_day(2, DAY_MIN_CLOSE)
            buy_safe_l = price_shift_lt_next_day(-10, DAY_MIN_CLOSE)
            buy_gain_c2 = price_shift_lt_next_day(20, DAY_MAX_CLOSE)
            buy_gain_c5 = price_shift_lt_next_day(50, DAY_MAX_CLOSE)
            cur_label = [[sell], [buy_sell_close], [buy_sell_open], [buy_safe], [buy_safe_l], [buy_gain_c2],
                         [buy_gain_c5]]
            if self.is_pre_train:
                yield self.make_pretrain_sample(cur_feature, is_next)
            else:
                yield cur_feature, cur_label

    def build(self):
        self.make_pre_model()
        self.make_model()

    def pre_train(self, df_min, df_day, epochs, workers=8):
        self.is_pre_train = True
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
        self.pre_model.fit(self.batch_feature_generator(train_min, train_day),
                           steps_per_epoch=train_steps,
                           epochs=epochs,
                           shuffle=True,
                           validation_data=self.batch_feature_generator(vals[0][0], vals[0][1]),
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
        lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=5e-6,
            first_decay_steps=20000,
            t_mul=2.0,
            m_mul=0.8,
            alpha=0.05,
            name=None
        )
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                           loss='binary_crossentropy',
                           metrics=tf.metrics.AUC())
        self.model.summary()
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
            model_name = f"model_{i}_{datetime.now().strftime('%Y%m%d')}"
            self.model.save(f"model/{model_name}")

    @staticmethod
    def predict(x):
        model = load_model("model/model")
        model.predict(x)
