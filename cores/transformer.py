import gc
import random
from datetime import datetime

import pandas as pd
import tensorflow as tf
from keras import layers, Model, regularizers
from keras.models import load_model
from cores.base_model import BaseModel
import numpy as np
import logging
from common.configs import config
from multiprocessing import Manager, Lock


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
        self.manager = Manager()
        self.lock = Lock()
        self.train_visited = self.manager.dict()
        self.val_visited = self.manager.dict()

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
        emb = layers.Embedding(11000, self.embedding_size)
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
        stock = layers.Dense(len(self.stock_code_list) + 1, activation='softmax', name='stock')(pool)
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
        lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=5e-6,
            first_decay_steps=10000,
            t_mul=2.0,
            m_mul=0.8,
            alpha=0.05,
            name=None
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss='binary_crossentropy',
                      metrics=tf.metrics.AUC())
        model.summary()
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

    @classmethod
    def transform_feature(cls, df, param):
        index = 4  # 0:NA, 1 MASK, 2:SEP, 3:START
        price_labels = np.arange(index, index + len(param['price_bounds']) - 1)
        df['o'] = np.array(pd.cut(df['open'], param['price_bounds'], labels=price_labels)).astype(np.int16)
        df['h'] = np.array(pd.cut(df['high'], param['price_bounds'], labels=price_labels)).astype(np.int16)
        df['l'] = np.array(pd.cut(df['low'], param['price_bounds'], labels=price_labels)).astype(np.int16)
        df['c'] = np.array(pd.cut(df['close'], param['price_bounds'], labels=price_labels)).astype(np.int16)
        df['p'] = np.array(pd.cut(df['pre_close'], param['price_bounds'], labels=price_labels)).astype(np.int16)
        if param['is_min']:
            df['do'] = np.array(pd.cut(df['day_open'], param['price_bounds'], labels=price_labels)).astype(np.int16)
            df['dc'] = np.array(pd.cut(df['day_close'], param['price_bounds'], labels=price_labels)).astype(np.int16)
            df['dp'] = np.array(pd.cut(df['day_pre_close'], param['price_bounds'], labels=price_labels)).astype(
                np.int16)
            df['d_min'] = np.array(pd.cut(df['day_min_close'], param['price_bounds'], labels=price_labels)).astype(
                np.int16)
            df['d_max'] = np.array(pd.cut(df['day_max_close'], param['price_bounds'], labels=price_labels)).astype(
                np.int16)
        index = index + len(param['price_bounds'])

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
        index = index + 13
        # vol
        vol_labels = np.arange(index + 1, index + len(param['vol_bounds']))
        df['v'] = np.array(pd.cut(df['vol'], param['vol_bounds'], labels=vol_labels)).astype(np.int16)
        df['v'].fillna(index, inplace=True)
        index = index + len(param['vol_bounds']) + 1

        if param['is_min']:
            df['t'] = np.array(pd.cut(df.trade_time.apply(lambda x: int("".join(x.split(" ")[1].split(":"))) / 100),
                                      param['time_bounds'],
                                      labels=np.arange(index, index + len(param['time_bounds']) - 1))).astype(np.int16)
        else:

            df['t'] = index + len(param['time_bounds']) - 1
        index = index + len(param['time_bounds'])
        df['s'] = df.ts_code.map(param['stock_code_mapping'])
        df['s'] = index + df['s'].apply(lambda x: x + index)
        df.fillna(0, inplace=True)
        return df

    def batch_feature_generator(self, tdf_min, tdf, last_only=False, is_train=True):
        n_round = 1
        if self.is_pre_train:
            f0, f1, l0, l1, l2, l3, l4, l5, l6, l7, l8 = [], [], [], [], [], [], [], [], [], [], []
            while True:
                for f, l in self.feature_generator(tdf_min, tdf, self.seq_len, n_round, is_train=is_train):
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
                n_round += 1
        else:
            features, l0, l1, l2, l3, l4, l5, l6 = [], [], [], [], [], [], [], []
            while True:
                for f, l in self.feature_generator(tdf_min, tdf, self.seq_len, n_round, last_only, is_train):
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
                n_round += 1

    def feature_generator(self, df_min, df_day, seq_len, n_round, last_only=False, is_train=True):
        """
        price: max:2589.0 min:0.6700000166893005
        vol: max:1946133833 min:1
        :param df_min: 分钟线子数据集
        :param df_day: 日线子数据集
        :param n_round: 第几次遍历股票。为了防止多进程时，有些进程过早进入STOP ITERRATION. 每次可以遍历到下一个ROUND
        :param seq_len 序列长度
        :param last_only 是否只生成收盘时刻样本
        :return:
        """
        min_cols = "o,h,l,c,p,v,t,md,w,m,s,sell,buy,close_t,do,dc,dp,d_min,d_max,trade_date".split(
            ",")
        cols = "o,h,l,c,p,v,t,md,w,m,s,trade_date".split(",")
        for ts_code in np.unique(df_min.ts_code):

            with self.lock:
                if is_train:
                    if ts_code in self.train_visited and self.train_visited[ts_code] == n_round:
                        continue
                    else:
                        self.train_visited[ts_code] = n_round
                else:
                    if ts_code in self.val_visited and self.val_visited[ts_code] == n_round:
                        continue
                    else:
                        self.val_visited[ts_code] = n_round
            tdf_min = df_min[df_min.ts_code == ts_code].sort_values(by='trade_time')[min_cols]
            tdf_day = df_day[df_day.ts_code == ts_code].sort_values(by='trade_date')[cols]
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
            last_date = date_min[last_pos]
            cur_min_feature = feature_min[start:last_pos + 1].tolist()
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

            cur_day_feature = tdf_day[tdf_day.trade_date < last_date][feature_col].values[-seq_len:].copy()
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

    def get_train_val_by_year(self, years=13, step_days=250, init_start_days=1000):
        df_min = self.load_min_data(years)
        print(df_min.head(3))
        df_day = self.load_data(years)
        print(df_day.head(3))
        dates = sorted(np.unique(df_min.trade_date))
        logging.info(f"length of dates is {len(dates)}")
        for i in range(init_start_days, len(dates), step_days):
            start = 0 if i == init_start_days else i - step_days - self.seq_len - self.seq_len // 9 - 2
            end = i
            val_start = i - self.seq_len - self.seq_len // 9 - 2
            val_end = min(end + step_days, len(dates) - 1)
            logging.info(f"train: {dates[start]} - {dates[end]}, val: {dates[val_start]} - {dates[val_end]}")
            train_min = df_min[(df_min.trade_date > dates[start]) & (df_min.trade_date < dates[end])]
            train_day = df_day[(df_day.trade_date > dates[start]) & (df_day.trade_date < dates[end])]
            val_min = df_min[(df_min.trade_date > dates[val_start]) & (df_min.trade_date < dates[val_end])]
            val_day = df_day[(df_day.trade_date > dates[val_start]) & (df_day.trade_date < dates[val_end])]
            yield train_min, train_day, val_min, val_day

    def pre_train(self, years, epochs, workers=8, pre_train_days=2000):
        self.is_pre_train = True
        data_set = self.get_train_val_by_year(years, init_start_days=pre_train_days)
        i = 0
        for train_min, train_day, val_min, val_day in data_set:
            i += 1
            if i != 2 and i != 3:
                continue
            train_steps = self.get_steps(train_min)
            val_steps = self.get_steps(val_min)
            train_gen = self.batch_feature_generator(train_min, train_day)
            val_gen = self.batch_feature_generator(val_min, val_day, is_train=False)
            logging.info(f"train_steps:{train_steps}, val_steps:{val_steps}")
            self.pre_model.fit(train_gen,
                               steps_per_epoch=train_steps,
                               epochs=epochs,
                               shuffle=True,
                               validation_data=val_gen,
                               validation_steps=val_steps,
                               workers=workers,
                               use_multiprocessing=True)
            model_name = f"pre_model_{datetime.now().strftime('%Y%m%d')}"
            self.pre_model.save(f"model/{model_name}")
            del train_min, train_day, train_steps, val_steps
            train_gen.close()
            val_gen.close()
            break
        data_set.close()
        gc.collect()

    def train(self, years, epochs, workers=8):
        self.is_pre_train = False
        years_round = 0
        for train_min, train_day, val_min, val_day in self.get_train_val_by_year(years, init_start_days=2000):
            train_steps = self.get_steps(train_min)
            val_steps = self.get_steps(val_min)
            with self.lock:
                self.train_visited.clear()
                self.val_visited.clear()
            logging.info(f"train_steps:{train_steps}, val_steps:{val_steps}")
            train_gen = self.batch_feature_generator(train_min, train_day)
            val_gen = self.batch_feature_generator(val_min, val_day, is_train=False)
            self.model.fit(train_gen,
                           steps_per_epoch=train_steps,
                           epochs=epochs,
                           shuffle=True,
                           validation_data=val_gen,
                           validation_steps=val_steps,
                           workers=workers,
                           use_multiprocessing=True)

            model_name = f"model_{years_round}_{datetime.now().strftime('%Y%m%d')}"
            self.model.save(f"model/{model_name}")
            years_round += 1
            del train_min, train_day, train_steps, val_steps
            train_gen.close()
            val_gen.close()
            gc.collect()

    @staticmethod
    def predict(x):
        model = load_model("model/model")
        model.predict(x)
