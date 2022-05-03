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
            ffn_out = layers.Dense(embedding_size, activation='relu')(out1)
            ffn_out = layers.Dense(embedding_size, activation='relu')(ffn_out)
            ffn_out = layers.Dropout(0.2)(ffn_out)
            out2 = (out1 + ffn_out)
            return out2

        layer_in = layers.Input(shape=(seq_len, 8))
        l1 = layers.Embedding(25000, embedding_size)(layer_in)
        l2 = layers.Lambda(lambda x: tf.reduce_sum(x, axis=-2))(l1)
        for i in range(5):
            l2 = encoder(l2)
        middle_model = Model(inputs=layer_in, outputs=l2)

        # finetune 训练模型
        flat = layers.Flatten()(middle_model(middle_model.inputs))
        h = layers.Dense(1024, activation='relu')(flat)
        h = layers.BatchNormalization()(h)
        h = layers.Dense(512, activation='relu')(h)
        sell_out = layers.Dense(1, activation='sigmoid')(h)
        model = Model(inputs=middle_model.inputs, outputs=sell_out)
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
                      metrics=[tf.metrics.AUC()])
        model.summary()

        # pretrain 模型
        mask_size = int(seq_len * self.mask_prob)
        soft_max_dim = len(self.price_bounds) + 1
        vectors = middle_model(middle_model.inputs)
        position = layers.Input(shape=(mask_size, 1), dtype=tf.int32)
        gather_vector = layers.Lambda(lambda x: tf.gather_nd(x[0], x[1], batch_dims=1))([vectors, position])
        l_open = layers.Dense(soft_max_dim, activation='softmax')(gather_vector)
        l_high = layers.Dense(soft_max_dim, activation='softmax')(gather_vector)
        l_low = layers.Dense(soft_max_dim, activation='softmax')(gather_vector)
        l_close = layers.Dense(soft_max_dim, activation='softmax')(gather_vector)
        pre_model = Model(middle_model.inputs + [position], [l_open, l_high, l_low, l_close])
        pre_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])
        pre_model.summary()
        return model, pre_model

    def batch_feature_generator(self, tdf):
        if self.is_pre_train:
            f0, f1, l0, l1, l2, l3 = [], [], [], [], [], []
            while True:
                for f, l in self.feature_generator(tdf, self.seq_len):
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
            features, labels = [], []
            while True:
                for f, l in self.feature_generator(tdf, self.seq_len):
                    features.append(f)
                    labels.append(l)
                    if len(features) >= self.batch_size:
                        yield np.array(features), np.array(labels)
                        features, labels = [], []

    def feature_generator(self, df, seq_len):
        """
        price: max:2589.0 min:0.6700000166893005
        vol: max:1946133833 min:1
        :param df: 子数据集
        :param seq_len 序列长度
        :return:
        """
        cols = "o,h,l,c,v,t,md,w,m,s,sell".split(",")
        feature_col = "o,h,l,c,v,t,md,w,m,s".split(",")
        label_col = "sell"
        for ts_code in np.unique(df.ts_code):
            tdf = df[df.ts_code == ts_code].sort_values(by='trade_time')[cols]
            feature = tdf[feature_col].values
            label = tdf[label_col].values
            size = len(feature) - seq_len
            for start in range(size):
                if self.is_pre_train:
                    cur_feature = feature[start:start + seq_len].copy()
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
                    yield feature[start:start + seq_len].tolist(), [label[start + seq_len - 1]]

    def build(self):
        self.model, self.pre_model = self.make_model(seq_len=self.seq_len, embedding_size=self.embedding_size)

    def pre_train(self, epochs, workers=8, years=10):
        self.is_pre_train = True
        df = self.load_min_data(years)
        train = df[df.trade_date < 20180101].copy()
        val = df[(df.trade_date > 20180101) & (df.trade_date < 20190101)].copy()
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

    def train(self, epochs, workers=8, years=10):
        self.is_pre_train = False
        df = self.load_min_data(years)
        train = df[df.trade_date < 20180101].copy()
        val = df[(df.trade_date > 20180101) & (df.trade_date < 20190101)].copy()
        train_steps = self.get_steps(train)
        val_steps = self.get_steps(val)
        logging.info(f"train_steps:{train_steps}, val_steps:{val_steps}")

        self.model.fit(self.batch_feature_generator(train),
                       steps_per_epoch=train_steps,
                       epochs=epochs,
                       shuffle=True,
                       validation_data=self.batch_feature_generator(val),
                       validation_steps=val_steps,
                       workers=workers,
                       use_multiprocessing=True)
        model_name = f"model_{datetime.now().strftime('%Y%m%d')}"
        self.model.save(f"model/{model_name}")

    @staticmethod
    def predict(x):
        model = load_model("model/model")
        model.predict(x)
