import logging
import pandas as pd
from cores.transformer import Transformer
import tensorflow as tf
from keras import backend as k
import os

# 指定第一块gpu可用
os.environ["cuda_visible_devices"] = "0"  # 指定gpu的第二种方法
phy_gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(phy_gpu[0], True)
logic_gpu = tf.config.list_logical_devices(['GPU'])
print(phy_gpu, logic_gpu)




if __name__ == '__main__':
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 30)
    # 显示所有行
    pd.set_option('display.max_rows', 100)  # 最多显示10行
    transformer = Transformer()
    transformer.load_data()
    transformer.train(epochs=1, workers=16)

    # model.save("model")


