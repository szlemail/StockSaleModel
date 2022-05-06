import logging
import pandas as pd
from cores.transformer import Transformer
import tensorflow as tf
import os

# 指定第一块gpu可用
os.environ["cuda_visible_devices"] = "0"  # 指定gpu的第二种方法
phy_gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(phy_gpu[0], True)
logic_gpu = tf.config.list_logical_devices(['GPU'])
print(phy_gpu, logic_gpu)




if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    logging.info("start")
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 30)
    # 显示所有行
    pd.set_option('display.max_rows', 100)  # 最多显示10行
    transformer = Transformer()
    data = transformer.load_min_data(years=10)
    data.head(3)
    transformer.build()
    transformer.pre_train(data, epochs=1, workers=8)
    transformer.train(data, epochs=1, workers=8)

    # model.save("model")


