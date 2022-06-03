import logging
import pandas as pd
from cores.transformer_pct import TransformerPct
from cores.transformer import Transformer
import tensorflow as tf
import os
import argparse

# 指定第一块gpu可用
os.environ["cuda_visible_devices"] = "0"  # 指定gpu的第二种方法
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
phy_gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(phy_gpu[0], True)
logic_gpu = tf.config.list_logical_devices(['GPU'])
print(phy_gpu, logic_gpu)


def killall():
    import os
    cmdout = os.popen("ps aux | grep 'python main.py'").read()
    pid = [[c for c in s.split(" ") if c != ''][1] for s in cmdout.split("\n")[0:-1]]
    pid_list = " ".join(pid)
    try:
        print(os.popen(f"kill -9 {pid_list}").read())
    except:
        print(f"kill pid:{pid_list} error")


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    logging.info("start")
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 30)
    # 显示所有行
    pd.set_option('display.max_rows', 100)  # 最多显示100行

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="price/pct", default="pct")
    parser.add_argument("-d", "--pre-train-days", help="pre-train-days", default=1750, type=int)
    parser.add_argument("-p", "--pre", help="pretrain", action="store_true")
    parser.add_argument("-n", "--pre-model-name", help="pre-model-name", default=None)
    args = parser.parse_args()
    if args.mode == "pct":
        transformer = TransformerPct()
    else:
        transformer = Transformer()
    transformer.build()
    if args.pre:
        transformer.pre_train(years=13, epochs=1, workers=15, pre_train_days=args.pre_train_days)
    else:
        if args.pre_model_name:
            transformer.load_middel_model(args.pre_model_name)
        transformer.train(years=13, epochs=1, workers=15)
    logging.info("Done")
    killall()
