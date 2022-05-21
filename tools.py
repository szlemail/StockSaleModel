import pandas as pd
import numpy as np


def run(x):
    data = x.split("\n")
    data = [d.split(" ") for d in data if d.strip() != ""]
    df = pd.DataFrame(data)
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
            print(f"{df[c].mean():0.4f}", end=' ')
        except:
            print(df[c][0], end=' ')


def killall():
    import os
    cmdout = os.popen("ps aux | grep 'python main.py'").read()
    print(cmdout)
    pid = [s.split("  ")[4] for s in cmdout.split("\n")[0:-1]]
    for p in pid:
        try:
            print(os.popen(f"kill -9 {p}").read())
        except:
            print(f"kill pid:{p} error")


if __name__ == "__main__":
    x = """
   ROUND 0, 5YEAR: loss: 0.7280 - auc: 0.7064 - val_loss: 0.5754 - val_auc: 0.6982
   ROUND 1, 1YEAR: loss: 0.5787 - auc: 0.7020 - val_loss: 0.5869 - val_auc: 0.7020
   ROUND 2, 1YEAR: loss: 0.5830 - auc: 0.7160 - val_loss: 0.5977 - val_auc: 0.6932
   """
    run(x)
    killall()
