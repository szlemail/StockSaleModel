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
    pid = [[c for c in s.split(" ") if c != ''][1] for s in cmdout.split("\n")[0:-1]]
    print([[c for c in s.split(" ") if c != ''][1] for s in cmdout.split("\n")[0:-1]])
    for p in pid:
        try:
            print(os.popen(f"kill -9 {p}").read())
        except:
            print(f"kill pid:{p} error")


if __name__ == "__main__":
    x = """
   ROUND 0, 5YEAR: loss: 1.1579 - auc: 0.6654 - val_loss: 0.5818 - val_auc: 0.6966
   ROUND 1, 1YEAR: loss: 0.5877 - auc: 0.6847 - val_loss: 0.5833 - val_auc: 0.7052
   ROUND 2, 1YEAR: loss: 0.5861 - auc: 0.6944 - val_loss: 0.5807 - val_auc: 0.7046
   ROUND 3, 1YEAR: loss: 0.5840 - auc: 0.6925 - val_loss: 0.5911 - val_auc: 0.6928
   ROUND 4, 1YEAR: loss: 0.5920 - auc: 0.6856 - val_loss: 0.5908 - val_auc: 0.6949
   """
    run(x)
    killall()
