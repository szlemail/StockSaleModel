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
   ROUND 0, 5YEAR: loss: 1.1394 - auc: 0.6952 - val_loss: 0.5766 - val_auc: 0.6980
   ROUND 1, 1YEAR: loss: 0.5771 - auc: 0.6952 - val_loss: 0.5764 - val_auc: 0.7055
   ROUND 2, 1YEAR: loss: 0.5757 - auc: 0.7049 - val_loss: 0.5741 - val_auc: 0.7063
   ROUND 3, 1YEAR: loss: 0.5738 - auc: 0.7040 - val_loss: 0.5848 - val_auc: 0.6946
   ROUND 4, 1YEAR: loss: 0.5819 - auc: 0.6982 - val_loss: 0.5853 - val_auc: 0.6962
   """
    run(x)
    killall()
