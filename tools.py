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
    ROUND 0, 5YEAR: loss: 1.1406 - auc: 0.6897 - val_loss: 0.5843 - val_auc: 0.6959
    ROUND 1, 1YEAR: loss: 0.5845 - auc: 0.6877 - val_loss: 0.5805 - val_auc: 0.7002
    ROUND 2, 1YEAR: loss: 0.5829 - auc: 0.6963 - val_loss: 0.5798 - val_auc: 0.6993
    ROUND 3, 1YEAR: loss: 0.5808 - auc: 0.6945 - val_loss: 0.5911 - val_auc: 0.6905
    ROUND 4, 1YEAR: loss: 0.5890 - auc: 0.6890 - val_loss: 0.5883 - val_auc: 0.6944
   """
    run(x)
    killall()
