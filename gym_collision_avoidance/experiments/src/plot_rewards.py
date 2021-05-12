import os
import csv
from datetime import datetime

import matplotlib.pyplot as plt

runs = (
    "Ntree30__Ncycles5__mcts_cp1.0__horizon4__xdt5_",
    "Ntree30__Ncycles5__mcts_cp2.0__horizon4__xdt5_",
    "Ntree30__Ncycles10__mcts_cp1.0__horizon4__xdt5_",
    "Ntree30__Ncycles10__mcts_cp2.0__horizon4__xdt5_",
    "Ntree30__Ncycles15__mcts_cp1.0__horizon4__xdt5_",
    "Ntree30__Ncycles15__mcts_cp2.0__horizon4__xdt5_"

)

if __name__ == '__main__':
    dir = os.path.dirname(os.path.realpath(__file__)) + '/../results/'
    results = []
    for run in runs:
        filename = dir + run + "/rewards.csv"
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            results.append((list(map(float, data[0])),run))

    fig = plt.figure()
    plt.rc('font', size=10)
    fig.set_size_inches(10, 5)
    ax = fig.add_subplot(111)
    for rewards, name in results:
        ax.plot(rewards, label=name)
    ax.set_xlabel('timesteps')
    ax.set_ylabel('cum. rewards [bits]')

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.legend()
    fig.tight_layout()

    dateObj = datetime.now()
    timestamp = dateObj.strftime("%Y%m%d_%H%M%S")
    fig.savefig(os.path.dirname(os.path.realpath(__file__)) + '/../results/__compare/_' + timestamp + ".png")