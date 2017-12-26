import time

time.sleep(2)
with open("./save_graph/scores.txt", 'r') as storage:
    data = list(map(float, storage.readline().split()))

from matplotlib import pylab

fig = pylab.figure(figsize=(20, 20))
pylab.plot(range(len(data)), data)
pylab.savefig("./save_graph/scores.png")
pylab.close()
