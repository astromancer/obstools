import multiprocessing as mp
from recipes.parallel.synced import SyncedCounter, SyncedArray

class Foo():
    def __init__(self):
        self.count = SyncedCounter(1)

    def inc(self, i):
        self.count.inc()
        return i

# def init()
#     foo = Foo()

rvec = SyncedArray(shape=(3, 2))
count = SyncedCounter(0)

def inc(i):
    count.inc()

with mp.Pool(
        # initializer=init,
             # initargs=(config_worker, )
            ) as pool:
    results = pool.map(inc, range(100))

    print(count.get_value())