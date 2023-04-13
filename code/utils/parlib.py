#!/usr/bin/python

import multiprocessing as mp
from multiprocessing import Manager
from time import sleep
import os
import signal


def worker_subprocess(function, devices, idx, lockd, results, lockr,
                      pids, lockp, args, kwargs, *others):
        device = None
        while device is None:
            with lockd:
                try:
                    device = devices.pop()
                except IndexError:
                    pass
            sleep(2)
        with lockp:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                    # print(f'{pid}-killing by {os.getpid()}')
                    pids.remove(pid)
                except ProcessLookupError:
                    pass
        # print(args, kwargs, device)
        output = function(*args, **kwargs, device=device, idx=idx)
        with lockd:
            devices.append(device)
        with lockr:
            results.append(output)
        with lockp:
            # print(os.getpid())
            pids.append(os.getpid())

def parallel_identical(function, *args, nruns=1, njobs=1, gpus=0, **kwargs):

    manager = Manager()
    devices = manager.list(['cuda:' + str(i%gpus)  if gpus !=-1
                            else 'cpu' for i in range(njobs)])
    results = manager.list()
    pids = manager.list()
    lockd = manager.Lock()
    lockr = manager.Lock()
    lockp = manager.Lock()
    poll = [mp.Process(target=worker_subprocess,
                       args=(function, devices, i,
                             lockd, results, lockr,
                             pids, lockp, args,
                             kwargs, i))
            for i in range(nruns)]
    for p in poll:
        p.start()
    for p in poll:
        p.join()

    return list(results)
