from multiprocessing import Process, Pipe
from itertools import izip, chain

def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]

def largeparmap(f,X):
    answer = []
    inc = 250
    start = 0
    end = min(inc, len(X))
    while True:
        #print start
        #print end
        answer.append(parmap(f,X[start:end]))
        #answer.append(map(f,X[start:end]))
        start = end
        if end == len(X):
            break
        end = min(start+inc, len(X))
        #print '--------'
    return list(chain.from_iterable(answer))