from inspect import currentframe, getframeinfo
from sys import getsizeof
import traceback, sys, code
import datetime
import time
import math
import os

def cp():
    '''
    prints the line number and filename wherever placed in code
    '''
    cf = currentframe()
    print 'checkpoint. line: {}, file: {}'.format(cf.f_back.f_lineno,cf.f_back.f_code.co_filename)

def echo_cp():
    '''
    like cp but instead echoes checkpoint info to '~/echo_cp.txt'
    '''
    cf = currentframe()
    ckpt_info = 'checkpoint. line: {}, file: {}'.format(cf.f_back.f_lineno,cf.f_back.f_code.co_filename)+str(now())
    os.system('echo {} >> ~/echo_cp.txt'.format(ckpt_info))


def humanizeFileSize(filesize):
    filesize = abs(filesize)
    if (filesize==0):
        return "0 Bytes"
    p = int(math.floor(math.log(filesize, 2)/10))
    return "%0.2f %s" % (filesize/math.pow(1024,p), ['Bytes','KiB','MiB','GiB','TiB','PiB','EiB','ZiB','YiB'][p])

def memsz(arr):
    '''
    returns memory usage of python array in human-readable format
    '''
    return humanizeFileSize(getsizeof(arr))

def now():
    return datetime.datetime.now()

def tic():
    global start
    start = now()

def toc():
    elapsed = now() - start
    min,secs=divmod(elapsed.days * 86400 + elapsed.seconds, 60)
    hour, minutes = divmod(min, 60)
    print 'elapsed: %.2d:%.2d:%.2d' % (hour,minutes,secs)


def count_to(n):
    for i in range(n):
        print str(i+1)+'...'
        time.sleep(1)

if __name__ == '__main__':
    code_f = sys.argv[1]
    other_args = sys.argv[1:]
    try:
        sys.argv = other_args
        execfile(code_f)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
        frame = last_frame().tb_frame
        ns = dict(frame.f_globals)
        ns.update(frame.f_locals)
        code.interact(local=ns)

