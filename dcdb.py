from inspect import currentframe, getframeinfo
import traceback, sys, code
import time

def cp():
    cf = currentframe()
    print 'checkpoint. line: {}, file: {}'.format(cf.f_back.f_lineno,cf.f_back.f_code.co_filename)

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

