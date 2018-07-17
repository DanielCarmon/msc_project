from inspect import currentframe, getframeinfo
frameinfo = getframeinfo(currentframe())

def cp():
    cf = currentframe()
    print 'checkpoint. line: {}, file: {}'.format(cf.f_back.f_lineno,cf.f_back.f_code.co_filename)
