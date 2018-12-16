def get_sqrt_seq(x, base = 2):
    import math
    l = []
    for i in range(1, int(math.log(x, base))):
        _ = int(x**(1/base**i))
        if _ > base:
            l.append(_)
    return l
