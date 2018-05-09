import math

def C(n, k):
    f = math.factorial
    return f(n) / f(k) / f(n - k)

def C_sum(n):
    sum=0
    ceil=math.ceil(n/2)
    for i in range(ceil):
        sum+=2*C(n,i)
    return int(sum)