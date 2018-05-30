import math


def combination(n, k):
    f = math.factorial
    return f(n) / f(k) / f(n - k)


def combinations_sum(n):
    summ = 0
    ceil = int(math.ceil(n / 2))
    for i in range(ceil):
        print(i)
        summ += 2 * combination(n, i)

    if (n % 2 == 0):
        summ += combination(n, n / 2)
    return int(summ)
