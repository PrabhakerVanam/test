
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


def gcd(x, y):
    if y == 0:
        return x
    else:
        return gcd(y, x % y)


def counter(c):
    return dict((i, c.count(i)) for i in c)


def cycle_count(c, n):
    cc = factorial(n)
    #print("cycle_count c", c)
    for a, b in counter(c).items():
        #print("a,b ", a,b)
        cc //= (a**b)*factorial(b)
    return cc


def cycle_partitions(n, i=1):
    yield [n]
    #print("cycle_partitions",n, i)
    for i in range(i, n//2+1):
        for p in cycle_partitions(n-i, i):
            yield [i]+p


def solution(w, h, s):
    grid = 0
    for cpw in cycle_partitions(w):
        #print("cpw",cpw)
        for cph in cycle_partitions(h):
            #print("cph", cph)
            m = cycle_count(cpw, w)*cycle_count(cph, h)
            grid += m*(s**sum([sum([gcd(i, j) for i in cpw]) for j in cph]))
    return str(grid//(factorial(w)*factorial(h)))


print(solution(4, 4, 5))

#Test Cases
'''
 2, 2, 2 - 7
 3, 3, 3 - 738
 3, 3, 4 - 8240
 3, 3, 5 - 57675
 4, 4, 4 - 7880456
 4, 4, 5 - 270656150
 5, 5, 5 - 20834113243925 
 '''