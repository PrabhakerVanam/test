def solution(n):
    n = int(n)
    steps = 0
    if n == 1:
        return steps
    while n != 1:
        if n % 2 == 0:
            n = n / 2
        elif (n % 4 == 1) or (n == 3):
            n = n - 1
        else:
            n = n + 1
        steps += 1
    return steps


print(solution('15'))

print(solution('4'))
