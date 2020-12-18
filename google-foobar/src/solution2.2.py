from fractions import Fraction


def solution(pegs):
    l = len(pegs)
    if ((not pegs) or l == 1):
        return [-1, -1]

    iseven = True if (l % 2 == 0) else False
    sum = (- pegs[0] + pegs[l - 1]) if iseven else (- pegs[0] - pegs[l - 1])

    if (l > 2):
        for i in range(1, l - 1):
            sum += 2 * (-1) ** (i + 1) * pegs[i]

    FirstRds = Fraction(2 * (float(sum) / 3 if iseven else sum)).limit_denominator()

    currentrds = FirstRds
    for idx in range(0, l - 2):
        currentdis = pegs[idx + 1] - pegs[idx]
        nextrds = currentdis - currentrds
        if (currentrds < 1 or nextrds < 1):
            return [-1, -1]
        else:
            currentrds = nextrds

    return [FirstRds.numerator, FirstRds.denominator]


print(solution([4, 30, 50]))

print(solution([4, 17, 50]))

print(solution([0, 20]))

print(solution([0, 21]))

print(solution([0,20,45,80]))

print(solution([0,20,45,80]))

print(solution([0,30,50,75]))

print(solution([4,10,12,64,90,100]))


