def solution(M, F):
    m = int(M)
    f = int(F)

    max_value = 10 ** 50

    if m > max_value or f > max_value:
        return "impossible"

    # print(max_value)

    steps = 0
    while (m != f):

        if is_possible(m, f):
            # print("possible")
            q = 0
            if m > f and f > 1:
                q = m // f
                m = m - (f * q)
                # print(f"m>f {m, f, steps}")
            elif f > m and m > 1:
                q = f // m
                f = f - (m * q)
                # print(f"f>m {m, f, steps}")
            steps = steps + q
            # print(f"{m, f, steps}")
            if m == 1 and f != 1:
                steps = steps + f - 1
                f = 1
                break
            elif m != 1 and f == 1:
                steps = steps + m - 1
                m = 1
                break
        else:
            break

    if m == 1 and f == 1:
        # print(f"f=m=1 {m, f, steps}")
        return str(steps)
    else:
        return "impossible"


def is_possible(m, f):
    # print(m,f)
    if m <= 0 or f <= 0 or (f != 1 and f == m):
        return False

    return True


print(solution('1', str(10 ** 50)))
print("----------------------------------------------")
print(solution('4', '7'))
print("----------------------------------------------")
print(solution('2', '1'))
print("----------------------------------------------")
print(solution('2', '4'))
print("----------------------------------------------")
