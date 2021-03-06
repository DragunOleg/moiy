import numpy as np
from numpy.linalg import linalg


def inv_matrix(A_inv, i, x, n):
    l = np.dot(A_inv, x)
    li = l[i]
    print(l)
    if l[i] == 0:
        print("False")
        return
    print("True")
    l[i] = -1
    l_ = [(-1 / li) * x for x in l]
    # print(l_)

    Q = np.eye(n)
    for j in range(0, n):
        Q[j][i] = l_[j]
    # print(Q)

    ans = np.dot(Q, A_inv)
    return ans


def simplex_method(c, A, b, x, jb):
    iter_count = 0
    n = len(x)
    m = len(jb)
    Ab = np.zeros((m, m))
    print("Матрицы:")
    print("Ab =\n", Ab)
    for i in range(0, m):
        for j in range(0, m):
            Ab[j][i] = A[j][jb[i]]

    Ab_inv = linalg.inv(Ab)
    print("Ab_inv=\n", Ab_inv)

    while True:
        iter_count += 1
        print("\nИтерация ", iter_count, "\n")
        cb = [c[i] for i in jb]
        print("cb =", cb)
        u = np.dot(cb, Ab_inv)
        print("Вектр потенциалов u =", u)
        uA = np.dot(u, A)
        print("uA", uA)
        delta = [uA[i] - c[i] for i in range(0, n)]
        print("delta (вектор оценок)", delta)

        is_optimal = True
        j0 = -1
        for i in range(len(delta)):
            if i not in jb:
                if delta[i] < 0:
                    is_optimal = False
                    j0 = i
                    break

        if is_optimal:
            print("Оптимальный план x: ", x)
            print("Базисные индексы: ", jb)
            s = 0
            for i in range(0, n):
                s += x[i] * c[i]
            print("c`x0 = ", s)
            return

        print("Выберем индекс j0: ", j0)

        Aj0 = [A[i][j0] for i in range(0, m)]
        z = np.dot(Ab_inv, Aj0)
        print("Построим вектор z", z)

        tetta = []
        for i in range(0, m):
            if z[i] > 0:
                tetta.append(x[jb[i]] / z[i])
            else:
                tetta.append(np.inf)

        print("\nTetta:\n", tetta)
        tetta0 = min(tetta)
        print("Tetta0:\n", tetta0)
        tetta_index = tetta.index(tetta0)
        print("Tetta index:\n", tetta_index)

        if tetta0 == np.inf:
            print("Неограничено")
            return

        jb[tetta_index] = j0

        for i in range(0, n):
            if i not in jb:
                x[i] = 0

        x[j0] = tetta0
        for i in range(0, m):
            if i != tetta_index:
                x[jb[i]] = x[jb[i]] - tetta0 * z[i]

        print("Новый план: ", x, "\n")
        new_col = [A[i][j0] for i in range(0, m)]
        Ab_inv = np.array(inv_matrix(Ab_inv, tetta_index, new_col, m))
        print("\nМатрица обратная базисной\n", Ab_inv, "\n")
        for i in range(0, m):
            Ab[i][tetta_index] = A[i][j0]
        print(Ab)
        print()


def main():
    # var1
    # c = [-5, 2, 3, -4, -6, 0, 1, -5]
    # A = [[0, 1, 4, 1, 0, -8, 1, 5],
    #      [0, -1, 0, -1, 0, 0, 0, 0],
    #      [0, 2, -1, 0, -1, 3, -1, 0],
    #      [1, 1, 1, 1, 0, 3, 1, 1]]
    # b = [36, -11, 10, 20]
    # x = [4, 5, 0, 6, 0, 0, 0, 5]
    # jb = [0, 1, 3, 7]

    # var3
    c = [-6, -9, -5, 2, -6, 0, 1, 3]
    A = [[0, -1, 1, -7.5, 0, 0, 0, 2],
         [0, 2, 1, 0, -1, 3, -1.5, 0],
         [1, -1, 1, -1, 0, 3, 1, 1]]
    b = [6, 1.5, 10]
    x = [4, 0, 6, 0, 4.5, 0, 0, 0]
    jb = [0, 2, 4]

    simplex_method(c, A, b, x, jb)


if __name__ == '__main__':
    main()
