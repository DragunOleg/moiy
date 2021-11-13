import numpy as np
from numpy.linalg import linalg


def inv_matrix(A_inv, i, x, n):
    l = np.dot(A_inv, x)
    li = l[i]
    if l[i] == 0:
        return
    l[i] = -1
    l_ = [(-1 / li) * x for x in l]
    Q = np.eye(n)
    for j in range(0, n):
        Q[j][i] = l_[j]
    ans = np.dot(Q, A_inv)
    return ans


def dual_simplex_method(a, b, c, y, jb):
    m, n = len(a), len(c)
    ab = (np.array([a[:, j] for j in jb])).transpose()
    ab_inv = linalg.inv(ab)

    koplan = np.dot(y, a) - c
    print("Вводные параметры задачи:")
    print("A=\n", a)
    print("b=", b)
    print("c =", c)
    print("Заданный начальный двойственный план yнач=", y)
    print("Jb =", jb)
    print("Соответствующий ему коплан=", koplan)

    iter = 0
    while True:
        iter += 1
        print('___________________________Итерация', iter, "__________________________")
        print('koplan текущий', koplan)
        kapa_b = np.dot(ab_inv, b)
        print('базисные компоненты псевдоплана χб=', kapa_b)
        if min(kapa_b) >= 0:
            print("Условие KSIb>=0 выполняется")
            kapa = [0] * n
            for j, i in zip(kapa_b, jb):
                kapa[i] = j
            print("\nОптимальный план: \n", kapa)
            cFloat = [float(i) for i in c]
            cx0 = np.dot(cFloat, kapa)
            print("c'x0=", cx0)
            return
        else:
            print("Условие KSIб>=0 не выполняется")
        k = np.argmin(kapa_b)
        j_n = [j for j in range(n) if j not in jb]
        e = np.zeros(m)
        e[k] = 1
        mu = e.dot(ab_inv.dot(a))
        print('Находим числа μ=', mu)
        if all(mu[i] >= 0 for i in j_n):
            print("Задача несовместна")
            return

        sigma = [np.inf] * n
        for j in j_n:
            if mu[j] < 0:
                sigma[j] = -koplan[j] / mu[j]
        print('Вычисляем шаги σ', sigma)
        j0 = np.argmin(sigma)
        print('j0=', j0 + 1)
        y = y + (sigma[j0] * e).dot(b)
        print('Построим новый двойственный план y_new=', y)
        koplan = koplan + sigma[j0] * mu
        print('Соответствующий ему новый коплан koplan_new =', koplan)
        jb[k] = j0
        print('Новый базис j_new=', jb)
        ab_inv = inv_matrix(ab_inv, k, a[:, j0], m)


def main():
    # 1
    c = [5, 2, 3, -16, 1, 3, - 3, -12]
    b = [-2, -4, -2]
    a = [[-2, -1, 1, -7, 0, 0, 0, 2],
         [4, 2, 1, 0, 1, 5, -1, -5],
         [1, 1, 0, -1, 0, 3, -1, 1]]
    y = [1, 2, -1]
    jb = [1, 2, 3]

    # 5
    c = [32, -12, 66, 76, -5, 77, -76, -7]
    b = [2, 5, -2]
    a = [
        [3, -1, 10, -7, 1, 0, 0, 2],
        [7, -2, 14, 8, 0, 12, -11, 0],
        [1, 1, 0, 1, -4, 3, -1, 1]
    ]
    y = [-3, 7, -1]
    jb = [6, 7, 3]

    # 3
    # c = [12, -2, -6, 20, -18, -5, -7, -20]
    # b = [-2, 8, -2]
    # a = [[-2, -1, 1, -7, 1, 0, 0, 2],
    #      [-4, 2, 1, 0, 5, 1, -1, 5],
    #      [1, 1, 0, 1, 4, 3, 1, 1]]
    # y = [-3, -2, -1]
    # jb = [1, 3, 5]

    a = np.array(a)
    dual_simplex_method(a, b, c, y, jb)


if __name__ == '__main__':
    main()
