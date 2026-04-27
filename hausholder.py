import numpy as np


def householder_qr(A):
    """
    Тут сам алгоритм Хаусхолдера.

    На вход:
    A : numpy.ndarray
        Исходная матрица размера m x n (m >= n)

    На выход:
    Q : numpy.ndarray
        Ортогональная матрица размера m x m
    R : numpy.ndarray
        Верхняя треугольная матрица размера m x n
    """
    R = A.copy().astype(float)
    m, n = R.shape
    Q = np.eye(m)

    for k in range(min(m, n)):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)

        if norm_x < 1e-14:
            continue

        e1 = np.zeros_like(x)
        e1[0] = 1.0

        # Выбираем знак для численной устойчивости
        if x[0] > 0:
            u = x + norm_x * e1
        else:
            u = x - norm_x * e1

        v = u / np.linalg.norm(u)

        # Считаем сами формулы Хаусхолдера
        R_k = R[k:, k:]
        R[k:, k:] = R_k - 2 * np.outer(v, v @ R_k)

        Q_k = Q[:, k:]
        Q[:, k:] = Q_k - 2 * np.outer(Q_k @ v, v)

    return Q, R


def householder_qr_check(A):
    """
    Экономная версия QR-разложения для проверки и решения систем.

    Параметры:
    A : numpy.ndarray
        Исходная матрица размера m x n

    Возвращает:
    Q : numpy.ndarray
        Ортогональная матрица размера m x n
    R : numpy.ndarray
        Верхняя треугольная матрица размера n x n
    """
    Q_full, R_full = householder_qr(A)
    m, n = A.shape
    return Q_full[:, :n], R_full[:n, :]


def solve_householder_qr(A, b):
    """
    Непосредственно решение системы линейных уравнений Ax = b методом QR-разложения.

    Параметры:
    A : numpy.ndarray
        Матрица коэффициентов  n x n
    b : numpy.ndarray
        Вектор правой части размерностью n

    Returns:
    x : numpy.ndarray
        Решение системы
    """
    Q, R = householder_qr_check(A)

    # y = Q^T @ b
    y = Q.T @ b

    # Обратная подстановка для Rx = y
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    return x


# Проверка
if __name__ == "__main__":
    print("Разложение методом Хаусхолдера")

    # Тестовая матрица
    np.random.seed(42)
    m, n = 5, 3
    A = np.random.randn(m, n)

    print(f"Исходная матрица A ({m}x{n}):")
    print(np.round(A, 4))

    # Выводим QR-разложение
    Q_full, R_full = householder_qr(A)
    Q, R = householder_qr_check(A)

    print(f"\nПолная Q ({m}x{m}):")
    print(np.round(Q_full, 4))
    print(f"\nЭкономная Q ({m}x{n}):")
    print(np.round(Q, 4))
    print(f"\nR ({n}x{n}):")
    print(np.round(R, 4))

    print("\n" + "=" * 60)
    print("Решение системы линейных уравнений")
    print("=" * 60)

    # Создаем тестовую систему
    n_solve = 3
    A_solve = np.array([
        [4, 2, 1],
        [2, 5, 3],
        [1, 3, 6]
    ])
    b_solve = np.array([1, 2, 3])

    print(f"Матрица A:\n{A_solve}")
    print(f"Вектор b: {b_solve}")

    # Решаем систему
    x = solve_householder_qr(A_solve, b_solve)
    print(f"\nРешение x: {np.round(x, 6)}")

    # Проверяем
    check = A_solve @ x
    print(f"Проверка A @ x: {np.round(check, 6)}")
    print(f"Должно быть:    {b_solve}")