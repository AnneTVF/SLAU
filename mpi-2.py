import numpy as np
import time

# Это код для заданий 4 и 5.
# Тут мы решаем одну и ту же систему четырьмя разными методами, а потом строим графики.
# Есть метод простой итерации (Ричардсона), чебышёвское ускорение к нему, Якоби, Гаусса-Зейделя.
class SparseMatrixCSR:
    """
    Хранение разреженной матрицы (CSR).
    """

    def __init__(self, data, indices, indptr, shape):
        self.data = np.array(data, dtype=float)
        self.indices = np.array(indices, dtype=int)
        self.indptr = np.array(indptr, dtype=int)
        self.shape = shape

    def dot(self, x):
        """Это умножение матрицы на вектор (A @ x) без перебора нулей."""
        result = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            total = 0.0
            for j in range(row_start, row_end):
                total += self.data[j] * x[self.indices[j]]
            result[i] = total
        return result

    @classmethod
    def from_dense(cls, dense_matrix, eps=1e-12):
        """Создание разреженной матрицы из плотной."""
        data = []
        indices = []
        indptr = [0]

        rows, cols = dense_matrix.shape
        for i in range(rows):
            row_nnz = 0
            for j in range(cols):
                if abs(dense_matrix[i, j]) > eps:
                    data.append(dense_matrix[i, j])
                    indices.append(j)
                    row_nnz += 1
            indptr.append(indptr[-1] + row_nnz)

        return cls(data, indices, indptr, dense_matrix.shape)

    def get_diagonal(self):
        """Извлечение диагональных элементов."""
        n = self.shape[0]
        diag = np.zeros(n)
        for i in range(n):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            for j in range(row_start, row_end):
                if self.indices[j] == i:
                    diag[i] = self.data[j]
                    break
        return diag


def gauss_seidel_sparse(A_csr, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Метод Гаусса-Зейделя для разреженных матриц.
    """
    n = A_csr.shape[0]

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    b = np.array(b, dtype=float)

    # Извлекаем диагональные элементы и сохраняем их позиции для быстрого доступа
    diag = np.zeros(n)
    diag_positions = np.full(n, -1, dtype=int)

    for i in range(n):
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]
        for j in range(row_start, row_end):
            if A_csr.indices[j] == i:
                diag[i] = A_csr.data[j]
                diag_positions[i] = j
                break
        if diag[i] == 0:
            raise ValueError("Нулевой диагональный элемент в строке", i)

    # Итерационный процесс Гаусса-Зейделя
    for iteration in range(max_iter):
        max_diff = 0.0

        # Последовательное обновление компонент
        for i in range(n):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]

            # Вычисляем сумму A[i,:] @ x, исключая диагональ
            sum_ax = 0.0
            for j in range(row_start, row_end):
                if j != diag_positions[i]:
                    sum_ax += A_csr.data[j] * x[A_csr.indices[j]]

            # Новое значение x[i]
            x_new_i = (b[i] - sum_ax) / diag[i]

            # Отслеживаем максимальное изменение
            diff = abs(x_new_i - x[i])
            max_diff = max(max_diff, diff)

            # Немедленное обновление
            x[i] = x_new_i

        # Проверка сходимости
        if max_diff < tol:
            return x

    return x


def simple_iteration_jacobi(A_csr, b, x0=None, tol=1e-6, max_iter=1000):
    """Метод Якоби для разреженных матриц."""
    n = A_csr.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    b = np.array(b, dtype=float)

    diag = A_csr.get_diagonal()

    for iteration in range(max_iter):
        Ax = A_csr.dot(x)
        r = b - Ax
        residual_norm = np.linalg.norm(r, ord=np.inf)

        if residual_norm < tol:
            return x

        x = x + r / diag

    return x


def simple_iteration_richardson(A_csr, b, tau=None, x0=None, tol=1e-6, max_iter=1000):
    """Метод простой итерации (Ричардсона)."""
    n = A_csr.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    b = np.array(b, dtype=float)

    if tau is None:
        # Оценка максимального собственного значения через степенной метод
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        for _ in range(100):
            v_new = A_csr.dot(v)
            v = v_new / np.linalg.norm(v_new)

        lambda_max = np.dot(v, A_csr.dot(v)) / np.dot(v, v)
        tau = 1.8 / lambda_max  # Более агрессивная оценка

    for iteration in range(max_iter):
        r = b - A_csr.dot(x)
        residual_norm = np.linalg.norm(r, ord=np.inf)

        if residual_norm < tol:
            return x

        x = x + tau * r

    return x


def richardson_chebyshev(A_csr, b, lambda_min=None, lambda_max=None, x0=None, tol=1e-6, max_iter=1000):
    """
    Метод Ричардсона с чебышевским ускорением.
    Использует оптимальные чебышевские параметры для ускорения сходимости.
    """
    n = A_csr.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    b = np.array(b, dtype=float)

    # Оценка границ спектра если не заданы
    if lambda_max is None:
        # Степенной метод для lambda_max
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        for _ in range(100):
            v_new = A_csr.dot(v)
            v = v_new / np.linalg.norm(v_new)
        lambda_max = np.dot(v, A_csr.dot(v)) / np.dot(v, v)

    if lambda_min is None:
        # Для нашей матрицы lambda_min можно оценить через отношение
        # Для пятидиагональной матрицы lambda_min примерно равна 0.5-1.0
        lambda_min = lambda_max * 0.05  # Более реалистичная оценка

    # Чебышевские параметры
    theta = (lambda_max + lambda_min) / (lambda_max - lambda_min)
    rho = (1 - lambda_min / lambda_max) / (1 + lambda_min / lambda_max)

    # Итерационный процесс
    r = b - A_csr.dot(x)
    p = np.zeros(n)

    for k in range(max_iter):
        residual = np.linalg.norm(r, ord=np.inf)

        if residual < tol:
            return x

        # Вычисление чебышевского параметра tau_k
        if k == 0:
            tau = 2.0 / (lambda_max + lambda_min)
        else:
            tau = 4.0 * theta / (lambda_max - lambda_min) / (2 * theta - rho * tau)

        # Шаг метода с использованием предыдущего направления
        if k == 0:
            x = x + tau * r
        else:
            beta = (tau / tau_old) * (rho / 2.0)
            p = r + beta * p
            x = x + tau * p

        tau_old = tau
        r = b - A_csr.dot(x)

        if residual > 1e10:
            break

    return x


class IterationHistory:
    """Технический класс для сбора статистики итераций."""

    def __init__(self):
        self.iterations = []
        self.residuals = []
        self.times = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def record(self, iteration, residual):
        self.iterations.append(iteration)
        self.residuals.append(residual)
        self.times.append(time.time() - self.start_time)


def solve_with_history(method_func, A_csr, b, **kwargs):
    """
    Тут мы решаем систему всеми методами и собираем историю сходимости, чтобы нарисовать графики.
    """
    n = A_csr.shape[0]
    x = np.zeros(n) if kwargs.get('x0') is None else np.array(kwargs.get('x0'), dtype=float)
    b = np.array(b, dtype=float)

    history = IterationHistory()
    history.start()

    method_name = method_func.__name__
    tol = kwargs.get('tol', 1e-6)
    max_iter = kwargs.get('max_iter', 1000)

    if 'gauss_seidel' in method_name:
        # Гаусс-Зейдель с историей
        diag = np.zeros(n)
        diag_positions = np.full(n, -1, dtype=int)

        for i in range(n):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            for j in range(row_start, row_end):
                if A_csr.indices[j] == i:
                    diag[i] = A_csr.data[j]
                    diag_positions[i] = j
                    break

        for iteration in range(max_iter):
            max_diff = 0.0

            for i in range(n):
                row_start = A_csr.indptr[i]
                row_end = A_csr.indptr[i + 1]

                sum_ax = 0.0
                for j in range(row_start, row_end):
                    if j != diag_positions[i]:
                        sum_ax += A_csr.data[j] * x[A_csr.indices[j]]

                x_new_i = (b[i] - sum_ax) / diag[i]
                max_diff = max(max_diff, abs(x_new_i - x[i]))
                x[i] = x_new_i

            r = b - A_csr.dot(x)
            residual = np.linalg.norm(r, ord=np.inf)
            history.record(iteration + 1, residual)

            if max_diff < tol:
                break

    elif 'jacobi' in method_name:
        # Якоби с историей
        diag = A_csr.get_diagonal()

        for iteration in range(max_iter):
            r = b - A_csr.dot(x)
            residual = np.linalg.norm(r, ord=np.inf)
            history.record(iteration + 1, residual)

            if residual < tol:
                break

            x = x + r / diag

    elif 'richardson' in method_name and 'chebyshev' not in method_name:
        # Ричардсон с историей
        tau = kwargs.get('tau')

        if tau is None:
            # Оценка максимального собственного значения
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            for _ in range(100):
                v_new = A_csr.dot(v)
                v = v_new / np.linalg.norm(v_new)

            lambda_max = np.dot(v, A_csr.dot(v)) / np.dot(v, v)
            tau = 1.8 / lambda_max

        for iteration in range(max_iter):
            r = b - A_csr.dot(x)
            residual = np.linalg.norm(r, ord=np.inf)
            history.record(iteration + 1, residual)

            if residual < tol:
                break

            x = x + tau * r

            if residual > 1e10:
                break

    elif 'chebyshev' in method_name:
        # Ричардсон с чебышевским ускорением
        lambda_min = kwargs.get('lambda_min')
        lambda_max = kwargs.get('lambda_max')

        # Оценка границ спектра
        if lambda_max is None:
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            for _ in range(100):
                v_new = A_csr.dot(v)
                v = v_new / np.linalg.norm(v_new)
            lambda_max = np.dot(v, A_csr.dot(v)) / np.dot(v, v)

        if lambda_min is None:
            lambda_min = lambda_max * 0.05

        theta = (lambda_max + lambda_min) / (lambda_max - lambda_min)
        rho = (1 - lambda_min / lambda_max) / (1 + lambda_min / lambda_max)

        r = b - A_csr.dot(x)
        p = np.zeros(n)

        for k in range(max_iter):
            residual = np.linalg.norm(r, ord=np.inf)
            history.record(k + 1, residual)

            if residual < tol:
                break

            if k == 0:
                tau = 2.0 / (lambda_max + lambda_min)
                x = x + tau * r
            else:
                tau = 4.0 * theta / (lambda_max - lambda_min) / (2 * theta - rho * tau_old)
                beta = (tau / tau_old) * (rho / 2.0)
                p = r + beta * p
                x = x + tau * p

            tau_old = tau
            r = b - A_csr.dot(x)

            if residual > 1e10:
                break

    return x, history


if __name__ == "__main__":
    # Создаем тестовую матрицу
    n = 100

    # Создаем пятидиагональную матрицу
    dense_A = np.zeros((n, n))
    for i in range(n):
        dense_A[i, i] = 4.0
        if i > 0:
            dense_A[i, i - 1] = -1.0
        if i < n - 1:
            dense_A[i, i + 1] = -1.0
        if i > 1:
            dense_A[i, i - 2] = -0.5
        if i < n - 2:
            dense_A[i, i + 2] = -0.5

    # Точное решение и правая часть
    x_exact = np.sin(np.linspace(0, 2 * np.pi, n))
    b = dense_A @ x_exact

    # Преобразуем в разреженную матрицу
    A_sparse = SparseMatrixCSR.from_dense(dense_A)

    print("Характеристики матрицы:")
    print("  Размер:", n, "×", n)
    print("  Ненулевых элементов:", len(A_sparse.data))
    print("  Плотность:", round(len(A_sparse.data) / (n * n) * 100, 2), "%")

    TOL = 1e-8
    MAX_ITER = 5000

    results = {}
    histories = {}

    # 1. Метод Якоби
    print("\n1. МЕТОД ЯКОБИ")
    start = time.time()
    x_jacobi, hist_jacobi = solve_with_history(
        simple_iteration_jacobi, A_sparse, b, tol=TOL, max_iter=MAX_ITER
    )
    time_jacobi = time.time() - start
    results['Якоби'] = {'x': x_jacobi, 'time': time_jacobi}
    histories['Якоби'] = hist_jacobi
    print("Итераций:", len(hist_jacobi.iterations),
          "Время:", round(time_jacobi, 4), "с",
          "Невязка:", format(hist_jacobi.residuals[-1], ".2e"))

    # 2. Метод Гаусса-Зейделя
    print("\n2. МЕТОД ГАУССА-ЗЕЙДЕЛЯ")
    start = time.time()
    x_gs, hist_gs = solve_with_history(
        gauss_seidel_sparse, A_sparse, b, tol=TOL, max_iter=MAX_ITER
    )
    time_gs = time.time() - start
    results['Гаусс-Зейдель'] = {'x': x_gs, 'time': time_gs}
    histories['Гаусс-Зейдель'] = hist_gs
    print("Итераций:", len(hist_gs.iterations),
          "Время:", round(time_gs, 4), "с",
          "Невязка:", format(hist_gs.residuals[-1], ".2e"))

    # 3. Метод Ричардсона
    print("\n3. МЕТОД РИЧАРДСОНА")
    start = time.time()
    x_rich, hist_rich = solve_with_history(
        simple_iteration_richardson, A_sparse, b, tol=TOL, max_iter=MAX_ITER
    )
    time_rich = time.time() - start
    results['Ричардсон'] = {'x': x_rich, 'time': time_rich}
    histories['Ричардсон'] = hist_rich
    print("Итераций:", len(hist_rich.iterations),
          "Время:", round(time_rich, 4), "с",
          "Невязка:", format(hist_rich.residuals[-1], ".2e"))

    # 4. Метод Ричардсона с чебышевским ускорением
    print("\n4. МЕТОД РИЧАРДСОНА (ЧЕБЫШЕВ)")
    start = time.time()
    x_cheb, hist_cheb = solve_with_history(
        richardson_chebyshev, A_sparse, b, tol=TOL, max_iter=MAX_ITER
    )
    time_cheb = time.time() - start
    results['Ричардсон (Чебышев)'] = {'x': x_cheb, 'time': time_cheb}
    histories['Ричардсон (Чебышев)'] = hist_cheb
    print("Итераций:", len(hist_cheb.iterations),
          "Время:", round(time_cheb, 4), "с",
          "Невязка:", format(hist_cheb.residuals[-1], ".2e"))

    # Визуализация
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'Якоби': 'blue', 'Гаусс-Зейдель': 'red', 'Ричардсон': 'purple', 'Ричардсон (Чебышев)': 'orange'}

    # зависимость невязки от номера итерации
    ax1 = axes[0]
    for method_name, hist in histories.items():
        ax1.semilogy(hist.iterations, hist.residuals, color=colors[method_name], label=method_name, linewidth=2)

    ax1.set_xlabel('Номер итерации')
    ax1.set_ylabel('Невязка (норма-inf)')
    ax1.set_title('Сходимость по итерациям')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # зависимость невязки от времени
    ax2 = axes[1]
    for method_name, hist in histories.items():
        ax2.semilogy(hist.times, hist.residuals, color=colors[method_name], label=method_name, linewidth=2)

    ax2.set_xlabel('Время (сек)')
    ax2.set_ylabel('Невязка (норма-inf)')
    ax2.set_title('Сходимость по времени')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()