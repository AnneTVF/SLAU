import numpy as np
import time


# Это код для задания 6 (на базе кода для заданий 4 и 5).
# Тут мы решаем одну и ту же систему шестью разными методами, а потом строим графики.
# Есть метод простой итерации (Ричардсона), чебышёвское ускорение к нему, Якоби, Гаусса-Зейделя,
# симметризованный Гаусс-Зейдель и чебышевское ускорение к нему.
# Есть подозрения, что на матрице, которую мы тут генерируем, симметризованный Гаусс-Зейдель уже настолько хорош, что
# чебышевское ускорение к нему уже излишне и только замедляет работу.
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

    def dot_lower(self, x):
        """Умножение нижней треугольной части (включая половину диагонали) на вектор."""
        result = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            total = 0.0
            for j in range(row_start, row_end):
                col = self.indices[j]
                if col <= i:
                    if col == i:
                        total += 0.5 * self.data[j] * x[col]  # половина диагонали
                    else:
                        total += self.data[j] * x[col]
            result[i] = total
        return result

    def dot_upper(self, x):
        """Умножение верхней треугольной части (включая половину диагонали) на вектор."""
        result = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            total = 0.0
            for j in range(row_start, row_end):
                col = self.indices[j]
                if col >= i:
                    if col == i:
                        total += 0.5 * self.data[j] * x[col]  # половина диагонали
                    else:
                        total += self.data[j] * x[col]
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


def solve_triangular_lower(A_csr, b, x_half=None):
    """
    Решение системы с нижней треугольной матрицей (L + D/2)*x = b
    методом прямой подстановки.
    """
    n = A_csr.shape[0]
    x = np.zeros(n)

    for i in range(n):
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]

        sum_lx = 0.0
        diag_val = 0.0
        for j in range(row_start, row_end):
            col = A_csr.indices[j]
            if col < i:
                sum_lx += A_csr.data[j] * x[col]
            elif col == i:
                diag_val = A_csr.data[j]

        # (D/2 + L) @ x = b  =>  x[i] = (b[i] - sum_lx) / (diag/2)
        x[i] = (b[i] - sum_lx) / (diag_val * 0.5)

    return x


def solve_triangular_upper(A_csr, b):
    """
    Решение системы с верхней треугольной матрицей (D/2 + U)*x = b
    Методом обратной подстановки.
    """
    n = A_csr.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]

        sum_ux = 0.0
        diag_val = 0.0
        for j in range(row_start, row_end):
            col = A_csr.indices[j]
            if col > i:
                sum_ux += A_csr.data[j] * x[col]
            elif col == i:
                diag_val = A_csr.data[j]

        # (D/2 + U) @ x = b  =>  x[i] = (b[i] - sum_ux) / (diag/2)
        x[i] = (b[i] - sum_ux) / (diag_val * 0.5)

    return x


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


def symmetric_gauss_seidel(A_csr, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Симметризованный метод Гаусса-Зейделя.

    Одна итерация состоит из двух полушагов:
    1. Прямой ход: (D + L)*x(k+1/2) = b - U*x(k)
    2. Обратный ход: (D + U)*x(k+1) = b - L*x(k+1/2)

    Эквивалентно: x(k+1) = x(k) + B_sym * (b - A*x(k)),
    где B_sym — симметризованный оператор Гаусса-Зейделя.
    """
    n = A_csr.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    b = np.array(b, dtype=float)

    # Извлекаем диагональ для прямого/обратного хода
    diag = A_csr.get_diagonal()

    for iteration in range(max_iter):
        # Сохраняем старое значение для проверки сходимости
        x_old = x.copy()

        # Прямой ход (forward sweep) — как обычный Гаусс-Зейдель
        for i in range(n):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]

            sum_ax = 0.0
            for j in range(row_start, row_end):
                sum_ax += A_csr.data[j] * x[A_csr.indices[j]]

            # Добавляем невязку
            x[i] = x[i] + (b[i] - sum_ax) / diag[i]

        # Обратный ход (backward sweep)
        for i in range(n - 1, -1, -1):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]

            sum_ax = 0.0
            for j in range(row_start, row_end):
                sum_ax += A_csr.data[j] * x[A_csr.indices[j]]

            x[i] = x[i] + (b[i] - sum_ax) / diag[i]

        # Проверка сходимости
        max_diff = np.max(np.abs(x - x_old))
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
        tau = 1.8 / lambda_max

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
    """
    n = A_csr.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    b = np.array(b, dtype=float)

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

        if residual < tol:
            return x

        if k == 0:
            tau = 2.0 / (lambda_max + lambda_min)
            x = x + tau * r
            tau_old = tau
        else:
            tau = 4.0 * theta / (lambda_max - lambda_min) / (2 * theta - rho * tau_old)
            beta = (tau / tau_old) * (rho / 2.0)
            p = r + beta * p
            x = x + tau * p
            tau_old = tau

        r = b - A_csr.dot(x)

        if residual > 1e10:
            break

    return x


def symmetric_gauss_seidel_chebyshev(A_csr, b, lambda_min=None, lambda_max=None,
                                     x0=None, tol=1e-6, max_iter=1000):
    """
    Симметризованный метод Гаусса-Зейделя с чебышевским ускорением.

    Использует SGS как предобусловливатель в методе Чебышева.
    x(k+1) = x(k) + tau(k) * M^(-1) * (b - A*x_k),
    где M^(-1) — одна итерация SGS.
    """
    n = A_csr.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    b = np.array(b, dtype=float)

    # Извлекаем диагональ
    diag = A_csr.get_diagonal()

    # Оценка границ спектра предобусловленной матрицы M^{-1}A
    if lambda_max is None or lambda_min is None:
        # Оцениваем через степенной метод на предобусловленной матрице
        def apply_preconditioned(A, x):
            """Применяет одну итерацию SGS как предобусловливатель."""
            # Прямой ход
            for i in range(n):
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]
                sum_ax = 0.0
                for j in range(row_start, row_end):
                    sum_ax += A.data[j] * x[A.indices[j]]
                x[i] = x[i] + (0 - sum_ax) / diag[i]  # b=0 для оценки спектра

            # Обратный ход
            for i in range(n - 1, -1, -1):
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]
                sum_ax = 0.0
                for j in range(row_start, row_end):
                    sum_ax += A.data[j] * x[A.indices[j]]
                x[i] = x[i] + (0 - sum_ax) / diag[i]

            return x

        # Степенной метод для оценки lambda_max предобусловленной матрицы
        if lambda_max is None:
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            for _ in range(50):
                Av = A_csr.dot(v)
                w = apply_preconditioned(A_csr, Av.copy())
                v = w / np.linalg.norm(w)
            lambda_max = np.dot(v, A_csr.dot(v)) / np.dot(v, v)

        if lambda_min is None:
            # Для SPD матрицы с SGS-предобусловливанием lambda_min ≈ 1
            lambda_min = 1.0

    theta = (lambda_max + lambda_min) / (lambda_max - lambda_min)
    rho = (1 - lambda_min / lambda_max) / (1 + lambda_min / lambda_max)

    r = b - A_csr.dot(x)
    p = np.zeros(n)

    for k in range(max_iter):
        residual = np.linalg.norm(r, ord=np.inf)

        if residual < tol:
            return x

        # Применяем SGS-предобусловливатель к невязке
        z = r.copy()

        # Прямой ход
        for i in range(n):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            sum_ax = 0.0
            for j in range(row_start, row_end):
                sum_ax += A_csr.data[j] * z[A_csr.indices[j]]
            z[i] = (z[i] - sum_ax) / diag[i]

        # Обратный ход
        for i in range(n - 1, -1, -1):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            sum_ax = 0.0
            for j in range(row_start, row_end):
                sum_ax += A_csr.data[j] * z[A_csr.indices[j]]
            z[i] = (z[i] - sum_ax) / diag[i]

        if k == 0:
            tau = 2.0 / (lambda_max + lambda_min)
            x = x + tau * z
            tau_old = tau
        else:
            tau = 4.0 * theta / (lambda_max - lambda_min) / (2 * theta - rho * tau_old)
            beta = (tau / tau_old) * (rho / 2.0)
            p = z + beta * p
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

    if 'gauss_seidel' in method_name and 'symmetric' not in method_name and 'chebyshev' not in method_name:
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

    elif 'symmetric' in method_name and 'chebyshev' not in method_name:
        # Симметризованный Гаусс-Зейдель с историей
        diag = A_csr.get_diagonal()

        for iteration in range(max_iter):
            x_old = x.copy()

            # Прямой ход
            for i in range(n):
                row_start = A_csr.indptr[i]
                row_end = A_csr.indptr[i + 1]

                sum_ax = 0.0
                for j in range(row_start, row_end):
                    sum_ax += A_csr.data[j] * x[A_csr.indices[j]]

                x[i] = x[i] + (b[i] - sum_ax) / diag[i]

            # Обратный ход
            for i in range(n - 1, -1, -1):
                row_start = A_csr.indptr[i]
                row_end = A_csr.indptr[i + 1]

                sum_ax = 0.0
                for j in range(row_start, row_end):
                    sum_ax += A_csr.data[j] * x[A_csr.indices[j]]

                x[i] = x[i] + (b[i] - sum_ax) / diag[i]

            r = b - A_csr.dot(x)
            residual = np.linalg.norm(r, ord=np.inf)
            history.record(iteration + 1, residual)

            max_diff = np.max(np.abs(x - x_old))
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

    elif 'chebyshev' in method_name and 'symmetric' not in method_name:
        # Ричардсон с чебышевским ускорением
        lambda_min = kwargs.get('lambda_min')
        lambda_max = kwargs.get('lambda_max')

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
                tau_old = tau
            else:
                tau = 4.0 * theta / (lambda_max - lambda_min) / (2 * theta - rho * tau_old)
                beta = (tau / tau_old) * (rho / 2.0)
                p = r + beta * p
                x = x + tau * p
                tau_old = tau

            r = b - A_csr.dot(x)

            if residual > 1e10:
                break

    elif 'symmetric' in method_name and 'chebyshev' in method_name:
        # Симметризованный Гаусс-Зейдель с чебышевским ускорением
        lambda_min = kwargs.get('lambda_min')
        lambda_max = kwargs.get('lambda_max')
        diag = A_csr.get_diagonal()

        if lambda_max is None or lambda_min is None:
            # Оценка через степенной метод
            def apply_sgs_preconditioner(A, x):
                n = A.shape[0]
                d = A.get_diagonal()
                # Прямой ход
                for i in range(n):
                    row_start = A.indptr[i]
                    row_end = A.indptr[i + 1]
                    sum_ax = 0.0
                    for j in range(row_start, row_end):
                        sum_ax += A.data[j] * x[A.indices[j]]
                    x[i] = (x[i] - sum_ax) / d[i]

                # Обратный ход
                for i in range(n - 1, -1, -1):
                    row_start = A.indptr[i]
                    row_end = A.indptr[i + 1]
                    sum_ax = 0.0
                    for j in range(row_start, row_end):
                        sum_ax += A.data[j] * x[A.indices[j]]
                    x[i] = (x[i] - sum_ax) / d[i]
                return x

            if lambda_max is None:
                v = np.random.randn(n)
                v = v / np.linalg.norm(v)
                for _ in range(50):
                    Av = A_csr.dot(v)
                    w = apply_sgs_preconditioner(A_csr, Av.copy())
                    v = w / np.linalg.norm(w)
                lambda_max = np.dot(v, A_csr.dot(v)) / np.dot(v, v)

            if lambda_min is None:
                lambda_min = 1.0

        theta = (lambda_max + lambda_min) / (lambda_max - lambda_min)
        rho = (1 - lambda_min / lambda_max) / (1 + lambda_min / lambda_max)

        r = b - A_csr.dot(x)
        p = np.zeros(n)

        for k in range(max_iter):
            residual = np.linalg.norm(r, ord=np.inf)
            history.record(k + 1, residual)

            if residual < tol:
                break

            # SGS-предобусловливатель
            z = r.copy()

            # Прямой ход
            for i in range(n):
                row_start = A_csr.indptr[i]
                row_end = A_csr.indptr[i + 1]
                sum_ax = 0.0
                for j in range(row_start, row_end):
                    sum_ax += A_csr.data[j] * z[A_csr.indices[j]]
                z[i] = (z[i] - sum_ax) / diag[i]

            # Обратный ход
            for i in range(n - 1, -1, -1):
                row_start = A_csr.indptr[i]
                row_end = A_csr.indptr[i + 1]
                sum_ax = 0.0
                for j in range(row_start, row_end):
                    sum_ax += A_csr.data[j] * z[A_csr.indices[j]]
                z[i] = (z[i] - sum_ax) / diag[i]

            if k == 0:
                tau = 2.0 / (lambda_max + lambda_min)
                x = x + tau * z
                tau_old = tau
            else:
                tau = 4.0 * theta / (lambda_max - lambda_min) / (2 * theta - rho * tau_old)
                beta = (tau / tau_old) * (rho / 2.0)
                p = z + beta * p
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

    # 3. Симметризованный Гаусс-Зейдель
    print("\n3. СИММЕТРИЗОВАННЫЙ ГАУСС-ЗЕЙДЕЛЬ")
    start = time.time()
    x_sgs, hist_sgs = solve_with_history(
        symmetric_gauss_seidel, A_sparse, b, tol=TOL, max_iter=MAX_ITER
    )
    time_sgs = time.time() - start
    results['Симм. GS'] = {'x': x_sgs, 'time': time_sgs}
    histories['Симм. GS'] = hist_sgs
    print("Итераций:", len(hist_sgs.iterations),
          "Время:", round(time_sgs, 4), "с",
          "Невязка:", format(hist_sgs.residuals[-1], ".2e"))

    # 4. Метод Ричардсона
    print("\n4. МЕТОД РИЧАРДСОНА")
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

    # 5. Метод Ричардсона с чебышевским ускорением
    print("\n5. МЕТОД РИЧАРДСОНА (ЧЕБЫШЕВ)")
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

    # 6. Симметризованный Гаусс-Зейдель с чебышевским ускорением
    print("\n6. СИММ. ГАУСС-ЗЕЙДЕЛЬ (ЧЕБЫШЕВ)")
    start = time.time()
    x_sgs_cheb, hist_sgs_cheb = solve_with_history(
        symmetric_gauss_seidel_chebyshev, A_sparse, b, tol=TOL, max_iter=MAX_ITER
    )
    time_sgs_cheb = time.time() - start
    results['Симм. GS (Чебышев)'] = {'x': x_sgs_cheb, 'time': time_sgs_cheb}
    histories['Симм. GS (Чебышев)'] = hist_sgs_cheb
    print("Итераций:", len(hist_sgs_cheb.iterations),
          "Время:", round(time_sgs_cheb, 4), "с",
          "Невязка:", format(hist_sgs_cheb.residuals[-1], ".2e"))

    # Визуализация
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = {
        'Якоби': 'blue',
        'Гаусс-Зейдель': 'red',
        'Симм. GS': 'green',
        'Ричардсон': 'purple',
        'Ричардсон (Чебышев)': 'orange',
        'Симм. GS (Чебышев)': 'brown'
    }

    # зависимость невязки от номера итерации
    ax1 = axes[0]
    for method_name, hist in histories.items():
        ax1.semilogy(hist.iterations, hist.residuals,
                     color=colors[method_name], label=method_name, linewidth=2)

    ax1.set_xlabel('Номер итерации')
    ax1.set_ylabel('Невязка (норма-inf)')
    ax1.set_title('Сходимость по итерациям')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # зависимость невязки от времени
    ax2 = axes[1]
    for method_name, hist in histories.items():
        ax2.semilogy(hist.times, hist.residuals,
                     color=colors[method_name], label=method_name, linewidth=2)

    ax2.set_xlabel('Время (сек)')
    ax2.set_ylabel('Невязка (норма-inf)')
    ax2.set_title('Сходимость по времени')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()