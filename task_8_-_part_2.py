import numpy as np
import time
import matplotlib.pyplot as plt

# Это вторая часть задания 8, где мы исследуем зависимость скорости сходимости по итерациям и времени от
# порядка m в методе GMRES. Выполнено на базе кода первой части.

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


def gmres(A_csr, b, restart=30, x0=None, tol=1e-6, max_iter=1000):
    """
    Обобщённый метод минимальных невязок (GMRES) с рестартами.

    Параметры:
    -----------
    A_csr : разреженная матрица
    b : правая часть
    restart : размер подпространства Крылова
    x0 : начальное приближение
    tol : требуемая точность
    max_iter : максимальное число итераций
    """
    n = A_csr.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    b = np.array(b, dtype=float)

    r = b - A_csr.dot(x)
    total_iter = 0

    while total_iter < max_iter:
        beta = np.linalg.norm(r)

        if beta < tol:
            return x

        m = min(restart, max_iter - total_iter)
        V = np.zeros((n, m + 1))
        H = np.zeros((m + 1, m))

        V[:, 0] = r / beta

        j = 0
        for j in range(m):
            w = A_csr.dot(V[:, j])

            for i in range(j + 1):
                H[i, j] = np.dot(w, V[:, i])
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = np.linalg.norm(w)

            if H[j + 1, j] < 1e-14:
                break

            V[:, j + 1] = w / H[j + 1, j]

        e1 = np.zeros(m + 1)
        e1[0] = beta

        for i in range(min(j + 1, m)):
            if H[i, i] == 0:
                c, s = 1.0, 0.0
            else:
                h1, h2 = H[i, i], H[i + 1, i]
                norm = np.sqrt(h1 ** 2 + h2 ** 2)
                c = h1 / norm
                s = h2 / norm

            for k in range(i, m):
                h1 = H[i, k]
                h2 = H[i + 1, k]
                H[i, k] = c * h1 + s * h2
                H[i + 1, k] = -s * h1 + c * h2

            e1_i = e1[i]
            e1_ip1 = e1[i + 1]
            e1[i] = c * e1_i + s * e1_ip1
            e1[i + 1] = -s * e1_i + c * e1_ip1

        y = np.zeros(m)
        for i in range(min(j, m - 1), -1, -1):
            sum_hy = 0.0
            for k in range(i + 1, min(j + 1, m)):
                sum_hy += H[i, k] * y[k]
            y[i] = (e1[i] - sum_hy) / H[i, i]

        for i in range(min(j + 1, m)):
            x = x + y[i] * V[:, i]

        total_iter += min(j + 1, m)
        r = b - A_csr.dot(x)

        if np.linalg.norm(r, ord=np.inf) < tol:
            return x

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


def solve_gmres_with_history(A_csr, b, restart=30, tol=1e-6, max_iter=5000):
    """
    Решает систему методом GMRES и собирает историю сходимости.
    """
    n = A_csr.shape[0]
    x = np.zeros(n)
    b = np.array(b, dtype=float)

    history = IterationHistory()
    history.start()

    r = b - A_csr.dot(x)
    total_iter = 0

    # Записываем начальную невязку
    history.record(0, np.linalg.norm(r, ord=np.inf))

    while total_iter < max_iter:
        beta = np.linalg.norm(r)

        if beta < tol:
            break

        m = min(restart, max_iter - total_iter)
        V = np.zeros((n, m + 1))
        H = np.zeros((m + 1, m))

        V[:, 0] = r / beta

        j = 0
        for j in range(m):
            w = A_csr.dot(V[:, j])

            for i in range(j + 1):
                H[i, j] = np.dot(w, V[:, i])
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = np.linalg.norm(w)

            if H[j + 1, j] < 1e-14:
                break

            V[:, j + 1] = w / H[j + 1, j]

        e1 = np.zeros(m + 1)
        e1[0] = beta

        for i in range(min(j + 1, m)):
            if H[i, i] == 0:
                c, s = 1.0, 0.0
            else:
                h1, h2 = H[i, i], H[i + 1, i]
                norm = np.sqrt(h1 ** 2 + h2 ** 2)
                c = h1 / norm
                s = h2 / norm

            for k in range(i, m):
                h1 = H[i, k]
                h2 = H[i + 1, k]
                H[i, k] = c * h1 + s * h2
                H[i + 1, k] = -s * h1 + c * h2

            e1_i = e1[i]
            e1_ip1 = e1[i + 1]
            e1[i] = c * e1_i + s * e1_ip1
            e1[i + 1] = -s * e1_i + c * e1_ip1

        y = np.zeros(m)
        for i in range(min(j, m - 1), -1, -1):
            sum_hy = 0.0
            for k in range(i + 1, min(j + 1, m)):
                sum_hy += H[i, k] * y[k]
            y[i] = (e1[i] - sum_hy) / H[i, i]

        for i in range(min(j + 1, m)):
            x = x + y[i] * V[:, i]

        total_iter += min(j + 1, m)
        r = b - A_csr.dot(x)

        residual = np.linalg.norm(r, ord=np.inf)
        history.record(total_iter, residual)

    return x, history


def build_poisson_matrix(N):
    """
    Построение матрицы для двумерного уравнения Пуассона на квадрате [0,1]*[0,1]

    Используется пятиточечный шаблон на сетке N*N.
    Размер матрицы - N^2*N^2.
    """
    n = N * N
    h = 1.0 / (N + 1)

    dense_A = np.zeros((n, n))

    for i in range(N):
        for j in range(N):
            k = i * N + j

            dense_A[k, k] = 4.0 / (h * h)

            if i > 0:
                dense_A[k, k - N] = -1.0 / (h * h)
            if i < N - 1:
                dense_A[k, k + N] = -1.0 / (h * h)
            if j > 0:
                dense_A[k, k - 1] = -1.0 / (h * h)
            if j < N - 1:
                dense_A[k, k + 1] = -1.0 / (h * h)

    return dense_A


if __name__ == "__main__":
    # Параметр сетки для уравнения Пуассона
    N = 30  # сетка N×N, матрица будет N²×N² = 900×900
    n = N * N

    print("Построение матрицы уравнения Пуассона на сетке", N, "×", N, "...")
    dense_A = build_poisson_matrix(N)

    # Точное решение и правая часть
    x_exact = np.sin(np.pi * np.arange(n) / n)
    b = dense_A @ x_exact

    # Преобразуем в разреженную матрицу
    A_sparse = SparseMatrixCSR.from_dense(dense_A)

    print("Характеристики матрицы:")
    print("  Размер:", n, "×", n)
    print("  Ненулевых элементов:", len(A_sparse.data))
    print("  Плотность:", round(len(A_sparse.data) / (n * n) * 100, 2), "%")
    print("  Число обусловленности:", round(np.linalg.cond(dense_A), 2))

    TOL = 1e-8
    MAX_ITER = 5000

    # Исследуем разные размеры подпространства Крылова
    restart_values = [1, 2, 3, 5, 7, 10]

    results = {}
    histories = {}

    print("Запуск GMRES с разными m:")

    for m in restart_values:
        print("\nДля m =", m, ":")
        start = time.time()
        x_gmres, hist_gmres = solve_gmres_with_history(
            A_sparse, b, restart=m, tol=TOL, max_iter=MAX_ITER
        )
        time_gmres = time.time() - start

        label = "m = " + str(m)
        results[label] = {'x': x_gmres, 'time': time_gmres}
        histories[label] = hist_gmres

        n_iters = len(hist_gmres.iterations) - 1
        final_residual = hist_gmres.residuals[-1]

        print("Итераций:", n_iters,
              "Время:", round(time_gmres, 4), "с",
              "Невязка:", format(final_residual, ".2e"))

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Цветовая схема для разных m
    colors = {
        "m = 1": 'red',
        "m = 2": 'orange',
        "m = 3": 'gold',
        "m = 5": 'green',
        "m = 7": 'blue',
        "m = 10": 'purple'
    }

    # зависимость невязки от номера итерации
    ax1 = axes[0]
    for m in restart_values:
        label = "m = " + str(m)
        hist = histories[label]
        ax1.semilogy(hist.iterations, hist.residuals,
                     color=colors[label], label=label, linewidth=2, marker='.', markersize=4)

    ax1.set_xlabel('Номер итерации')
    ax1.set_ylabel('Невязка (норма-inf)')
    ax1.set_title(f'Сходимость GMRES по итерациям для разных m (матрица {n}×{n})')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # зависимость невязки от времени
    ax2 = axes[1]
    for m in restart_values:
        label = "m = " + str(m)
        hist = histories[label]
        ax2.semilogy(hist.times, hist.residuals,
                     color=colors[label], label=label, linewidth=2)

    ax2.set_xlabel('Время (сек)')
    ax2.set_ylabel('Невязка (норма-inf)')
    ax2.set_title(f'Сходимость GMRES по времени для разных m (матрица {n}×{n})')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.show()