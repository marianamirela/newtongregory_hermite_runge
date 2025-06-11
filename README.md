# Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, BarycentricInterpolator, KroghInterpolator

# -----------------------------------------------------------------------------
# 1) Aplicação da Interpolação de Hermite (PARTE II)
# -----------------------------------------------------------------------------
x = np.array([0.0, 1.0, 2.0])
f = np.array([1.0, 2.0, 0.0])
df = np.array([0.0, 1.0, -1.0])

interp = KroghInterpolator(x, np.vstack((f, df)).T)

x_vals = np.linspace(x.min() - 0.5, x.max() + 0.5, 400)
y_vals = interp(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r"Polinômio de Hermite $H_{2n+1}(x)$")
plt.plot(x, f, 'ro', label="Pontos $(x_i, f_i)$")

for xi, fi, dfi in zip(x, f, df):
    plt.arrow(xi, fi, 0.2, 0.2 * dfi,
              head_width=0.05, head_length=0.1, color='green')

plt.title("Interpolação de Hermite")
plt.xlabel("x")
plt.ylabel("H(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 2) Aplicação do Fenômeno de Runge (PARTE III)
# -----------------------------------------------------------------------------
def f_runge(x):
    return 1 / (1 + 25 * x**2)

z = np.linspace(-5, 5, 51)
f_z = f_runge(z)

ks = [5, 10, 20]
erros = {}

for k in ks:
    xk = np.linspace(-5, 5, k + 1)
    yk = f_runge(xk)

    # Polinômio de Barycentric
    p = BarycentricInterpolator(xk, yk)
    p_z = p(z)

    # Splines linear e cúbica
    S1 = interp1d(xk, yk, kind='linear')
    S1_z = S1(z)

    S3 = CubicSpline(xk, yk)
    S3_z = S3(z)

    # Erros máximos
    erro_p = np.max(np.abs(f_z - p_z))
    erro_s1 = np.max(np.abs(f_z - S1_z))
    erro_s3 = np.max(np.abs(f_z - S3_z))
    erros[k] = {
        'polinomio': erro_p,
        'spline_linear': erro_s1,
        'spline_cubica': erro_s3
    }

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(z, f_z, 'k-', label='f(x)')
    plt.plot(z, p_z, 'r--', label='Polinômio Interp.')
    plt.plot(z, S1_z, 'b-.', label='Spline Linear')
    plt.plot(z, S3_z, 'g:', label='Spline Cúbica')
    plt.scatter(xk, yk, c='k', marker='o', label='Pontos de Interpolação')

    plt.title(f'Interpolação de Runge com k={k}')
    plt.xlabel("x")
    plt.ylabel("valor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Impressão dos erros
for k in ks:
    print(f"\nk = {k}:")
    print(f"  Erro máx. polinômio:    {erros[k]['polinomio']:.6f}")
    print(f"  Erro máx. spline linear: {erros[k]['spline_linear']:.6f}")
    print(f"  Erro máx. spline cúbica: {erros[k]['spline_cubica']:.6f}")
text
Copiar
Editar

