import math
import matplotlib.pyplot as plt
import numpy as np

def func1(x, y):
    return (x**4) + (y**4) + (2*x**2*y**2) + (6*x*y) - (4*x) - (4*y) + 1

# Gradiente da função
def gradient_func1(x, y):
    df_dx = 4*x**3 + 4*x*y**2 + 6*y - 4
    df_dy = 4*y**3 + 4*y*x**2 + 6*x - 4
    return df_dx, df_dy

# Função a ser minimizada
def func2(x, y):
    return 100*(y - x**2)**2 + (x - 1)**2

# Gradiente da função
def gradient_func2(x, y):
    df_dx = -400*x*(y - x**2) + 2*(x - 1)
    df_dy = 200*(y - x**2)
    return df_dx, df_dy

# Função MIPS
def mips(f, r, delta, strategy='recent'):
    s = r - delta
    t = r + delta
    for i in range(50):
        d = 2 * ((s - r) * (f(t) - f(s)) - (f(s) - f(r)) * (t - s))
        if abs(d) < 1e-10:
            a = (r + s + t) / 3.0
        else:
            a = (r + s) / 2.0 - ((f(s) - f(r)) * (t - r) * (t - s)) / d
        
        if abs(f(s) - f(t)) <= 1e-6:
            return (s + t) / 2.0
        
        if strategy == 'recent':
            if f(a) < f(s):
                r, s, t = s, t, a
            else:
                r = s
                s = t
                t = a
        elif strategy == 'worst':
            if f(a) < f(s):
                if f(s) > f(t):
                    s = a
                else:
                    t = a
            else:
                if f(r) > f(t):
                    r = a
                else:
                    s = a
    return (s + t) / 2.0

# Método do Gradiente Descendente com IPS
def gradient_descent_ips(f, grad_f, x0, delta, max_iter, tol=1e-6, strategy='recent'):
    x = x0
    trajectory = [x0]
    prev_f_val = f(*x)
    for i in range(max_iter):
        grad = grad_f(*x)
        
        def f_alpha(alpha):
            x_temp = [xi - alpha * gi for xi, gi in zip(x, grad)]
            return f(*x_temp)
        
        # Calcular o passo usando IPS
        alpha = mips(f_alpha, 0, delta, strategy=strategy)
        
        x = [xi - alpha * gi for xi, gi in zip(x, grad)]
        
        # Verificar convergência
        current_f_val = f(*x)
        if abs(current_f_val - prev_f_val) < tol:
            break
        prev_f_val = current_f_val
        
        # Normalização dos valores
        norm = np.linalg.norm(x)
        if norm > 1e10:
            x = [xi / norm * 1e10 for xi in x]
        
        trajectory.append(x)
    return x, trajectory, i + 1

# Método do Gradiente Descendente com Passos Constantes
def gradient_descent_constant_step(f, grad_f, x0, learning_rate, max_iter, tol=1e-6):
    x = x0
    trajectory = [x0]
    prev_f_val = f(*x)
    for i in range(max_iter):
        grad = grad_f(*x)
        x = [xi - learning_rate * gi for xi, gi in zip(x, grad)]
        
        # Verificar convergência
        current_f_val = f(*x)
        if abs(current_f_val - prev_f_val) < tol:
            break
        prev_f_val = current_f_val
        
        # Normalização dos valores
        norm = np.linalg.norm(x)
        if norm > 1e10:
            x = [xi / norm * 1e10 for xi in x]
        
        trajectory.append(x)
    return x, trajectory, i + 1

def plot_trajectory_3d(f, trajectory, title):
    x_vals = [p[0] for p in trajectory]
    y_vals = [p[1] for p in trajectory]
    z_vals = [f(x, y) for x, y in trajectory]

    X = np.linspace(min(x_vals)-1, max(x_vals)+1, 400)
    Y = np.linspace(min(y_vals)-1, max(y_vals)+1, 400)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.plot(x_vals, y_vals, z_vals, 'r.-', label='Gradient Descent Path', markersize=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(title)
    ax.legend()
    plt.show()

# Teste com visualização usando passos constantes
x0 = [-1, -1]
learning_rate = 0.001  # Ajustar a taxa de aprendizado para um valor menor
max_iter = 10000

result, trajectory, iterations = gradient_descent_constant_step(func2, gradient_func2, x0, learning_rate, max_iter)
print(f"Número de iterações (passos constantes): {iterations}")
plot_trajectory_3d(func2, trajectory, "Function 2 with Constant Steps")

# Teste com visualização usando IPS e substituição da estimativa mais recente
delta = 0.1  # Ajustar o delta

result, trajectory, iterations = gradient_descent_ips(func2, gradient_func2, x0, delta, max_iter, strategy='recent')
print(f"Número de iterações (IPS, estratégia recente): {iterations}")
plot_trajectory_3d(func2, trajectory, "Function 2 with IPS (Estimativa Mais Recente)")

# Teste com visualização usando IPS e substituição da pior estimativa
result, trajectory, iterations = gradient_descent_ips(func2, gradient_func2, x0, delta, max_iter, strategy='worst')
print(f"Número de iterações (IPS, pior estimativa): {iterations}")
plot_trajectory_3d(func2, trajectory, "Function 2 with IPS (Pior Estimativa)")
