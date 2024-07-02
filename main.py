import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

STEP = 1e-2
TOL = 5e-2
MAX_STEPS = 50000

def f(v):
    x, y = v
    return x**4 + y**4 + 2 * x**2 * y**2 + 6 * x * y - 4 * x - 4 * y + 1

def rosenbrock_2d(v):
    x, y = v
    return 100 * (y - x**2)**2 + (x - 1)**2

def rosenbrock_3d(v):
    x, y, z = v
    return 100 * (y - x**2)**2 + (x - 1)**2 + 100 * (z - y**2)**2 + (y - 1)**2

def partial_derivative(f, v, h, i):
    temp = v[i]
    v[i] += h
    f1 = f(v)
    v[i] = temp - h
    f2 = f(v)
    v[i] = temp
    return (f1 - f2) / (2 * h)

def compute_gradient(f, v, h):
    n = len(v)
    gradient = np.array([partial_derivative(f, v.copy(), h, i) for i in range(n)])
    norm = np.linalg.norm(gradient)
    if norm > 1:
        gradient /= norm
    return gradient

def sub_x_av(a, x, v):
    return x - a * v

def worst_estimate(fr, fs, ft):
    return np.argmax([fr, fs, ft])

def mips_mult_v(r, s, t, x, v, f, tol, c):
    count = 0
    w = np.zeros_like(x)
    fa, a = 0, 0

    fr = f(sub_x_av(r, x, v))
    fs = f(sub_x_av(s, x, v))
    ft = f(sub_x_av(t, x, v))

    while True:
        if count > 50:
            return a

        if count >= 5:
            criterion = abs(fs - ft)
            if criterion <= tol:
                return (s + t) / 2

        d = 2 * ((s - r) * (ft - fs) - (fs - fr) * (t - s))

        if d < 10e-10:
            a = (r + s + t) / 3
        else:
            a = ((r + s) / 2) - ((fs - fr) * (t - r) * (t - s)) / d

        fa = f(sub_x_av(a, x, v))

        if c:
            r, fr = s, fs
            s, fs = t, ft
            t, ft = a, fa
        else:
            worst = worst_estimate(fr, fs, ft)
            if worst == 0:
                r, fr = a, fa
            elif worst == 1:
                s, fs = a, fa
            elif worst == 2:
                t, ft = a, fa

        count += 1

def check_convergence(g, tol):
    return all(abs(g_i) < tol for g_i in g)

def descgrad_ips(n, r, s, t, c, f, v):
    count = 0
    grad = compute_gradient(f, v, STEP)
    history = [v.copy()]

    while not check_convergence(grad, TOL) and count < MAX_STEPS:
        grad = compute_gradient(f, v, STEP)
        a = mips_mult_v(r, s, t, v, grad, f, 1e-7, c)
        v -= a * grad
        history.append(v.copy())
        count += 1

    return count, history

def descgrad_constant(n, f, v, a):
    count = 0
    grad = compute_gradient(f, v, STEP)
    history = [v.copy()]

    while not check_convergence(grad, TOL) and count < MAX_STEPS:
        grad = compute_gradient(f, v, STEP)
        v -= a * grad
        history.append(v.copy())
        count += 1

    return count, history

def plot_2d_function(f, history, title):
    history = np.array(history)
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    plt.figure()
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
    plt.plot(history[:, 0], history[:, 1], marker='o', color='r')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.grid(True)
    plt.show()

def plot_3d_function(f, history, title):
    history = np.array(history)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.plot(history[:, 0], history[:, 1], [f(v) for v in history], marker='o', color='r')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def main():
    d = int(input("Enter the number of dimensions: "))
    v = np.array([float(input(f"Enter the value for v[{i}]: ")) for i in range(d)])
    temp = v.copy()

    n = 2
    if n == d:
        print("Function f(x,y) = x^4 + y^4 + 6xy - 4x - 4y + 1")
        print("Evaluating with constant step size:")
        count, history = descgrad_constant(n, f, v, 0.1)
        print(f"Number of iterations: {count}")
        print(v)
        plot_3d_function(f, history, "Constant Step Size")
        v = temp.copy()

        print("Evaluating with step size defined by MIPS (least recent):")
        count, history = descgrad_ips(n, -1, 0, 1, 1, f, v)
        print(f"Number of iterations: {count}")
        print(v)
        plot_3d_function(f, history, "MIPS (Least Recent)")
        v = temp.copy()

        print("Evaluating with step size defined by MIPS (worst estimate):")
        count, history = descgrad_ips(n, -1, 0, 1, 0, f, v)
        print(f"Number of iterations: {count}")
        print(v)
        plot_3d_function(f, history, "MIPS (Worst Estimate)")
        v = temp.copy()
        print()

    else:
        print(f"This function cannot be tested with these parameters. n must be equal to {n}")

    n = 2
    if n == d:
        print("Function Rosenbrock 2D")
        print("Evaluating with constant step size:")
        count, history = descgrad_constant(n, rosenbrock_2d, v, 1e-3)
        print(f"Number of iterations: {count}")
        print(v)
        plot_3d_function(rosenbrock_2d, history, "Constant Step Size (Rosenbrock 2D)")
        v = temp.copy()

        print("Evaluating with step size defined by MIPS (least recent):")
        count, history = descgrad_ips(n, 0, 0.5, 1, 1, rosenbrock_2d, v)
        print(f"Number of iterations: {count}")
        print(v)
        plot_3d_function(rosenbrock_2d, history, "MIPS (Least Recent, Rosenbrock 2D)")
        v = temp.copy()

        print("Evaluating with step size defined by MIPS (worst estimate):")
        count, history = descgrad_ips(n, 0, 0.5, 1, 0, rosenbrock_2d, v)
        print(f"Number of iterations: {count}")
        print(v)
        plot_3d_function(rosenbrock_2d, history, "MIPS (Worst Estimate, Rosenbrock 2D)")
        v = temp.copy()
        print()

    else:
        print(f"This function cannot be tested with these parameters. n must be equal to {n}")

    n = 3
    if n == d:
        print("Function Rosenbrock 3D")
        print("Evaluating with constant step size:")
        count, history = descgrad_constant(n, rosenbrock_3d, v, 1e-3)
        print(f"Number of iterations: {count}")
        print(v)
        v = temp.copy()

        print("Evaluating with step size defined by MIPS (least recent):")
        count, history = descgrad_ips(n, 1, 2, 3, 1, rosenbrock_3d, v)
        print(f"Number of iterations: {count}")
        print(v)
        v = temp.copy()

        print("Evaluating with step size defined by MIPS (worst estimate):")
        count, history = descgrad_ips(n, 1, 2, 3, 0, rosenbrock_3d, v)
        print(f"Number of iterations: {count}")
        print(v)
        v = temp.copy()
        print()

    else:
        print(f"This function cannot be tested with these parameters. n must be equal to {n}")

if __name__ == "__main__":
    main()
