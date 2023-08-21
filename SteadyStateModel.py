import numpy as np
from scipy.integrate import simpson
from typing import Dict


class BECModel:

    def __init__(self, N: int = 2**8, L: float = 25.0, parameter_config: Dict = {}, rotating=False) -> None:
        self.N = N
        self.L = L
        self.dx = 2*L / (N - 1)
        self.rotating = rotating

        self.parameter_config = self.default_config()

        if parameter_config:
            self.update_config(parameter_config)

        self.psi = np.zeros([N, N], np.complex128)
        self.H_operator = self.create_H_operator(self.N, self.L, self.dx)
        self._init_approximation_wavefunction()

    @staticmethod
    def default_config():

        h = 6.626*1e-34/(2*np.pi)
        m = 1.44*1e-25
        w = 219*2*np.pi/2**0.5

        config = {
            'd': 2,

            "h": h,
            "m": m,
            "w": w,

            "a": 5.1*1e-9,
            "w0": 0.1001,
            "a0": np.sqrt(h / (m*w)),
            "N0": 1e5,
            "k": 1e2
        }

        return config

    def _init_approximation_wavefunction(self) -> None:
        x = np.linspace(-self.L, self.L, self.N)

        k = self.parameter_config['k']
        d = self.parameter_config['d']

        Cd = 2*np.pi
        u = 0.5*(((d + 1)**2 - 1) * k / Cd)**(2.0 / (d + 2))

        for i in range(len(self.psi)):
            for j in range(len(self.psi[0])):
                y_ = x[i]
                x_ = x[j]

                r = (x_**2 + y_**2)**0.5
                V = 0.5*r**2

                if V < u:
                    self.psi[i, j] = ((u-V)/k)**0.5

                if V > u:
                    self.psi[i, j] = 0

    def load_psi(self, psi):
        if psi.shape != self.psi.shape:
            raise ValueError("shape different")

        self.psi = psi

    def update_config(self, config: Dict) -> None:
        self.parameter_config.update(config)

    def create_second_derivative_operator(self, N, dx):
        D2 = np.zeros([N, N], np.complex128)

        for i in range(N):
            D2[i, i] = -2

        for i in range(N - 1):
            D2[i, i + 1] = 1

        for i in range(1, N):
            D2[i, i - 1] = 1

        D2 = (-D2 / dx ** 2)
        return D2

    def create_first_order_derivative(self, N, dx):
        D1 = np.zeros([N, N], np.complex128)

        for i in range(1, N):
            D1[i, i-1] = -1j

        for i in range(N-1):
            D1[i, i+1] = 1j

        return (D1/dx)

    def create_potential_energy_operator(self, x):
        return np.diag(x**2)

    def create_H_operator(self, N, L, dx):
        x = np.linspace(-L, L, N)

        D2 = self.create_second_derivative_operator(N, dx)
        V = self.create_potential_energy_operator(x)

        return 0.5 * (D2 + V)

    def update(self, steps, dt=1e-3):

        # Using gradient decent method to update energy functional.

        # Create a location coordinate reference for calculation.
        x = np.linspace(-self.L, self.L, self.N)
        xx, yy = np.meshgrid(x, x)

        w0 = self.parameter_config['w0']
        k = self.parameter_config['k']

        if self.rotating == True:
            D1 = self.create_first_order_derivative(self.N, self.dx)

        H = self.create_H_operator(self.N, self.L, self.dx)

        for _ in range(steps):

            abs_term = self.psi * k * np.abs(self.psi)**2

            if self.rotating == True:
                # rotation operator apply on psi
                rotate_term = -w0*(xx*(np.dot(D1, self.psi)) -
                                   yy*(np.dot(D1, self.psi.T).T))

                dpsi = np.dot(H, self.psi.T).T + \
                    np.dot(H, self.psi) + abs_term + rotate_term
            else:

                dpsi = np.dot(H, self.psi.T).T + \
                    np.dot(H, self.psi) + abs_term

            np.subtract(self.psi, dpsi * dt, out=self.psi)

            # normalized coeffcient using 2d fixed point integration
            C = simpson(np.abs(self.psi)**2, x=x)
            C = simpson(C, x=x)

            np.divide(self.psi, (C**0.5), out=self.psi)

        return self.psi
