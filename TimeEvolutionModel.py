from typing import Dict
import numpy as np


class DissipationModel:

    def __init__(
        self,
        N: int = 2**8,
        L: float = 25.0,
        parameter_config: Dict = {},
    ) -> None:

        self.N = N
        self.L = L
        self.dx = 2*L / (N - 1)

        self.parameter_config = self.default_config()

        if parameter_config:
            self.update_config(parameter_config)

        self.psi = np.zeros([N, N], np.complex128)
        self._init_approximation_wavefunction()

    @staticmethod
    def default_config():

        config = {
            'd': 2,
            "w0": 0.1,
            "k": 1e2,

            # potential energy parameter and dissipation
            "ex": 0.03,
            "ey": 0.09,
            "r": 0.03,
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

    def create_H_operator(self):

        x = np.linspace(-self.L, self.L, self.N)

        H = np.zeros(shape=[self.N, self.N])
        Hx = np.zeros(shape=[self.N, self.N])
        Hy = np.zeros(shape=[self.N, self.N])

        ex = self.parameter_config["ex"]
        ey = self.parameter_config["ey"]

        for i in range(self.N):
            H[i, i] = -2
            Hx[i, i] = (1 + ex) * x[i] ** 2
            Hy[i, i] = (1 + ey) * x[i] ** 2

        for i in range(self.N-1):
            H[i, i+1] = 1

        for i in range(1, self.N):
            H[i, i-1] = 1

        np.divide(H, self.dx**2, out=H)

        return H, Hx, Hy

    def update(self, t_start, t_end, dt=1e-3):

        ex = self.parameter_config["ex"]
        ey = self.parameter_config["ey"]
        r = self.parameter_config["r"]

        k = self.parameter_config["k"]
        w0 = self.parameter_config["w0"]

        x = np.linspace(-self.L, self.L, self.N, dtype=np.complex128)
        y = np.linspace(-self.L, self.L, self.N, dtype=np.complex128)
        mx, my = np.meshgrid(x, y)

        ux = 2*np.pi*np.fft.fftfreq(self.N, d=1.0/self.N)/(2 * self.L)
        mux, muy = np.meshgrid(ux, ux)

        Ux = np.exp(1/(1j-r)*dt*(mux**2 + 2*w0*my*mux)/4).astype(np.complex128)
        Uy = np.exp(1/(1j-r)*dt*(muy**2 - 2*w0*mx*muy)/4).astype(np.complex128)
        Cd = 2*np.pi
        U = 0.5*(((3)**2-1)*k/Cd)**(2./(4))
        t = t_start

        while t <= t_end:

            temp_psi = np.fft.fft(self.psi, axis=1)
            temp2_psi = np.multiply(temp_psi, Ux)
            self.psi = np.fft.ifft(temp2_psi, axis=1)

            temp_psi = np.fft.fft(self.psi, axis=0)
            temp2_psi = np.multiply(temp_psi, Uy)
            self.psi = np.fft.ifft(temp2_psi, axis=0)

            self.psi = np.exp(1/(1j-r)*dt*(((1 + ex)*mx**2 + (1 + ey) * my**2)/2
                                           + k*np.abs(self.psi)**2 - U)) * self.psi

            temp_psi = np.fft.fft(self.psi, axis=0)
            temp2_psi = np.multiply(temp_psi, Uy)
            self.psi = np.fft.ifft(temp2_psi, axis=0)

            temp_psi = np.fft.fft(self.psi, axis=1)
            temp2_psi = np.multiply(temp_psi, Ux)
            self.psi = np.fft.ifft(temp2_psi, axis=1)

            t += dt

        return self.psi
