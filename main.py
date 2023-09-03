from SteadyStateModel import BECModel
from TimeEvolutionModel_gpu import DissipationModel_GPU
from TimeEvolutionModel import DissipationModel
import matplotlib.pyplot as plt
import numpy as np
import time
import cupy


def time_comparesion(psi):

    psi = psi

    w0 = 0.7

    time_evolution_config = {
        "k": k,
        "w0": w0,

        "ex": 0.03,
        "ey": 0.09,
        "r": 0.03,
    }

    time_model_gpu = DissipationModel_GPU(
        N=N,
        L=L,
        parameter_config=time_evolution_config
    )

    time_model_gpu.load_psi(psi)

    time_model_cpu = DissipationModel(
        N=N,
        L=L,
        parameter_config=time_evolution_config
    )

    time_model_cpu.load_psi(psi)

    dt = 1e-3
    t_s = 0
    T_final = 1

    time_start = time.time()
    psi = time_model_gpu.update(t_s, T_final, dt=dt)

    print(f"GPU time elapsed: {time.time() - time_start} s. ")

    time_start = time.time()
    psi = time_model_cpu.update(t_s, T_final, dt=dt)

    print(f"CPU time elapsed: {time.time() - time_start} s. ")


N = 512
L = 15
k = 2000

ground_state_config = {
    "k": k
}

groud_state_model = BECModel(
    N=N, L=L, parameter_config=ground_state_config, rotating=False)

steps = 100_000
dt = 1e-3

psi = groud_state_model.update(steps=steps, dt=dt)

filename = f"N_{N}_k_{k}_L_{L}.npy"
np.save(filename, psi)

psi = np.load(filename)

w0 = 0.7
time_evolution_config = {
    "k": k,
    "w0": w0,

    "ex": 0.03,
    "ey": 0.09,
    "r": 0.03,
}

time_model = DissipationModel_GPU(
    N=N,
    L=L,
    parameter_config=time_evolution_config
)

time_model.load_psi(psi)

dt = 1e-3
t_s = 0
frame = 0
T_final = 250

for T in np.arange(0, T_final, 2):

    psi = time_model.update(t_s, T, dt=dt)

    plt.figure()
    plt.imshow((cupy.abs(psi)**2).get(), cmap="gray")
    plt.title(f't = {T}')
    plt.savefig(f"./img/frame_{frame}.png", dpi=300)
    plt.close()

    frame += 1
    t_s = T

print(f"Simulation finisehd!!")
