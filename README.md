# Vortices in Bose–Einstein condensates.

This simulation shows the appearance of vortices in a rotating Bose-Einstein condensate. The entire simulation considers a two-dimensional anisotropic harmonic trap acting on a rotating plane. 

### Ground state solution
The time-independent dimensionless two-dimensional GPE can be solved using the **gradient descent method** to minimize the non-dimensional energy functional of the time-independent GPE.

$$
\mu \phi(r) = \left( - \frac{1}{2}\nabla^{2} + V_{d}(r) + k |\psi(r)|^{2} - \Omega \hat{L_{z}} \right) \phi(r)
$$

$$
E_{k,\Omega}(\phi) = \int_{R^{d}} [\frac{1}{2}|\nabla\phi|^{2} + V_{d}(r)|\phi|^{2} + \frac{k}{2} |\phi|^{4} - \Omega\phi^{*}L_{z}\phi]
$$


###  Dynamics Of Vortex Lattice Formation

Dissipative dynamics of a trapped condensate can be described by following the GP equation, in which $\gamma$ is the dimensionless parameter of the dissipation.

$$
(i - \gamma)\hbar \frac{\psi(r, t)}{\partial t} = \left( - \frac{1}{2}\nabla^{2} + V_{d}(r) + k |\psi(r, t)|^{2} - \Omega \hat{L_{z}} \right) \psi(r, t)
$$

Time evolution model using the **time-split Fourier pseudospectral method** to solve a dimensionless 2D GPE. The GPU can perform the entire computation with fast matrix multiplication and FFT operations.


## Requirements

- numpy
- scipy
- cupy

# Reference

1. Kasamatsu K, Tsubota M and Ueda M. 2003 *Phys. Rev.* A **67** 033610
2. W. Bao et al. Ground-state solution of Bose–Einstein condensate by directly minimizing the energy functional. J. Comput. Phys. (2003)
3. Alexander L. Fetter. *Rev. Mod. Phys.* **81**, 647
4. Makoto Tsubota 2006 *J. Phys.: Conf. Ser.* 31 88
