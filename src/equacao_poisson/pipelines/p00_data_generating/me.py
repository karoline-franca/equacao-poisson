"""
Classe para resolver a equação de Poisson 3D em eletrostática usando o método
pseudo-espectral Galerkin-Colocação com polinômios de Chebyshev.
"""

import numpy as np


class PoissonSpectral3D:
    """
    Implementação do método pseudo-espectral Galerkin-Colocação para resolver
    a equação de Poisson 3D em eletrostática:

        ∇²φ(x,y,z) = -ρ(x,y,z)/ε₀,   (x,y,z) ∈ Ω,

    com condições de contorno de Dirichlet gerais prescritas em ∂Ω.
    """

    def __init__(self, device='cpu'):
        """
        Inicializa o solver espectral.

        Args:
            device (str): dispositivo para computação ('cpu' ou 'cuda')
        """
        self.device = device

    def _build_diff_matrix(self, N, a, b):
        """
        Constrói a matriz de diferenciação espectral de segunda ordem em 1D.

        Args:
            N (int): grau do polinômio (N+1 pontos de colocação)
            a, b (float): extremos do domínio físico

        Returns:
            tuple: (xi, D2) pontos GLC em [-1,1] e matriz D2 no domínio físico
        """
        j    = np.arange(N + 1)
        xi   = np.cos(np.pi * j / N)

        c       = np.ones(N + 1)
        c[0]    = 2.0
        c[-1]   = 2.0
        c       = c * ((-1.0) ** j)

        X       = np.tile(xi, (N + 1, 1)).T
        dX      = X - X.T
        D_ref   = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
        D_ref  -= np.diag(D_ref.sum(axis=1))

        scale   = 2.0 / (b - a)
        D       = scale * D_ref
        D2      = D @ D

        return xi, D2

    def solve(self, densidade_carga, condicao_contorno, params):
        """
        Resolve a equação de Poisson 3D em eletrostática para uma dada distribuição
        de carga e condições de contorno de Dirichlet gerais.

        Args:
            densidade_carga (callable): Função que define a densidade de carga ρ(x,y,z).
                            Deve ter assinatura: densidade_carga(Xg, Yg, Zg) -> ρ
            condicao_contorno (callable): Função que define o potencial prescrito em ∂Ω.
                            Deve ter assinatura: condicao_contorno(Xg, Yg, Zg) -> g
                            onde g é o valor de Dirichlet em cada ponto da malha.
            params (dict): Dicionário de parâmetros lido do parameters.yml, contendo:
                            - 'N'        : grau do polinômio (N+1 pontos por direção)
                            - 'dominio'  : dict com xmin, xmax, ymin, ymax, zmin, zmax
                            - 'eps0'     : permissividade do vácuo

        Returns:
            dict: Dicionário com a malha de colocação, o potencial elétrico calculado
                  e o resíduo da equação de Poisson nos pontos interiores.
        """
        N    = int(params['N'])
        dom  = params['dominio']
        eps0 = params['eps0']

        xmin, xmax = dom['xmin'], dom['xmax']
        ymin, ymax = dom['ymin'], dom['ymax']
        zmin, zmax = dom['zmin'], dom['zmax']

        xi_x, D2x = self._build_diff_matrix(N, xmin, xmax)
        xi_y, D2y = self._build_diff_matrix(N, ymin, ymax)
        xi_z, D2z = self._build_diff_matrix(N, zmin, zmax)

        x = 0.5 * (xmin + xmax) + 0.5 * (xmax - xmin) * xi_x
        y = 0.5 * (ymin + ymax) + 0.5 * (ymax - ymin) * xi_y
        z = 0.5 * (zmin + zmax) + 0.5 * (zmax - zmin) * xi_z

        Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")

        # termo fonte: f = -ρ/ε₀
        rho    = densidade_carga(Xg, Yg, Zg)
        f      = -rho / eps0
        f_flat = f.ravel(order="F")

        # CC de Dirichlet: φ = g(x,y,z) em ∂Ω
        g       = condicao_contorno(Xg, Yg, Zg)
        g_flat  = g.ravel(order="F")

        I    = np.eye(N + 1)
        Lap  = (np.kron(np.kron(D2x, I), I)
              + np.kron(np.kron(I, D2y), I)
              + np.kron(np.kron(I, I), D2z))

        mask_bc = np.zeros((N + 1, N + 1, N + 1), dtype=bool)
        mask_bc[0, :, :] = True;  mask_bc[-1, :, :] = True
        mask_bc[:, 0, :] = True;  mask_bc[:, -1, :] = True
        mask_bc[:, :, 0] = True;  mask_bc[:, :, -1] = True
        idx_bc = np.where(mask_bc.ravel(order="F"))[0]

        A   = Lap.copy()
        rhs = f_flat.copy()
        A[idx_bc, :]      = 0.0
        A[idx_bc, idx_bc] = 1.0
        rhs[idx_bc]       = g_flat[idx_bc]

        phi_flat = np.linalg.solve(A, rhs)
        phi      = phi_flat.reshape((N + 1, N + 1, N + 1), order="F")

        # resíduo da equação de Poisson nos pontos interiores
        residuo = np.linalg.norm((Lap @ phi_flat - f_flat)[~mask_bc.ravel(order="F")],
                                 ord=np.inf)

        return {
            'x': x,
            'y': y,
            'z': z,
            'Xg': Xg,
            'Yg': Yg,
            'Zg': Zg,
            'phi': phi,
            'rho': rho,
            'residuo': residuo,
            'N': N,
            'dominio': dom
        }