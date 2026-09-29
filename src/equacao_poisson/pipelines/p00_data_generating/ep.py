"""
Classe para resolver a equação de Poisson 3D em eletrostática usando o método
pseudo-espectral de Galerkin-Colocação com polinômios de Chebyshev.
A equação de Poisson em eletrostática descreve o potencial elétrico gerado por
uma distribuição de carga ρ.
Sistema: ∇²φ = -ρ/ε₀,  φ = g em ∂Ω
"""

import numpy as np
from .me import PoissonSpectral3D


class PoissonEletrostatico:
    """
    Classe para resolver a equação de Poisson 3D em eletrostática usando o
    método pseudo-espectral de Galerkin-Colocação.
    """

    def __init__(self, parametros_eps0, device='cpu'):
        """
        Args:
            parametros_eps0 (float, list ou array): permissividade(s) do vácuo (ε₀)
            device (str): dispositivo para computação ('cpu' ou 'cuda')
        """
        if np.isscalar(parametros_eps0):
            parametros_eps0 = [parametros_eps0]
        self.parametros_eps0 = np.asarray(parametros_eps0, dtype=np.float64)
        self.n_sistemas = len(self.parametros_eps0)

        self.device = device
        self.solver = PoissonSpectral3D(device=device)

    def resolve(self, densidade_carga, condicao_contorno, params):
        """
        Resolve a equação de Poisson para uma ou várias distribuições de carga.

        Args:
            densidade_carga: função ρ(x,y,z) OU lista de funções
            condicao_contorno: função g(x,y,z) OU lista de funções (uma por carga)
            params: dicionário com 'N', 'dominio', 'eps0'

        Returns:
            dict com 'potencial', 'cargas', 'residuos' e metadados
        """
        if callable(densidade_carga):
            densidade_carga = [densidade_carga]
        if callable(condicao_contorno):
            condicao_contorno = [condicao_contorno] * len(densidade_carga)

        n_cargas = len(densidade_carga)

        # resolve para cada distribuição de carga
        solucoes = [
            self.solver.solve(rho, bc, params)
            for rho, bc in zip(densidade_carga, condicao_contorno)
        ]

        potencial = np.stack([s['phi'] for s in solucoes], axis=0)
        cargas    = np.stack([s['rho'] for s in solucoes], axis=0)
        residuos  = np.array([s['residuo'] for s in solucoes])

        return {
            'potencial': potencial,
            'cargas': cargas,
            'x': solucoes[0]['x'],
            'y': solucoes[0]['y'],
            'z': solucoes[0]['z'],
            'N': solucoes[0]['N'],
            'dominio': solucoes[0]['dominio'],
            'residuos': residuos,
            'parametros_eps0': self.parametros_eps0,
            'n_cargas': n_cargas,
            'n_sistemas': self.n_sistemas
        }