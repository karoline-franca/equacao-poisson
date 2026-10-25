"""
Classe para resolver a equação de Poisson 2D (adimensional) usando o método
pseudo-espectral de Galerkin-Colocação com polinômios de Chebyshev.
A equação de Poisson adimensionalizada descreve o potencial elétrico gerado
por uma distribuição de carga ρ, ambos em unidades puramente numéricas.
Sistema: ∇²φ̃ = -ρ̃,  φ̃ = g̃ em ∂Ω
"""

import numpy as np
from .me import PoissonSpectral2D


class PoissonEletrostatico:
    """
    Classe para resolver a equação de Poisson 2D adimensional usando o
    método pseudo-espectral de Galerkin-Colocação.

    A adimensionalização absorve a permissividade ε₀ na escala do potencial
    (φ = φ̃ · φ₀, com φ₀ = ρ₀·L₀²/ε₀), de modo que a equação se reduz a
    ∇²φ̃ = -ρ̃ e todos os resultados numéricos são adimensionais.
    """

    def __init__(self, device='cpu'):
        """
        Args:
            device (str): dispositivo para computação ('cpu' ou 'cuda')
        """
        self.device = device
        self.solver = PoissonSpectral2D(device=device)

    def resolve(self, densidade_carga, condicao_contorno, params):
        """
        Resolve a equação de Poisson adimensional para uma ou várias
        distribuições de carga.

        Args:
            densidade_carga: função ρ̃(x,y) OU lista de funções
            condicao_contorno: função g̃(x,y) OU lista de funções (uma por carga)
            params: dicionário com 'N', 'dominio'

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
            'N': solucoes[0]['N'],
            'dominio': solucoes[0]['dominio'],
            'residuos': residuos,
            'n_cargas': n_cargas,
        }