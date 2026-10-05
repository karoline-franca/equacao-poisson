"""
Nodes do pipeline Kedro para a equação de Poisson 3D em eletrostática.
"""

import numpy as np
import pandas as pd
import torch
from equacao_poisson.utils import cria_grafico_2d, cria_grafico_3d
import os
from typing import Dict, Any, Tuple, Callable

from .ep import PoissonEletrostatico


def le_distribuicao_carga_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Lê a distribuição de carga definida explicitamente no parameters.yml.
    Não sorteia nada — apenas empacota a configuração em um DataFrame.
    """
    cfg     = parameters['densidade_carga']
    tipo    = cfg['tipo']
    params  = cfg['parametros']

    return pd.DataFrame([{
        'tipo':       tipo,
        'amplitude':  params.get('amplitude'),
        'amplitude_fator': params.get('amplitude_fator'),
        'largura':    params.get('largura'),
        'centro_x':   params.get('centro_x'),
        'centro_y':   params.get('centro_y'),
        'centro_z':   params.get('centro_z'),
        'n':          params.get('n'),
        'm':          params.get('m'),
        'l':          params.get('l'),
    }])


def constroi_densidade_carga(row, eps0, dom) -> Callable:
    """
    Constrói a função ρ(x,y,z) a partir de uma linha do DataFrame de
    distribuições de carga. O tipo é definido no parameters.yml.

    Para famílias que dependem do domínio (senoide, polinomial), as coordenadas
    são normalizadas para [0,1] usando os extremos do domínio físico, de modo
    que a densidade se anule corretamente nas faces de [a,b]³.

    Args:
        row  : linha do DataFrame com 'tipo' e os parâmetros da carga.
        eps0 : permissividade do vácuo (usada quando a amplitude é dada
               em múltiplos de ε₀ via 'amplitude_fator').
        dom  : dicionário com xmin, xmax, ymin, ymax, zmin, zmax.

    Returns:
        callable ρ(Xg, Yg, Zg)
    """
    tipo = row.tipo

    xmin, xmax = dom['xmin'], dom['xmax']
    ymin, ymax = dom['ymin'], dom['ymax']
    zmin, zmax = dom['zmin'], dom['zmax']

    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin

    if tipo == 'gaussiana':
        amp, larg = row.amplitude, row.largura
        cx, cy, cz = row.centro_x, row.centro_y, row.centro_z
        def rho(Xg, Yg, Zg):
            r2 = (Xg - cx)**2 + (Yg - cy)**2 + (Zg - cz)**2
            return amp * np.exp(-r2 / (2.0 * larg**2))
        return rho

    elif tipo == 'gaussiana_dupla':
        A1, A2 = row.A1, row.A2
        sigma1, sigma2 = row.sigma1, row.sigma2
        c1x, c1y, c1z = row.c1
        c2x, c2y, c2z = row.c2
        def rho(Xg, Yg, Zg):
            r2_1 = (Xg - c1x)**2 + (Yg - c1y)**2 + (Zg - c1z)**2
            r2_2 = (Xg - c2x)**2 + (Yg - c2y)**2 + (Zg - c2z)**2
            g1 = A1 * np.exp(-r2_1 / (2.0 * sigma1**2))
            g2 = A2 * np.exp(-r2_2 / (2.0 * sigma2**2))
            return g1 + g2
        return rho

    elif tipo == 'senoide':
        amp = row.amplitude_fator * eps0 if row.amplitude_fator is not None else row.amplitude
        n, m, l = row.n, row.m, row.l
        def rho(Xg, Yg, Zg):
            sx = np.sin(n * np.pi * (Xg - xmin) / Lx)
            sy = np.sin(m * np.pi * (Yg - ymin) / Ly)
            sz = np.sin(l * np.pi * (Zg - zmin) / Lz)
            return amp * sx * sy * sz
        return rho

    elif tipo == 'polinomial':
        amp = row.amplitude
        n, m, l = row.n, row.m, row.l
        def rho(Xg, Yg, Zg):
            ux = (Xg - xmin) / Lx
            uy = (Yg - ymin) / Ly
            uz = (Zg - zmin) / Lz
            return amp * (ux**n) * (uy**m) * (uz**l)
        return rho

    else:
        raise ValueError(f"Tipo de densidade de carga desconhecido: {tipo}")

def executa_simulacao_espectral_node(
    distribuicoes_carga: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executa a resolução espectral da equação de Poisson.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N     = int(parameters['solver']['N'])
    dom   = parameters['solver']['dominio']
    eps0 = float(parameters['intervals']['parametro_eps0'])

    params_solver = {'N': N, 'dominio': dom, 'eps0': eps0}

    def contorno_aterrado(Xg, Yg, Zg):
        return np.zeros_like(Xg)

    densidades = [constroi_densidade_carga(row, eps0, dom)
                  for row in distribuicoes_carga.itertuples()]
    contornos  = [contorno_aterrado] * len(densidades)

    solver = PoissonEletrostatico(parametros_eps0=[eps0], device=device)
    solucao = solver.resolve(
        densidade_carga=densidades,
        condicao_contorno=contornos,
        params=params_solver
    )

    metadados = {
        'dispositivo': device,
        'N': N,
        'n_cargas': int(len(distribuicoes_carga)),
        'n_sistemas': int(solver.n_sistemas),
        'dominio': dom,
        'parametros': {'eps0': eps0}
    }

    return solucao, metadados


def gera_base_consolidada_node(
    solucao: Dict[str, Any],
) -> pd.DataFrame:
    """
    Constrói a base de dados consolidada com os campos de potencial e carga.
    """
    dados = []
    n_cargas = solucao['n_cargas']
    N = solucao['N']

    x = solucao['x']; y = solucao['y']; z = solucao['z']

    for i_carga in range(n_cargas):
        for ix in range(N + 1):
            for iy in range(N + 1):
                for iz in range(N + 1):
                    dados.append({
                        'carga_id': i_carga,
                        'x_col': ix, 'y_col': iy, 'z_col': iz,
                        'x': float(x[ix]),
                        'y': float(y[iy]),
                        'z': float(z[iz]),
                        'potencial': float(solucao['potencial'][i_carga, ix, iy, iz]),
                        'densidade_carga': float(solucao['cargas'][i_carga, ix, iy, iz])
                    })

    return pd.DataFrame(dados)


def cria_visualizacoes_node(
    solucao: Dict[str, Any],
) -> None:
    """
    Cria visualizações 2D e 3D do potencial elétrico.
    """
    data_version = os.environ.get('DATA_VERSION', 'base_01')
    output_dir = f"data/05_model_input/{data_version}"
    os.makedirs(output_dir, exist_ok=True)

    fig2d = cria_grafico_2d(solucao)
    fig3d = cria_grafico_3d(solucao)

    fig2d.write_html(f"{output_dir}/potencial_eletrostatico_2d.html")
    fig3d.write_html(f"{output_dir}/potencial_eletrostatico_3d.html")

    fig2d.show()
    fig3d.show()