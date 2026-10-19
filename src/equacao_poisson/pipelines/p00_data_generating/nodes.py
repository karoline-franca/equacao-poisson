"""
Nodes do pipeline Kedro para a equação de Poisson 2D em eletrostática.
"""

import numpy as np
import pandas as pd
import torch
from equacao_poisson.utils import cria_grafico_2d
import os
from typing import Dict, Any, Tuple, Callable

from .ep import PoissonEletrostatico


def le_distribuicao_carga_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Gera n distribuições de carga.
    """
    cfg      = parameters['densidade_carga']
    tipo     = cfg['tipo']
    params   = cfg['parametros']
    n_cargas = parameters['simulation']['n_cargas']
    seed     = parameters.get('seed', 42)

    rng = np.random.default_rng(seed)

    if tipo == 'gaussiana':
        amplitude = rng.uniform(params['amplitude_min'], params['amplitude_max'], n_cargas)
        largura   = rng.uniform(params['largura_min'],   params['largura_max'],   n_cargas)
        centro_x  = rng.uniform(params['centro_x_min'],  params['centro_x_max'],  n_cargas)
        centro_y  = rng.uniform(params['centro_y_min'],  params['centro_y_max'],  n_cargas)

        return pd.DataFrame({
            'tipo':       tipo,
            'amplitude':  amplitude,
            'largura':    largura,
            'centro_x':   centro_x,
            'centro_y':   centro_y,
        })

    elif tipo == 'gaussiana_dupla':
        A1     = rng.uniform(params['A1_min'],     params['A1_max'],     n_cargas)
        A2     = rng.uniform(params['A2_min'],     params['A2_max'],     n_cargas)
        sigma1 = rng.uniform(params['sigma1_min'], params['sigma1_max'], n_cargas)
        sigma2 = rng.uniform(params['sigma2_min'], params['sigma2_max'], n_cargas)
        c1x    = rng.uniform(params['c1x_min'],    params['c1x_max'],    n_cargas)
        c1y    = rng.uniform(params['c1y_min'],    params['c1y_max'],    n_cargas)
        c2x    = rng.uniform(params['c2x_min'],    params['c2x_max'],    n_cargas)
        c2y    = rng.uniform(params['c2y_min'],    params['c2y_max'],    n_cargas)

        return pd.DataFrame({
            'tipo': tipo,
            'A1': A1, 'A2': A2,
            'sigma1': sigma1, 'sigma2': sigma2,
            'c1': list(zip(c1x, c1y)),
            'c2': list(zip(c2x, c2y)),
        })

    else:
        raise ValueError(f"Tipo de densidade de carga não suportado para amostragem: {tipo}")


def constroi_densidade_carga(row, eps0, dom) -> Callable:
    """
    Constrói a função ρ(x,y) a partir de uma linha do DataFrame de
    distribuições de carga. O tipo é definido no parameters.yml.

    Para famílias que dependem do domínio (senoide, polinomial), as coordenadas
    são normalizadas para [0,1] usando os extremos do domínio físico, de modo
    que a densidade se anule corretamente nas arestas de [a,b]².

    Args:
        row  : linha do DataFrame com 'tipo' e os parâmetros da carga.
        eps0 : permissividade do vácuo (usada quando a amplitude é dada
               em múltiplos de ε₀ via 'amplitude_fator').
        dom  : dicionário com xmin, xmax, ymin, ymax.

    Returns:
        callable ρ(Xg, Yg)
    """
    tipo = row.tipo

    xmin, xmax = dom['xmin'], dom['xmax']
    ymin, ymax = dom['ymin'], dom['ymax']

    Lx = xmax - xmin
    Ly = ymax - ymin

    if tipo == 'gaussiana':
        amp, larg = row.amplitude, row.largura
        cx, cy = row.centro_x, row.centro_y
        def rho(Xg, Yg):
            r2 = (Xg - cx)**2 + (Yg - cy)**2
            return amp * np.exp(-r2 / (2.0 * larg**2))
        return rho

    elif tipo == 'gaussiana_dupla':
        A1, A2 = row.A1, row.A2
        sigma1, sigma2 = row.sigma1, row.sigma2
        c1x, c1y = row.c1
        c2x, c2y = row.c2
        def rho(Xg, Yg):
            r2_1 = (Xg - c1x)**2 + (Yg - c1y)**2
            r2_2 = (Xg - c2x)**2 + (Yg - c2y)**2
            g1 = A1 * np.exp(-r2_1 / (2.0 * sigma1**2))
            g2 = A2 * np.exp(-r2_2 / (2.0 * sigma2**2))
            return g1 + g2
        return rho

    elif tipo == 'senoide':
        amp = row.amplitude_fator * eps0 if row.amplitude_fator is not None else row.amplitude
        n, m = row.n, row.m
        def rho(Xg, Yg):
            sx = np.sin(n * np.pi * (Xg - xmin) / Lx)
            sy = np.sin(m * np.pi * (Yg - ymin) / Ly)
            return amp * sx * sy
        return rho

    elif tipo == 'polinomial':
        amp = row.amplitude
        n, m = row.n, row.m
        def rho(Xg, Yg):
            ux = (Xg - xmin) / Lx
            uy = (Yg - ymin) / Ly
            return amp * (ux**n) * (uy**m)
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
    eps0  = float(parameters['intervals']['parametro_eps0'])

    params_solver = {'N': N, 'dominio': dom, 'eps0': eps0}

    def contorno_aterrado(Xg, Yg):
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
    distribuicoes_carga: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói a base consolidada com potencial, densidade de carga e os
    parâmetros sorteados da gaussiana de cada carga.
    """
    dados = []
    n_cargas = solucao['n_cargas']
    N = solucao['N']

    x = solucao['x']; y = solucao['y']

    colunas_param = [c for c in ['amplitude', 'largura', 'centro_x', 'centro_y']
                     if c in distribuicoes_carga.columns]

    for i_carga in range(n_cargas):
        param_row = distribuicoes_carga.iloc[i_carga]
        params_i = {c: param_row[c] for c in colunas_param}

        for ix in range(N + 1):
            for iy in range(N + 1):
                linha = {
                    'carga_id': i_carga,
                    'x_col': ix, 'y_col': iy,
                    'x': float(x[ix]),
                    'y': float(y[iy]),
                    'potencial': float(solucao['potencial'][i_carga, ix, iy]),
                    'densidade_carga': float(solucao['cargas'][i_carga, ix, iy]),
                }
                linha.update(params_i)
                dados.append(linha)

    return pd.DataFrame(dados)


def cria_visualizacoes_node(solucao):
    data_version = os.environ.get('DATA_VERSION', 'base_01')
    output_dir = f"data/05_model_input/{data_version}"
    os.makedirs(output_dir, exist_ok=True)

    MAX_CARGAS_VIS = 10
    solucao_reduzida = dict(solucao)
    solucao_reduzida['potencial'] = solucao['potencial'][:MAX_CARGAS_VIS]
    solucao_reduzida['n_cargas']  = min(solucao['n_cargas'], MAX_CARGAS_VIS)

    fig2d = cria_grafico_2d(solucao_reduzida)
    fig2d.write_html(f"{output_dir}/potencial_eletrostatico_2d.html")
    fig2d.show()