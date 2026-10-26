"""
Utilitários para o pipeline da equação de Poisson 2D em eletrostática.
"""

import numpy as np
from typing import Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score

CORES_PALETA = [
    '#C41E3A','#2E7D32','#1565C0','#E65100','#6A1B9A',
    '#00695C','#AD1457','#F57F17','#37474F','#D84315',
    '#1B5E20','#0D47A1','#4A148C','#BF360C','#1A237E',
    '#004D40','#880E4F','#4E342E','#263238','#B71C1C'
]


def formatar_numero_pt_br(numero):
    """Formata número no padrão brasileiro com 3 casas decimais."""
    try:
        valor = float(numero)
        return f"{valor:.3f}".replace('.', ',')
    except (ValueError, TypeError):
        return str(numero)


def viridis_deslocado(i_carga):
    """
    Gera uma escala de cores baseada no Viridis, mas deslocada no espectro
    conforme o índice da carga.
    """
    viridis = [
        [0.0,  '#440154'],
        [0.1,  '#482777'],
        [0.2,  '#3F4A8A'],
        [0.3,  '#31688E'],
        [0.4,  '#26828E'],
        [0.5,  '#1F9E89'],
        [0.6,  '#35B779'],
        [0.7,  '#6CCE59'],
        [0.8,  '#B4DE2C'],
        [0.9,  '#FDE725'],
        [1.0,  '#FDE725'],
    ]

    n = len(viridis)
    shift = (i_carga * 3) % n

    reordenada = viridis[shift:] + viridis[:shift]
    escala = []
    for k, (_, cor) in enumerate(reordenada):
        escala.append([k / (n - 1), cor])

    return escala


def cria_grafico_2d(solucao):
    """Cria o gráfico da distribuição dos dados da base consolidada."""
    n_cargas = solucao['n_cargas']
    N        = solucao['N']
    x        = solucao['x']
    y        = solucao['y']

    rho_vmin = float(solucao['cargas'].min())
    rho_vmax = float(solucao['cargas'].max())
    phi_vmin = float(solucao['potencial'].min())
    phi_vmax = float(solucao['potencial'].max())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Densidade de Carga ρ̃(x,y)",
                        "Potencial Elétrico φ̃(x,y)"),
        horizontal_spacing=0.20,
    )

    for i_carga in range(n_cargas):
        rho_plano = solucao['cargas'][i_carga, :, :]
        phi_plano = solucao['potencial'][i_carga, :, :]
        visivel   = (i_carga == 0)

        # ρ̃(x,y)
        fig.add_trace(
            go.Contour(
                x=x, y=y,
                z=rho_plano.T,
                colorscale="Viridis",
                zmin=rho_vmin, zmax=rho_vmax,
                contours=dict(coloring='heatmap', showlabels=True),
                line=dict(width=0.5, color='rgba(255,255,255,0.4)'),
                name=f"Densidade de Carga {i_carga} ρ̃(x,y)",
                showlegend=False,
                showscale=False,
                visible=visivel,
                opacity=0.85,
                hovertemplate=(
                    f"<b>ρ̃ (x,y) {i_carga}</b><br>"
                    "x = %{x:.3f}<br>"
                    "y = %{y:.3f}<br>"
                    "ρ̃ = %{z:.3e}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

        # φ̃(x,y)
        fig.add_trace(
            go.Contour(
                x=x, y=y,
                z=phi_plano.T,
                colorscale="Viridis",
                zmin=phi_vmin, zmax=phi_vmax,
                contours=dict(coloring='heatmap', showlabels=True),
                line=dict(width=0.5, color='rgba(255,255,255,0.4)'),
                name=f"Potencial Elétrico {i_carga} φ̃(x,y)",
                showlegend=False,
                showscale=False,
                visible=visivel,
                opacity=0.85,
                hovertemplate=(
                    f"<b>ρ̃ (x,y) {i_carga}</b><br>"
                    "x = %{x:.3f}<br>"
                    "y = %{y:.3f}<br>"
                    "φ̃ = %{z:.3e}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1, col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale="Viridis",
                cmin=rho_vmin, cmax=rho_vmax,
                showscale=True,
                colorbar=dict(
                    title="ρ̃(x,y)",
                    x=-0.10,
                    len=0.55,
                    xanchor="right",
                ),
            ),
            showlegend=False,
            hoverinfo='skip',
            name='colorbar_rho',
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale="Viridis",
                cmin=phi_vmin, cmax=phi_vmax,
                showscale=True,
                colorbar=dict(
                    title="φ̃(x,y)",
                    x=1.05,
                    len=0.55,
                    xanchor="left",
                ),
            ),
            showlegend=False,
            hoverinfo='skip',
            name='colorbar_phi',
        ),
        row=1, col=2,
    )

    n_tracos_dados = 2 * n_cargas
    botoes = []
    for i_carga in range(n_cargas):
        visibilidade = [False] * n_tracos_dados + [True, True]
        visibilidade[2 * i_carga]     = True
        visibilidade[2 * i_carga + 1] = True

        botoes.append(dict(
            label=f"ρ̃ (x,y) {i_carga}",
            method="update",
            args=[
                {"visible": visibilidade},
                {"title.text":
                    f"<span style='font-size:20px; font-weight:bold;'>"
                    f"Gráficos de Densidade de Carga e Potencial Elétrico"
                    f"</span><br><br>"
                    f"<span style='font-size:18px; color:#555555;'>"
                    f"Densidade de Carga ρ̃ (x,y) {i_carga} | Grau N = {N}</span>"}
            ],
        ))

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>"
                 f"Densidade de Carga e Potencial Elétrico"
                 f"</span><br><br>"
                 f"<span style='font-size:18px; color:#555555;'>"
                 f"Densidade de Carga ρ̃ (x,y) 0 | Grau N = {N}</span>",
            x=0.50, y=0.95, font=dict(size=20),
        ),
        updatemenus=[dict(
            type="dropdown",
            buttons=botoes,
            direction="down",
            x=1.05, y=1.00,
            xanchor="left", yanchor="top",
            showactive=True,
            bgcolor="white", bordercolor="black",
            font=dict(size=13),
        )],
        width=1700, height=800,
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(t=170, r=250, b=60, l=140),
    )

    fig.update_xaxes(
        title_text="x", showgrid=False, zeroline=False,
        title_font=dict(size=16), tickfont=dict(size=16),
        range=[x.min(), x.max()],
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="y", showgrid=False, zeroline=False,
        title_font=dict(size=16), tickfont=dict(size=16),
        range=[y.min(), y.max()],
        scaleanchor="x", scaleratio=1,
        row=1, col=1,
    )
    fig.update_xaxes(
        title_text="x", showgrid=False, zeroline=False,
        title_font=dict(size=16), tickfont=dict(size=16),
        range=[x.min(), x.max()],
        row=1, col=2,
    )
    fig.update_yaxes(
        title_text="y", showgrid=False, zeroline=False,
        title_font=dict(size=16), tickfont=dict(size=16),
        range=[y.min(), y.max()],
        scaleanchor="x", scaleratio=1,
        row=1, col=2,
    )

    return fig


def cria_grafico_distribuicao_max_rho(
    cargas_train, cargas_val, cargas_test,
    parametros_carga,
    amplitudes_carga,
    titulo="Distribuição das Amplitudes Máximas de Densidade de Carga por Conjunto"
):
    """
    Cria gráfico 2D com a distribuição espacial (centro_x, centro_y) das
    cargas de cada conjunto (treino/validação/teste), com o marcador de cada
    carga colorido pelo valor máximo de ρ̃ (x,y).

    Uma cor fixa por conjunto (verde, azul, laranja) distingue treino,
    validação e teste. A intensidade da cor (na escala Viridis) reflete
    a amplitude máxima de ρ̃ (x,y) da carga.
    """

    n_train = len(cargas_train)
    n_val   = len(cargas_val)
    n_test  = len(cargas_test)
    total   = n_train + n_val + n_test

    # escala de cores global (todas as cargas compartilham o mesmo range)
    todas_amplitudes = list(amplitudes_carga.values())
    amin = float(min(todas_amplitudes))
    amax = float(max(todas_amplitudes))

    def _coords_centro(carga_id):
        if hasattr(parametros_carga, 'loc'):
            if carga_id in parametros_carga.index:
                row = parametros_carga.loc[carga_id]
            elif 'carga_id' in parametros_carga.columns:
                row = parametros_carga[parametros_carga['carga_id'] == carga_id].iloc[0]
            else:
                row = parametros_carga.iloc[int(carga_id)]
            return float(row['centro_x']), float(row['centro_y'])
        else:
            return (float(parametros_carga['centro_x'][carga_id]),
                    float(parametros_carga['centro_y'][carga_id]))

    def _coords_lista(cargas):
        xs, ys, amps = [], [], []
        for c in cargas:
            xc, yc = _coords_centro(c)
            xs.append(xc)
            ys.append(yc)
            amps.append(amplitudes_carga[c])
        return xs, ys, amps

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale="Viridis",
            cmin=amin, cmax=amax,
            showscale=True,
            colorbar=dict(
                title="ρ̃ (x,y) máximo",
                x=1.02,
                len=0.55,
            ),
        ),
        showlegend=False,
        hoverinfo='skip',
        name='colorbar_rho_max',
    ))

    # treino (círculos, borda verde)
    x_train, y_train, a_train = _coords_lista(cargas_train)
    fig.add_trace(go.Scatter(
        x=x_train, y=y_train,
        mode='markers',
        name=f'Treino ({n_train})',
        marker=dict(
            color=a_train,
            colorscale="Viridis",
            cmin=amin, cmax=amax,
            size=11, symbol='circle',
            line=dict(width=2, color='#2E7D32'),
            showscale=False,
        ),
        hovertemplate=(
            "<b>Treino</b><br>"
            "centro_x = %{x:.3f}<br>"
            "centro_y = %{y:.3f}<br>"
            "ρ̃ (x,y) máxima = %{marker.color:.3e}<br>"
            "<extra></extra>"
        )
    ))

    # validação (quadrados, borda azul)
    x_val, y_val, a_val = _coords_lista(cargas_val)
    fig.add_trace(go.Scatter(
        x=x_val, y=y_val,
        mode='markers',
        name=f'Validação ({n_val})',
        marker=dict(
            color=a_val,
            colorscale="Viridis",
            cmin=amin, cmax=amax,
            size=11, symbol='square',
            line=dict(width=2, color='#0D47A1'),
            showscale=False,
        ),
        hovertemplate=(
            "<b>Validação</b><br>"
            "centro_x = %{x:.3f}<br>"
            "centro_y = %{y:.3f}<br>"
            "ρ̃ (x,y) máxima = %{marker.color:.3e}<br>"
            "<extra></extra>"
        )
    ))

    # teste (losangos, borda laranja)
    x_test, y_test, a_test = _coords_lista(cargas_test)
    fig.add_trace(go.Scatter(
        x=x_test, y=y_test,
        mode='markers',
        name=f'Teste ({n_test})',
        marker=dict(
            color=a_test,
            colorscale="Viridis",
            cmin=amin, cmax=amax,
            size=11, symbol='diamond',
            line=dict(width=2, color='#F57F17'),
            showscale=False,
        ),
        hovertemplate=(
            "<b>Teste</b><br>"
            "centro_x = %{x:.3f}<br>"
            "centro_y = %{y:.3f}<br>"
            "ρ̃ (x,y) = %{marker.color:.3e}<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>"
                 f"<span style='font-size:16px; color:#555555;'>"
                 f"Total: {total} Distribuições de Densidade de Carga | "
                 f"Treino: {n_train} ({n_train/total*100:.1f}%) | "
                 f"Validação: {n_val} ({n_val/total*100:.1f}%) | "
                 f"Teste: {n_test} ({n_test/total*100:.1f}%)</span>",
            x=0.5, y=0.95, font=dict(size=20)
        ),
        xaxis_title="x (Centro da Gaussiana)",
        yaxis_title="y (Centro da Gaussiana)",
        width=1200, height=900,
        legend=dict(
            title="Legenda",
            x=1.02, y=1.00,
            xanchor="left", yanchor="top",
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(t=140, r=200),
        xaxis=dict(
            showgrid=False, zeroline=False,
            title_font=dict(size=16), tickfont=dict(size=14),
            scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            title_font=dict(size=16), tickfont=dict(size=14),
        ),
    )

    return fig


def cria_grafico_distribuicao_max_phi(
    cargas_train, cargas_val, cargas_test,
    parametros_carga,
    amplitudes_potencial,
    titulo="Distribuição das Amplitudes Máximas de Potencial Elétrico por Conjunto"
):
    """
    Cria gráfico 2D com a distribuição espacial (centro_x, centro_y) das
    cargas de cada conjunto (treino/validação/teste), com o marcador de cada
    carga colorido pelo valor máximo de φ̃.

    Uma cor fixa por conjunto (verde, azul, laranja) distingue treino,
    validação e teste. A intensidade da cor (na escala Viridis) reflete
    a amplitude máxima de φ̃ (x,y) da carga.
    """

    n_train = len(cargas_train)
    n_val   = len(cargas_val)
    n_test  = len(cargas_test)
    total   = n_train + n_val + n_test

    # escala de cores global (todas as cargas compartilham o mesmo range)
    todas_amplitudes = list(amplitudes_potencial.values())
    amin = float(min(todas_amplitudes))
    amax = float(max(todas_amplitudes))

    def _coords_centro(carga_id):
        if hasattr(parametros_carga, 'loc'):
            if carga_id in parametros_carga.index:
                row = parametros_carga.loc[carga_id]
            elif 'carga_id' in parametros_carga.columns:
                row = parametros_carga[parametros_carga['carga_id'] == carga_id].iloc[0]
            else:
                row = parametros_carga.iloc[int(carga_id)]
            return float(row['centro_x']), float(row['centro_y'])
        else:
            return (float(parametros_carga['centro_x'][carga_id]),
                    float(parametros_carga['centro_y'][carga_id]))

    def _coords_lista(cargas):
        xs, ys, amps = [], [], []
        for c in cargas:
            xc, yc = _coords_centro(c)
            xs.append(xc)
            ys.append(yc)
            amps.append(amplitudes_potencial[c])
        return xs, ys, amps

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale="Viridis",
            cmin=amin, cmax=amax,
            showscale=True,
            colorbar=dict(title="φ̃ (x,y) máximo", x=1.02, len=0.55),
        ),
        showlegend=False,
        hoverinfo='skip',
        name='colorbar_phi_max',
    ))

    # treino (círculos, borda verde)
    x_train, y_train, a_train = _coords_lista(cargas_train)
    fig.add_trace(go.Scatter(
        x=x_train, y=y_train,
        mode='markers',
        name=f'Treino ({n_train})',
        marker=dict(
            color=a_train,
            colorscale="Viridis",
            cmin=amin, cmax=amax,
            size=11, symbol='circle',
            line=dict(width=2, color='#2E7D32'),
            showscale=False,
        ),
        hovertemplate=(
            "<b>Treino</b><br>"
            "centro_x = %{x:.3f}<br>"
            "centro_y = %{y:.3f}<br>"
            "φ̃ (x,y) máxima = %{marker.color:.3e}<br>"
            "<extra></extra>"
        )
    ))

    # validação (quadrados, borda azul)
    x_val, y_val, a_val = _coords_lista(cargas_val)
    fig.add_trace(go.Scatter(
        x=x_val, y=y_val,
        mode='markers',
        name=f'Validação ({n_val})',
        marker=dict(
            color=a_val,
            colorscale="Viridis",
            cmin=amin, cmax=amax,
            size=11, symbol='square',
            line=dict(width=2, color='#0D47A1'),
            showscale=False,
        ),
        hovertemplate=(
            "<b>Validação</b><br>"
            "centro_x = %{x:.3f}<br>"
            "centro_y = %{y:.3f}<br>"
            "φ̃ (x,y) máxima = %{marker.color:.3e}<br>"
            "<extra></extra>"
        )
    ))

    # teste (losangos, borda laranja)
    x_test, y_test, a_test = _coords_lista(cargas_test)
    fig.add_trace(go.Scatter(
        x=x_test, y=y_test,
        mode='markers',
        name=f'Teste ({n_test})',
        marker=dict(
            color=a_test,
            colorscale="Viridis",
            cmin=amin, cmax=amax,
            size=11, symbol='diamond',
            line=dict(width=2, color='#F57F17'),
            showscale=False,
        ),
        hovertemplate=(
            "<b>Teste</b><br>"
            "centro_x = %{x:.3f}<br>"
            "centro_y = %{y:.3f}<br>"
            "φ̃ (x,y) = %{marker.color:.3e}<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>"
                 f"<span style='font-size:16px; color:#555555;'>"
                 f"Total: {total} Distribuições de Potencial Elétrico | "
                 f"Treino: {n_train} ({n_train/total*100:.1f}%) | "
                 f"Validação: {n_val} ({n_val/total*100:.1f}%) | "
                 f"Teste: {n_test} ({n_test/total*100:.1f}%)</span>",
            x=0.5, y=0.95, font=dict(size=20)
        ),
        xaxis_title="x (Centro da Gaussiana)",
        yaxis_title="y (Centro da Gaussiana)",
        width=1200, height=900,
        legend=dict(
            title="Legenda",
            x=1.02, y=1.00,
            xanchor="left", yanchor="top",
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(t=140, r=200),
        xaxis=dict(
            showgrid=False, zeroline=False,
            title_font=dict(size=16), tickfont=dict(size=14),
            scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            title_font=dict(size=16), tickfont=dict(size=14),
        ),
    )

    return fig


def cria_grafico_distribuicao_cargas(
    cargas_train, cargas_val, cargas_test,
    densidades_por_carga,
    parametros_carga,
    grid_x,
    grid_y,
    titulo="Distribuição de ρ̃ (x,y) por Conjunto de Dados"
):
    """
    Cria gráfico com TODOS os pontos da malha de integração numérica, coloridos
    pelo valor de ρ̃(x,y) da carga atualmente selecionada no dropdown.
    """

    cargas_ids = list(densidades_por_carga.keys())
    n_cargas   = len(cargas_ids)

    n_train = len(cargas_train)
    n_val   = len(cargas_val)
    n_test  = len(cargas_test)
    total   = n_train + n_val + n_test

    amps_global = [float(np.max(np.abs(densidades_por_carga[c]))) for c in cargas_ids]
    amin = float(min(amps_global))
    amax = float(max(amps_global))

    Xg, Yg = np.meshgrid(grid_x, grid_y, indexing="ij")
    xs = Xg.ravel(order="F")
    ys = Yg.ravel(order="F")

    set_train = set(cargas_train.tolist() if hasattr(cargas_train, 'tolist') else list(cargas_train))
    set_val   = set(cargas_val.tolist()   if hasattr(cargas_val,   'tolist') else list(cargas_val))
    set_test  = set(cargas_test.tolist()  if hasattr(cargas_test,  'tolist') else list(cargas_test))

    def _conjunto_de(carga_id):
        if carga_id in set_train:
            return "Treino"
        elif carga_id in set_val:
            return "Validação"
        elif carga_id in set_test:
            return "Teste"
        return "Desconhecido"

    fig = go.Figure()

    for i_carga, carga_id in enumerate(cargas_ids):
        rho2d = densidades_por_carga[carga_id]
        zs = rho2d.ravel(order="F")
        visivel = (i_carga == 0)
        conjunto = _conjunto_de(carga_id)

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            marker=dict(
                size=15,
                color=zs,
                colorscale="Viridis",
                cmin=amin, cmax=amax,
                showscale=False,
                line=dict(width=0),
            ),
            visible=visivel,
            name=f"Densidade de Carga {carga_id} (Dados de {conjunto})",
            showlegend=False,
            hovertemplate=(
                f"<b>Densidade de Carga {carga_id} (Dados de {conjunto})</b><br>"
                "x = %{x:.3f}<br>"
                "y = %{y:.3f}<br>"
                "ρ̃ = %{marker.color:.3e}<br>"
                "<extra></extra>"
            ),
        ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale="Viridis",
            cmin=amin, cmax=amax,
            showscale=True,
            colorbar=dict(
                title="ρ̃ (x,y)",
                x=1.02,
                len=0.55,
            ),
        ),
        showlegend=False,
        hoverinfo='skip',
        name='colorbar_rho',
    ))

    n_tracos_dados = n_cargas
    botoes = []
    for i_carga, carga_id in enumerate(cargas_ids):
        visibilidade = [False] * n_tracos_dados + [True]
        visibilidade[i_carga] = True

        conjunto = _conjunto_de(carga_id)

        botoes.append(dict(
            label=f"ρ̃ (x,y) {carga_id} (Dados de {conjunto})",
            method="update",
            args=[
                {"visible": visibilidade},
                {"title.text":
                    f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>"
                    f"<span style='font-size:16px; color:#555555;'>"
                    f"Densidade de Carga {carga_id}: Dados de <b>{conjunto}</b> | "
                    f"Treino: {n_train} | Validação: {n_val} | Teste: {n_test}</span>"}
            ],
        ))

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>"
                 f"<span style='font-size:16px; color:#555555;'>"
                 f"Treino: {n_train} | Validação: {n_val} | Teste: {n_test}</span>",
            x=0.5, y=0.95, font=dict(size=20)
        ),
        updatemenus=[dict(
            type="dropdown",
            buttons=botoes,
            direction="down",
            x=1.02, y=0.95,
            xanchor="left", yanchor="top",
            showactive=True,
            bgcolor="white", bordercolor="black",
            font=dict(size=13),
        )],
        xaxis_title="x",
        yaxis_title="y",
        width=1200, height=900,
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(t=140, r=180),
        xaxis=dict(
            showgrid=False, zeroline=False,
            title_font=dict(size=16), tickfont=dict(size=14),
            scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            title_font=dict(size=16), tickfont=dict(size=14),
        ),
    )

    return fig


def cria_grafico_historico_treinamento(
    history: Dict,
    titulo: str = "Evolução da Função de Custo Durante o Treinamento"
) -> go.Figure:
    """
    Cria gráfico da evolução das funções de custo de treino e validação ao
    longo das épocas (loss em escala).
    """

    epochs = list(range(1, len(history['train_loss']) + 1))

    fig = go.Figure()

    # função de custo de treino
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['train_loss'],
        mode='lines',
        name='Loss de Treino',
        line=dict(color='#2E7D32', width=2),
        hovertemplate=(
            "<b>Loss de Treino</b><br>" +
            "Época: %{x}<br>" +
            "Loss: %{y:.6f}<br>" +
            "<extra></extra>"
        )
    ))

    # loss de validação
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['val_loss'],
        mode='lines',
        name='Loss de Validação',
        line=dict(color='#B71C1C', width=2),
        hovertemplate=(
            "<b>Loss de Validação</b><br>" +
            "Época: %{x}<br>" +
            "Loss: %{y:.6f}<br>" +
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>",
            x=0.5,
            y=0.95,
            font=dict(size=16)
        ),
        xaxis_title="Época",
        yaxis_title="Função de Custo (MSE)",
        width=1400,
        height=800,
        legend=dict(
            title="Legenda",
            x=0.85,
            y=0.95,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=150, b=80, l=80, r=80),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            title_font=dict(size=16),
            tickfont=dict(size=14),
            type='log'
        )
    )

    return fig


def cria_grafico_previsoes_espaco_fases(
    phi_pred,
    phi_true,
    rho_inputs,
    parametros_cargas,
    x_grid,
    y_grid,
    titulo="Dados Reais vs Previsões do Modelo"
):
    """
    Painel esquerdo: φ̃ real (solução espectral). Painel direito: φ̃ previsto (MLP).
    Cada ponto da malha de integração numérica é colorido pelo valor de φ̃ naquele ponto.
    """

    n_cargas = phi_pred.shape[0]
    N_plus_1 = phi_pred.shape[1]

    phi_vmin = float(min(phi_pred.min(), phi_true.min()))
    phi_vmax = float(max(phi_pred.max(), phi_true.max()))

    MAX_CARGAS_VIS = min(10, n_cargas)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Potencial Elétrico φ̃ Real (Solução Espectral)",
                        "Potencial Elétrico φ̃ Previsto (MLP)"),
        horizontal_spacing=0.12,
    )

    Xg, Yg = np.meshgrid(x_grid, y_grid, indexing="ij")
    xs = Xg.ravel(order="F")
    ys = Yg.ravel(order="F")

    for i_carga in range(MAX_CARGAS_VIS):
        visivel = (i_carga == 0)

        # φ̃ real (painel esquerdo)
        zs_true = phi_true[i_carga].ravel(order="F")
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode='markers',
                marker=dict(
                    size=12,
                    color=zs_true,
                    colorscale="Viridis",
                    cmin=phi_vmin, cmax=phi_vmax,
                    showscale=False,
                    line=dict(width=0),
                ),
                visible=visivel,
                showlegend=False,
                name=f"ρ̃(x,y) {i_carga} (Real (Método Espectral))",
                hovertemplate=(
                    f"<b>ρ̃(x,y) {i_carga} (Real (Método Espectral))</b><br>"
                    "x = %{x:.3f}<br>"
                    "y = %{y:.3f}<br>"
                    "φ̃ = %{marker.color:.3e}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

        # φ̃ previsto (painel direito)
        zs_pred = phi_pred[i_carga].ravel(order="F")
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode='markers',
                marker=dict(
                    size=12,
                    color=zs_pred,
                    colorscale="Viridis",
                    cmin=phi_vmin, cmax=phi_vmax,
                    showscale=False,
                    line=dict(width=0),
                ),
                visible=visivel,
                showlegend=False,
                name=f"ρ̃(x,y) {i_carga} (Previsto (MLP))",
                hovertemplate=(
                    f"<b>ρ̃(x,y) {i_carga} (Previsto (MLP))</b><br>"
                    "x = %{x:.3f}<br>"
                    "y = %{y:.3f}<br>"
                    "φ̃ = %{marker.color:.3e}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1, col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale="Viridis",
                cmin=phi_vmin, cmax=phi_vmax,
                showscale=True,
                colorbar=dict(
                    title="φ̃ (x,y) Real",
                    x=-0.10,
                    len=0.65,
                    xanchor="right",
                ),
            ),
            showlegend=False,
            hoverinfo='skip',
            name='colorbar_phi_real',
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale="Viridis",
                cmin=phi_vmin, cmax=phi_vmax,
                showscale=True,
                colorbar=dict(
                    title="φ̃ (x,y) Previsto",
                    x=1.10,
                    len=0.65,
                    xanchor="left",
                ),
            ),
            showlegend=False,
            hoverinfo='skip',
            name='colorbar_phi_pred',
        ),
        row=1, col=2,
    )

    def _param(carga_idx, chave):
        if hasattr(parametros_cargas, 'iloc'):
            return parametros_cargas.iloc[carga_idx][chave]
        else:
            return parametros_cargas[chave][carga_idx]

    def _metricas_carga(i_carga):
        y_true = phi_true[i_carga].flatten()
        y_pred = phi_pred[i_carga].flatten()
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        return rmse, r2

    def _subtitulo_carga(i_carga):
        rmse, r2 = _metricas_carga(i_carga)
        return (
            f"Densidade de Carga ρ̃(x,y) {i_carga} | "
            f"RMSE = {rmse:.3e} | R² = {r2:.4f}"
        )

    n_tracos_dados = 2 * MAX_CARGAS_VIS
    botoes = []
    for i_carga in range(MAX_CARGAS_VIS):
        visibilidade = [False] * n_tracos_dados + [True, True]
        visibilidade[2 * i_carga]     = True
        visibilidade[2 * i_carga + 1] = True

        botoes.append(dict(
            label=f"ρ̃(x,y) {i_carga} - Parâmetros: A={_param(i_carga, 'amplitude'):.2e}, "
                  f"σ={_param(i_carga, 'largura'):.2f}, "
                  f"c=({_param(i_carga, 'centro_x'):.2f}, "
                  f"{_param(i_carga, 'centro_y'):.2f})",
            method="update",
            args=[
                {"visible": visibilidade},
                {"title.text":
                    f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>"
                    f"<span style='font-size:16px; color:#555555;'>"
                    f"{_subtitulo_carga(i_carga)}</span>"}
            ],
        ))

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>"
                 f"<span style='font-size:16px; color:#555555;'>"
                 f"{_subtitulo_carga(0)}</span>",
            x=0.5, y=0.97, font=dict(size=20),
        ),
        updatemenus=[dict(
            type="dropdown",
            buttons=botoes,
            direction="down",
            x=1.10, y=1.00,
            xanchor="left", yanchor="top",
            showactive=True,
            bgcolor="white", bordercolor="black",
            font=dict(size=13),
        )],
        width=1600, height=700,
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(t=170, r=300, b=60, l=130),
    )

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_yaxes(title_text="y", row=1, col=2, scaleanchor="x", scaleratio=1)

    return fig


def cria_grafico_real_previsto_mlp(
    predictions,
    y_true,
    titulo="Previsões do Modelo MLP"
):
    """
    Cria uma figura com dois painéis lado a lado:
      - esquerda: dispersão real vs. previsto (y=x)
      - direita:  dispersão do erro absoluto |previsto − real| em função
                  do valor real φ̃

    Cada linha de `predictions` e `y_true` corresponde a uma distribuição de
    carga completa (malha inteira achatada). Os arrays são achatados para
    agregação visual ponto a ponto de todas as cargas × todos os pontos da
    malha.
    """

    y_pred_flat = predictions.flatten() if predictions.ndim > 1 else predictions
    y_true_flat = y_true.flatten()     if y_true.ndim     > 1 else y_true

    # erro absoluto ponto a ponto
    erro_absoluto = np.abs(y_pred_flat - y_true_flat)

    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    r2   = float(r2_score(y_true_flat, y_pred_flat))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Real vs. Previsto",
                        "Erro Absoluto vs. Valor Real"),
        horizontal_spacing=0.12,
    )

    # -----------------------------------------------------------------
    # dispersão real vs. previsto
    # -----------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=y_true_flat,
            y=y_pred_flat,
            mode='markers',
            name='Real vs. Previsto',
            marker=dict(color='#BF360C', size=3, opacity=0.5, symbol='circle'),
            hovertemplate=(
                "Real: %{x:.3e}<br>" +
                "Previsto: %{y:.3e}<br>" +
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    min_val = float(min(y_true_flat.min(), y_pred_flat.min()))
    max_val = float(max(y_true_flat.max(), y_pred_flat.max()))

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Referência (y=x)',
            line=dict(color='#1B5E20', width=2, dash='dot'),
            hovertemplate='Referência: %{x:.3e}<extra></extra>',
        ),
        row=1, col=1,
    )

    # -----------------------------------------------------------------
    # erro absoluto vs. valor real
    # -----------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=y_true_flat,
            y=erro_absoluto,
            mode='markers',
            name='Erro Absoluto vs. φ̃ real',
            marker=dict(color='#1565C0', size=3, opacity=0.5, symbol='diamond'),
            hovertemplate=(
                "φ̃ real: %{x:.3e}<br>" +
                "Erro absoluto: %{y:.3e}<br>" +
                "<extra></extra>"
            ),
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>"
                 f"<span style='font-size:20px; color:#555555;'>"
                 f"<sup>RMSE: {rmse:.4e} | R²: {r2:.4f}</sup></span>",
            x=0.5,
            y=0.95,
            font=dict(size=20),
        ),
        width=1500,
        height=700,
        legend=dict(
            title="Legenda",
            x=1.02, y=0.95,
            xanchor="left", yanchor="top",
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=140, r=160, b=60, l=60),
    )

    # eixos do painel 1
    fig.update_xaxes(title_text="φ̃ (x,y) Real", row=1, col=1,
                     showgrid=False, zeroline=True,
                     zerolinecolor='darkgray', zerolinewidth=1,
                     title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(title_text="φ̃ (x,y) Previsto", row=1, col=1,
                     showgrid=False, zeroline=True,
                     zerolinecolor='darkgray', zerolinewidth=1,
                     scaleanchor="x", scaleratio=1,
                     title_font=dict(size=14), tickfont=dict(size=12))

    # eixos do painel 2
    fig.update_xaxes(title_text="φ̃ (x,y) Real", row=1, col=2,
                     showgrid=True, gridcolor='lightgray',
                     title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(title_text="Erro Absoluto", row=1, col=2,
                     showgrid=True, gridcolor='lightgray',
                     title_font=dict(size=14), tickfont=dict(size=12),
                     type='log')

    return fig


# def cria_grafico_trajetorias_completas(
#     posicoes_true, velocidades_true,
#     posicoes_pred, velocidades_pred,
#     tempos,
#     num_trajetorias=5,
#     titulo="Trajetórias Completas: Real vs Previsto"
# ):
#     """
#     Cria gráfico mostrando trajetórias completas no espaço de fases.
    
#     Args:
#         posicoes_true: Posições reais (n_trajetorias, n_timesteps)
#         velocidades_true: Velocidades reais (n_trajetorias, n_timesteps)
#         posicoes_pred: Posições previstas (n_trajetorias, n_timesteps)
#         velocidades_pred: Velocidades previstas (n_trajetorias, n_timesteps)
#         tempos: Array com os tempos
#         num_trajetorias: Número de trajetórias a serem exibidas
#         titulo: Título do gráfico
        
#     Returns:
#         Figura Plotly
#     """
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=('Espaço de Fases', 'Posição e Velocidade vs Tempo'),
#         horizontal_spacing=0.15
#     )
    
#     n_trajetorias = min(num_trajetorias, len(posicoes_true))
#     indices = np.random.choice(len(posicoes_true), n_trajetorias, replace=False)
    
#     for idx, i in enumerate(indices):
#         cor = CORES_PALETA[idx % len(CORES_PALETA)]
        
#         # Espaço de fases
#         # Real
#         fig.add_trace(
#             go.Scatter(
#                 x=posicoes_true[i],
#                 y=velocidades_true[i],
#                 mode='lines',
#                 name=f'Real {i}',
#                 line=dict(color=cor, width=2, dash='solid'),
#                 legendgroup=f'traj_{i}',
#                 showlegend=True,
#                 hovertemplate=(
#                     f"<b>Trajetória Real {i}</b><br>" +
#                     f"Posição: %{{x:.3f}} m<br>" +
#                     f"Velocidade: %{{y:.3f}} m/s<br>" +
#                     f"<extra></extra>"
#                 )
#             ),
#             row=1, col=1
#         )
        
#         # Previsto
#         fig.add_trace(
#             go.Scatter(
#                 x=posicoes_pred[i],
#                 y=velocidades_pred[i],
#                 mode='lines',
#                 name=f'Previsto {i}',
#                 line=dict(color=cor, width=2, dash='dash'),
#                 legendgroup=f'traj_{i}',
#                 showlegend=False,
#                 hovertemplate=(
#                     f"<b>Trajetória Prevista {i}</b><br>" +
#                     f"Posição: %{{x:.3f}} m<br>" +
#                     f"Velocidade: %{{y:.3f}} m/s<br>" +
#                     f"<extra></extra>"
#                 )
#             ),
#             row=1, col=1
#         )
        
#         # Posição vs Tempo
#         fig.add_trace(
#             go.Scatter(
#                 x=tempos,
#                 y=posicoes_true[i],
#                 mode='lines',
#                 name=f'Posição Real {i}',
#                 line=dict(color=cor, width=2, dash='solid'),
#                 legendgroup=f'traj_{i}',
#                 showlegend=False,
#                 hovertemplate=(
#                     f"<b>Posição Real {i}</b><br>" +
#                     f"Tempo: %{{x:.3f}} s<br>" +
#                     f"Posição: %{{y:.3f}} m<br>" +
#                     f"<extra></extra>"
#                 )
#             ),
#             row=1, col=2
#         )
        
#         fig.add_trace(
#             go.Scatter(
#                 x=tempos,
#                 y=posicoes_pred[i],
#                 mode='lines',
#                 name=f'Posição Prevista {i}',
#                 line=dict(color=cor, width=2, dash='dash'),
#                 legendgroup=f'traj_{i}',
#                 showlegend=False,
#                 hovertemplate=(
#                     f"<b>Posição Prevista {i}</b><br>" +
#                     f"Tempo: %{{x:.3f}} s<br>" +
#                     f"Posição: %{{y:.3f}} m<br>" +
#                     f"<extra></extra>"
#                 )
#             ),
#             row=1, col=2
#         )
    
#     fig.update_xaxes(title_text="Posição (m)", row=1, col=1)
#     fig.update_yaxes(title_text="Velocidade (m/s)", row=1, col=1)
#     fig.update_xaxes(title_text="Tempo (s)", row=1, col=2)
#     fig.update_yaxes(title_text="Posição (m)", row=1, col=2)
    
#     fig.update_layout(
#         title=dict(
#             text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span>",
#             x=0.5,
#             y=0.95,
#             font=dict(size=20)
#         ),
#         width=1600,
#         height=800,
#         showlegend=True,
#         legend=dict(
#             title="Legenda",
#             x=1.02,
#             y=0.5,
#             xanchor='left',
#             yanchor='middle',
#             bgcolor='rgba(255, 255, 255, 0.9)',
#             bordercolor='black',
#             borderwidth=1,
#             font=dict(size=12)
#         ),
#         hovermode='closest',
#         plot_bgcolor='white',
#         margin=dict(t=100)
#     )
    
#     return fig


# def cria_grafico_interpolacao_completo(
#     tempos_lista,
#     posicao_lista,
#     velocidade_lista,
#     casos_info,
#     titulo="Interpolação do Modelo: Posição e Velocidade vs Tempo - Oscilador de Van der Pol"
# ):
#     """
#     Cria gráfico 2D combinando posição e velocidade no tempo.
#     Trabalha com trajetórias completas previstas a partir das condições iniciais.
    
#     Args:
#         tempos_lista: Lista de arrays com os tempos para cada caso
#         posicao_lista: Lista de arrays com as posições previstas para cada caso
#         velocidade_lista: Lista de arrays com as velocidades previstas para cada caso
#         casos_info: Lista de dicionários com informações dos casos (nome, x0, y0, cor, nome_legenda)
#         titulo: Título do gráfico
        
#     Returns:
#         Figura Plotly
#     """
#     fig = go.Figure()
    
#     def clarear_cor(cor, fator=0.5):
#         """Clareia uma cor hexadecimal."""
#         cor = cor.lstrip('#')
#         r, g, b = int(cor[0:2], 16), int(cor[2:4], 16), int(cor[4:6], 16)
#         r = min(255, int(r + (255 - r) * fator))
#         g = min(255, int(g + (255 - g) * fator))
#         b = min(255, int(b + (255 - b) * fator))
#         return f'#{r:02x}{g:02x}{b:02x}'
    
#     for i, (tempos, posicao, velocidade) in enumerate(zip(tempos_lista, posicao_lista, velocidade_lista)):
#         caso = casos_info[i]
        
#         nome_legenda = caso.get('nome_legenda', caso['nome'])
        
#         cor_velocidade = clarear_cor(caso['cor'], fator=0.6)
        
#         # posição
#         fig.add_trace(go.Scatter(
#             x=tempos,
#             y=posicao,
#             mode='lines',
#             name=f"{nome_legenda} - Posição",
#             line=dict(color=caso['cor'], width=2, dash='solid'),
#             legendgroup=f"posicao_{i}",
#             hovertemplate=(
#                 f"<b>{nome_legenda} - Posição</b><br>" +
#                 f"x₀ = {caso['x0']:.3f}<br>" +
#                 f"y₀ = {caso['y0']:.3f}<br>" +
#                 f"Tempo: %{{x:.3f}} s<br>" +
#                 f"Posição: %{{y:.3f}}<br>" +
#                 f"<extra></extra>"
#             )
#         ))
        
#         # velocidade
#         fig.add_trace(go.Scatter(
#             x=tempos,
#             y=velocidade,
#             mode='lines',
#             name=f"{nome_legenda} - Velocidade",
#             line=dict(color=cor_velocidade, width=2, dash='solid'),
#             legendgroup=f"velocidade_{i}",
#             hovertemplate=(
#                 f"<b>{nome_legenda} - Velocidade</b><br>" +
#                 f"x₀ = {caso['x0']:.3f}<br>" +
#                 f"y₀ = {caso['y0']:.3f}<br>" +
#                 f"Tempo: %{{x:.3f}} s<br>" +
#                 f"Velocidade: %{{y:.3f}}<br>" +
#                 f"<extra></extra>"
#             )
#         ))
    
#     fig.update_layout(
#         title=dict(
#             text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>",
#             x=0.35,
#             font=dict(size=16)
#         ),
#         xaxis_title="Tempo (s)",
#         yaxis_title="Estado",
#         width=1400,
#         height=900,
#         legend=dict(
#             title="Legenda",
#             x=1.05,
#             y=0.75,
#             xanchor='left',
#             yanchor='top',
#             bgcolor='rgba(255, 255, 255, 0.9)',
#             bordercolor='black',
#             borderwidth=1,
#             font=dict(size=14),
#             itemclick="toggle",
#             itemdoubleclick="toggleothers"
#         ),
#         hovermode='closest',
#         plot_bgcolor='white',
#         paper_bgcolor='white',
#         xaxis=dict(
#             showgrid=True,
#             gridcolor='darkgray',
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         ),
#         yaxis=dict(
#             showgrid=True,
#             gridcolor='darkgray',
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         )
#     )
    
#     return fig


# def cria_grafico_interpolacao_espaco_fases(
#     posicao_lista,
#     velocidade_lista,
#     casos_info,
#     titulo="Interpolação do Modelo no Espaço de Fases - Oscilador de Van der Pol"
# ):
#     """
#     Cria gráfico 2D mostrando as trajetórias previstas no espaço de fases.
#     Trabalha com trajetórias completas previstas a partir das condições iniciais.
    
#     Args:
#         posicao_lista: Lista de arrays com as posições previstas para cada caso
#         velocidade_lista: Lista de arrays com as velocidades previstas para cada caso
#         casos_info: Lista de dicionários com informações dos casos (nome, x0, y0, cor, nome_legenda)
#         titulo: Título do gráfico
        
#     Returns:
#         Figura Plotly
#     """
#     fig = go.Figure()
    
#     for i, (posicao, velocidade) in enumerate(zip(posicao_lista, velocidade_lista)):
#         caso = casos_info[i]
        
#         nome_legenda = caso.get('nome_legenda', caso['nome'])
        
#         # trajetória completa no espaço de fases
#         fig.add_trace(go.Scatter(
#             x=posicao,
#             y=velocidade,
#             mode='lines',
#             name=nome_legenda,
#             line=dict(color=caso['cor'], width=2),
#             hovertemplate=(
#                 f"<b>{nome_legenda}</b><br>" +
#                 f"x₀ = {caso['x0']:.3f}<br>" +
#                 f"y₀ = {caso['y0']:.3f}<br>" +
#                 f"Posição: %{{x:.3f}}<br>" +
#                 f"Velocidade: %{{y:.3f}}<br>" +
#                 f"<extra></extra>"
#             )
#         ))
        
#         # ponto inicial (condição inicial)
#         fig.add_trace(go.Scatter(
#             x=[posicao[0]],
#             y=[velocidade[0]],
#             mode='markers',
#             marker=dict(
#                 color=caso['cor'],
#                 size=12,
#                 symbol='circle',
#                 line=dict(color='white', width=1.5)
#             ),
#             name=f"Início - {nome_legenda}",
#             showlegend=False,
#             hovertemplate=(
#                 f"<b>Condição Inicial - {nome_legenda}</b><br>" +
#                 f"x₀ = {caso['x0']:.3f}<br>" +
#                 f"y₀ = {caso['y0']:.3f}<br>" +
#                 f"<extra></extra>"
#             )
#         ))
        
#         # ponto final (fim da trajetória)
#         fig.add_trace(go.Scatter(
#             x=[posicao[-1]],
#             y=[velocidade[-1]],
#             mode='markers',
#             marker=dict(
#                 color=caso['cor'],
#                 size=10,
#                 symbol='x',
#                 line=dict(color='white', width=1)
#             ),
#             name=f"Fim - {nome_legenda}",
#             showlegend=False,
#             hovertemplate=(
#                 f"<b>Fim da Trajetória - {nome_legenda}</b><br>" +
#                 f"Posição: %{{x:.3f}}<br>" +
#                 f"Velocidade: %{{y:.3f}}<br>" +
#                 f"<extra></extra>"
#             )
#         ))
    
#     fig.update_layout(
#         title=dict(
#             text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
#                  f"<span style='font-size:16px; color:#555555;'>",
#             x=0.35,
#             font=dict(size=16)
#         ),
#         xaxis_title="Posição",
#         yaxis_title="Velocidade",
#         width=1400,
#         height=1000,
#         legend=dict(
#             title="Legenda",
#             x=1.05,
#             y=0.75,
#             bgcolor='rgba(255, 255, 255, 0.9)',
#             bordercolor='black',
#             borderwidth=1,
#             font=dict(size=14),
#             itemclick="toggle",
#             itemdoubleclick="toggleothers"
#         ),
#         hovermode='closest',
#         plot_bgcolor='white',
#         paper_bgcolor='white',
#         xaxis=dict(
#             showgrid=False,
#             gridcolor='darkgray',
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         ),
#         yaxis=dict(
#             showgrid=False,
#             gridcolor='darkgray',
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         )
#     )
    
#     return fig


# def cria_grafico_interpolacao_pontual_mlp(
#     predictions,
#     y_true,
#     titulo="Interpolação Pontual de Posição e Velocidade - Oscilador de Van der Pol"
# ):
#     """
#     Cria gráficos de dispersão para visualizar a interpolação pontual do modelo MLP.
#     Agora trabalha com pontos achatados de trajetórias completas.
    
#     Args:
#         predictions: array com as previsões (n_samples, 2) - [x, y] (pontos achatados)
#         y_true: array com os valores reais (n_samples, 2) - [x, y] (pontos achatados)
#         titulo: título do gráfico
        
#     Returns:
#         Figura Plotly
#     """
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=('Posição', 'Velocidade'),
#         horizontal_spacing=0.15
#     )
    
#     cores = ['blue', 'green']
#     nomes = ['Posição', 'Velocidade']
    
#     for i in range(2):
#         fig.add_trace(
#             go.Scatter(
#                 x=y_true[:, i],
#                 y=predictions[:, i],
#                 mode='markers',
#                 name=f'{nomes[i]}',
#                 marker=dict(
#                     color=cores[i],
#                     size=3,
#                     opacity=0.5
#                 ),
#                 hovertemplate=(
#                     f"<b>{nomes[i]}</b><br>" +
#                     f"Valor Real: %{{x:.3f}}<br>" +
#                     f"Valor Previsto: %{{y:.3f}}<br>" +
#                     f"<extra></extra>"
#                 )
#             ),
#             row=1, col=i+1
#         )
        
#         # linha y=x (referência)
#         min_val = min(y_true[:, i].min(), predictions[:, i].min())
#         max_val = max(y_true[:, i].max(), predictions[:, i].max())
        
#         fig.add_trace(
#             go.Scatter(
#                 x=[min_val, max_val],
#                 y=[min_val, max_val],
#                 mode='lines',
#                 name='Referência (y=x)',
#                 line=dict(color='red', width=2, dash='dash'),
#                 showlegend=(i == 0),  # mostra apenas na primeira coluna
#                 hovertemplate='Referência: %{x:.3f}<extra></extra>'
#             ),
#             row=1, col=i+1
#         )
        
#         fig.update_xaxes(
#             title_text=f'Valor Real {nomes[i]}',
#             row=1, col=i+1,
#             showgrid=True,
#             gridcolor='lightgray',
#             zeroline=True,
#             zerolinecolor='lightgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         )
        
#         fig.update_yaxes(
#             title_text=f'Valor Previsto {nomes[i]}',
#             row=1, col=i+1,
#             showgrid=True,
#             gridcolor='lightgray',
#             zeroline=True,
#             zerolinecolor='lightgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         )
    
#     rmse_posicao = np.sqrt(mean_squared_error(y_true[:, 0], predictions[:, 0]))
#     rmse_velocidade = np.sqrt(mean_squared_error(y_true[:, 1], predictions[:, 1]))
#     r2_posicao = r2_score(y_true[:, 0], predictions[:, 0])
#     r2_velocidade = r2_score(y_true[:, 1], predictions[:, 1])
    
#     fig.update_layout(
#         title=dict(
#             text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
#                  f"<span style='font-size:20px; color:#555555;'>" +
#                  f"<sup>RMSE Posição: {rmse_posicao:.4f} | RMSE Velocidade: {rmse_velocidade:.4f}</sup><br>" +
#                  f"<sup>R² Posição: {r2_posicao:.4f} | R² Velocidade: {r2_velocidade:.4f}</sup><br>",
#             x=0.45,
#             y=0.92,
#             font=dict(size=16)
#         ),
#         width=1400,
#         height=700,
#         showlegend=True,
#         legend=dict(
#             title="Legenda",
#             x=1.02,
#             y=0.5,
#             xanchor='left',
#             yanchor='middle',
#             bgcolor='rgba(255, 255, 255, 0.9)',
#             bordercolor='black',
#             borderwidth=1,
#             font=dict(size=14),
#             itemclick="toggle",
#             itemdoubleclick="toggleothers"
#         ),
#         plot_bgcolor='white',
#         paper_bgcolor='white',
#         hovermode='closest',
#         margin=dict(t=180)
#     )
    
#     return fig


# def cria_grafico_interpolacao_pontual_espaco_fases(
#     y_pos_true, y_vel_true,
#     y_pos_pred, y_vel_pred,
#     titulo="Interpolação Pontual no Espaço de Fases - Oscilador de Van der Pol"
# ):
#     """
#     Cria gráfico 2D mostrando a interpolação pontual do modelo no espaço de fases.
#     Agora mostra trajetórias completas previstas a partir das condições iniciais.
    
#     Args:
#         y_pos_true: Posições reais (pontos achatados)
#         y_vel_true: Velocidades reais (pontos achatados)
#         y_pos_pred: Posições previstas (pontos achatados)
#         y_vel_pred: Velocidades previstas (pontos achatados)
#         titulo: Título do gráfico
        
#     Returns:
#         Figura Plotly
#     """
#     fig = go.Figure()
    
#     # previsões do modelo
#     fig.add_trace(go.Scatter(
#         x=y_pos_pred.flatten(),
#         y=y_vel_pred.flatten(),
#         mode='markers',
#         name='MLP',
#         marker=dict(
#             color='#BF360C',
#             size=3,
#             opacity=0.6,
#             symbol='diamond'
#         ),
#         hovertemplate=(
#             f"<b>MLP</b><br>" +
#             f"Posição: %{{x:.3f}}<br>" +
#             f"Velocidade: %{{y:.3f}}<br>" +
#             f"<extra></extra>"
#         )
#     ))

#     # dados reais (solução RK4)
#     fig.add_trace(go.Scatter(
#         x=y_pos_true.flatten(),
#         y=y_vel_true.flatten(),
#         mode='markers',
#         name='Solução RK4',
#         marker=dict(
#             color='#1A237E',
#             size=3,
#             opacity=0.6,
#             symbol='circle'
#         ),
#         hovertemplate=(
#             f"<b>Solução RK4</b><br>" +
#             f"Posição: %{{x:.3f}}<br>" +
#             f"Velocidade: %{{y:.3f}}<br>" +
#             f"<extra></extra>"
#         )
#     ))

#     rmse_posicao = np.sqrt(mean_squared_error(y_pos_true.flatten(), y_pos_pred.flatten()))
#     rmse_velocidade = np.sqrt(mean_squared_error(y_vel_true.flatten(), y_vel_pred.flatten()))
#     r2_posicao = r2_score(y_pos_true.flatten(), y_pos_pred.flatten())
#     r2_velocidade = r2_score(y_vel_true.flatten(), y_vel_pred.flatten())
    
#     fig.update_layout(
#         title=dict(
#             text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
#                  f"<span style='font-size:20px; color:#555555;'>" +
#                  f"<sup>RMSE Posição: {rmse_posicao:.4f} | RMSE Velocidade: {rmse_velocidade:.4f}</sup><br>" +
#                  f"<sup>R² Posição: {r2_posicao:.4f} | R² Velocidade: {r2_velocidade:.4f}</sup><br>",
#             x=0.5,
#             y=0.92,
#             font=dict(size=16)
#         ),
#         xaxis_title="Posição",
#         yaxis_title="Velocidade",
#         width=1400,
#         height=1000,
#         legend=dict(
#             title="Legenda",
#             x=0.95,
#             y=0.85,
#             xanchor='left',
#             yanchor='middle',
#             bgcolor='rgba(255, 255, 255, 0.9)',
#             bordercolor='black',
#             borderwidth=1,
#             font=dict(size=14),
#             itemclick="toggle",
#             itemdoubleclick="toggleothers"
#         ),
#         hovermode='closest',
#         plot_bgcolor='white',
#         paper_bgcolor='white',
#         margin=dict(t=180),
#         xaxis=dict(
#             showgrid=False,
#             gridcolor='darkgray',
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         ),
#         yaxis=dict(
#             showgrid=False,
#             gridcolor='darkgray',
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         )
#     )
    
#     return fig


# def cria_grafico_interpolacao_pontual_completo(
#     tempos_lista,
#     posicoes_previstas_lista,
#     velocidades_previstas_lista,
#     posicoes_reais_lista,
#     velocidades_reais_lista,
#     casos_info,
#     titulo="Interpolação Pontual: Posição e Velocidade vs Tempo - Oscilador de Van der Pol"
# ):
#     """
#     Cria gráfico 2D combinando posição e velocidade no tempo para interpolação pontual.
#     Mostra trajetórias completas previstas a partir das condições iniciais.
    
#     Args:
#         tempos_lista: Lista de arrays com os tempos para cada caso
#         posicoes_previstas_lista: Lista de arrays com as posições previstas pelo MLP
#         velocidades_previstas_lista: Lista de arrays com as velocidades previstas pelo MLP
#         posicoes_reais_lista: Lista de arrays com as posições reais (solução RK4)
#         velocidades_reais_lista: Lista de arrays com as velocidades reais (solução RK4)
#         casos_info: Lista de dicionários com informações dos casos (x0, y0, cor)
#         titulo: Título do gráfico
        
#     Returns:
#         Figura Plotly
#     """
#     fig = go.Figure()
    
#     def clarear_cor(cor, fator=0.5):
#         """Clareia uma cor hexadecimal."""
#         cor = cor.lstrip('#')
#         r, g, b = int(cor[0:2], 16), int(cor[2:4], 16), int(cor[4:6], 16)
#         r = min(255, int(r + (255 - r) * fator))
#         g = min(255, int(g + (255 - g) * fator))
#         b = min(255, int(b + (255 - b) * fator))
#         return f'#{r:02x}{g:02x}{b:02x}'
    
#     for i, (tempos, pos_prev, vel_prev, pos_real, vel_real) in enumerate(zip(
#         tempos_lista, posicoes_previstas_lista, velocidades_previstas_lista,
#         posicoes_reais_lista, velocidades_reais_lista
#     )):
#         caso = casos_info[i]
        
#         nome_sistema = f"x₀={caso['x0']:.2f}, y₀={caso['y0']:.2f}"
        
#         cor_velocidade = clarear_cor(caso['cor'], fator=0.6)
        
#         # posição prevista pelo MLP
#         fig.add_trace(go.Scatter(
#             x=tempos,
#             y=pos_prev,
#             mode='lines',
#             name=f"{nome_sistema} - Posição (MLP)",
#             line=dict(color=caso['cor'], width=2, dash='solid'),
#             legendgroup=f"posicao_mlp_{i}",
#             hovertemplate=(
#                 f"<b>Posição (MLP)</b><br>" +
#                 f"x₀ = {caso['x0']:.3f}<br>" +
#                 f"y₀ = {caso['y0']:.3f}<br>" +
#                 f"Tempo: %{{x:.3f}} s<br>" +
#                 f"Posição Prevista: %{{y:.3f}}<br>" +
#                 f"<extra></extra>"
#             )
#         ))
        
#         # posição real (solução RK4)
#         fig.add_trace(go.Scatter(
#             x=tempos,
#             y=pos_real,
#             mode='lines',
#             name=f"{nome_sistema} - Posição (Real)",
#             line=dict(color=caso['cor'], width=1.5, dash='dot'),
#             legendgroup=f"posicao_real_{i}",
#             hovertemplate=(
#                 f"<b>Posição (Real)</b><br>" +
#                 f"x₀ = {caso['x0']:.3f}<br>" +
#                 f"y₀ = {caso['y0']:.3f}<br>" +
#                 f"Tempo: %{{x:.3f}} s<br>" +
#                 f"Posição Real: %{{y:.3f}}<br>" +
#                 f"<extra></extra>"
#             )
#         ))
        
#         # velocidade prevista pelo MLP
#         fig.add_trace(go.Scatter(
#             x=tempos,
#             y=vel_prev,
#             mode='lines',
#             name=f"{nome_sistema} - Velocidade (MLP)",
#             line=dict(color=cor_velocidade, width=2, dash='solid'),
#             legendgroup=f"velocidade_mlp_{i}",
#             hovertemplate=(
#                 f"<b>Velocidade (MLP)</b><br>" +
#                 f"x₀ = {caso['x0']:.3f}<br>" +
#                 f"y₀ = {caso['y0']:.3f}<br>" +
#                 f"Tempo: %{{x:.3f}} s<br>" +
#                 f"Velocidade Prevista: %{{y:.3f}}<br>" +
#                 f"<extra></extra>"
#             )
#         ))
        
#         # velocidade real (solução RK4)
#         fig.add_trace(go.Scatter(
#             x=tempos,
#             y=vel_real,
#             mode='lines',
#             name=f"{nome_sistema} - Velocidade (Real)",
#             line=dict(color=cor_velocidade, width=1.5, dash='dot'),
#             legendgroup=f"velocidade_real_{i}",
#             hovertemplate=(
#                 f"<b>Velocidade (Real)</b><br>" +
#                 f"x₀ = {caso['x0']:.3f}<br>" +
#                 f"y₀ = {caso['y0']:.3f}<br>" +
#                 f"Tempo: %{{x:.3f}} s<br>" +
#                 f"Velocidade Real: %{{y:.3f}}<br>" +
#                 f"<extra></extra>"
#             )
#         ))
    
#     fig.update_layout(
#         title=dict(
#             text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
#                  f"<span style='font-size:16px; color:#555555;'>",
#             x=0.5,
#             y=0.95,
#             font=dict(size=16)
#         ),
#         xaxis_title="Tempo (s)",
#         yaxis_title="Estado",
#         width=1400,
#         height=900,
#         legend=dict(
#             title="Legenda",
#             x=1.02,
#             y=0.75,
#             xanchor='left',
#             yanchor='top',
#             bgcolor='rgba(255, 255, 255, 0.9)',
#             bordercolor='black',
#             borderwidth=1,
#             font=dict(size=14),
#             itemclick="toggle",
#             itemdoubleclick="toggleothers"
#         ),
#         hovermode='closest',
#         plot_bgcolor='white',
#         paper_bgcolor='white',
#         margin=dict(t=150),
#         xaxis=dict(
#             showgrid=True,
#             gridcolor='darkgray',
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         ),
#         yaxis=dict(
#             showgrid=True,
#             gridcolor='darkgray',
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         )
#     )
    
#     return fig


# def cria_grafico_interpolacao_entre_trajetorias_espaco_fases(
#     trajetoria1_pos,
#     trajetoria1_vel,
#     trajetoria2_pos,
#     trajetoria2_vel,
#     interpolacoes_lista,
#     casos_info,
#     titulo="Interpolação entre Trajetórias no Espaço de Fases - Oscilador de Van der Pol",
#     cores_paleta=CORES_PALETA
# ):
#     """
#     Cria gráfico 2D mostrando as duas trajetórias originais e as trajetórias interpoladas no espaço de fases.
#     Trabalha com trajetórias completas previstas a partir das condições iniciais.
#     Mostra os pontos inicial e final de cada trajetória.
    
#     Args:
#         trajetoria1_pos: Array com as posições da primeira trajetória
#         trajetoria1_vel: Array com as velocidades da primeira trajetória
#         trajetoria2_pos: Array com as posições da segunda trajetória
#         trajetoria2_vel: Array com as velocidades da segunda trajetória
#         interpolacoes_lista: Lista de dicionários contendo alpha, posicoes, velocidades, x0_interp, v0_interp
#         casos_info: Lista de dicionários com informações dos casos interpolados
#         titulo: Título do gráfico
#         cores_paleta: Lista de cores para as trajetórias interpoladas
        
#     Returns:
#         Figura Plotly
#     """
    
#     fig = go.Figure()
    
#     # trajetória 1 (alpha = 0)
#     caso_info = casos_info[0] if casos_info else {}
#     fig.add_trace(go.Scatter(
#         x=trajetoria1_pos,
#         y=trajetoria1_vel,
#         mode='lines',
#         name=f"Trajetória 1: x₀={caso_info.get('x0_1', 0):.3f}, y₀={caso_info.get('v0_1', 0):.3f}",
#         line=dict(color='blue', width=3, dash='dash'),
#         hovertemplate=(
#             f"<b>Trajetória 1</b><br>" +
#             f"x₀ = {caso_info.get('x0_1', 0):.3f}<br>" +
#             f"y₀ = {caso_info.get('v0_1', 0):.3f}<br>" +
#             f"Posição: %{{x:.3f}}<br>" +
#             f"Velocidade: %{{y:.3f}}<br>" +
#             f"<extra></extra>"
#         )
#     ))
    
#     # trajetória 2 (alpha = 1)
#     fig.add_trace(go.Scatter(
#         x=trajetoria2_pos,
#         y=trajetoria2_vel,
#         mode='lines',
#         name=f"Trajetória 2: x₀={caso_info.get('x0_2', 0):.3f}, y₀={caso_info.get('v0_2', 0):.3f}",
#         line=dict(color='red', width=3, dash='dash'),
#         hovertemplate=(
#             f"<b>Trajetória 2</b><br>" +
#             f"x₀ = {caso_info.get('x0_2', 0):.3f}<br>" +
#             f"y₀ = {caso_info.get('v0_2', 0):.3f}<br>" +
#             f"Posição: %{{x:.3f}}<br>" +
#             f"Velocidade: %{{y:.3f}}<br>" +
#             f"<extra></extra>"
#         )
#     ))
    
#     # trajetórias interpoladas (0 < alpha < 1) - trajetórias completas
#     n_interpolacoes = len(interpolacoes_lista)
#     if n_interpolacoes > 0:
#         indices_cores = np.linspace(0, len(cores_paleta) - 1, n_interpolacoes, dtype=int)
        
#         for i, interpolacao in enumerate(interpolacoes_lista):
#             alpha = interpolacao['alpha']
#             posicoes = interpolacao['posicoes']
#             velocidades = interpolacao['velocidades']
#             x0_interp = interpolacao['x0_interp']
#             v0_interp = interpolacao['v0_interp']
            
#             cor = cores_paleta[indices_cores[i] % len(cores_paleta)]
            
#             # linha da trajetória interpolada
#             fig.add_trace(go.Scatter(
#                 x=posicoes,
#                 y=velocidades,
#                 mode='lines',
#                 name=f"Trajetória Interpolada: x₀={x0_interp:.3f}, y₀={v0_interp:.3f}",
#                 line=dict(color=cor, width=3, dash='solid'),
#                 opacity=0.7,
#                 hovertemplate=(
#                     f"<b>Trajetória Interpolada</b><br>" +
#                     f"x₀_interp = {x0_interp:.3f}<br>" +
#                     f"y₀_interp = {v0_interp:.3f}<br>" +
#                     f"Posição: %{{x:.3f}}<br>" +
#                     f"Velocidade: %{{y:.3f}}<br>" +
#                     f"<extra></extra>"
#                 )
#             ))
            
#             # ponto inicial da trajetória interpolada
#             fig.add_trace(go.Scatter(
#                 x=[posicoes[0]],
#                 y=[velocidades[0]],
#                 mode='markers',
#                 marker=dict(
#                     color=cor,
#                     size=8,
#                     symbol='circle',
#                     line=dict(color='white', width=1)
#                 ),
#                 name=f"Início Trajetória Interpolada",
#                 showlegend=False,
#                 hovertemplate=(
#                     f"<b>Início Trajetória Interpolada</b><br>" +
#                     f"x₀ = {x0_interp:.3f}<br>" +
#                     f"y₀ = {v0_interp:.3f}<br>" +
#                     f"<extra></extra>"
#                 )
#             ))
            
#             # ponto final da trajetória interpolada
#             fig.add_trace(go.Scatter(
#                 x=[posicoes[-1]],
#                 y=[velocidades[-1]],
#                 mode='markers',
#                 marker=dict(
#                     color=cor,
#                     size=8,
#                     symbol='x',
#                     line=dict(color='white', width=1)
#                 ),
#                 name=f"Fim Trajetória Interpolada",
#                 showlegend=False,
#                 hovertemplate=(
#                     f"<b>Fim Trajetória Interpolada</b><br>" +
#                     f"x_final = {posicoes[-1]:.3f}<br>" +
#                     f"y_final = {velocidades[-1]:.3f}<br>" +
#                     f"<extra></extra>"
#                 )
#             ))
    
#     # ponto inicial da trajetória 1
#     fig.add_trace(go.Scatter(
#         x=[trajetoria1_pos[0]],
#         y=[trajetoria1_vel[0]],
#         mode='markers',
#         marker=dict(color='blue', size=12, symbol='circle', line=dict(color='white', width=2)),
#         name="Início Trajetória 1",
#         showlegend=False,
#         hovertemplate=f"<b>Início Trajetória 1</b><br>x₀ = {caso_info.get('x0_1', 0):.3f}<br>y₀ = {caso_info.get('v0_1', 0):.3f}<br><extra></extra>"
#     ))
    
#     # ponto final da trajetória 1
#     fig.add_trace(go.Scatter(
#         x=[trajetoria1_pos[-1]],
#         y=[trajetoria1_vel[-1]],
#         mode='markers',
#         marker=dict(color='blue', size=12, symbol='x', line=dict(color='white', width=2)),
#         name="Fim Trajetória 1",
#         showlegend=False,
#         hovertemplate=f"<b>Fim Trajetória 1</b><br>x_final = {trajetoria1_pos[-1]:.3f}<br>y_final = {trajetoria1_vel[-1]:.3f}<br><extra></extra>"
#     ))
    
#     # ponto inicial da trajetória 2
#     fig.add_trace(go.Scatter(
#         x=[trajetoria2_pos[0]],
#         y=[trajetoria2_vel[0]],
#         mode='markers',
#         marker=dict(color='red', size=12, symbol='circle', line=dict(color='white', width=2)),
#         name="Início Trajetória 2",
#         showlegend=False,
#         hovertemplate=f"<b>Início Trajetória 2</b><br>x₀ = {caso_info.get('x0_2', 0):.3f}<br>y₀ = {caso_info.get('v0_2', 0):.3f}<br><extra></extra>"
#     ))
    
#     # ponto final da trajetória 2
#     fig.add_trace(go.Scatter(
#         x=[trajetoria2_pos[-1]],
#         y=[trajetoria2_vel[-1]],
#         mode='markers',
#         marker=dict(color='red', size=12, symbol='x', line=dict(color='white', width=2)),
#         name="Fim Trajetória 2",
#         showlegend=False,
#         hovertemplate=f"<b>Fim Trajetória 2</b><br>x_final = {trajetoria2_pos[-1]:.3f}<br>y_final = {trajetoria2_vel[-1]:.3f}<br><extra></extra>"
#     ))
     
#     fig.update_layout(
#         title=dict(
#             text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
#                  f"<span style='font-size:16px; color:#555555;'>",
#             x=0.45,
#             y=0.92,
#             font=dict(size=16)
#         ),
#         xaxis_title="Posição",
#         yaxis_title="Velocidade",
#         width=1400,
#         height=1000,
#         legend=dict(
#             title="Legenda",
#             x=1.00,
#             y=0.75,
#             xanchor='left',
#             yanchor='middle',
#             bgcolor='rgba(255, 255, 255, 0.95)',
#             bordercolor='gray',
#             borderwidth=1,
#             font=dict(size=14),
#             itemclick="toggle",
#             itemdoubleclick="toggleothers"
#         ),
#         hovermode='closest',
#         plot_bgcolor='white',
#         paper_bgcolor='white',
#         margin=dict(t=150),
#         xaxis=dict(
#             showgrid=False,
#             gridcolor='darkgray',
#             gridwidth=0.5,
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         ),
#         yaxis=dict(
#             showgrid=False,
#             gridcolor='darkgray',
#             gridwidth=0.5,
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         )
#     )
    
#     return fig


# def cria_grafico_interpolacao_trajetorias_espaco_fases(
#     trajetoria_base_pos,
#     trajetoria_base_vel,
#     novas_trajetorias_lista,
#     casos_info,
#     titulo="Trajetória Base vs Novas Condições Iniciais no Espaço de Fases - Oscilador de Van der Pol",
#     cores_paleta=CORES_PALETA
# ):
#     """
#     Cria gráfico 2D mostrando a trajetória base e as novas trajetórias geradas a partir de diferentes condições iniciais no espaço de fases.
#     Trabalha com trajetórias completas previstas a partir das condições iniciais.
#     Mostra os pontos inicial e final de cada trajetória.
    
#     Args:
#         trajetoria_base_pos: Array com as posições da trajetória base
#         trajetoria_base_vel: Array com as velocidades da trajetória base
#         novas_trajetorias_lista: Lista de dicionários contendo posicoes, velocidades, x0, v0, variacao_id
#         casos_info: Dicionário com informações da trajetória base
#         titulo: Título do gráfico
#         cores_paleta: Lista de cores para as novas trajetórias
        
#     Returns:
#         Figura Plotly
#     """
    
#     fig = go.Figure()
    
#     # trajetória base
#     caso_info = casos_info if casos_info else {}
#     fig.add_trace(go.Scatter(
#         x=trajetoria_base_pos,
#         y=trajetoria_base_vel,
#         mode='lines',
#         name=f"Trajetória Base: x₀={caso_info.get('x0_base', 0):.3f}, y₀={caso_info.get('v0_base', 0):.3f}",
#         line=dict(color='blue', width=3, dash='dash'),
#         hovertemplate=(
#             f"<b>Trajetória Base</b><br>" +
#             f"x₀ = {caso_info.get('x0_base', 0):.3f}<br>" +
#             f"y₀ = {caso_info.get('v0_base', 0):.3f}<br>" +
#             f"Posição: %{{x:.3f}}<br>" +
#             f"Velocidade: %{{y:.3f}}<br>" +
#             f"<extra></extra>"
#         )
#     ))
    
#     # novas trajetórias - trajetórias completas
#     n_novas = len(novas_trajetorias_lista)
#     if n_novas > 0:
#         indices_cores = np.linspace(0, len(cores_paleta) - 1, n_novas, dtype=int)
        
#         for i, trajetoria in enumerate(novas_trajetorias_lista):
#             posicoes = trajetoria['posicoes']
#             velocidades = trajetoria['velocidades']
#             x0_novo = trajetoria['x0']
#             v0_novo = trajetoria['v0']
#             variacao_id = trajetoria.get('variacao_id', i)
            
#             cor = cores_paleta[indices_cores[i] % len(cores_paleta)]
            
#             # linha da nova trajetória
#             fig.add_trace(go.Scatter(
#                 x=posicoes,
#                 y=velocidades,
#                 mode='lines',
#                 name=f"Trajetória Interpolada: x₀={x0_novo:.3f}, y₀={v0_novo:.3f}",
#                 line=dict(color=cor, width=2, dash='solid'),
#                 opacity=0.7,
#                 hovertemplate=(
#                     f"<b>Trajetória Interpolada</b><br>" +
#                     f"x₀ = {x0_novo:.3f}<br>" +
#                     f"y₀ = {v0_novo:.3f}<br>" +
#                     f"Posição: %{{x:.3f}}<br>" +
#                     f"Velocidade: %{{y:.3f}}<br>" +
#                     f"<extra></extra>"
#                 )
#             ))
            
#             # ponto inicial da nova trajetória
#             fig.add_trace(go.Scatter(
#                 x=[posicoes[0]],
#                 y=[velocidades[0]],
#                 mode='markers',
#                 marker=dict(
#                     color=cor,
#                     size=8,
#                     symbol='circle',
#                     line=dict(color='white', width=1)
#                 ),
#                 name=f"Início Trajetória Interpolada",
#                 showlegend=False,
#                 hovertemplate=(
#                     f"<b>Início Trajetória Interpolada</b><br>" +
#                     f"x₀ = {x0_novo:.3f}<br>" +
#                     f"y₀ = {v0_novo:.3f}<br>" +
#                     f"<extra></extra>"
#                 )
#             ))
            
#             # ponto final da nova trajetória
#             fig.add_trace(go.Scatter(
#                 x=[posicoes[-1]],
#                 y=[velocidades[-1]],
#                 mode='markers',
#                 marker=dict(
#                     color=cor,
#                     size=8,
#                     symbol='x',
#                     line=dict(color='white', width=1)
#                 ),
#                 name=f"Fim Trajetória Interpolada",
#                 showlegend=False,
#                 hovertemplate=(
#                     f"<b>Fim Trajetória Interpolada</b><br>" +
#                     f"x_final = {posicoes[-1]:.3f}<br>" +
#                     f"y_final = {velocidades[-1]:.3f}<br>" +
#                     f"<extra></extra>"
#                 )
#             ))
    
#     # ponto inicial da trajetória base
#     fig.add_trace(go.Scatter(
#         x=[trajetoria_base_pos[0]],
#         y=[trajetoria_base_vel[0]],
#         mode='markers',
#         marker=dict(color='blue', size=12, symbol='circle', line=dict(color='white', width=2)),
#         name="Início Trajetória Base",
#         showlegend=False,
#         hovertemplate=f"<b>Início Trajetória Base</b><br>x₀ = {caso_info.get('x0_base', 0):.3f}<br>y₀ = {caso_info.get('v0_base', 0):.3f}<br><extra></extra>"
#     ))
    
#     # ponto final da trajetória base
#     fig.add_trace(go.Scatter(
#         x=[trajetoria_base_pos[-1]],
#         y=[trajetoria_base_vel[-1]],
#         mode='markers',
#         marker=dict(color='blue', size=12, symbol='x', line=dict(color='white', width=2)),
#         name="Fim Trajetória Base",
#         showlegend=False,
#         hovertemplate=f"<b>Fim Trajetória Base</b><br>x_final = {trajetoria_base_pos[-1]:.3f}<br>y_final = {trajetoria_base_vel[-1]:.3f}<br><extra></extra>"
#     ))
    
#     fig.update_layout(
#         title=dict(
#             text=f"<span style='font-size:20px; font-weight:bold;'>{titulo}</span><br><br>" +
#                  f"<span style='font-size:16px; color:#555555;'>",
#             x=0.45,
#             y=0.92,
#             font=dict(size=16)
#         ),
#         xaxis_title="Posição",
#         yaxis_title="Velocidade",
#         width=1400,
#         height=1000,
#         legend=dict(
#             title="Legenda",
#             x=1.00,
#             y=0.75,
#             xanchor='left',
#             yanchor='middle',
#             bgcolor='rgba(255, 255, 255, 0.95)',
#             bordercolor='gray',
#             borderwidth=1,
#             font=dict(size=14),
#             itemclick="toggle",
#             itemdoubleclick="toggleothers"
#         ),
#         hovermode='closest',
#         plot_bgcolor='white',
#         paper_bgcolor='white',
#         margin=dict(t=150),
#         xaxis=dict(
#             showgrid=False,
#             gridcolor='darkgray',
#             gridwidth=0.5,
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         ),
#         yaxis=dict(
#             showgrid=False,
#             gridcolor='darkgray',
#             gridwidth=0.5,
#             zeroline=True,
#             zerolinecolor='darkgray',
#             zerolinewidth=1,
#             title_font=dict(size=16),
#             tickfont=dict(size=16)
#         )
#     )
    
#     return fig