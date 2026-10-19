"""
Nodes do pipeline MLP para previsão do potencial elétrico φ(x,y) a partir
da densidade de carga ρ(x,y) na equação de Poisson 2D em eletrostática.

O modelo recebe como entrada os valores de ρ(x,y) em toda a malha de colocação
(achatados em um vetor) e devolve os valores de φ(x,y) nos mesmos pontos
(achatados no mesmo formato). A divisão treino/validação/teste é feita por
distribuição de carga como amostra completa.
"""

import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# from scipy.interpolate import interp1d
from typing import Dict, Any, Tuple
from .model import MLP
# from equacao_poisson.pipelines.p00_data_generating.ep import PoissonEletrostatico
from equacao_poisson.utils import (
    CORES_PALETA,
    cria_grafico_historico_treinamento,
    cria_grafico_real_previsto_mlp,
    cria_grafico_previsoes_espaco_fases,
    # cria_grafico_interpolacao_completo,
    # cria_grafico_interpolacao_espaco_fases,
    # cria_grafico_interpolacao_pontual_mlp,
    # cria_grafico_interpolacao_pontual_espaco_fases,
    # cria_grafico_interpolacao_pontual_completo,
    # cria_grafico_interpolacao_entre_trajetorias_espaco_fases,
    # cria_grafico_interpolacao_trajetorias_espaco_fases,
)


def fixa_sementes(seed: int = 42):
    """Fixa todas as sementes para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepara_dados_mlp_node(base_eletrostatica: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple:
    """
    Prepara os dados para treinamento do MLP para a equação de Poisson 2D.

    Cada distribuição de carga é uma amostra:
        Entrada: ρ(x,y) em toda a malha, achatado em (N+1)²
        Saída:   φ(x,y) em toda a malha, achatado em (N+1)²

    Também propaga os parâmetros sorteados de cada carga (amplitude, largura,
    centro_x, centro_y), que ficam disponíveis para anotar as visualizações
    posteriores — análogo ao que o caso do oscilador fazia com x0, y0.

    A divisão treino/validação/teste é feita por amostra completa de carga.
    """
    seed = parameters.get('seed', 42)
    fixa_sementes(seed)

    for col in base_eletrostatica.columns:
        if base_eletrostatica[col].dtype == 'object':
            try:
                base_eletrostatica[col] = pd.to_numeric(
                    base_eletrostatica[col].astype(str).str.replace(',', '.'),
                    errors='coerce'
                )
            except Exception:
                pass

    if base_eletrostatica[['x', 'y', 'densidade_carga', 'potencial']].isnull().any().any():
        print("  AVISO: Valores NaN detectados nas colunas numéricas!")
        base_eletrostatica = base_eletrostatica.dropna(
            subset=['x', 'y', 'densidade_carga', 'potencial']
        )

    print(f"\n=== BASE DE DADOS ===")
    print(f"  Total de linhas da base: {len(base_eletrostatica)}")

    cargas_unicas = base_eletrostatica['carga_id'].unique()
    print(f"  Total de distribuições de carga únicas: {len(cargas_unicas)}")

    print(f"\n=== PREPARAÇÃO DOS DADOS ===")

    # agrupa por carga e achata ρ(x,y) e φ(x,y) de cada uma em vetores
    # além disso, extrai os parâmetros sorteados da gaussiana de cada carga
    X_list = []
    y_list = []
    params_list = []

    # colunas de parâmetros que serão propagadas (se existirem na base)
    colunas_param = [c for c in ['amplitude', 'largura', 'centro_x', 'centro_y']
                     if c in base_eletrostatica.columns]

    for carga_id in cargas_unicas:
        grupo = base_eletrostatica[base_eletrostatica['carga_id'] == carga_id]
        grupo = grupo.sort_values(['x_col', 'y_col'])

        rho_carga = grupo['densidade_carga'].values.astype(np.float32)
        phi_carga = grupo['potencial'].values.astype(np.float32)

        X_list.append(rho_carga)
        y_list.append(phi_carga)

        # extrai os parâmetros da primeira linha do grupo (idênticos em todas)
        if colunas_param:
            primeira_linha = grupo.iloc[0]
            params_carga = {c: primeira_linha[c] for c in colunas_param}
        else:
            params_carga = {}
        params_list.append(params_carga)

    X_raw = np.array(X_list, dtype=np.float32)   # (n_cargas, (N+1)²)
    y_raw = np.array(y_list, dtype=np.float32)   # (n_cargas, (N+1)²)
    params_df = pd.DataFrame(params_list)        # (n_cargas, n_parametros)

    print(f"\n  Distribuições válidas: {len(X_raw)}")
    print(f"  Dimensão entrada: {X_raw.shape[1]} (pontos de colocação)")
    print(f"  Dimensão saída: {y_raw.shape[1]} (pontos de colocação)")
    if colunas_param:
        print(f"  Parâmetros propagados por carga: {colunas_param}")

    # divisão por carga (70/20/10)
    n_cargas = len(X_raw)
    indices = np.random.permutation(n_cargas)
    n_train = int(0.7 * n_cargas)
    n_val   = int(0.2 * n_cargas)

    train_indices = indices[:n_train]
    val_indices   = indices[n_train:n_train + n_val]
    test_indices  = indices[n_train + n_val:]

    X_train = X_raw[train_indices]
    y_train = y_raw[train_indices]
    X_val   = X_raw[val_indices]
    y_val   = y_raw[val_indices]
    X_test  = X_raw[test_indices]
    y_test  = y_raw[test_indices]

    cargas_train = np.array(cargas_unicas)[train_indices]
    cargas_val   = np.array(cargas_unicas)[val_indices]
    cargas_test  = np.array(cargas_unicas)[test_indices]

    # separa os parâmetros por conjunto, na mesma ordem das cargas
    params_train = params_df.iloc[train_indices].reset_index(drop=True)
    params_val   = params_df.iloc[val_indices].reset_index(drop=True)
    params_test  = params_df.iloc[test_indices].reset_index(drop=True)

    print(f"\n  Cargas de treino: {len(X_train)}")
    print(f"  Cargas de validação: {len(X_val)}")
    print(f"  Cargas de teste: {len(X_test)}")

    # -------------------------------------------------------------
    # Normalização com tratamento de colunas de variância zero
    # -------------------------------------------------------------
    scaler_X = StandardScaler()
    scaler_X.fit(X_train)
    zero_scale_X = (scaler_X.scale_ == 0)
    if zero_scale_X.any():
        print(f"  AVISO: {zero_scale_X.sum()} colunas de X têm variância zero. "
              f"Ajustando scale para 1.0.")
        scaler_X.scale_[zero_scale_X] = 1.0

    scaler_y = StandardScaler()
    scaler_y.fit(y_train)
    zero_scale_y = (scaler_y.scale_ == 0)
    if zero_scale_y.any():
        print(f"  AVISO: {zero_scale_y.sum()} colunas de y têm variância zero. "
              f"Ajustando scale para 1.0.")
        scaler_y.scale_[zero_scale_y] = 1.0

    X_scaled_train = scaler_X.transform(X_train)
    X_scaled_val   = scaler_X.transform(X_val)
    X_scaled_test  = scaler_X.transform(X_test)

    y_scaled_train = scaler_y.transform(y_train)
    y_scaled_val   = scaler_y.transform(y_val)
    y_scaled_test  = scaler_y.transform(y_test)

    input_dim  = X_raw.shape[1]
    output_dim = y_raw.shape[1]

    print(f"\n  Dimensão entrada: {input_dim}")
    print(f"  Dimensão saída: {output_dim}")

    return (X_scaled_train, y_scaled_train,
            X_scaled_val, y_scaled_val,
            X_scaled_test, y_scaled_test,
            input_dim, output_dim, scaler_X, scaler_y,
            cargas_train, cargas_val, cargas_test,
            params_train, params_val, params_test,
            n_cargas)


def visualiza_distribuicao_dados_separado(
    base_eletrostatica: pd.DataFrame,
    parameters: Dict[str, Any]
) -> None:
    """
    Node separado para visualizar a distribuição dos dados.

    Carrega a base consolidada, calcula amplitudes de ρ e φ por carga e gera
    um gráfico das cargas por conjunto (treino/validação/teste) em função
    da amplitude de ρ, para checar balanceamento da divisão.
    """
    data_version = parameters.get('data_version', 'default_v1')
    output_dir = f"data/08_reporting/{data_version}"
    os.makedirs(output_dir, exist_ok=True)

    for col in base_eletrostatica.columns:
        if base_eletrostatica[col].dtype == 'object':
            try:
                base_eletrostatica[col] = base_eletrostatica[col].astype(str).str.replace(',', '.').astype(float)
            except Exception:
                pass

    cargas_unicas = base_eletrostatica['carga_id'].unique()

    # amplitudes por carga
    amplitudes_carga = {}
    amplitudes_potencial = {}
    for carga_id in cargas_unicas:
        grupo = base_eletrostatica[base_eletrostatica['carga_id'] == carga_id]
        amplitudes_carga[carga_id]     = np.max(np.abs(grupo['densidade_carga'].values))
        amplitudes_potencial[carga_id] = np.max(np.abs(grupo['potencial'].values))

    cargas_ordenadas = sorted(amplitudes_carga.items(), key=lambda x: x[1])
    amplitudes_carga_ordenadas = [t[1] for t in cargas_ordenadas]

    cargas_ordenadas_pot = sorted(amplitudes_potencial.items(), key=lambda x: x[1])
    amplitudes_pot_ordenadas = [t[1] for t in cargas_ordenadas_pot]

    print(f"\n=== DISTRIBUIÇÃO DAS CARGAS POR AMPLITUDE ===")
    print(f"  Amplitude de ρ mínima: {amplitudes_carga_ordenadas[0]:.4e}")
    print(f"  Amplitude de ρ máxima: {amplitudes_carga_ordenadas[-1]:.4e}")
    print(f"  Amplitude de ρ mediana: {amplitudes_carga_ordenadas[len(amplitudes_carga_ordenadas)//2]:.4e}")

    print(f"\n=== DISTRIBUIÇÃO DOS POTENCIAIS POR AMPLITUDE ===")
    print(f"  Amplitude de φ mínima: {amplitudes_pot_ordenadas[0]:.4e}")
    print(f"  Amplitude de φ máxima: {amplitudes_pot_ordenadas[-1]:.4e}")
    print(f"  Amplitude de φ mediana: {amplitudes_pot_ordenadas[len(amplitudes_pot_ordenadas)//2]:.4e}")

    # divisão por carga (70/20/10)
    n_cargas = len(cargas_unicas)
    indices = np.random.RandomState(42).permutation(n_cargas)
    n_train = int(0.7 * n_cargas)
    n_val   = int(0.2 * n_cargas)

    train_indices = indices[:n_train]
    val_indices   = indices[n_train:n_train + n_val]
    test_indices  = indices[n_train + n_val:]

    cargas_train = cargas_unicas[train_indices]
    cargas_val   = cargas_unicas[val_indices]
    cargas_test  = cargas_unicas[test_indices]

    print(f"\n=== DIVISÃO DOS DADOS (POR CARGA) ===")
    print(f"  Cargas de treino: {len(cargas_train)}")
    print(f"  Cargas de validação: {len(cargas_val)}")
    print(f"  Cargas de teste: {len(cargas_test)}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[int(c) for c in cargas_train],
        y=[amplitudes_carga[c] for c in cargas_train],
        mode='markers',
        name=f'Treino ({len(cargas_train)})',
        marker=dict(color='#2E7D32', size=10, symbol='circle'),
        hovertemplate="Carga: %{x}<br>Amplitude ρ: %{y:.4e}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[int(c) for c in cargas_val],
        y=[amplitudes_carga[c] for c in cargas_val],
        mode='markers',
        name=f'Validação ({len(cargas_val)})',
        marker=dict(color='#0D47A1', size=10, symbol='square'),
        hovertemplate="Carga: %{x}<br>Amplitude ρ: %{y:.4e}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[int(c) for c in cargas_test],
        y=[amplitudes_carga[c] for c in cargas_test],
        mode='markers',
        name=f'Teste ({len(cargas_test)})',
        marker=dict(color='#F57F17', size=10, symbol='diamond'),
        hovertemplate="Carga: %{x}<br>Amplitude ρ: %{y:.4e}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:20px; font-weight:bold;'>"
                 f"Distribuição de Cargas por Conjunto</span><br><br>"
                 f"<span style='font-size:16px; color:#555555;'>"
                 f"Total: {n_cargas} distribuições de cargas | "
                 f"Treino: {len(cargas_train)} | "
                 f"Validação: {len(cargas_val)} | "
                 f"Teste: {len(cargas_test)}</span>",
            x=0.5, y=0.95, font=dict(size=20)
        ),
        xaxis_title="ID da carga",
        yaxis_title="Amplitude Máxima de ρ",
        width=1400,
        height=700,
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(t=140),
    )

    fig.write_html(f"{output_dir}/distribuicao_cargas_conjuntos.html")
    fig.show()

    return None


def cria_modelo_mlp_node(input_dim: int, output_dim: int, parameters: Dict[str, Any]) -> nn.Module:
    """Cria o modelo MLP para previsão do potencial elétrico φ(x,y) em toda a malha a partir de ρ(x,y)."""

    mlp_config = parameters.get('mlp', {})
    seed = parameters.get('seed', 42)

    fixa_sementes(seed)

    hidden_dims = mlp_config.get('hidden_dims', [64, 128, 64])
    activation  = mlp_config.get('activation', 'relu')

    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
        seed=seed
    )

    print("\n=== MODELO MLP CRIADO ===")
    print(f"  Dimensão entrada: {input_dim} (pontos de colocação da malha)")
    print(f"  Camadas ocultas: {hidden_dims}")
    print(f"  Dimensão saída: {output_dim} (pontos de colocação da malha)")
    print(f"  Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"  Função de ativação: {activation.capitalize()}")

    return model


def treina_mlp_node(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    parameters: Dict[str, Any]
) -> Tuple[nn.Module, Dict]:
    """Treina o modelo MLP para prever φ(x,y) em toda a malha a partir de ρ(x,y)."""

    mlp_config = parameters.get('mlp', {})
    seed = parameters.get('seed', 42)

    batch_size    = mlp_config.get('batch_size', 16)
    epochs        = mlp_config.get('epochs', 500)
    learning_rate = mlp_config.get('learning_rate', 0.005)
    weight_decay  = mlp_config.get('weight_decay', 0.0001)

    exp_name     = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')

    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)

    grafico_historico_loss = f"{output_dir}/historico_treinamento_loss.html"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    model = model.to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    history = {
        'train_loss': [],
        'val_loss': []
    }

    print("\n=== INICIANDO TREINAMENTO DO MLP ===")
    print(f"  Entrada: ρ(x,y) em toda a malha -> Saída: φ(x,y) em toda a malha")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Função loss: MSE (Mean Squared Error)")
    print(f"  Seed: {seed}")

    for epoch in range(epochs):
        # treino
        model.train()
        epoch_train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        # validação
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)

        history['train_loss'].append(float(epoch_train_loss))
        history['val_loss'].append(float(epoch_val_loss))

        scheduler.step(epoch_val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    # ============================================
    # GRÁFICO: Histórico de Treinamento
    # ============================================

    fig = cria_grafico_historico_treinamento(
        history=history,
        titulo="Evolução da Função de Custo durante o Treinamento do MLP - Equação de Poisson 2D"
    )

    fig.write_html(grafico_historico_loss)

    print(f"\n=== TREINAMENTO CONCLUÍDO ===")
    print(f"  Loss final de treino: {history['train_loss'][-1]:.6f}")
    print(f"  Loss final de validação: {history['val_loss'][-1]:.6f}")

    fig.show()

    return model, history


def avalia_metricas_mlp_node(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
) -> Dict[str, float]:
    """
    Avalia o modelo MLP nos dados de validação e teste para a equação de Poisson 2D.

    Avalia o potencial elétrico φ(x,y) previsto pelo modelo em toda a malha
    contra o potencial de referência (solução espectral), amostra a amostra.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X_val_tensor  = torch.tensor(X_val,  dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions_scaled_val  = model(X_val_tensor).cpu().numpy()
        predictions_scaled_test = model(X_test_tensor).cpu().numpy()

    # desnormaliza previsões
    predictions_val  = scaler_y.inverse_transform(predictions_scaled_val)
    y_val_original   = scaler_y.inverse_transform(y_val)
    predictions_test = scaler_y.inverse_transform(predictions_scaled_test)
    y_test_original  = scaler_y.inverse_transform(y_test)

    # avalia sobre toda a malha
    rmse_val  = float(np.sqrt(mean_squared_error(y_val_original.flatten(),  predictions_val.flatten())))
    r2_val    = float(r2_score(y_val_original.flatten(), predictions_val.flatten()))

    rmse_test = float(np.sqrt(mean_squared_error(y_test_original.flatten(), predictions_test.flatten())))
    r2_test   = float(r2_score(y_test_original.flatten(), predictions_test.flatten()))

    erro_max_val  = float(np.max(np.abs(y_val_original  - predictions_val)))
    erro_max_test = float(np.max(np.abs(y_test_original - predictions_test)))

    metrics = {
        'rmse_val':  rmse_val,
        'r2_val':    r2_val,
        'rmse_test': rmse_test,
        'r2_test':   r2_test,
        'erro_max_val':  erro_max_val,
        'erro_max_test': erro_max_test,
    }

    print("\n=== AVALIAÇÃO DO MODELO MLP - EQUAÇÃO DE POISSON 2D ===")
    print(f"  RMSE Validação: {rmse_val:.6e}")
    print(f"  R²   Validação: {r2_val:.4f}")
    print(f"  Erro máx. Validação: {erro_max_val:.6e}")
    print(f"  RMSE Teste:     {rmse_test:.6e}")
    print(f"  R²   Teste:     {r2_test:.4f}")
    print(f"  Erro máx. Teste: {erro_max_test:.6e}")

    return metrics


def visualiza_previsoes_mlp_node(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
    parameters: Dict[str, Any]
) -> None:
    """
    Visualiza as previsões do modelo MLP nos dados de teste.
    Compara o potencial elétrico previsto com o de referência em toda a malha.

    Args:
        model: Modelo treinado
        X_test: Dados de teste (ρ(x,y) em toda a malha, achatado)
        y_test: Targets de teste (potenciais φ(x,y) de referência, achatados)
        scaler_y: Scaler dos targets
        parameters: Parâmetros do pipeline
    """
    exp_name     = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'base_01')
    seed         = parameters.get('seed', 42)

    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)

    grafico_previsoes_mlp = f"{output_dir}/real_previsto_mlp.html"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()

    # desnormaliza previsões
    predictions     = scaler_y.inverse_transform(predictions_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)

    # achata toda a malha para visualização ponto a ponto
    predictions_flat = predictions.flatten().reshape(-1, 1)
    y_true_flat      = y_test_original.flatten().reshape(-1, 1)

    fig = cria_grafico_real_previsto_mlp(
        predictions=predictions_flat,
        y_true=y_true_flat,
        titulo="Real vs Previsto - Equação de Poisson 2D (Dados de Teste)"
    )

    fig.write_html(grafico_previsoes_mlp)
    fig.show()

    return None


def visualiza_previsoes_espaco_fases_node(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
    params_test: pd.DataFrame,
    parameters: Dict[str, Any]
) -> None:
    """
    Node: Visualiza as previsões do modelo para o potencial elétrico φ(x,y)
    como uma malha 2D completa por carga, anotando os parâmetros sorteados
    da gaussiana em cada carga (amplitude, largura, centros).
    """
    exp_name     = parameters.get('exp_name', 'default_exp')
    data_version = parameters.get('data_version', 'default_v1')

    output_dir = f"data/08_reporting/{exp_name}/{data_version}"
    os.makedirs(output_dir, exist_ok=True)

    grafico_previsoes_espaco_fases = f"{output_dir}/previsoes_espaco_fases.html"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    n_pontos = X_test.shape[1]
    N_plus_1 = int(np.sqrt(n_pontos))
    if N_plus_1 * N_plus_1 != n_pontos:
        raise ValueError(
            f"X_test.shape[1] = {n_pontos} não é um quadrado perfeito. "
            f"Esperado (N+1)² para a malha 2D."
        )
    N = N_plus_1 - 1

    dom = parameters['solver']['dominio']
    x_grid = np.linspace(dom['xmin'], dom['xmax'], N_plus_1)
    y_grid = np.linspace(dom['ymin'], dom['ymax'], N_plus_1)

    print("\n=== VISUALIZAÇÃO DAS PREVISÕES ===")
    print(f"  Número de cargas de teste: {len(X_test)}")
    print(f"  Pontos por carga: {n_pontos}  →  malha {N_plus_1} × {N_plus_1}")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()

    predictions     = scaler_y.inverse_transform(predictions_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)

    n_cargas = predictions.shape[0]
    phi_pred = np.array([predictions[i].reshape(N_plus_1, N_plus_1, order='F')
                         for i in range(n_cargas)])
    phi_true = np.array([y_test_original[i].reshape(N_plus_1, N_plus_1, order='F')
                         for i in range(n_cargas)])

    # ρ reshaped para (n_cargas, N+1, N+1)
    rho_inputs = np.array([X_test[i].reshape(N_plus_1, N_plus_1, order='F')
                           for i in range(n_cargas)])

    fig = cria_grafico_previsoes_espaco_fases(
        phi_pred=phi_pred,
        phi_true=phi_true,
        rho_inputs=rho_inputs,
        parametros_cargas=params_test,
        x_grid=x_grid,
        y_grid=y_grid,
        titulo="Previsões do Modelo — Equação de Poisson 2D"
    )

    fig.write_html(grafico_previsoes_espaco_fases)
    fig.show()

    return None


# def interpola_trajetorias_avulsas_node(
#     model: nn.Module,
#     scaler_X: StandardScaler,
#     scaler_y: StandardScaler,
#     parameters: Dict[str, Any],
#     tempos_referencia: np.ndarray = None
# ) -> None:
#     """
#     Node: Usa o modelo treinado para fazer interpolações e prever trajetórias completas
#     para novas condições iniciais não vistas durante o treinamento.
#     Nota: O parâmetro do sistema (mu) é fixo para todos os casos.
    
#     Args:
#         model: Modelo MLP treinado (prevê trajetórias completas)
#         scaler_X: Scaler das features de entrada
#         scaler_y: Scaler dos targets
#         parameters: Parâmetros do pipeline
#         tempos_referencia: Array com os tempos para plotagem (opcional)
#     """

#     exp_name = parameters.get('exp_name', 'default_exp')
#     data_version = parameters.get('data_version', 'base_01')
    
#     output_dir = f"data/08_reporting/{exp_name}/{data_version}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     grafico_interpolacao_completa = f"{output_dir}/interpolacao_avulsa_posicao_velocidade_vs_t.html"
#     grafico_interpolacao_espaco_fases = f"{output_dir}/interpolacao_avulsa_espaco_fases.html"
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
    
#     intervals = parameters.get('intervals', {})
    
#     # Parâmetro do sistema Van der Pol
#     mu = intervals.get('parametro_mu', 1.0)
    
#     # Ponto de equilíbrio (0,0) para o Van der Pol
#     x_eq = 0.0
#     y_eq = 0.0
    
#     if tempos_referencia is None:
#         # período aproximado para definir tempo de simulação
#         # Para mu pequeno: T ≈ 2π
#         # Para mu grande: T ≈ (3 - 2*ln(2))/mu
#         if mu < 0.1:
#             T_aprox = 2.0 * np.pi
#         else:
#             T_aprox = (3.0 - 2.0 * np.log(2.0)) / mu
        
#         sim_params = parameters.get('simulation', {})
#         dt = sim_params.get('dt', 0.01)
#         num_periodos = sim_params.get('num_periodos', 3)
#         t_final = num_periodos * T_aprox
#         tempos_referencia = np.arange(0, t_final + dt, dt)
#         print(f"  Tempos de referência criados: {len(tempos_referencia)} pontos")
#         print(f"  Período aproximado: {T_aprox:.4f} s")
#         print(f"  Tempo final: {t_final:.4f} s")
#     else:
#         print(f"\n  Nós de saída do modelo por trajetória: {len(tempos_referencia)} pontos")
    
#     num_timesteps = len(tempos_referencia)
    
#     # casos de teste para interpolação (apenas condições iniciais variam)
#     casos_teste = [
#         {
#             "nome": "Caso 1",
#             "x0": 0.5,
#             "y0": 0.0,
#             "cor": CORES_PALETA[0]
#         },
#         {
#             "nome": "Caso 2",
#             "x0": 1.5,
#             "y0": 0.0,
#             "cor": CORES_PALETA[1]
#         },
#         {
#             "nome": "Caso 3",
#             "x0": 0.0,
#             "y0": 1.0,
#             "cor": CORES_PALETA[2]
#         },
#         {
#             "nome": "Caso 4",
#             "x0": 2.0,
#             "y0": 0.5,
#             "cor": CORES_PALETA[3]
#         },
#         {
#             "nome": "Caso 5",
#             "x0": -1.0,
#             "y0": 0.0,
#             "cor": CORES_PALETA[4]
#         },
#     ]
    
#     # gera os nomes das legendas dinamicamente
#     for caso in casos_teste:
#         caso["nome_legenda"] = (
#             f"{caso['nome']}: x0={caso['x0']:.1f}, "
#             f"y0={caso['y0']:.1f}"
#         )
#         # informações do sistema
#         caso["mu"] = mu
#         caso["x_eq"] = x_eq
#         caso["y_eq"] = y_eq
    
#     tempos_lista = []
#     posicao_lista = []
#     velocidade_lista = []
    
#     for caso in casos_teste:
#         # verifica se o número de timesteps é compatível com o modelo
#         if len(tempos_referencia) != num_timesteps:
#             print(f"  AVISO: {caso['nome']} - Ajustando tempos para {num_timesteps} pontos")
#             tempos = np.linspace(tempos_referencia[0], tempos_referencia[-1], num_timesteps)
#         else:
#             tempos = tempos_referencia
        
#         # entrada: [x0, y0] - posição e velocidade iniciais
#         X_caso = np.array([[caso["x0"], caso["y0"]]], dtype=np.float32)
        
#         # normaliza a entrada
#         X_caso_scaled = scaler_X.transform(X_caso)
#         X_tensor = torch.tensor(X_caso_scaled, dtype=torch.float32).to(device)
        
#         # previsão: trajetória completa
#         with torch.no_grad():
#             predictions_scaled = model(X_tensor).cpu().numpy()
        
#         # desnormaliza a trajetória completa
#         predictions = scaler_y.inverse_transform(predictions_scaled)
        
#         # separa posição e velocidade da trajetória completa
#         posicao = predictions[0, 0::2]  # posição (índices pares)
#         velocidade = predictions[0, 1::2]  # velocidade (índices ímpares)
        
#         # se os tempos não têm o mesmo tamanho, ajusta
#         if len(posicao) != len(tempos):
#             print(f"  AVISO: {caso['nome']} - Ajustando tempos para {len(posicao)} pontos")
#             tempos = np.linspace(tempos[0], tempos[-1], len(posicao))
        
#         tempos_lista.append(tempos)
#         posicao_lista.append(posicao)
#         velocidade_lista.append(velocidade)
        
#         caso["num_pontos"] = len(posicao)
#         caso["dt"] = tempos[1] - tempos[0] if len(tempos) > 1 else 0
    
#     print("\n=== INTERPOLAÇÃO DE TRAJETÓRIAS AVULSAS ===")
#     print(f"  Parâmetro do sistema: mu={mu:.2f}")
#     print(f"  Ponto de equilíbrio: x*={x_eq:.2f}, y*={y_eq:.2f}")
#     for caso in casos_teste:
#         print(f"    {caso['nome']}: x0={caso['x0']:.1f}, y0={caso['y0']:.1f}")
    
#     fig_completo = cria_grafico_interpolacao_completo(
#         tempos_lista=tempos_lista,
#         posicao_lista=posicao_lista,
#         velocidade_lista=velocidade_lista,
#         casos_info=casos_teste,
#         titulo="Interpolação Avulsa: Posição e Velocidade vs Tempo - Oscilador de Van der Pol"
#     )
    
#     fig_fases = cria_grafico_interpolacao_espaco_fases(
#         posicao_lista=posicao_lista,
#         velocidade_lista=velocidade_lista,
#         casos_info=casos_teste,
#         titulo="Interpolação Avulsa: Espaço de Fases - Oscilador de Van der Pol"
#     )
    
#     fig_completo.write_html(grafico_interpolacao_completa)
#     fig_fases.write_html(grafico_interpolacao_espaco_fases)
    
#     fig_completo.show()
#     fig_fases.show()
    
#     return None

# def interpolacoes_pontuais_mlp_node(
#     model: nn.Module,
#     scaler_X: StandardScaler,
#     scaler_y: StandardScaler,
#     parameters: Dict[str, Any],
#     tempos_referencia: np.ndarray = None
# ) -> pd.DataFrame:
#     """
#     Node: Usa o modelo treinado para fazer interpolação entre pontos de dados gerados aleatoriamente.
#     Faz previsões de trajetórias completas a partir de condições iniciais aleatórias.
#     Nota: A interpolação é feita dentro da mesma trajetória, variando apenas o tempo.
#     O parâmetro do sistema (mu) é constante.
    
#     Args:
#         model: Modelo MLP treinado (prevê trajetórias completas)
#         scaler_X: Scaler das features de entrada
#         scaler_y: Scaler dos targets
#         parameters: Parâmetros do pipeline
#         tempos_referencia: Array com os tempos de referência (opcional)
        
#     Returns:
#         DataFrame com os dados interpolados e previsões do modelo
#     """
    
#     exp_name = parameters.get('exp_name', 'default_exp')
#     data_version = parameters.get('data_version', 'base_01')
    
#     output_dir = f"data/08_reporting/{exp_name}/{data_version}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     grafico_interpolacao_pontual = f"{output_dir}/interpolacoes_pontuais_real_previsto_mlp.html"
#     grafico_interpolacao_pontual_espaco_fases = f"{output_dir}/interpolacao_pontual_espaco_fases.html"
#     grafico_interpolacao_pontual_temporal = f"{output_dir}/interpolacao_pontual_posicao_velocidade_vs_t.html"
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
    
#     intervals = parameters.get('intervals', {})
#     seed = parameters.get('seed', 42)
    
#     mu = intervals.get('parametro_mu', 1.0)
    
#     # ponto de equilíbrio do Van der Pol (0,0)
#     x_eq = 0.0
#     y_eq = 0.0
    
#     # fixa a semente para reprodutibilidade
#     np.random.seed(seed)
    
#     print("\n=== INTERPOLAÇÃO PONTUAL - OSCILADOR DE VAN DER POL ===")
#     print(f"  Parâmetro do sistema: μ={mu:.2f}")
#     print(f"  Ponto de equilíbrio: x*={x_eq:.2f}, y*={y_eq:.2f}")
#     print("\n  A interpolação é feita variando o tempo para uma mesma trajetória")
    
#     # ============================================
#     # GERAÇÃO DE DADOS ALEATÓRIOS
#     # ============================================
    
#     # limites dos intervalos dos parâmetros
#     x0_min = intervals.get('x0_min', -3.0)
#     x0_max = intervals.get('x0_max', 3.0)
#     y0_min = intervals.get('y0_min', -3.0)
#     y0_max = intervals.get('y0_max', 3.0)
    
#     # número de trajetórias a serem geradas
#     num_trajetorias = 2
    
#     print(f"\n  Gerando {num_trajetorias} trajetórias aleatórias:")
#     print(f"    x0 no intervalo [{x0_min:.3f}, {x0_max:.3f}]")
#     print(f"    y0 no intervalo [{y0_min:.3f}, {y0_max:.3f}]")
    
#     # gera condições iniciais aleatórias
#     x0_values = np.random.uniform(x0_min, x0_max, num_trajetorias)
#     y0_values = np.random.uniform(y0_min, y0_max, num_trajetorias)
    
#     # número de pontos baseado nos tempos de referência ou padrão
#     if tempos_referencia is not None:
#         num_pontos_por_trajetoria = len(tempos_referencia)
#         tempos_interpolados = tempos_referencia
#         tempo_maximo = tempos_interpolados[-1]
#         dt_interpolacao = tempos_interpolados[1] - tempos_interpolados[0] if len(tempos_interpolados) > 1 else 0.01
#         print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
#         print(f"  Tempo máximo: {tempo_maximo:.3f} s")
#         print(f"  Passo temporal: {dt_interpolacao:.6f} s")
#     else:
#         # Período aproximado para o Van der Pol
#         if mu < 0.1:
#             T_aprox = 2.0 * np.pi
#         else:
#             T_aprox = (3.0 - 2.0 * np.log(2.0)) / mu
        
#         sim_params = parameters.get('simulation', {})
#         dt = sim_params.get('dt', 0.01)
#         num_periodos = sim_params.get('num_periodos', 3)
#         tempo_maximo = num_periodos * T_aprox
#         num_pontos_por_trajetoria = int(tempo_maximo / dt) + 1
#         tempos_interpolados = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
#         dt_interpolacao = tempos_interpolados[1] - tempos_interpolados[0]
        
#         print(f"\n  Configuração da interpolação:")
#         print(f"    Período aproximado: {T_aprox:.3f} s")
#         print(f"    Tempo máximo: {tempo_maximo:.3f} s")
#         print(f"    Passo temporal: {dt_interpolacao:.6f} s")
#         print(f"    Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria}")
    
#     todas_previsoes = []
#     todos_reais_interpolados = []
    
#     # listas para o gráfico temporal
#     tempos_lista = []
#     posicao_prevista_lista = []
#     velocidade_prevista_lista = []
#     posicao_real_lista = []
#     velocidade_real_lista = []
#     casos_info_lista = []
#     dados_interpolados = []
    
#     for idx in range(num_trajetorias):
#         x0 = x0_values[idx]
#         y0 = y0_values[idx]
        
#         print(f"\n  Processando trajetória {idx}: x0={x0:.3f}, y0={y0:.3f}")
        
#         # Usa o oscilador de Van der Pol para gerar a solução "real"
#         osc = OsciladorVanDerPol(
#             parametros_mu=[mu],
#             device='cpu'
#         )
        
#         cond_curta = torch.tensor([[x0, y0]], dtype=torch.float32)
#         solucao_curta = osc.resolve_multi_condicoes_sistemas(
#             condicoes_iniciais=cond_curta,
#             t_final=tempo_maximo,
#             dt=dt_interpolacao
#         )
        
#         # valores reais da simulação
#         posicao_real = solucao_curta['posicao'][:, 0, 0]
#         velocidade_real = solucao_curta['velocidade'][:, 0, 0]
#         tempos_reais = solucao_curta['tempo']
        
#         if len(tempos_reais) != len(tempos_interpolados):
#             interp_posicao = interp1d(tempos_reais, posicao_real, kind='linear', fill_value='extrapolate')
#             interp_velocidade = interp1d(tempos_reais, velocidade_real, kind='linear', fill_value='extrapolate')
#             posicao_real_interpolado = interp_posicao(tempos_interpolados)
#             velocidade_real_interpolado = interp_velocidade(tempos_interpolados)
#         else:
#             posicao_real_interpolado = posicao_real
#             velocidade_real_interpolado = velocidade_real
        
#         X_interpolado = np.array([[x0, y0]], dtype=np.float32)
        
#         X_interpolado_scaled = scaler_X.transform(X_interpolado)
#         X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
        
#         with torch.no_grad():
#             pred_scaled = model(X_tensor).cpu().numpy()
        
#         pred = scaler_y.inverse_transform(pred_scaled)
        
#         # separa posição e velocidade da trajetória completa
#         # a saída está no formato: [x0, y0, x1, y1, ..., xN, yN]
#         posicao_prevista = pred[0, 0::2]  # Pega a posição (índices pares)
#         velocidade_prevista = pred[0, 1::2]  # Pega a velocidade (índices ímpares)
        
#         if len(posicao_prevista) != len(tempos_interpolados):
#             print(f"    AVISO: Ajustando tempos para {len(posicao_prevista)} pontos")
#             tempos_ajustados = np.linspace(tempos_interpolados[0], tempos_interpolados[-1], len(posicao_prevista))
#         else:
#             tempos_ajustados = tempos_interpolados
        
#         pred_pontos = np.column_stack([posicao_prevista, velocidade_prevista])
#         real_pontos = np.column_stack([posicao_real_interpolado, velocidade_real_interpolado])
        
#         if len(posicao_prevista) != len(posicao_real_interpolado):
#             interp_posicao = interp1d(tempos_interpolados, posicao_real_interpolado, kind='linear', fill_value='extrapolate')
#             interp_velocidade = interp1d(tempos_interpolados, velocidade_real_interpolado, kind='linear', fill_value='extrapolate')
#             posicao_real_ajustado = interp_posicao(tempos_ajustados)
#             velocidade_real_ajustado = interp_velocidade(tempos_ajustados)
#             real_pontos = np.column_stack([posicao_real_ajustado, velocidade_real_ajustado])
#             tempos_para_grafico = tempos_ajustados
#         else:
#             tempos_para_grafico = tempos_interpolados
        
#         todas_previsoes.append(pred_pontos)
#         todos_reais_interpolados.append(real_pontos)
        
#         tempos_lista.append(tempos_para_grafico)
#         posicao_prevista_lista.append(posicao_prevista)
#         velocidade_prevista_lista.append(velocidade_prevista)
#         posicao_real_lista.append(posicao_real_interpolado[:len(tempos_para_grafico)])
#         velocidade_real_lista.append(velocidade_real_interpolado[:len(tempos_para_grafico)])
        
#         cor = CORES_PALETA[idx % len(CORES_PALETA)]
        
#         casos_info_lista.append({
#             'x0': x0,
#             'y0': y0,
#             'cor': cor
#         })
        
#         for k in range(len(tempos_para_grafico)):
#             dados_interpolados.append({
#                 'id_trajetoria': f"x0_{x0:.3f}_y0_{y0:.3f}",
#                 'x0': x0,
#                 'y0': y0,
#                 'parametro_mu': mu,
#                 'posicao_eq': x_eq,
#                 'velocidade_eq': y_eq,
#                 'tempo_interpolado': tempos_para_grafico[k],
#                 'posicao_real': real_pontos[k, 0],
#                 'velocidade_real': real_pontos[k, 1],
#                 'posicao_previsto_mlp': pred_pontos[k, 0],
#                 'velocidade_previsto_mlp': pred_pontos[k, 1],
#                 'erro_posicao': pred_pontos[k, 0] - real_pontos[k, 0],
#                 'erro_velocidade': pred_pontos[k, 1] - real_pontos[k, 1],
#                 'erro_abs_posicao': abs(pred_pontos[k, 0] - real_pontos[k, 0]),
#                 'erro_abs_velocidade': abs(pred_pontos[k, 1] - real_pontos[k, 1]),
#             })
    
#     if len(todas_previsoes) == 0:
#         print("  ERRO: Nenhuma trajetória válida encontrada para interpolação")
#         return pd.DataFrame()
        
#     predictions_all = np.vstack(todas_previsoes)
#     y_true_all = np.vstack(todos_reais_interpolados)
    
#     rmse_posicao = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
#     rmse_velocidade = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
#     r2_posicao = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
#     r2_velocidade = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
#     print(f"\n  Total de pontos interpolados: {len(predictions_all)}")
#     print(f"  RMSE Posição (vs solução RK4): {rmse_posicao:.6f}")
#     print(f"  RMSE Velocidade (vs solução RK4): {rmse_velocidade:.6f}")
#     print(f"  R² Posição (vs solução RK4): {r2_posicao:.4f}")
#     print(f"  R² Velocidade (vs solução RK4): {r2_velocidade:.4f}")
    
#     # ============================================
#     # GRÁFICO 1: Real vs Previsto
#     # ============================================
    
#     fig1 = cria_grafico_interpolacao_pontual_mlp(
#         predictions=predictions_all,
#         y_true=y_true_all,
#         titulo="Interpolação Pontual: RK4 vs MLP - Dados Gerados Aleatoriamente - Oscilador de Van der Pol"
#     )
    
#     fig1.write_html(grafico_interpolacao_pontual)
    
#     # ============================================
#     # GRÁFICO 2: Espaço de Fases
#     # ============================================
    
#     y_posicao_true = y_true_all[:, 0].reshape(-1, 1)
#     y_velocidade_true = y_true_all[:, 1].reshape(-1, 1)
#     y_posicao_pred = predictions_all[:, 0].reshape(-1, 1)
#     y_velocidade_pred = predictions_all[:, 1].reshape(-1, 1)
    
#     fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
#         y_pos_true=y_posicao_true,
#         y_vel_true=y_velocidade_true,
#         y_pos_pred=y_posicao_pred,
#         y_vel_pred=y_velocidade_pred,
#         titulo="Interpolação Pontual: MLP vs RK4 - Espaço de Fases - Oscilador de Van der Pol"
#     )
    
#     fig2.write_html(grafico_interpolacao_pontual_espaco_fases)
    
#     # ============================================
#     # GRÁFICO 3: Posição e Velocidade vs Tempo
#     # ============================================
    
#     fig3 = cria_grafico_interpolacao_pontual_completo(
#         tempos_lista=tempos_lista,
#         posicoes_previstas_lista=posicao_prevista_lista,
#         velocidades_previstas_lista=velocidade_prevista_lista,
#         posicoes_reais_lista=posicao_real_lista,
#         velocidades_reais_lista=velocidade_real_lista,
#         casos_info=casos_info_lista,
#         titulo="Interpolação Pontual: MLP vs RK4 - Posição e Velocidade vs Tempo - Oscilador de Van der Pol"
#     )
    
#     fig3.write_html(grafico_interpolacao_pontual_temporal)
    
#     fig1.show()
#     fig2.show()
#     fig3.show()
    
#     df_interpolado = pd.DataFrame(dados_interpolados)
    
#     df_interpolado.attrs['rmse_posicao'] = rmse_posicao
#     df_interpolado.attrs['rmse_velocidade'] = rmse_velocidade
#     df_interpolado.attrs['r2_posicao'] = r2_posicao
#     df_interpolado.attrs['r2_velocidade'] = r2_velocidade
#     df_interpolado.attrs['total_pontos'] = len(predictions_all)
#     df_interpolado.attrs['num_trajetorias'] = num_trajetorias
#     df_interpolado.attrs['pontos_por_trajetoria'] = num_pontos_por_trajetoria
#     df_interpolado.attrs['parametro_mu'] = mu
#     df_interpolado.attrs['posicao_eq'] = x_eq
#     df_interpolado.attrs['velocidade_eq'] = y_eq
#     df_interpolado.attrs['tempo_maximo'] = tempo_maximo
#     df_interpolado.attrs['dt_interpolacao'] = dt_interpolacao
#     df_interpolado.attrs['x0_min'] = x0_min
#     df_interpolado.attrs['x0_max'] = x0_max
#     df_interpolado.attrs['y0_min'] = y0_min
#     df_interpolado.attrs['y0_max'] = y0_max
    
#     print(f"\n  Base de dados com interpolação temporal dentro de {num_trajetorias} trajetórias gerada com {len(df_interpolado)} registros")
    
#     return df_interpolado


# def interpola_entre_trajetorias_mlp_node(
#     model: nn.Module,
#     scaler_X: StandardScaler,
#     scaler_y: StandardScaler,
#     parameters: Dict[str, Any],
#     tempos_referencia: np.ndarray = None
# ) -> pd.DataFrame:
#     """
#     Node: Usa o modelo treinado para fazer interpolação entre trajetórias.
#     Para cada instante de tempo, interpola entre duas trajetórias diferentes (variando x0 e y0).
#     Não mistura dados de treino/validação/teste pois usa dados gerados aleatoriamente.
#     Agora prevê trajetórias completas a partir das condições iniciais.
#     """
    
#     exp_name = parameters.get('exp_name', 'default_exp')
#     data_version = parameters.get('data_version', 'base_01')
    
#     output_dir = f"data/08_reporting/{exp_name}/{data_version}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     grafico_interpolacao_entre_trajetorias = f"{output_dir}/interpolacoes_entre_trajetorias_real_previsto_mlp.html"
#     grafico_interpolacao_entre_trajetorias_espaco_fases = f"{output_dir}/interpolacao_entre_trajetorias_espaco_fases.html"
#     grafico_interpolacao_entre_trajetorias_temporal = f"{output_dir}/interpolacao_entre_trajetorias_posicao_velocidade_vs_t.html"
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
    
#     intervals = parameters.get('intervals', {})
    
#     mu = intervals.get('parametro_mu', 1.0)
    
#     # ponto de equilíbrio do Van der Pol (0,0)
#     x_eq = 0.0
#     y_eq = 0.0
    
#     seed = parameters.get('seed', 42)
#     np.random.seed(seed)
    
#     print("\n=== INTERPOLAÇÃO ENTRE TRAJETÓRIAS - OSCILADOR DE VAN DER POL ===")
#     print(f"  Parâmetro do sistema: μ={mu:.2f}")
#     print(f"  Ponto de equilíbrio: x*={x_eq:.2f}, y*={y_eq:.2f}")
#     print("\n  Para cada instante de tempo, interpola entre duas trajetórias diferentes")
    
#     # ============================================
#     # GERAÇÃO DE DADOS ALEATÓRIOS
#     # ============================================
    
#     # limites dos intervalos dos parâmetros
#     x0_min = intervals.get('x0_min', -3.0)
#     x0_max = intervals.get('x0_max', 3.0)
#     y0_min = intervals.get('y0_min', -3.0)
#     y0_max = intervals.get('y0_max', 3.0)
    
#     # define o número de pontos baseado nos tempos de referência ou padrão
#     if tempos_referencia is not None:
#         num_pontos_por_trajetoria = len(tempos_referencia)
#         tempos_unicos = tempos_referencia
#         tempo_maximo = tempos_unicos[-1]
#         dt_interpolacao = tempos_unicos[1] - tempos_unicos[0] if len(tempos_unicos) > 1 else 0.01
#         print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
#         print(f"  Tempo máximo: {tempo_maximo:.3f} s")
#     else:
#         # Período aproximado para o Van der Pol
#         if mu < 0.1:
#             T_aprox = 2.0 * np.pi
#         else:
#             T_aprox = (3.0 - 2.0 * np.log(2.0)) / mu
        
#         sim_params = parameters.get('simulation', {})
#         dt = sim_params.get('dt', 0.01)
#         num_periodos = sim_params.get('num_periodos', 3)
#         tempo_maximo = num_periodos * T_aprox
#         num_pontos_por_trajetoria = int(tempo_maximo / dt) + 1
#         tempos_unicos = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
#         dt_interpolacao = tempos_unicos[1] - tempos_unicos[0]
        
#         print(f"\n  Configuração da interpolação:")
#         print(f"    Período aproximado: {T_aprox:.3f} s")
#         print(f"    Tempo máximo: {tempo_maximo:.3f} s")
#         print(f"    Passo temporal: {dt_interpolacao:.6f} s")
#         print(f"    Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria}")
    
#     # gera trajetórias com diferentes amplitudes (distância do equilíbrio)
#     x0_candidates = np.random.uniform(x0_min, x0_max, 100)
#     y0_candidates = np.random.uniform(y0_min, y0_max, 100)
    
#     # calcula distância do equilíbrio (0,0)
#     distancias = np.sqrt(x0_candidates**2 + y0_candidates**2)
    
#     # trajetória de menor amplitude (mais próxima do equilíbrio)
#     idx_pequena = np.argmin(distancias)
#     x0_1 = x0_candidates[idx_pequena]
#     y0_1 = y0_candidates[idx_pequena]
    
#     # trajetória de maior amplitude (mais distante do equilíbrio)
#     idx_grande = np.argmax(distancias)
#     x0_2 = x0_candidates[idx_grande]
#     y0_2 = y0_candidates[idx_grande]
    
#     print(f"\n  Trajetória 1 (próxima ao equilíbrio): x0={x0_1:.3f}, y0={y0_1:.3f}")
#     print(f"  Trajetória 2 (distante do equilíbrio): x0={x0_2:.3f}, y0={y0_2:.3f}")
#     print(f"  Distâncias do equilíbrio: d1={distancias[idx_pequena]:.3f}, d2={distancias[idx_grande]:.3f}")
    
#     # define os níveis de interpolação
#     alphas = np.linspace(0, 1, 5)  # 5 níveis de interpolação
    
#     todas_previsoes = []
#     todos_reais_interpolados = []
    
#     tempos_lista = []
#     posicao_prevista_lista = []
#     velocidade_prevista_lista = []
#     posicao_real_lista = []
#     velocidade_real_lista = []
#     casos_info_lista = []
#     dados_interpolados = []
    
#     for alpha in alphas:
#         x0_interp = (1 - alpha) * x0_1 + alpha * x0_2
#         y0_interp = (1 - alpha) * y0_1 + alpha * y0_2
        
#         # entrada para o modelo: x0, y0
#         X_interpolado = np.array([[x0_interp, y0_interp]], dtype=np.float32)
        
#         # normaliza e faz previsão
#         X_interpolado_scaled = scaler_X.transform(X_interpolado)
#         X_tensor = torch.tensor(X_interpolado_scaled, dtype=torch.float32).to(device)
        
#         with torch.no_grad():
#             pred_scaled = model(X_tensor).cpu().numpy()
        
#         # desnormaliza a trajetória completa
#         pred = scaler_y.inverse_transform(pred_scaled)
        
#         # separa posição e velocidade da trajetória completa
#         # a saída está no formato: [x0, y0, x1, y1, ..., xN, yN]
#         posicao_prevista = pred[0, 0::2]  # posição (índices pares)
#         velocidade_prevista = pred[0, 1::2]  # velocidade (índices ímpares)
        
#         # verifica se o número de pontos coincide com os tempos
#         if len(posicao_prevista) != len(tempos_unicos):
#             print(f"  AVISO: Ajustando tempos para {len(posicao_prevista)} pontos")
#             tempos_ajustados = np.linspace(tempos_unicos[0], tempos_unicos[-1], len(posicao_prevista))
#         else:
#             tempos_ajustados = tempos_unicos
        
#         osc = OsciladorVanDerPol(
#             parametros_mu=[mu],
#             device='cpu'
#         )
        
#         cond_curta = torch.tensor([[x0_interp, y0_interp]], dtype=torch.float32)
#         solucao_curta = osc.resolve_multi_condicoes_sistemas(
#             condicoes_iniciais=cond_curta,
#             t_final=tempo_maximo,
#             dt=dt_interpolacao
#         )
        
#         posicao_real = solucao_curta['posicao'][:, 0, 0]
#         velocidade_real = solucao_curta['velocidade'][:, 0, 0]
#         tempos_reais = solucao_curta['tempo']
        
#         if len(tempos_reais) != len(tempos_ajustados):
#             interp_posicao = interp1d(tempos_reais, posicao_real, kind='linear', fill_value='extrapolate')
#             interp_velocidade = interp1d(tempos_reais, velocidade_real, kind='linear', fill_value='extrapolate')
#             posicao_real_ajustado = interp_posicao(tempos_ajustados)
#             velocidade_real_ajustado = interp_velocidade(tempos_ajustados)
#         else:
#             posicao_real_ajustado = posicao_real
#             velocidade_real_ajustado = velocidade_real
        
#         pred_pontos = np.column_stack([posicao_prevista, velocidade_prevista])
#         real_pontos = np.column_stack([posicao_real_ajustado, velocidade_real_ajustado])
        
#         todas_previsoes.append(pred_pontos)
#         todos_reais_interpolados.append(real_pontos)
        
#         tempos_lista.append(tempos_ajustados)
#         posicao_prevista_lista.append(posicao_prevista)
#         velocidade_prevista_lista.append(velocidade_prevista)
#         posicao_real_lista.append(posicao_real_ajustado)
#         velocidade_real_lista.append(velocidade_real_ajustado)
        
#         cor_idx = int(alpha * (len(CORES_PALETA) - 1))
#         cor = CORES_PALETA[cor_idx]
        
#         casos_info_lista.append({
#             'alpha': alpha,
#             'x0': x0_interp,
#             'y0': y0_interp,
#             'cor': cor
#         })
        
#         for k in range(len(tempos_ajustados)):
#             dados_interpolados.append({
#                 'alpha_interpolacao': alpha,
#                 'x0_original_1': x0_1,
#                 'y0_original_1': y0_1,
#                 'x0_original_2': x0_2,
#                 'y0_original_2': y0_2,
#                 'x0_interpolado': x0_interp,
#                 'y0_interpolado': y0_interp,
#                 'parametro_mu': mu,
#                 'posicao_eq': x_eq,
#                 'velocidade_eq': y_eq,
#                 'tempo': tempos_ajustados[k],
#                 'posicao_real': posicao_real_ajustado[k],
#                 'velocidade_real': velocidade_real_ajustado[k],
#                 'posicao_previsto_mlp': pred_pontos[k, 0],
#                 'velocidade_previsto_mlp': pred_pontos[k, 1],
#                 'erro_posicao': pred_pontos[k, 0] - posicao_real_ajustado[k],
#                 'erro_velocidade': pred_pontos[k, 1] - velocidade_real_ajustado[k],
#                 'erro_abs_posicao': abs(pred_pontos[k, 0] - posicao_real_ajustado[k]),
#                 'erro_abs_velocidade': abs(pred_pontos[k, 1] - velocidade_real_ajustado[k]),
#                 'erro_rel_posicao_pct': (abs(pred_pontos[k, 0] - posicao_real_ajustado[k]) / (abs(posicao_real_ajustado[k]) + 1e-6)) * 100,
#                 'erro_rel_velocidade_pct': (abs(pred_pontos[k, 1] - velocidade_real_ajustado[k]) / (abs(velocidade_real_ajustado[k]) + 1e-6)) * 100,
#             })
    
#     if len(todas_previsoes) == 0:
#         print("  ERRO: Nenhuma interpolação realizada")
#         return pd.DataFrame()
    
#     predictions_all = np.vstack(todas_previsoes)
#     y_true_all = np.vstack(todos_reais_interpolados)
    
#     rmse_posicao = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
#     rmse_velocidade = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
#     r2_posicao = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
#     r2_velocidade = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
#     print(f"\n  Total de pontos interpolados: {len(predictions_all)}")
#     print(f"  RMSE Posição (vs solução RK4): {rmse_posicao:.6f}")
#     print(f"  RMSE Velocidade (vs solução RK4): {rmse_velocidade:.6f}")
#     print(f"  R² Posição (vs solução RK4): {r2_posicao:.4f}")
#     print(f"  R² Velocidade (vs solução RK4): {r2_velocidade:.4f}")
    
#     # ============================================
#     # GRÁFICO 1: Real vs Previsto
#     # ============================================
    
#     fig1 = cria_grafico_interpolacao_pontual_mlp(
#         predictions=predictions_all,
#         y_true=y_true_all,
#         titulo="Interpolação entre Trajetórias: RK4 vs MLP - Oscilador de Van der Pol"
#     )
    
#     fig1.write_html(grafico_interpolacao_entre_trajetorias)
    
#     # ============================================
#     # GRÁFICO 2: Espaço de Fases
#     # ============================================
    
#     y_posicao_true = y_true_all[:, 0].reshape(-1, 1)
#     y_velocidade_true = y_true_all[:, 1].reshape(-1, 1)
#     y_posicao_pred = predictions_all[:, 0].reshape(-1, 1)
#     y_velocidade_pred = predictions_all[:, 1].reshape(-1, 1)
    
#     fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
#         y_pos_true=y_posicao_true,
#         y_vel_true=y_velocidade_true,
#         y_pos_pred=y_posicao_pred,
#         y_vel_pred=y_velocidade_pred,
#         titulo="Interpolação entre Trajetórias: MLP vs RK4 - Espaço de Fases - Oscilador de Van der Pol"
#     )
    
#     fig2.write_html(grafico_interpolacao_entre_trajetorias_espaco_fases)
    
#     # ============================================
#     # GRÁFICO 3: Posição e Velocidade vs Tempo
#     # ============================================
    
#     fig3 = cria_grafico_interpolacao_pontual_completo(
#         tempos_lista=tempos_lista,
#         posicoes_previstas_lista=posicao_prevista_lista,
#         velocidades_previstas_lista=velocidade_prevista_lista,
#         posicoes_reais_lista=posicao_real_lista,
#         velocidades_reais_lista=velocidade_real_lista,
#         casos_info=casos_info_lista,
#         titulo="Interpolação entre Trajetórias: MLP vs RK4 - Posição e Velocidade vs Tempo - Oscilador de Van der Pol"
#     )
    
#     fig3.write_html(grafico_interpolacao_entre_trajetorias_temporal)
    
#     fig1.show()
#     fig2.show()
#     fig3.show()
    
#     df_interpolado = pd.DataFrame(dados_interpolados)
    
#     df_interpolado.attrs['rmse_posicao'] = rmse_posicao
#     df_interpolado.attrs['rmse_velocidade'] = rmse_velocidade
#     df_interpolado.attrs['r2_posicao'] = r2_posicao
#     df_interpolado.attrs['r2_velocidade'] = r2_velocidade
#     df_interpolado.attrs['total_pontos'] = len(predictions_all)
#     df_interpolado.attrs['num_trajetorias'] = 2
#     df_interpolado.attrs['num_alpha'] = len(alphas)
#     df_interpolado.attrs['num_tempos'] = len(tempos_ajustados)
#     df_interpolado.attrs['parametro_mu'] = mu
#     df_interpolado.attrs['posicao_eq'] = x_eq
#     df_interpolado.attrs['velocidade_eq'] = y_eq
#     df_interpolado.attrs['x0_min'] = x0_min
#     df_interpolado.attrs['x0_max'] = x0_max
#     df_interpolado.attrs['y0_min'] = y0_min
#     df_interpolado.attrs['y0_max'] = y0_max
    
#     # ========================================================================
#     # GRÁFICO 4: Trajetórias Originais e Interpoladas no Espaço de Fases
#     # ========================================================================
    
#     osc1 = OsciladorVanDerPol(
#         parametros_mu=[mu],
#         device='cpu'
#     )
    
#     cond1 = torch.tensor([[x0_1, y0_1]], dtype=torch.float32)
#     sol1 = osc1.resolve_multi_condicoes_sistemas(
#         condicoes_iniciais=cond1,
#         t_final=tempo_maximo,
#         dt=dt_interpolacao
#     )
#     posicao_traj1 = sol1['posicao'][:, 0, 0]
#     velocidade_traj1 = sol1['velocidade'][:, 0, 0]
#     tempos_reais1 = sol1['tempo']
    
#     interp_posicao1 = interp1d(tempos_reais1, posicao_traj1, kind='linear', fill_value='extrapolate')
#     interp_velocidade1 = interp1d(tempos_reais1, velocidade_traj1, kind='linear', fill_value='extrapolate')
#     posicao_traj1 = interp_posicao1(tempos_ajustados)
#     velocidade_traj1 = interp_velocidade1(tempos_ajustados)
    
#     cond2 = torch.tensor([[x0_2, y0_2]], dtype=torch.float32)
#     sol2 = osc1.resolve_multi_condicoes_sistemas(
#         condicoes_iniciais=cond2,
#         t_final=tempo_maximo,
#         dt=dt_interpolacao
#     )
#     posicao_traj2 = sol2['posicao'][:, 0, 0]
#     velocidade_traj2 = sol2['velocidade'][:, 0, 0]
#     tempos_reais2 = sol2['tempo']
    
#     interp_posicao2 = interp1d(tempos_reais2, posicao_traj2, kind='linear', fill_value='extrapolate')
#     interp_velocidade2 = interp1d(tempos_reais2, velocidade_traj2, kind='linear', fill_value='extrapolate')
#     posicao_traj2 = interp_posicao2(tempos_ajustados)
#     velocidade_traj2 = interp_velocidade2(tempos_ajustados)
    
#     # interpolações
#     interpolacoes_para_grafico = []
#     alphas_unicos = np.sort(df_interpolado['alpha_interpolacao'].unique())
    
#     for alpha in alphas_unicos:
#         if alpha == 0 or alpha == 1:
#             continue
        
#         mask_alpha = df_interpolado['alpha_interpolacao'] == alpha
#         dados_alpha = df_interpolado[mask_alpha].sort_values('tempo')
        
#         x0_interp = dados_alpha['x0_interpolado'].iloc[0]
#         y0_interp = dados_alpha['y0_interpolado'].iloc[0]
        
#         interpolacoes_para_grafico.append({
#             'alpha': alpha,
#             'posicoes': dados_alpha['posicao_previsto_mlp'].values,
#             'velocidades': dados_alpha['velocidade_previsto_mlp'].values,
#             'x0_interp': x0_interp,
#             'v0_interp': y0_interp
#         })
    
#     casos_info_grafico = [{
#         'x0_1': x0_1,
#         'v0_1': y0_1,
#         'x0_2': x0_2,
#         'v0_2': y0_2
#     }]
    
#     fig4 = cria_grafico_interpolacao_entre_trajetorias_espaco_fases(
#         trajetoria1_pos=posicao_traj1,
#         trajetoria1_vel=velocidade_traj1,
#         trajetoria2_pos=posicao_traj2,
#         trajetoria2_vel=velocidade_traj2,
#         interpolacoes_lista=interpolacoes_para_grafico,
#         casos_info=casos_info_grafico,
#         titulo="Interpolação entre Trajetórias no Espaço de Fases - Oscilador de Van der Pol"
#     )
    
#     grafico_entre_trajetorias_espaco_fases = f"{output_dir}/interpolacao_entre_trajetorias_espaco_fases_detalhado.html"
#     fig4.write_html(grafico_entre_trajetorias_espaco_fases)
    
#     fig4.show()

#     return df_interpolado


# def interpola_trajetorias_mlp_node(
#     model: nn.Module,
#     scaler_X: StandardScaler,
#     scaler_y: StandardScaler,
#     parameters: Dict[str, Any],
#     tempos_referencia: np.ndarray = None
# ) -> pd.DataFrame:
#     """
#     Node: Usa o modelo treinado para gerar diferentes condições iniciais a partir de uma trajetória base.
#     A partir de uma trajetória escolhida aleatoriamente, gera novas condições iniciais variando x0 e y0.
#     Não mistura dados de treino/validação/teste pois usa dados gerados aleatoriamente.
#     Prevê trajetórias completas a partir das condições iniciais.
#     """
    
#     exp_name = parameters.get('exp_name', 'default_exp')
#     data_version = parameters.get('data_version', 'base_01')
    
#     output_dir = f"data/08_reporting/{exp_name}/{data_version}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     grafico_interpolacao_trajetorias = f"{output_dir}/interpolacoes_trajetorias_real_previsto_mlp.html"
#     grafico_interpolacao_trajetorias_espaco_fases = f"{output_dir}/interpolacao_trajetorias_espaco_fases.html"
#     grafico_interpolacao_trajetorias_temporal = f"{output_dir}/interpolacao_trajetorias_posicao_velocidade_vs_t.html"
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
    
#     intervals = parameters.get('intervals', {})
    
#     mu = intervals.get('parametro_mu', 1.0)
    
#     # ponto de equilíbrio do Van der Pol (0,0)
#     x_eq = 0.0
#     y_eq = 0.0
    
#     x0_min = intervals.get('x0_min', -3.0)
#     x0_max = intervals.get('x0_max', 3.0)
#     y0_min = intervals.get('y0_min', -3.0)
#     y0_max = intervals.get('y0_max', 3.0)
    
#     seed = parameters.get('seed', 42)
#     np.random.seed(seed)
    
#     print("\n=== GERAÇÃO DE CONDIÇÕES INICIAIS A PARTIR DA TRAJETÓRIA BASE - OSCILADOR DE VAN DER POL ===")
#     print(f"  Parâmetro do sistema: μ={mu:.2f}")
#     print(f"  Ponto de equilíbrio: x*={x_eq:.2f}, y*={y_eq:.2f}")
#     print("  Gerando novas condições iniciais variando x0 e y0 dentro dos limites de treino do modelo")
    
#     # ============================================
#     # GERAÇÃO DE DADOS ALEATÓRIOS
#     # ============================================
    
#     # define o número de pontos baseado nos tempos de referência ou padrão
#     if tempos_referencia is not None:
#         num_pontos_por_trajetoria = len(tempos_referencia)
#         tempos_unicos = tempos_referencia
#         tempo_maximo = tempos_unicos[-1]
#         dt_interpolacao = tempos_unicos[1] - tempos_unicos[0] if len(tempos_unicos) > 1 else 0.01
#         print(f"\n  Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria} pontos")
#         print(f"  Tempo máximo: {tempo_maximo:.3f} s")
#     else:
#         # Período aproximado para o Van der Pol
#         if mu < 0.1:
#             T_aprox = 2.0 * np.pi
#         else:
#             T_aprox = (3.0 - 2.0 * np.log(2.0)) / mu
        
#         sim_params = parameters.get('simulation', {})
#         dt = sim_params.get('dt', 0.01)
#         num_periodos = sim_params.get('num_periodos', 3)
#         tempo_maximo = num_periodos * T_aprox
#         num_pontos_por_trajetoria = int(tempo_maximo / dt) + 1
#         tempos_unicos = np.linspace(0, tempo_maximo, num_pontos_por_trajetoria)
#         dt_interpolacao = tempos_unicos[1] - tempos_unicos[0]
        
#         print(f"\n  Configuração da interpolação:")
#         print(f"    Período aproximado: {T_aprox:.3f} s")
#         print(f"    Tempo máximo: {tempo_maximo:.3f} s")
#         print(f"    Passo temporal: {dt_interpolacao:.6f} s")
#         print(f"    Nós de saída do modelo por trajetória: {num_pontos_por_trajetoria}")
    
#     # gera uma trajetória base aleatória
#     x0_base = np.random.uniform(x0_min, x0_max)
#     y0_base = np.random.uniform(y0_min, y0_max)
    
#     print(f"\n  Trajetória Base Selecionada:")
#     print(f"    x0 = {x0_base:.3f}")
#     print(f"    y0 = {y0_base:.3f}")
#     print(f"    Distância do equilíbrio: {np.sqrt(x0_base**2 + y0_base**2):.3f}")
#     print(f"    Tempo máximo: {tempo_maximo:.3f} s")
    
#     num_variacoes = 5
#     variacoes = []
    
#     # geração aleatória dentro dos limites
#     np.random.seed(seed)
#     x0_variacoes = np.random.uniform(x0_min, x0_max, num_variacoes)
#     y0_variacoes = np.random.uniform(y0_min, y0_max, num_variacoes)
        
#     print(f"\n  Gerando {num_variacoes} novas condições iniciais:")
#     for i in range(num_variacoes):
#         distancia = np.sqrt(x0_variacoes[i]**2 + y0_variacoes[i]**2)
#         variacoes.append({
#             'x0': x0_variacoes[i],
#             'y0': y0_variacoes[i],
#             'distancia_eq': distancia
#         })
#         print(f"    Caso {i+1}: x0={x0_variacoes[i]:.3f}, y0={y0_variacoes[i]:.3f}, distância={distancia:.3f}")
    
#     todas_previsoes = []
#     todos_reais_interpolados = []
    
#     tempos_lista = []
#     posicao_prevista_lista = []
#     velocidade_prevista_lista = []
#     posicao_real_lista = []
#     velocidade_real_lista = []
#     casos_info_lista = []
#     dados_interpolados = []
    
#     # para cada nova condição inicial, faz a previsão
#     for var_idx, var in enumerate(variacoes):
#         x0_novo = var['x0']
#         y0_novo = var['y0']
        
#         # entrada para o modelo: x0, y0
#         X_novo = np.array([[x0_novo, y0_novo]], dtype=np.float32)
        
#         # normaliza e faz previsão
#         X_novo_scaled = scaler_X.transform(X_novo)
#         X_tensor = torch.tensor(X_novo_scaled, dtype=torch.float32).to(device)
        
#         with torch.no_grad():
#             pred_scaled = model(X_tensor).cpu().numpy()
        
#         # desnormaliza a trajetória completa
#         pred = scaler_y.inverse_transform(pred_scaled)
        
#         # separa posição e velocidade da trajetória completa
#         # saída: [x0, y0, x1, y1, ..., xN, yN]
#         posicao_prevista = pred[0, 0::2]  # posição (índices pares)
#         velocidade_prevista = pred[0, 1::2]  # velocidade (índices ímpares)
        
#         # verifica se o número de pontos coincide com os tempos
#         if len(posicao_prevista) != len(tempos_unicos):
#             print(f"  AVISO: Ajustando tempos para {len(posicao_prevista)} pontos")
#             tempos_ajustados = np.linspace(tempos_unicos[0], tempos_unicos[-1], len(posicao_prevista))
#         else:
#             tempos_ajustados = tempos_unicos
        
#         osc = OsciladorVanDerPol(
#             parametros_mu=[mu],
#             device='cpu'
#         )
        
#         cond_curta = torch.tensor([[x0_novo, y0_novo]], dtype=torch.float32)
#         solucao_curta = osc.resolve_multi_condicoes_sistemas(
#             condicoes_iniciais=cond_curta,
#             t_final=tempo_maximo,
#             dt=dt_interpolacao
#         )
        
#         posicao_real = solucao_curta['posicao'][:, 0, 0]
#         velocidade_real = solucao_curta['velocidade'][:, 0, 0]
#         tempos_reais = solucao_curta['tempo']
        
#         if len(tempos_reais) != len(tempos_ajustados):
#             interp_posicao = interp1d(tempos_reais, posicao_real, kind='linear', fill_value='extrapolate')
#             interp_velocidade = interp1d(tempos_reais, velocidade_real, kind='linear', fill_value='extrapolate')
#             posicao_real_ajustado = interp_posicao(tempos_ajustados)
#             velocidade_real_ajustado = interp_velocidade(tempos_ajustados)
#         else:
#             posicao_real_ajustado = posicao_real
#             velocidade_real_ajustado = velocidade_real
        
#         # métricas globais (ponto a ponto para compatibilidade)
#         pred_pontos = np.column_stack([posicao_prevista, velocidade_prevista])
#         real_pontos = np.column_stack([posicao_real_ajustado, velocidade_real_ajustado])
        
#         todas_previsoes.append(pred_pontos)
#         todos_reais_interpolados.append(real_pontos)
        
#         tempos_lista.append(tempos_ajustados)
#         posicao_prevista_lista.append(posicao_prevista)
#         velocidade_prevista_lista.append(velocidade_prevista)
#         posicao_real_lista.append(posicao_real_ajustado)
#         velocidade_real_lista.append(velocidade_real_ajustado)
        
#         cor = CORES_PALETA[var_idx % len(CORES_PALETA)]
        
#         casos_info_lista.append({
#             'x0': x0_novo,
#             'y0': y0_novo,
#             'cor': cor,
#             'variation_id': var_idx
#         })
        
#         for k in range(len(tempos_ajustados)):
#             dados_interpolados.append({
#                 'variacao_id': var_idx,
#                 'x0': x0_novo,
#                 'y0': y0_novo,
#                 'parametro_mu': mu,
#                 'posicao_eq': x_eq,
#                 'velocidade_eq': y_eq,
#                 'tempo': tempos_ajustados[k],
#                 'posicao_real': posicao_real_ajustado[k],
#                 'velocidade_real': velocidade_real_ajustado[k],
#                 'posicao_previsto_mlp': pred_pontos[k, 0],
#                 'velocidade_previsto_mlp': pred_pontos[k, 1],
#                 'erro_posicao': pred_pontos[k, 0] - posicao_real_ajustado[k],
#                 'erro_velocidade': pred_pontos[k, 1] - velocidade_real_ajustado[k],
#                 'erro_abs_posicao': abs(pred_pontos[k, 0] - posicao_real_ajustado[k]),
#                 'erro_abs_velocidade': abs(pred_pontos[k, 1] - velocidade_real_ajustado[k]),
#                 'erro_rel_posicao_pct': (abs(pred_pontos[k, 0] - posicao_real_ajustado[k]) / (abs(posicao_real_ajustado[k]) + 1e-6)) * 100,
#                 'erro_rel_velocidade_pct': (abs(pred_pontos[k, 1] - velocidade_real_ajustado[k]) / (abs(velocidade_real_ajustado[k]) + 1e-6)) * 100,
#             })
    
#     if len(todas_previsoes) == 0:
#         print("  ERRO: Nenhuma previsão realizada")
#         return pd.DataFrame()
    
#     predictions_all = np.vstack(todas_previsoes)
#     y_true_all = np.vstack(todos_reais_interpolados)
    
#     rmse_posicao = float(np.sqrt(mean_squared_error(y_true_all[:, 0], predictions_all[:, 0])))
#     rmse_velocidade = float(np.sqrt(mean_squared_error(y_true_all[:, 1], predictions_all[:, 1])))
#     r2_posicao = float(r2_score(y_true_all[:, 0], predictions_all[:, 0]))
#     r2_velocidade = float(r2_score(y_true_all[:, 1], predictions_all[:, 1]))
    
#     print(f"\n  Total de pontos previstos: {len(predictions_all)}")
#     print(f"  RMSE Posição (vs solução RK4): {rmse_posicao:.6f}")
#     print(f"  RMSE Velocidade (vs solução RK4): {rmse_velocidade:.6f}")
#     print(f"  R² Posição (vs solução RK4): {r2_posicao:.4f}")
#     print(f"  R² Velocidade (vs solução RK4): {r2_velocidade:.4f}")
    
#     # ============================================
#     # GRÁFICO 1: Real vs Previsto
#     # ============================================
    
#     fig1 = cria_grafico_interpolacao_pontual_mlp(
#         predictions=predictions_all,
#         y_true=y_true_all,
#         titulo="Novas Condições Iniciais: RK4 vs MLP - Oscilador de Van der Pol"
#     )
    
#     fig1.write_html(grafico_interpolacao_trajetorias)
    
#     # ============================================
#     # GRÁFICO 2: Espaço de Fases
#     # ============================================
    
#     y_posicao_true = y_true_all[:, 0].reshape(-1, 1)
#     y_velocidade_true = y_true_all[:, 1].reshape(-1, 1)
#     y_posicao_pred = predictions_all[:, 0].reshape(-1, 1)
#     y_velocidade_pred = predictions_all[:, 1].reshape(-1, 1)
    
#     fig2 = cria_grafico_interpolacao_pontual_espaco_fases(
#         y_pos_true=y_posicao_true,
#         y_vel_true=y_velocidade_true,
#         y_pos_pred=y_posicao_pred,
#         y_vel_pred=y_velocidade_pred,
#         titulo="Novas Condições Iniciais: MLP vs RK4 - Espaço de Fases - Oscilador de Van der Pol"
#     )
    
#     fig2.write_html(grafico_interpolacao_trajetorias_espaco_fases)
    
#     # ============================================
#     # GRÁFICO 3: Posição e Velocidade vs Tempo
#     # ============================================
    
#     fig3 = cria_grafico_interpolacao_pontual_completo(
#         tempos_lista=tempos_lista,
#         posicoes_previstas_lista=posicao_prevista_lista,
#         velocidades_previstas_lista=velocidade_prevista_lista,
#         posicoes_reais_lista=posicao_real_lista,
#         velocidades_reais_lista=velocidade_real_lista,
#         casos_info=casos_info_lista,
#         titulo="Novas Condições Iniciais: MLP vs RK4 - Posição e Velocidade vs Tempo - Oscilador de Van der Pol"
#     )
    
#     fig3.write_html(grafico_interpolacao_trajetorias_temporal)
    
#     fig1.show()
#     fig2.show()
#     fig3.show()
    
#     df_interpolado = pd.DataFrame(dados_interpolados)
    
#     df_interpolado.attrs['rmse_posicao'] = rmse_posicao
#     df_interpolado.attrs['rmse_velocidade'] = rmse_velocidade
#     df_interpolado.attrs['r2_posicao'] = r2_posicao
#     df_interpolado.attrs['r2_velocidade'] = r2_velocidade
#     df_interpolado.attrs['total_pontos'] = len(predictions_all)
#     df_interpolado.attrs['num_variacoes'] = num_variacoes
#     df_interpolado.attrs['num_tempos'] = len(tempos_ajustados)
#     df_interpolado.attrs['parametro_mu'] = mu
#     df_interpolado.attrs['posicao_eq'] = x_eq
#     df_interpolado.attrs['velocidade_eq'] = y_eq
#     df_interpolado.attrs['x0_min'] = x0_min
#     df_interpolado.attrs['x0_max'] = x0_max
#     df_interpolado.attrs['y0_min'] = y0_min
#     df_interpolado.attrs['y0_max'] = y0_max
    
#     # ========================================================================
#     # GRÁFICO 4: Trajetória Base e Novas Condições Iniciais no Espaço de Fases
#     # ========================================================================
    
#     # gera a trajetória base via RK4
#     osc_base = OsciladorVanDerPol(
#         parametros_mu=[mu],
#         device='cpu'
#     )
    
#     cond_base = torch.tensor([[x0_base, y0_base]], dtype=torch.float32)
#     sol_base = osc_base.resolve_multi_condicoes_sistemas(
#         condicoes_iniciais=cond_base,
#         t_final=tempo_maximo,
#         dt=dt_interpolacao
#     )
    
#     posicao_base = sol_base['posicao'][:, 0, 0]
#     velocidade_base = sol_base['velocidade'][:, 0, 0]
#     tempos_base = sol_base['tempo']
    
#     if len(tempos_base) != len(tempos_ajustados):
#         interp_posicao_base = interp1d(tempos_base, posicao_base, kind='linear', fill_value='extrapolate')
#         interp_velocidade_base = interp1d(tempos_base, velocidade_base, kind='linear', fill_value='extrapolate')
#         posicao_base = interp_posicao_base(tempos_ajustados)
#         velocidade_base = interp_velocidade_base(tempos_ajustados)
    
#     novas_trajetorias_para_grafico = []
    
#     for var_idx in range(num_variacoes):
#         mask_var = df_interpolado['variacao_id'] == var_idx
#         dados_var = df_interpolado[mask_var].sort_values('tempo')
        
#         if len(dados_var) > 0:
#             x0_var = dados_var['x0'].iloc[0]
#             y0_var = dados_var['y0'].iloc[0]
            
#             novas_trajetorias_para_grafico.append({
#                 'variacao_id': var_idx,
#                 'posicoes': dados_var['posicao_previsto_mlp'].values,
#                 'velocidades': dados_var['velocidade_previsto_mlp'].values,
#                 'x0': x0_var,
#                 'v0': y0_var
#             })
    
#     casos_info_grafico = {
#         'x0_base': x0_base,
#         'v0_base': y0_base
#     }
    
#     fig4 = cria_grafico_interpolacao_trajetorias_espaco_fases(
#         trajetoria_base_pos=posicao_base,
#         trajetoria_base_vel=velocidade_base,
#         novas_trajetorias_lista=novas_trajetorias_para_grafico,
#         casos_info=casos_info_grafico,
#         titulo="Trajetória Base vs Novas Condições Iniciais no Espaço de Fases - Oscilador de Van der Pol"
#     )
    
#     grafico_novas_trajetorias = f"{output_dir}/trajetoria_base_vs_novas_condicoes.html"
#     fig4.write_html(grafico_novas_trajetorias)
    
#     fig4.show()

#     return df_interpolado