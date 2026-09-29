"""
Definição da arquitetura do MLP para previsão do potencial elétrico φ(x,y,z)
a partir das coordenadas espaciais (x, y, z) e da densidade de carga ρ(x,y,z)
na equação de Poisson 3D em eletrostática.
"""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron para prever o potencial elétrico φ(x,y,z) ponto a
    ponto, a partir das coordenadas espaciais (x, y, z) e da densidade de
    carga ρ(x,y,z), resolvendo a equação de Poisson 3D em eletrostática.

    Entrada: [x, y, z, ρ(x,y,z)] — 4 features por ponto de colocação.
    Saída:   φ(x,y,z) — 1 valor por ponto de colocação.
    """

    def __init__(self, input_dim=4, hidden_dims=[128, 256, 512], output_dim=1,
                 activation='tanh', seed=None):
        """
        Args:
            input_dim:   dimensão da entrada (4: x, y, z, ρ).
            hidden_dims: dimensões das camadas ocultas.
            output_dim:  dimensão da saída (1: φ).
            activation:  função de ativação ('relu', 'tanh', 'sigmoid').
            seed:        semente para reprodutibilidade.
        """
        super(MLP, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        if output_dim is None:
            output_dim = 1
        self.input_dim  = input_dim
        self.output_dim = output_dim

        activation_functions = {
            'sigmoid': nn.Sigmoid(),
            'relu':    nn.ReLU(),
            'tanh':    nn.Tanh(),
        }
        self.activation = activation_functions.get(activation.lower(), nn.Tanh())

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: tensor de entrada (batch_size, 4) — [x, y, z, ρ] por ponto.

        Returns:
            tensor de saída (batch_size, 1) — φ(x,y,z) por ponto.
        """
        return self.network(x)

    def predict_potential(self, x, y, z, rho):
        """
        Prevê o potencial elétrico φ em um conjunto de pontos (x, y, z) para
        uma dada distribuição de carga ρ avaliada nesses mesmos pontos.

        Args:
            x, y, z: coordenadas dos pontos, arrays de shape (n_pontos,) ou
                     escalares.
            rho:     valores de ρ nos mesmos pontos, mesmo shape de x/y/z.

        Returns:
            array de φ com o mesmo shape de x/y/z.
        """
        self.eval()

        x_arr   = np.asarray(x).ravel()
        y_arr   = np.asarray(y).ravel()
        z_arr   = np.asarray(z).ravel()
        rho_arr = np.asarray(rho).ravel()

        features = np.column_stack([x_arr, y_arr, z_arr, rho_arr]).astype(np.float32)

        with torch.no_grad():
            t = torch.tensor(features, dtype=torch.float32)
            output = self.forward(t).numpy().flatten()

        return output.reshape(np.asarray(x).shape)

    def predict_potential_batch(self, x, y, z, rho):
        """
        Prevê o potencial elétrico φ para múltiplos conjuntos de pontos,
        cada um associado a uma distribuição de carga ρ diferente.

        Args:
            x, y, z: coordenadas dos pontos, arrays de shape
                     (batch_size, n_pontos) ou (batch_size,).
            rho:     valores de ρ nos mesmos pontos, shape compatível com x/y/z.

        Returns:
            array de φ com o mesmo shape de x/y/z (batch_size, n_pontos).
        """
        self.eval()

        x_arr   = np.asarray(x)
        y_arr   = np.asarray(y)
        z_arr   = np.asarray(z)
        rho_arr = np.asarray(rho)

        # achata para (n_total, 4)
        features = np.column_stack([
            x_arr.ravel(),
            y_arr.ravel(),
            z_arr.ravel(),
            rho_arr.ravel(),
        ]).astype(np.float32)

        with torch.no_grad():
            t = torch.tensor(features, dtype=torch.float32)
            outputs = self.forward(t).numpy().flatten()

        return outputs.reshape(x_arr.shape)