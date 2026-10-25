"""
Definição da arquitetura do MLP para previsão do potencial elétrico adimensional
φ̃(x,y) a partir da densidade de carga adimensional ρ̃(x,y) na equação de
Poisson 2D adimensional.

O modelo recebe como entrada os valores de ρ̃(x,y) em todos os pontos da malha
de colocação (achatados em um vetor) e devolve os valores de φ̃(x,y) nos
mesmos pontos (achatados no mesmo formato).

As coordenadas (x, y) são apenas referência para montar a malha; não entram
como features de entrada nem como alvos de saída. Todos os valores são
adimensionais (grandezas puras).
"""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron para prever o potencial elétrico adimensional φ̃(x,y)
    em todos os pontos da malha a partir da densidade de carga adimensional
    ρ̃(x,y) avaliada na mesma malha.

    Entrada: ρ̃(x,y) nos N+1 pontos de colocação em cada direção, achatados
             em um vetor de dimensão (N+1)² (adimensional).
    Saída:   φ̃(x,y) nos mesmos pontos, achatados em um vetor de dimensão (N+1)²
             (adimensional).
    """

    def __init__(self, input_dim, hidden_dims=[128, 256, 512], output_dim=None,
                 activation='tanh', seed=None):
        """
        Args:
            input_dim:   dimensão da entrada, igual a (N+1)² (número de pontos
                         da malha espectral de colocação).
            hidden_dims: dimensões das camadas ocultas.
            output_dim:  dimensão da saída; se None, é igual a input_dim.
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
            output_dim = input_dim
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
            x: tensor de entrada (batch_size, (N+1)²) — valores adimensionais
               de ρ̃(x,y) nos pontos de colocação, achatados.

        Returns:
            tensor de saída (batch_size, (N+1)²) — valores adimensionais de
            φ̃(x,y) nos mesmos pontos, achatados.
        """
        return self.network(x)

    def predict_potential(self, rho):
        """
        Prevê o potencial elétrico adimensional φ̃(x,y) em toda a malha para
        uma única distribuição de carga adimensional ρ̃(x,y).

        Args:
            rho: array de densidade de carga adimensional nos pontos de
                 colocação, com shape (N+1, N+1) ou já achatado em (input_dim,).

        Returns:
            array de potencial elétrico adimensional com o mesmo shape de rho.
        """
        self.eval()
        shape_original = np.asarray(rho).shape
        with torch.no_grad():
            x = torch.tensor(np.asarray(rho).ravel(order='F'),
                             dtype=torch.float32).unsqueeze(0)
            output = self.forward(x).numpy().flatten()
        return output.reshape(shape_original, order='F')

    def predict_potential_batch(self, rho_batch):
        """
        Prevê o potencial elétrico adimensional φ̃(x,y) em toda a malha para
        múltiplas distribuições de carga adimensional ρ̃(x,y).

        Args:
            rho_batch: array de densidade de carga adimensional com shape
                       (batch_size, N+1, N+1) ou (batch_size, input_dim).

        Returns:
            array de potencial elétrico adimensional com o mesmo shape de
            rho_batch.
        """
        self.eval()
        shape_original = np.asarray(rho_batch).shape
        n = shape_original[0]
        with torch.no_grad():
            x = torch.tensor(np.asarray(rho_batch).reshape(n, -1, order='F'),
                             dtype=torch.float32)
            outputs = self.forward(x).numpy()
        return outputs.reshape(shape_original, order='F')