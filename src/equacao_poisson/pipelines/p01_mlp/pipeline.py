"""
Definição do pipeline MLP para a equação de Poisson 2D.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    prepara_dados_mlp_node,
    visualiza_distribuicao_max_rho_node,
    visualiza_distribuicao_dados_separado,
    cria_modelo_mlp_node,
    treina_mlp_node,
    avalia_metricas_mlp_node,
    visualiza_previsoes_espaco_fases_node,
    visualiza_previsoes_mlp_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Cria o pipeline de treinamento do MLP para a equação de Poisson 2D
   .

    Pipeline:
        1. Prepara dados (treino, validação, teste) a partir da base consolidada
        2. Visualiza distribuição de amplitudes máximas de densidade de carga
        3. Visualiza distribuição dos dados
        4. Cria modelo MLP
        5. Treina modelo
        6. Avalia métricas
        7. Visualiza previsões por densidade de carga
        8. Visualiza previsões do modelos vs. valores reais
    """

    return Pipeline([

        node(
            func=prepara_dados_mlp_node,
            inputs=["base_eletrostatica", "parameters"],
            outputs=["X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
                     "input_dim", "output_dim", "scaler_X", "scaler_y",
                     "cargas_train", "cargas_val", "cargas_test",
                     "params_train", "params_val", "params_test",
                     "n_cargas"],
            name="node_prepara_dados_mlp",
            tags=["data_preparation", "mlp"]
        ),

    node(
        func=visualiza_distribuicao_max_rho_node,
        inputs=["base_eletrostatica", "parameters"],
        outputs=None,
        name="node_visualiza_distribuicao_max_rho",
        tags=["visualization", "eda"]
    ),

        node(
            func=visualiza_distribuicao_dados_separado,
            inputs=["base_eletrostatica", "parameters"],
            outputs=None,
            name="node_visualiza_distribuicao_dados",
            tags=["visualization", "eda"]
        ),

        node(
            func=cria_modelo_mlp_node,
            inputs=["input_dim", "output_dim", "parameters"],
            outputs="modelo_mlp",
            name="node_cria_modelo_mlp",
            tags=["model_creation", "mlp"]
        ),

        node(
            func=treina_mlp_node,
            inputs=["modelo_mlp", "X_train", "y_train", "X_val", "y_val", "parameters"],
            outputs=["modelo_mlp_treinado", "history_mlp"],
            name="node_treina_mlp",
            tags=["training", "mlp"]
        ),

        node(
            func=avalia_metricas_mlp_node,
            inputs=["modelo_mlp_treinado", "X_val", "y_val", "X_test", "y_test", "scaler_y"],
            outputs="metricas_mlp",
            name="node_avalia_metricas_mlp",
            tags=["evaluation", "mlp"]
        ),

        node(
            func=visualiza_previsoes_mlp_node,
            inputs=["modelo_mlp_treinado", "X_test", "y_test", "scaler_y", "parameters"],
            outputs=None,
            name="node_visualiza_previsoes_mlp",
            tags=["visualization", "predictions"]
        ),

        node(
            func=visualiza_previsoes_espaco_fases_node,
            inputs=["modelo_mlp_treinado", "X_test", "y_test", "scaler_y",
                    "params_test", "parameters"],
            outputs=None,
            name="node_visualiza_previsoes_dispersao",
            tags=["visualization", "scatter"]
        ),

    ])