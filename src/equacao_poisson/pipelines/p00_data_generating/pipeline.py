from kedro.pipeline import Pipeline, node

from .nodes import (
    le_distribuicao_carga_node,
    executa_simulacao_espectral_node,
    gera_base_consolidada_node,
    cria_visualizacoes_node
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=le_distribuicao_carga_node,
            inputs="parameters",
            outputs="distribuicoes_carga",
            name="node_le_distribuicao_carga",
            tags=["generation", "charge_distribution"]
        ),

        node(
            func=executa_simulacao_espectral_node,
            inputs=["distribuicoes_carga", "parameters"],
            outputs=["solucao_espectral", "metadata_simulacao"],
            name="node_executa_simulacao_espectral",
            tags=["simulation", "spectral"]
        ),

        node(
            func=gera_base_consolidada_node,
            inputs=["solucao_espectral"],
            outputs="base_eletrostatica",
            name="node_gera_base_consolidada",
            tags=["data", "database"]
        ),

        node(
            func=cria_visualizacoes_node,
            inputs=["solucao_espectral"],
            outputs=None,
            name="node_cria_visualizacoes",
            tags=["visualization", "plotly"]
        ),

    ])