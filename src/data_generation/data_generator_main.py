import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from utils.logging_results import setup_logger
from utils.saving_results import save_to_pickle
from data_generation.data_simulator import IndividualDataSimulator
from data_generation.network_generator import GraphGenerator, NetworkFeatureExtractor, StochasticBlockModel
from data_generation.corrupt_network import make_corrupt_network
from utils.warnings import validate_argument


def run_data_simulations(
    assignment_types: List[str],
    treatment_effects: List[Tuple[float, int]],
    network_configs: List[tuple[str, float | None]],
    neighbour_influences: List[float],
    n_nodes: int,
    n_features: int,
    error_mean: float,
    error_std: float,
    beta_mean: float,
    beta_std: float,
    share_treatment: float,
    n_edges: int,
    n_sim: int,
    network_structure_type: str = "random",
    n_blocks: int = None,
    inter_block_probs: int = None,
    add_network_features: bool = False,
    add_embeddings: bool = False,
    add_count_treated: bool = False,
    dimensions: int = None,
    walk_length: int = None,
    num_walks: int = None,
    p_random_walk: float = None,
    q_random_walk: float = None,
    vec_window: int = None,
    min_count: int = None,
    batch_words: int = None,
    corrupt_network: bool = False,
    save_data: bool = True
) -> Dict[str, Any]:
    """Function for data generation simulations

    Args:
        assignment_types (List[str]): List of treatment assignment types 
        treatment_effects (List[Tuple[float, int]]): List of tuples representing treatment effects
        network_configs (List[Tuple[str, Optional[float]]]): List of network types and their parameters
            The first value is the network type (e.g., "barabasi_albert_graph"), 
            and the second is an optional parameter
        neighbour_influences (List[float]): List of influence values from neighbors
        n_nodes (int): Number of nodes in the network
        n_features (int): Number of features
        error_mean (float): Mean of the error term
        error_std (float): Standard deviation of the error term
        beta_mean (float): Mean of the beta
        beta_std (float): Standard deviation of the beta
        share_treatment (float): Proportion of treated nodes in the network
        n_edges (int): Number of edges in the network
        n_sim (int): Number of simulation runs to perform
        add_network_features (bool, optional): Whether to include network-based features. Defaults to False.
        add_embeddings (bool, optional): Whether to compute and include node embeddings. Defaults to False.
        dimensions (Optional[int], optional): Number of dimensions for node embeddings. Defaults to None.
        walk_length (Optional[int], optional): Length of random walks for node embeddings. Defaults to None.
        num_walks (Optional[int], optional): Number of random walks per node. Defaults to None.
        p_random_walk (Optional[float], optional): Return parameter for random walks. Defaults to None.
        q_random_walk (Optional[float], optional): In-out parameter for random walks. Defaults to None.
        vec_window (Optional[int], optional): Context window size for Node2Vec embeddings. Defaults to None.
        min_count (Optional[int], optional): Minimum word count for embeddings training. Defaults to None.
        batch_words (Optional[int], optional): Batch size for Node2Vec training. Defaults to None.
        corrupt_network (bool, optional): Whether to corrupt network / add aditional noise, 
            remove or add some edges with given probabilities. Defaults to False.
        save_data (bool, optional): Whether to save the generated data to a file. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing the generated simulation data, including network structures, 
            treatment assignments, feature values, and results.
    """
    setup_logger()
    logging.info("Checking validity...")
    validate_argument(
        value=network_structure_type,
        allowed_values=['random', 'covariate_block'],
        argument_name='network_structure_type',
    )
    logging.info("Starting data simulations...")
    results = []
    
    total_iterations = (
        len(assignment_types) * len(treatment_effects) * len(network_configs) * len(neighbour_influences) * n_sim
    )
    
    with tqdm(total=total_iterations, desc="Simulations Progress") as pbar:
        for assignment_type in assignment_types:
            individual_data_simulator = IndividualDataSimulator(
                n_nodes=n_nodes,
                n_features=n_features,
                error_mean=error_mean,
                error_std=error_std,
                beta_mean=beta_mean,
                beta_std=beta_std,
                share_treatment=share_treatment,
                group_assignment_type=assignment_type
            )
            
            for treatment_effect_mean, treatment_effect_std in treatment_effects:
                individual_data_simulator.set_treatment_effect(
                    treatment_effect_mean=treatment_effect_mean,
                    treatment_effect_std=treatment_effect_std
                )
                
                for network_type, p_edges in network_configs:
                    network_feature_extractor = NetworkFeatureExtractor()
                    if network_structure_type == "random":
                        graph_generator = GraphGenerator(
                            n_nodes=n_nodes,
                            n_edges=n_edges,
                            p_edges=p_edges
                        )
                    else:
                        graph_generator = StochasticBlockModel(
                            n_blocks=n_blocks,
                            inter_block_probs=inter_block_probs
                        )
                                            
                    for influence in neighbour_influences:
                        simulation_artifacts = []
                        
                        for i_sim in range(n_sim):
                            logging.info(f"Running simulation {i_sim+1}/{n_sim} for {assignment_type}, {network_type}, influence {influence}")                            
                            error = individual_data_simulator.generate_random_error()
                            beta = individual_data_simulator.generate_random_beta_matrix()
                            covariates = individual_data_simulator.generate_random_covariates()
                            if network_structure_type == "random":
                                adj_matrix = graph_generator.generate_network(network_type)
                            else:
                                adj_matrix, block_assignments = graph_generator.generate_network(covariates)
                            if corrupt_network:
                                adj_matrix_corrupt = make_corrupt_network(adj_matrix=adj_matrix)

                            group_assignment = individual_data_simulator.generate_group_assignment(
                                covariates=covariates,
                                adj_matrix=adj_matrix
                            )

                            treatment_effect = individual_data_simulator.generate_treatment_effect()
                            
                            outcome = individual_data_simulator.generate_SAR_outcome(
                                adj_matrix=adj_matrix,
                                neighbour_influence=influence,
                                group_assignment=group_assignment,
                                beta=beta,
                                covariates=covariates,
                                error=error,
                                treatment_effect=treatment_effect
                            )
                            
                            individ_data = individual_data_simulator.generate_individ_data(
                                outcome=outcome,
                                covariates=covariates,
                                group_assignment=group_assignment
                            )
                            
                            if add_network_features:
                                if not corrupt_network:
                                    df_features, df_communities = network_feature_extractor.extract_network_features(adj_matrix=adj_matrix)
                                else:
                                    df_features, df_communities = network_feature_extractor.extract_network_features(adj_matrix=adj_matrix_corrupt)
                                individ_data = individ_data.join(df_features, how="left")
                                individ_data = individ_data.merge(df_communities, on="node", how="left")

                            if add_count_treated:
                                if not corrupt_network:
                                    df_treated_neighbors = network_feature_extractor.count_treated_neighbors(
                                        adj_matrix=adj_matrix, 
                                        treatment_assignment=group_assignment
                                    )
                                else:
                                    df_treated_neighbors = network_feature_extractor.count_treated_neighbors(
                                        adj_matrix=adj_matrix_corrupt, 
                                        treatment_assignment=group_assignment
                                    )
                                individ_data = individ_data.merge(df_treated_neighbors, on="node", how="left")

                            if add_embeddings:
                                if not corrupt_network:
                                    df_embeddings = network_feature_extractor.generate_node_embeddings(
                                        adj_matrix=adj_matrix,
                                        dimensions=dimensions,
                                        walk_length=walk_length,
                                        num_walks=num_walks,
                                        p=p_random_walk,
                                        q=q_random_walk,
                                        window=vec_window,
                                        min_count=min_count,
                                        batch_words=batch_words
                                    )
                                else:
                                    df_embeddings = network_feature_extractor.generate_node_embeddings(
                                        adj_matrix=adj_matrix_corrupt,
                                        dimensions=dimensions,
                                        walk_length=walk_length,
                                        num_walks=num_walks,
                                        p=p_random_walk,
                                        q=q_random_walk,
                                        window=vec_window,
                                        min_count=min_count,
                                        batch_words=batch_words
                                    )
                                individ_data = individ_data.merge(df_embeddings, on="node", how="left")
                            
                            artifact = {
                                "i_sim": i_sim,
                                "individ_data": individ_data,
                                "treatment_effect": treatment_effect,
                                "adj_matrix": adj_matrix
                            }
                            simulation_artifacts.append(artifact)
                            pbar.update(1)
                        
                        param_summary = {
                            "assignment_type": assignment_type,
                            "treatment_effect_mean": treatment_effect_mean,
                            "treatment_effect_std": treatment_effect_std,
                            "network_type": network_type,
                            "p_edges": p_edges,
                            "neighbour_influence": influence
                        }
                        
                        if save_data:
                            corrupt_network_path = "corrupt_network" if corrupt_network else ""
                            filename = f"sim_{network_structure_type}_{assignment_type}_{network_type}_influence{influence}_{corrupt_network_path}_.pkl"
                            filepath = save_to_pickle(simulation_artifacts, filename)
                            param_summary["filename"] = filepath
                        
                        results.append({"parameters": param_summary, "simulations": simulation_artifacts})
                        
    save_to_pickle(results, "all_simulations.pkl")
    logging.info("Data simulations completed!")
    return results
