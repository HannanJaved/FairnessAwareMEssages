import pandas as pd
import numpy as np
import os
import torch
from torch_geometric.data import Data

def load_nba_data(
        base_path: str = os.getcwd(),
        class_labels: str = 'SALARY',
        sens_attribute: str = 'country'
        ):
    """
    Load the NBA dataset from the .csv files and prepare it for GNNs.
    """

    # Set the path to the data files
    if base_path is None:
        base_path = os.getcwd()
    input_data_path = base_path + "/Datasets/nba"

    # Load the data files
    user_labels_path = os.path.join(input_data_path, "nba.csv")
    user_edges_path = os.path.join(input_data_path, "nba_relationship.csv")

    # Create dataframes to store the information from the .csv files
    user_labels = pd.read_csv(user_labels_path)
    user_edges = pd.read_csv(user_edges_path)

    # Prepare the data for GNNs
    node_features = torch.tensor(user_labels.iloc[:, 1:].values, dtype=torch.float)
    edge_index = torch.tensor(user_edges.values, dtype=torch.long).t().contiguous()

    user_edges = user_edges[user_edges['uid1'].isin(user_labels['user_id']) & user_edges['uid2'].isin(user_labels['user_id'])]

    # Extract node features from user_labels dataframe
    node_features = user_labels.iloc[:, 1:] 
    node_features = torch.tensor(node_features.values, dtype=torch.float)

    # Extract edges from user_edges dataframe
    edges = user_edges[['uid1', 'uid2']]
    edges['uid1'] = edges['uid1'].map(dict(zip(user_labels['user_id'], range(len(user_labels)))))
    edges['uid2'] = edges['uid2'].map(dict(zip(user_labels['user_id'], range(len(user_labels)))))

    # Convert edges dataframe to tensor
    edges_tensor = torch.tensor(edges.values, dtype=torch.long).t().contiguous()

    # Create edge_index tensor
    edge_index = edges_tensor

    user_labels['SALARY'] = user_labels['SALARY'].map({-1: 0, 0: 1, 1: 1})

    # Create torch-geometric data
    data = Data(x=node_features, edge_index=edge_index)

    num_nodes = node_features.size(0)
    num_classes = 2 
    num_node_features = data.num_node_features

    # Create masks for training, and testing
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 60-20-20 Train and Test data split
    num_train = int(num_nodes * 0.6)
    num_val = int(num_nodes * 0.8)
    train_mask[:num_train] = True
    val_mask[num_train:num_val] = True
    test_mask[num_val:] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask

    # Labels from the data 
    labels_values = user_labels[class_labels].values
    data.y = torch.tensor(labels_values, dtype=torch.long)

    # Sensitivite attribute
    sens_values = user_labels[sens_attribute].values
    sens_attribute_tensor = torch.tensor(sens_values, dtype=torch.long)

    return data, num_classes, num_node_features, sens_attribute_tensor, labels_values, sens_values

def load_german_data(
        base_path: str = os.getcwd(),
        class_labels: str = 'GoodCustomer',
        sens_attribute: str = 'Gender'
        ):
    """
    Load the German dataset from the .csv files and prepare it for GNNs.
    """
    if base_path is None:
        base_path = os.getcwd()
    input_data_path = base_path + "/Datasets/german"

    # Load the data files
    user_labels_path = os.path.join(input_data_path, "german.csv")
    user_edges_path = os.path.join(input_data_path, "german_edges.csv")

    # Create dataframes to store the information from the .csv files
    user_labels = pd.read_csv(user_labels_path)
    user_edges = pd.read_csv(user_edges_path)

    user_labels['Gender'] = user_labels['Gender'].replace({'Female': 1, 'Male': 0})
    user_labels['GoodCustomer'] = user_labels['GoodCustomer'].replace({1: 1, -1: 0})
    user_labels.insert(0, 'user_id', user_labels.index)
    user_labels = user_labels.drop('PurposeOfLoan', axis=1)

    user_edges = user_edges[user_edges['uid1'].isin(user_labels['user_id']) & user_edges['uid2'].isin(user_labels['user_id'])]
    user_labels_train = user_labels
    user_labels_train = user_labels_train.drop(columns=['GoodCustomer'])

    # Extract node features from user_labels dataframe
    node_features = user_labels_train.iloc[:, 1:] 
    node_features = torch.tensor(node_features.values, dtype=torch.float)

    # Extract edges from user_edges dataframe
    edges = user_edges[['uid1', 'uid2']]
    edges['uid1'] = edges['uid1'].map(dict(zip(user_labels['user_id'], range(len(user_labels)))))
    edges['uid2'] = edges['uid2'].map(dict(zip(user_labels['user_id'], range(len(user_labels)))))

    # Convert edges dataframe to tensor
    edges_tensor = torch.tensor(edges.values, dtype=torch.long).t().contiguous()

    # Create edge_index tensor
    edge_index = edges_tensor

    # Create torch-geometric data
    data = Data(x=node_features, edge_index=edge_index)

    num_nodes = node_features.size(0)
    num_classes = 2 
    num_node_features = data.num_node_features

    # Create masks for training, and testing
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 80 - 20 Train and Test data split
    num_train = int(num_nodes * 0.6)
    num_val = int(num_nodes * 0.8)
    train_mask[:num_train] = True
    val_mask[num_train:num_val] = True
    test_mask[num_val:] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask

    # Labels from the data 
    labels_values = user_labels[class_labels].values
    data.y = torch.tensor(labels_values, dtype=torch.long)

    # Sensitivite attribute
    sens_values = user_labels[sens_attribute].values
    sens_attribute_tensor = torch.tensor(sens_values, dtype=torch.long)

    return data, num_classes, num_node_features, sens_attribute_tensor, labels_values, sens_values