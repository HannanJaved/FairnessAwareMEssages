from load_data import load_german_data, load_nba_data
from FAME import FAME_GCN
from A_FAME import A_FAME_GAT
from GAT import GAT
from GCN import GCN
from utils import set_device, fair_metric, train, test, print_metrics
import argparse 
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Bias Mitigation Experiments')
    parser.add_argument('--model', type=str, default='GCN', help='Model to train: FAME, A_FAME, GCN, GAT')
    parser.add_argument('--fairness', type=bool, default=False, help='Whether to use fairness-aware loss')
    parser.add_argument('--alpha', type=float, default=0, help='Alpha parameter for fairness-aware loss')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter for fairness-aware loss')
    parser.add_argument('--gamma', type=float, default=0, help='Gamma parameter for fairness-aware loss')
    parser.add_argument('--delta', type=float, default=0, help='Delta parameter for fairness-aware loss')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train the model')
    parser.add_argument('--dataset', type=str, default='german', help='Dataset to use: german, nba')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: cuda, cpu')
    parser.add_argument('--val', type=bool, default=True, help='Whether to use validation set')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use: Adam, SGD')

    args = parser.parse_args()

    if args.dataset == 'german':
        data, num_classes, num_node_features, sens_attribute_tensor, labels_values, sens_values = load_german_data()
    elif args.dataset == 'nba':
        data, num_classes, num_node_features, sens_attribute_tensor, labels_values, sens_values = load_nba_data()

    if args.model == 'FAME':
        model = FAME_GCN(data, sens_attribute_tensor)
    elif args.model == 'A_FAME':
        model = A_FAME_GAT(data, sens_attribute_tensor)
    elif args.model == 'GCN':
        model = GCN(num_node_features, num_classes)
    elif args.model == 'GAT':
        model = GAT(num_node_features, num_classes)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters())

    train(model, data, optimizer, labels_values=labels_values, sens_values=sens_values ,epochs=args.epochs, fairness=args.fairness, alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta)
    fairness_metrics = test(model, data, labels_values, sens_values, val=args.val)
    print_metrics(fairness_metrics)
