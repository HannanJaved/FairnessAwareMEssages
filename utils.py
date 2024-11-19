import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_df_info(df):
    print(df.info())
    print('####### Repeat ####### \n', df.duplicated().any())
    print('####### Count ####### \n', df.nunique())
    print('####### Example ####### \n',df.head())

def label_statics(label_df, label_list):
    print("####### nCount #######")
    for label in label_list:
        print(label_df[label].value_counts())
    print("####### nPercent #######")
    for label in label_list:
        print(label_df[label].value_counts()/label_df.shape[0])

def fair_metric(labels, pred, sens):
    ''''
    Compute the fairness metrics: parity and equality.
    
    Args:
    labels: The ground truth labels.
    pred: The predicted labels.
    sens: The sensitive attribute.
    '''
	# labels = user_labels[labels].values
	# sens = user_labels[sens].values
	
    idx_s0 = sens==0
    idx_s1 = sens==1

    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)

    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))

    return {"parity": parity.item(), "equality": equality.item()}

def fairness_aware_loss(output, data, sensitive_attr, alpha=0, beta=0, gamma=0, delta=0):
    target = data.y[data.train_mask]
    
    # standard_loss = F.cross_entropy(output, target)
    standard_loss = F.nll_loss(output, target)

    labels = data.y[train_mask]
    pos_prob = torch.exp(output[:, 1])
    neg_prob = torch.exp(output[:, 0])

    predictions = output.argmax(dim=1)

    # Statistical Parity Regularization
    sp_reg = torch.abs(pos_prob[sensitive_attr == 1].mean() - pos_prob[sensitive_attr == 0].mean())

    # Treatment Equality Regularization
    fp_diff = (neg_prob * (labels == 0) * (sensitive_attr == 1)).float().mean() - \
              (neg_prob * (labels == 0) * (sensitive_attr == 0)).float().mean()
    fn_diff = (pos_prob * (labels == 1) * (sensitive_attr == 1)).float().mean() - \
              (pos_prob * (labels == 1) * (sensitive_attr == 0)).float().mean()
    treatment_reg = torch.abs(fp_diff) + torch.abs(fn_diff)

    # Equal Opportunity Difference Regularization
    eod_reg = torch.abs((pos_prob * (labels == 1) * (sensitive_attr == 1)).float().mean() - \
                        (pos_prob * (labels == 1) * (sensitive_attr == 0)).float().mean())

    # Overall Accuracy Equality Difference Regularization
    oaed_reg = torch.abs((pos_prob * (sensitive_attr == 1)).float().mean() - \
                         (pos_prob * (sensitive_attr == 0)).float().mean())

    penalty = alpha + beta + gamma + delta
    
    # Combine losses
    combined_loss = (1-penalty)*standard_loss
    + alpha * sp_reg
    + beta * treatment_reg
    + gamma * eod_reg
    + delta * oaed_reg
    
    return combined_loss

def train(
        model, data, optimizer,  labels_values, sens_values, epochs=1000, fairness=False, alpha=0, beta=0, gamma=0, delta=0
        ):
    model.to(set_device())
    data.to(set_device())
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        if fairness:
            loss = fairness_aware_loss(out[data.train_mask], data, data.x[data.train_mask, -1],
                                       alpha=alpha, beta=beta, gamma=gamma, delta=delta)
            
        else:
            # criterion = torch.nn.CrossEntropyLoss()
            # criterion = torch.nn.BCELoss()
            criterion = torch.nn.NLLLoss()
            loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        metrics = test(model, data, labels_values, sens_values)

        if epoch % 10 == 0:
            print(f'Epoch {epoch} | Loss: {loss.item()} | \n AUC_ROC: {metrics["AUC_ROC"]} | F1 Score: {metrics["F1_Score"]} | SPD: {metrics["parity"]} | EOD: {metrics["equality"]}')

def test(model, data, labels_values, sens_values, val=True):
    model.to(set_device())
    data.to(set_device())
    
    if val==True:
      mask = data.val_mask
    else:
      mask = data.test_mask

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        predictions = out.argmax(dim=1)

    # Compute accuracy
    correct = int(predictions[mask].eq(data.y[mask]).sum().item())
    accuracy = correct / int(mask.sum())
    
    # Extract the predictions and the true labels
    y_true = data.y[mask].cpu().numpy()
    y_pred = predictions[mask].cpu().numpy()
    
    # Compute F1 score
    f1 = f1_score(y_true, y_pred, average='binary')

    # Compute AUC-ROC score
    y_probs = out[mask][:, 1].cpu().numpy() 
    auc_roc = roc_auc_score(y_true, y_probs)
    
    fairness_metrics = fair_metric(labels_values, predictions, sens_values)
    fairness_metrics['Accuracy'] = accuracy
    fairness_metrics['F1_Score'] = f1
    fairness_metrics['AUC_ROC'] = auc_roc

    return fairness_metrics

def print_metrics(metrics):
    print("\n####### Metrics #######")
    for key, value in metrics.items():
        print(f"{key} : {value:.5f}")