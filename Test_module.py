## target: true label, output: predictions
from tqdm import tqdm
from sklearn.metrics import classification_report,confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import seaborn as sns
from sklearn.metrics import roc_curve
def evaluate(test_moment_dataloader, tokenizer, Discriminator, Roberta_model):
    fin_targets = []
    fin_outputs = []
    fin_preds = []
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    Discriminator.eval()
    Roberta_model.eval()
    with torch.no_grad():
        for data in tqdm(test_moment_dataloader):
            source = data['comment']
            target = data['hate']
            mask = (source != PAD_INDEX).type(torch.uint8)
            y_pred_embedding = Roberta_model(input_ids=source,
                            attention_mask=mask)
#             target = target.long()
            y_pred = Discriminator(y_pred_embedding)
            fin_targets.extend(target.tolist())
            fin_outputs.extend(torch.argmax(y_pred,1).tolist())
            fin_preds.extend(y_pred.tolist())

    fin_targets1 = np.array(fin_targets)
    fin_outputs1 = np.array(fin_outputs)
    print('Classification Report:')
    y_true1 = fin_targets1
    y_pred1 = fin_outputs1
    target_names = [0, 1]
    print(classification_report(y_true1, y_pred1, digits=4))
    y_prob_final = []
    for i in range(len(fin_preds)):
        tempA = abs(fin_preds[i][0])
        tempB = abs(fin_preds[i][1])
        y_prob_final.append(tempA / (tempA + tempB))

    cm = confusion_matrix(y_true1, y_pred1, labels=[1, 0])
    plt.figure(1, figsize=(20, 8))

    ax = plt.subplot(121)
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['Hate', 'Not Hate'])
    ax.yaxis.set_ticklabels(['Hate', 'Not Hate'])
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_true1, y_prob_final)
    plt.subplot(122)
    lw = 2
    plt.plot(fpr_rt_lm, tpr_rt_lm, color='darkorange',
             lw=lw, label='roc curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    # auc score
    print('auc score:', roc_auc_score(y_true1, y_pred1))













