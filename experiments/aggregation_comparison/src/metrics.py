import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score

NAMES = ['roc_auc','pr_auc','accuracy','recall','specificity','precision','f1','mcc','balanced_accuracy']

def metrics(y,p,t=.5):
    y,p = np.asarray(y),np.asarray(p)
    pred = p >= t
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),accuracy=float(accuracy_score(y,pred)), recall=float(recall_score(y,pred,zero_division=0)), specificity=float(tn/(tn+fp)), precision=float(precision_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),mcc=float(matthews_corrcoef(y,pred)), balanced_accuracy=float(balanced_accuracy_score(y,pred)))

def threshold(y,p):
    # Exhaustive validation breakpoints; deterministic tie-break closest to .5 then smaller.
    candidates = sorted(set([0.,.5,1.] + list(map(float,p))))
    return max(candidates,key=lambda t:(f1_score(y,np.asarray(p)>=t,zero_division=0),-abs(t-.5),-t))
