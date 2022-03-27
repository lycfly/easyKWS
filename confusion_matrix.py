import numpy as np 
from sklearn.metrics import confusion_matrix as cm

def confusion_matrix_generate(label, pred, category):
    list_l = []
    list_p = []
    num = len(label)

    confusion_matrix_lp = cm(list(label), list(pred), labels=list(category))

    return confusion_matrix_lp

def FAFR_rating(confusion_matrix_lp):
    total_count = np.sum(confusion_matrix_lp)
    triu = np.triu(confusion_matrix_lp)
    tril = np.tril(confusion_matrix_lp)
    diag = np.diag(confusion_matrix_lp) * np.eye(confusion_matrix_lp.shape[0])
    FN_count = np.sum(triu-diag)
    FP_count = np.sum(tril-diag)
    FAR = FP_count / total_count
    FRR = FN_count / total_count

    return FAR,FRR
    

