from decimal import DivisionByZero
import numpy as np


def get_performance_metrics(cnf: list):
    f1s = []
    TPs = []
    FNs = []
    FPs = []
    Ps = []
    Rs = []
    class_counts = [sum(x) for x in cnf]
    for i, _ in enumerate(class_counts):
        FP = class_counts[i] - cnf[i][i]
        FPs.append(FP)
        TP = cnf[i][i]
        TPs.append(TP)
        P = TP / (TP + FP)
        Ps.append(P)
        FN = 0
        for j in range(len(class_counts)):
            if i != j:
                FN += cnf[j][i]
        FNs.append(FN)
        try:
            R = TP / (TP + FN)
        except ZeroDivisionError:
            R = 0
        Rs.append(R)
        try:
            f1s.append(2 * (P * R) / (P + R))
        except ZeroDivisionError:
            f1s.append(0)

    P = sum(TPs) / (sum(TPs) + sum(FPs))
    R = sum(TPs) / (sum(TPs) + sum(FNs))
    f1_micro = 2 * (P * R) / (P + R)
    weighted_f1 = (np.sum(np.multiply(f1s, class_counts))) / np.sum(class_counts)
    overall_accuracy = sum(TPs) / sum(class_counts)
    return {
        "macro": {
            "P": np.mean(Ps),
            "R": np.mean(Rs),
            "f1": np.mean(f1s),
        },
        "micro": {"P": P, "R": R, "f1": f1_micro},
        "weighted": {"f1": weighted_f1},
        "accuracy": overall_accuracy,
    }
