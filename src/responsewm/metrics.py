"""Patient-aware classification and probabilistic diagnostics, without efficacy claims."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score


def classification_metrics(y,p):
    y,p = np.asarray(y),np.asarray(p,dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("Metrics need matched nonempty vectors")
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Invalid labels/probabilities")
    clipped = np.clip(p,1e-12,1-1e-12)
    nll = float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean())
    ece,bins = 0.,[]
    edges = np.linspace(0,1,11)
    for index in range(10):
        mask = (p>=edges[index]) & ((p<=edges[index+1]) if index == 9 else (p<edges[index+1]))
        if mask.any():
            acc,conf = float(y[mask].mean()),float(p[mask].mean())
            ece += mask.mean()*abs(acc-conf)
            bins.append({"count":int(mask.sum()),"mean_probability":conf,"event_rate":acc})
    both = len(np.unique(y)) == 2
    return {"n":len(y),"positives":int(y.sum()),"auroc":float(roc_auc_score(y,p)) if both else None,
            "auprc":float(average_precision_score(y,p)) if y.sum() else None,
            "brier":float(((p-y)**2).mean()),"nll":nll,"ece_10_equal_width":float(ece),"reliability_bins":bins}


def patient_bootstrap(y,p,patient_ids,repetitions=1000,seed=0):
    """Cluster bootstrap. Multiple landmarks from a patient stay together."""
    y,p,ids = np.asarray(y),np.asarray(p),np.asarray(patient_ids)
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    groups = {i:np.where(ids == i)[0] for i in unique}
    values = {k:[] for k in ("auroc","auprc","brier","nll")}
    for _ in range(repetitions):
        chosen = rng.choice(unique,len(unique),replace=True)
        idx = np.concatenate([groups[i] for i in chosen])
        result = classification_metrics(y[idx],p[idx])
        for k in values:
            if result[k] is not None:
                values[k].append(result[k])
    return {k:{"lower":float(np.quantile(v,.025)),"upper":float(np.quantile(v,.975)),"valid_replicates":len(v)}
            if v else {"lower":None,"upper":None,"valid_replicates":0} for k,v in values.items()}


def paired_bootstrap_delta(y,pa,pb,patient_ids,repetitions=1000,seed=0):
    """Paired patient resampling; same indices for both models."""
    y,pa,pb,ids = map(np.asarray,(y,pa,pb,patient_ids))
    unique = np.unique(ids); rng = np.random.default_rng(seed)
    groups = {i:np.where(ids == i)[0] for i in unique}
    values = {k:[] for k in ("auroc","brier","nll")}
    for _ in range(repetitions):
        idx = np.concatenate([groups[i] for i in rng.choice(unique,len(unique),replace=True)])
        a,b = classification_metrics(y[idx],pa[idx]),classification_metrics(y[idx],pb[idx])
        for k in values:
            if a[k] is not None and b[k] is not None:
                values[k].append(a[k]-b[k])
    return {k:{"mean_delta_A_minus_B":float(np.mean(v)),"lower":float(np.quantile(v,.025)),
               "upper":float(np.quantile(v,.975)),"valid_replicates":len(v)} if v else None for k,v in values.items()}
