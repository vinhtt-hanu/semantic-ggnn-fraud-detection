import pandas as pd, numpy as np, json, os
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from scipy import stats
E='/mnt/data/fraud_final_work/evidence'; r=pd.read_csv(E+'/results_per_seed.csv'); t=pd.read_csv(E+'/tabular_baselines_same_inputs.csv')
assert len(r)==30 and r.groupby('model')['seed'].nunique().eq(5).all()
assert len(t)==15
issues=[]; prev_y=None; prev_rows=None
for _,z in r.iterrows():
  a=np.load(f"{E}/test_predictions/{z['model']}_{int(z.seed)}.npz")
  y=a['test_y']; sc=a['test_scores']; ids=a['test_orig_rows']
  assert len(y)==len(sc)==len(ids)==68848
  if prev_y is not None:assert np.array_equal(y,prev_y) and np.array_equal(ids,prev_rows)
  prev_y,prev_rows=y,ids
  a_ap=average_precision_score(y,sc); a_roc=roc_auc_score(y,sc)
  if abs(a_ap-z.pr_auc)>5e-6 or abs(a_roc-z.roc_auc)>5e-6:issues.append(f"{z['model']}-{z.seed}: AP diff {a_ap-z.pr_auc} ROC diff {a_roc-z.roc_auc}")
  predictions=sc>=z.threshold_from_validation
  p,rec,f,_=precision_recall_fscore_support(y,predictions,average='binary',zero_division=0)
  if abs(p-z.precision)>5e-6 or abs(f-z.f1)>5e-6: issues.append(f"{z['model']}-{z.seed}: f1 mismatch")
assert int(prev_y.sum())==846
fr=r.set_index(['model','seed'])
a=fr.loc['semantic-first'].sort_index();b=fr.loc['semantic-last'].sort_index(); delta=(a.pr_auc-b.pr_auc)
ci=stats.t.interval(.95, len(delta)-1, loc=delta.mean(),scale=stats.sem(delta))
print('GPU verification errors:',issues)
print('test size',len(prev_y),'positives',int(prev_y.astype(np.int64).sum()))
print('paired delta',delta.mean(),ci)
print('GPU table')
print(r.groupby('model')[['pr_auc','roc_auc','precision','recall','f1']].agg(['mean','std']).round(6).to_string())
print('TABULAR table')
print(t.groupby('model')[['pr_auc','roc_auc','precision','recall','f1']].agg(['mean','std']).round(6).to_string())
result={'verified_models':len(r),'verified_baselines':len(t),'test_n':len(prev_y),'test_fraud_n':int(prev_y.astype(np.int64).sum()), 'paired_mean_delta_pr_auc':float(delta.mean()),'paired_t_ci_95':list(map(float,ci)), 'pr_auc_recomputation_issues':issues}
open(E+'/independent_verification.json','w').write(json.dumps(result,indent=2))
