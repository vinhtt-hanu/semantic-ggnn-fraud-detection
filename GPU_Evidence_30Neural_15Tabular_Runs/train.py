"""GPU training + statistics for causal ego-event GGNN revision experiments.

No unseen test labels or future transaction attributes are included in graph inputs.
The final results remain distinct from original P3/P4 because the graph is redefined.
"""
from __future__ import annotations
import os,time,json,random,copy,math,hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,precision_score,recall_score,f1_score
from scipy.stats import t as student_t
import torch
from torch import nn
import torch.nn.functional as F
from experiment import MODES,SEEDS,NODE_DIM,EDGE_DIM


def normalize_train_only(data:dict,bounds:list):
    train_end=bounds[1];node=data['node'];edge=data['edge'];mask=data['mask']
    train_nodes=node[:train_end][mask[:train_end]]
    scaler=StandardScaler().fit(train_nodes)
    normnode=np.clip(scaler.transform(node.reshape(-1,NODE_DIM)).reshape(node.shape),-12,12).astype('float32')
    # Keep absent padded nodes exactly zero. Mask is the only source of node validity.
    normnode[~mask]=0
    escaler=StandardScaler().fit(edge[:train_end,:4])
    e=edge.copy();e[:,:4]=np.clip(escaler.transform(e[:,:4]),-12,12)
    return normnode,e.astype('float32'),{'node_mean':scaler.mean_.tolist(),'node_scale':scaler.scale_.tolist(),
                                    'edge_mean':escaler.mean_.tolist(),'edge_scale':escaler.scale_.tolist()}


def encode_frozen_lm(texts, output_dir,checkpoint='sentence-transformers/all-MiniLM-L6-v2',batch_size=128):
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import model_info
    out=Path(output_dir);out.mkdir(exist_ok=True,parents=True)
    path=out/'frozen_minilm_text_bank.npy'
    meta=out/'frozen_minilm_text_bank.json'
    # Cache invalidated if text list/checkpoint changes.
    digest=hashlib.sha256((checkpoint+'\n'+'\n'.join(texts)).encode('utf-8')).hexdigest()
    if path.exists() and meta.exists():
        old=json.loads(meta.read_text())
        if old.get('digest')==digest:
            print('Use cached frozen LM',path)
            return np.load(path),old
    revision=model_info(checkpoint).sha
    lm=SentenceTransformer(checkpoint,revision=revision,device='cuda' if torch.cuda.is_available() else 'cpu')
    em=lm.encode(texts,batch_size=batch_size,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True).astype('float32')
    np.save(path,em)
    m={'checkpoint':checkpoint,'revision':revision,'digest':digest,'rows':len(texts),'dim':int(em.shape[1]),
       'text_policy':'Frozen MiniLM on bucketed causal historical profiles; no account IDs or labels'}
    meta.write_text(json.dumps(m,indent=2,ensure_ascii=False))
    del lm
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return em,m


class FeatureTransformer(nn.Module):
    def __init__(self,d=32):
        super().__init__()
        self.token_weight=nn.Parameter(torch.empty(NODE_DIM,d));nn.init.xavier_uniform_(self.token_weight)
        self.token_bias=nn.Parameter(torch.zeros(NODE_DIM,d))
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=64,dropout=0.1,
                                         batch_first=True,activation='gelu',norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False)
        self.proj=nn.Linear(d,64)
    def forward(self,x):
        B,S,D=x.shape
        tokens=x.reshape(B*S,D,1)*self.token_weight[None,:,:]+self.token_bias[None,:,:]
        return self.proj(self.encoder(tokens).mean(dim=1)).reshape(B,S,64)

class StarGGNN(nn.Module):
    """Actual trainable, five-step gated graph propagation on historical ego-event stars."""
    def __init__(self,k=3,d=64,steps=5):
        super().__init__();self.k=k;self.steps=steps;self.cell=nn.GRUCell(d,d)
    def forward(self,h,mask):
        B,S,D=h.shape;k=self.k
        active=mask[:,:,None].float();h=h*active
        for _ in range(self.steps):
            messages=torch.zeros_like(h)
            left=h[:,2:2+k,:]*active[:,2:2+k,:]
            right=h[:,2+k:2+2*k,:]*active[:,2+k:2+2*k,:]
            messages[:,0,:]=left.sum(1)/active[:,2:2+k,:].sum(1).clamp(min=1)
            messages[:,1,:]=right.sum(1)/active[:,2+k:2+2*k,:].sum(1).clamp(min=1)
            messages[:,2:2+k,:]=h[:,0:1,:]
            messages[:,2+k:2+2*k,:]=h[:,1:2,:]
            h=self.cell(messages.reshape(-1,D),h.reshape(-1,D)).reshape(B,S,D)*active
        return h

class CausalHybrid(nn.Module):
    def __init__(self,mode:str,lmemb:np.ndarray,k=3,steps=5):
        super().__init__();self.mode=mode
        self.numeric=nn.Sequential(nn.Linear(NODE_DIM,64),nn.GELU(),nn.Linear(64,64))
        self.transformer=FeatureTransformer()
        self.lm_table=nn.Embedding.from_pretrained(torch.as_tensor(lmemb,dtype=torch.float32),freeze=True)
        self.lm_projection=nn.Linear(lmemb.shape[1],64)
        self.semantic_fusion=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.LayerNorm(64))
        self.merger=nn.Linear(128,64)
        self.ggnn=StarGGNN(k=k,steps=steps)
        self.decoder=nn.Sequential(nn.Linear(64*4+EDGE_DIM,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,1))
    def semantic(self,x,textids):
        b,s,_=x.shape
        if self.mode in ('no-lm','numeric-ggnn'):
            le=x.new_zeros((b,s,64))
        else:
            le=self.lm_projection(self.lm_table(textids))
        if self.mode in ('no-transformer','numeric-ggnn'):
            te=x.new_zeros((b,s,64))
        else:
            te=self.transformer(x)
        return self.semantic_fusion(torch.cat([le,te],dim=-1))
    def forward(self,x,mask,textids,edge):
        n=self.numeric(x)
        if self.mode=='numeric-ggnn':
            h=self.ggnn(n,mask)
        elif self.mode=='semantic-last':
            g=self.ggnn(n,mask)
            s=self.semantic(x,textids)
            h=self.merger(torch.cat([g,s],dim=-1))
        else:
            s=self.semantic(x,textids)
            initial=self.merger(torch.cat([n,s],dim=-1))
            h=initial if self.mode=='no-ggnn' else self.ggnn(initial,mask)
        a=h[:,0];b=h[:,1]
        return self.decoder(torch.cat([a,b,torch.abs(a-b),a*b,edge],dim=-1)).squeeze(-1)


def _metrics(y,prob,threshold):
    y=np.asarray(y).astype('int8');prob=np.asarray(prob).astype('float64')
    pred=prob>=threshold
    return {'pr_auc':float(average_precision_score(y,prob)),
      'roc_auc':float(roc_auc_score(y,prob)),
      'precision':float(precision_score(y,pred,zero_division=0)),
      'recall':float(recall_score(y,pred,zero_division=0)),
      'f1':float(f1_score(y,pred,zero_division=0))}

def _val_threshold(y,prob):
    pp,rr,th=precision_recall_curve(y,prob)
    score=2*pp[:-1]*rr[:-1]/np.maximum(pp[:-1]+rr[:-1],1e-10)
    return float(th[np.argmax(score)])


def _predict(model,X,M,T,E,ids,device,batch_size=1024):
    model.eval();out=[]
    with torch.inference_mode():
        for j in range(0,len(ids),batch_size):
            inds=ids[j:j+batch_size]
            x=torch.from_numpy(X[inds]).to(device)
            mask=torch.from_numpy(M[inds]).to(device)
            tid=torch.from_numpy(T[inds]).long().to(device)
            e=torch.from_numpy(E[inds]).to(device)
            z=model(x,mask,tid,e)
            out.append(torch.sigmoid(z).cpu().numpy().astype('float32'))
    return np.concatenate(out)


def _append_result(out_dir,result):
    path=Path(out_dir)/'results_per_seed.csv'
    if path.exists():
        old=pd.read_csv(path)
        old=old[~((old['model']==result['model'])&(old['seed']==result['seed']))]
        res=pd.concat([old,pd.DataFrame([result])],ignore_index=True)
    else:res=pd.DataFrame([result])
    res.sort_values(['model','seed']).to_csv(path,index=False)
    return res


def _summaries(out_dir):
    out=Path(out_dir);path=out/'results_per_seed.csv'
    if not path.exists():return
    r=pd.read_csv(path)
    statcols=[c for c in ['pr_auc','roc_auc','precision','recall','f1','best_val_pr_auc','seconds'] if c in r]
    r.groupby('model')[statcols].agg(['mean','std']).to_csv(out/'results_mean_std.csv')
    a=r[r.model=='semantic-first'].set_index('seed')
    b=r[r.model=='semantic-last'].set_index('seed')
    common=sorted(set(a.index)&set(b.index));rows=[]
    for seed in common:rows.append({'seed':seed,'delta_pr_auc_first_minus_last':float(a.loc[seed,'pr_auc']-b.loc[seed,'pr_auc']),
                                    'delta_f1_first_minus_last':float(a.loc[seed,'f1']-b.loc[seed,'f1'])})
    pd.DataFrame(rows).to_csv(out/'paired_differences.csv',index=False)
    if len(rows)>=2:
        z=np.array([v['delta_pr_auc_first_minus_last'] for v in rows]);n=len(z)
        mean=float(z.mean());std=float(z.std(ddof=1));crit=float(student_t.ppf(.975,n-1))
        p={'matched_seeds':n,'mean_delta_pr_auc':mean,'std_delta_pr_auc':std,
           'paired_t_95pct_ci':[mean-crit*std/math.sqrt(n),mean+crit*std/math.sqrt(n)],
           'note':'CI across seed pairs only; not a full uncertainty treatment for a target deployment'}
        (out/'paired_uncertainty.json').write_text(json.dumps(p,indent=2))


def train_all(data,qs,emb,out_dir, modes=MODES,seeds=SEEDS,epochs=8,batch_size=512,lr=.0015,save_models=True):
    """Fully matched time split/input/tokenizer/cached LM and training budgets for both main architectures.
       Results written after EACH completed seed so Colab disconnects can resume.
    """
    if not torch.cuda.is_available():raise RuntimeError('Colab GPU is disabled. Runtime > Change runtime type > T4 GPU')
    device='cuda';out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    X,E,scaler_meta=normalize_train_only(data,qs['bounds'])
    (out/'train_only_scalers.json').write_text(json.dumps(scaler_meta,indent=2))
    mask=data['mask'];text=data['text_id'].astype('int32');y=data['y'].astype('int8')
    ntrain,nval,ntest=qs['bounds'][1],qs['bounds'][2]-qs['bounds'][1],qs['bounds'][3]-qs['bounds'][2]
    va=np.arange(ntrain,ntrain+nval);te=np.arange(ntrain+nval,ntrain+nval+ntest)
    label_train=y[:ntrain];pos=int(label_train.sum());neg=ntrain-pos
    positive_weight=min(25.0,max(1.0,neg/max(1,pos)))
    print(f'train={ntrain:,} (positive {pos}), validation={nval:,}, test={ntest:,}, pos_weight={positive_weight:.2f}')
    cdir=out/'model_checkpoints';cdir.mkdir(exist_ok=True)
    pdir=out/'test_predictions';pdir.mkdir(exist_ok=True)
    existing=set()
    if (out/'results_per_seed.csv').exists():
        prev=pd.read_csv(out/'results_per_seed.csv');existing=set(zip(prev.model,prev.seed))
    for mode in modes:
      for seed in seeds:
        if (mode,seed) in existing:
            print('Resume: already completed',mode,seed);continue
        tic=time.time();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=CausalHybrid(mode,emb,k=(mask.shape[1]-2)//2,steps=5).to(device)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
        bceloss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight,device=device))
        gen=np.random.default_rng(seed);best_ap=-float('inf');best=None;best_epoch=0
        for ep in range(1,epochs+1):
            model.train();perm=gen.permutation(ntrain);losses=[]
            for j in range(0,ntrain,batch_size):
                ix=perm[j:j+batch_size];opt.zero_grad(set_to_none=True)
                z=model(torch.from_numpy(X[ix]).to(device),torch.from_numpy(mask[ix]).to(device),
                        torch.from_numpy(text[ix]).long().to(device),torch.from_numpy(E[ix]).to(device))
                yy=torch.from_numpy(y[ix].astype('float32')).to(device)
                loss=bceloss(z,yy);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                losses.append(float(loss.item()))
            val_prob=_predict(model,X,mask,text,E,va,device)
            val_ap=float(average_precision_score(y[va],val_prob))
            if val_ap>best_ap:
                best_ap=val_ap;best={key:t.detach().cpu().clone() for key,t in model.state_dict().items()};best_epoch=ep
            print(f'{mode} seed={seed} epoch={ep}/{epochs} train_loss={np.mean(losses):.4f} full_val_ap={val_ap:.5f}',flush=True)
        model.load_state_dict(best)
        val_prob=_predict(model,X,mask,text,E,va,device)
        threshold=_val_threshold(y[va],val_prob)
        test_prob=_predict(model,X,mask,text,E,te,device)
        metric=_metrics(y[te],test_prob,threshold)
        result={'model':mode,'seed':seed,'epochs_budget':epochs,'best_epoch':best_epoch,
                'train_samples':ntrain,'val_samples':nval,'test_samples':ntest,
                'threshold_from_validation':threshold,'best_val_pr_auc':best_ap,
                'seconds':round(time.time()-tic,1),**metric}
        np.savez_compressed(pdir/f'{mode}_{seed}.npz',test_scores=test_prob,val_scores=val_prob,
                            test_y=y[te],val_y=y[va],test_orig_rows=qs['te'],val_orig_rows=qs['va'])
        if save_models:torch.save({'model_state':best,'mode':mode,'seed':seed,'embeddings_checkpoint':'see frozen_minilm_text_bank.json'},cdir/f'{mode}_{seed}.pt')
        _append_result(out,result);_summaries(out)
        print('COMPLETE',result,flush=True)
        del model,opt,best
        torch.cuda.empty_cache()
    final=pd.read_csv(out/'results_per_seed.csv')
    n_expected=len(modes)*len(seeds)
    print('Completed',len(final),'of',n_expected,'GPU model runs')
    return final


def run_tabular_baselines(data,qs,out_dir,seeds=SEEDS,device='cpu'):
    """Same labeled train/val/test, same root/event/edge inputs flattened for tabular comparators."""
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    out=Path(out_dir);X,E,_=normalize_train_only(data,qs['bounds'])
    Z=np.column_stack([X.reshape(len(X),-1),E]).astype('float32');y=data['y']
    ntr,nva,nte=qs['bounds'][1],qs['bounds'][2]-qs['bounds'][1],qs['bounds'][3]-qs['bounds'][2]
    results_path=out/'tabular_baselines_same_inputs.csv'
    old=pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    seen=set(zip(old.model,old.seed)) if len(old) else set();rows=[]
    for name in ['RandomForest','XGBoost','LightGBM']:
      for seed in seeds:
        if (name,seed) in seen:continue
        tic=time.time()
        if name=='RandomForest':model=RandomForestClassifier(n_estimators=250,max_depth=14,min_samples_leaf=2,
                                                             class_weight='balanced_subsample',n_jobs=2,random_state=seed)
        elif name=='XGBoost':model=XGBClassifier(n_estimators=350,max_depth=6,learning_rate=.05,subsample=.85,
                                                colsample_bytree=.85,objective='binary:logistic',eval_metric='aucpr',
                                                tree_method='hist',n_jobs=2,random_state=seed)
        else:model=LGBMClassifier(n_estimators=350,max_depth=-1,num_leaves=31,learning_rate=.05,
                                   class_weight='balanced',n_jobs=2,verbosity=-1,random_state=seed)
        model.fit(Z[:ntr],y[:ntr]);vp=model.predict_proba(Z[ntr:ntr+nva])[:,1]
        threshold=_val_threshold(y[ntr:ntr+nva],vp)
        tp=model.predict_proba(Z[ntr+nva:])[:,1]
        record={'model':name,'seed':seed,'train_samples':ntr,'val_samples':nva,'test_samples':nte,
                'threshold_from_validation':threshold,'seconds':round(time.time()-tic,1),
                **_metrics(y[ntr+nva:],tp,threshold)}
        rows.append(record)
        pd.concat([old,pd.DataFrame(rows)],ignore_index=True).to_csv(results_path,index=False)
        print('BASELINE DONE',record,flush=True)
    return pd.read_csv(results_path)


def create_summary(out_dir,qs,data,source_path,config,language_model_meta):
    out=Path(out_dir);(out/'study_protocol.json').write_text(json.dumps({
      'study':'controlled redefinition of semantic-first/semantic-last, NOT original P3/P4 reproduction',
      'data_file':str(source_path),'data_file_size':Path(source_path).stat().st_size,
      'protocol':'Same PaySim temporal cutoffs. Account event history only from steps STRICTLY earlier than prediction step; same-step excluded. Train negatives sampled; full validation and test. Same frozen LM checkpoint, fixed template encoder, same inputs and training budget. Only fusion placement changes for matched P3/P4-analogues.',
      'time_cutoffs':qs['cuts'],'rows':{'sampled_labeled_train':len(qs['tr']),'validation_full':len(qs['va']),'test_full':len(qs['te'])},
      'fraud_counts':{'train':int(data['y'][:qs['bounds'][1]].sum()),'validation':int(data['y'][qs['bounds'][1]:qs['bounds'][2]].sum()),'test':int(data['y'][qs['bounds'][2]:].sum())},
      'architecture_note':'Historical *transaction-event* star-graph, not exact account-level GGNN of archived original P3/P4',
      'encoder':language_model_meta,'config':config,'graph_audit':data['info']
    },indent=2,ensure_ascii=False))

