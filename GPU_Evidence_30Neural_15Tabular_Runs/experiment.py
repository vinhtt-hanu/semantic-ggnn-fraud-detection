"""Controlled, chronological causal ego-event GGNN experiments for PaySim.

Important: this is a REDEFINED controlled experiment, not a faithful reproduction of the
original P3/P4 notebooks. Historical graph nodes here are prior *transaction events*.
All account history and historical neighbor events are strictly before prediction step.
"""
from __future__ import annotations
import os, json, gc, hashlib, random, time, math, zipfile
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

SEEDS=[11,22,33,44,55]
MODES=['semantic-first','semantic-last','no-lm','no-transformer','numeric-ggnn','no-ggnn']
TX_TYPES=['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']
NODE_DIM=10
EDGE_DIM=4+len(TX_TYPES)


def open_paysim(path: str, rows: Optional[int]=None) -> pd.DataFrame:
    """Read the actual PaySim table, optionally first N rows for a smoke test."""
    p=Path(path)
    cols=['step','type','amount','nameOrig','nameDest','oldbalanceOrg','oldbalanceDest','isFraud']
    dtypes={'step':'int16','amount':'float32','nameOrig':'string','nameDest':'string',
            'oldbalanceOrg':'float32','oldbalanceDest':'float32','isFraud':'int8'}
    if str(p).lower().endswith('.zip'):
        with zipfile.ZipFile(p) as z:
            csvs=[n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX')]
            if not csvs:raise ValueError('ZIP does not contain a PaySim CSV')
            filename=max(csvs,key=lambda s:z.getinfo(s).file_size)
            with z.open(filename) as f:
                df=pd.read_csv(f,usecols=cols,dtype=dtypes,nrows=rows,low_memory=False)
    else:
        df=pd.read_csv(p,usecols=cols,dtype=dtypes,nrows=rows,low_memory=False)
    if df.empty:raise ValueError('Empty dataset')
    if not df['step'].is_monotonic_increasing:
        raise ValueError('CSV is not ordered by PaySim step; sort by step with a stable sort first')
    if not set(df['type'].unique()).issubset(set(TX_TYPES)):raise ValueError('Unexpected transaction type')
    return df


def select_queries(df:pd.DataFrame,history_cut=483,train_cut=595,val_cut=669,n_train_neg=18000,seed=2026):
    step=df.step.to_numpy();label=df.isFraud.to_numpy()
    train_mask=(step>=history_cut)&(step<train_cut)
    pos=np.flatnonzero(train_mask&(label==1))
    neg=np.flatnonzero(train_mask&(label==0))
    if len(pos)<1 or len(neg)<1:raise ValueError('No labeled training examples in time window')
    rng=np.random.default_rng(seed)
    if len(neg)>n_train_neg:neg=rng.choice(neg,n_train_neg,replace=False)
    tr=np.sort(np.concatenate([pos,neg]));va=np.flatnonzero((step>=train_cut)&(step<val_cut))
    te=np.flatnonzero(step>=val_cut)
    if not (len(va) and len(te) and label[va].sum() and label[te].sum()):
        raise ValueError('Incomplete time windows; use full PaySim CSV, not a truncated subset')
    query_ids=np.concatenate([tr,va,te])
    return dict(tr=tr,va=va,te=te,query_ids=query_ids,bounds=[0,len(tr),len(tr)+len(va),len(tr)+len(va)+len(te)],
                cuts=dict(history_cut=history_cut,train_cut=train_cut,val_cut=val_cut),neg_subsample=seed)


def _descriptions(nodes:np.ndarray,valid:np.ndarray):
    """Coarsely bucketized account/event text without account IDs or fraud labels."""
    z=np.rint(np.clip(nodes,0,16)).astype(np.int16)
    out=[]
    for i in range(len(z)):
        if not valid[i]:out.append('padding absent historical transaction');continue
        r=z[i]
        if r[0]==0:
            out.append(f'account; merchant {r[1]}; outgoing count log bucket {r[2]}; outgoing amount log bucket {r[3]}; incoming count log bucket {r[4]}; incoming amount log bucket {r[5]}; outgoing mean bucket {r[6]}; incoming mean bucket {r[7]}.')
        else:
            out.append(f'past transaction event; outgoing direction {r[1]}; amount log bucket {r[2]}; transaction type code {r[3]}; age log bucket {r[4]}; counterparty merchant {r[5]}; old balance log bucket {r[6]}.')
    return out


def build_causal_examples(df:pd.DataFrame,qs:dict,k:int=3, verbose=True):
    """Construct fixed-size genuine historical ego-event stars from past events only.

    Store at most k most recent transaction events for each query endpoint. At step t,
    the example is frozen before processing ANY transaction at t. Then records at t
    may update state for later steps. This prevents same-step/future leakage.
    """
    t0=time.time();n_rows=len(df);allq=qs['query_ids'];S=2+2*k
    q=df.iloc[allq]
    names=pd.Index(pd.unique(np.r_[q.nameOrig.to_numpy(dtype=object),q.nameDest.to_numpy(dtype=object)]))
    n_acc=len(names)
    if verbose:print(f'Query transactions={len(allq):,}, unique query endpoint accounts={n_acc:,}')
    src_id=names.get_indexer(df.nameOrig).astype('int32')
    dst_id=names.get_indexer(df.nameDest).astype('int32')
    qpos=np.full(n_rows,-1,dtype=np.int32);qpos[allq]=np.arange(len(allq),dtype=np.int32)
    steps=df.step.to_numpy(dtype=np.int16);amount=df.amount.to_numpy(dtype=np.float32)
    old_src=df.oldbalanceOrg.to_numpy(dtype=np.float32);old_dst=df.oldbalanceDest.to_numpy(dtype=np.float32)
    tp=pd.Categorical(df.type,categories=TX_TYPES).codes.astype(np.int8)
    mer_src=df.nameOrig.str.startswith('M').fillna(False).to_numpy(dtype=np.bool_)
    mer_dst=df.nameDest.str.startswith('M').fillna(False).to_numpy(dtype=np.bool_)
    root_merchant=np.asarray(names.str.startswith('M'),dtype=np.float32)
    # Account count/sum arrays only for endpoints of the selected predictions.
    out_cnt=np.zeros(n_acc,np.float32);out_sum=np.zeros(n_acc,np.float64)
    in_cnt=np.zeros(n_acc,np.float32);in_sum=np.zeros(n_acc,np.float64)
    # For each account retain k most recent event features in a ring buffer.
    hist=np.zeros((n_acc,k,7),dtype=np.float32)
    ptr=np.zeros(n_acc,dtype=np.int32)
    NX=np.zeros((len(allq),S,NODE_DIM),dtype=np.float32)
    MASK=np.zeros((len(allq),S),dtype=bool)
    kinds=np.empty((len(allq),S),dtype=np.int32)
    # At most one account text per role-quantized description; no IDs/labels in text.
    pool={'padding absent historical transaction':0}; texts=['padding absent historical transaction']
    edge=np.zeros((len(allq),EDGE_DIM),dtype=np.float32)
    qlabel=df.isFraud.iloc[allq].to_numpy(dtype=np.int8)
    step_starts=np.r_[0,np.flatnonzero(np.diff(steps)!=0)+1,n_rows]

    def get_root_feat(a):
        a=np.asarray(a,dtype=np.int32)
        feat=np.zeros((len(a),NODE_DIM),np.float32)
        feat[:,0]=0;feat[:,1]=root_merchant[a]
        feat[:,2]=np.log1p(out_cnt[a]);feat[:,3]=np.log1p(out_sum[a])
        feat[:,4]=np.log1p(in_cnt[a]);feat[:,5]=np.log1p(in_sum[a])
        feat[:,6]=np.log1p(out_sum[a]/np.maximum(out_cnt[a],1))
        feat[:,7]=np.log1p(in_sum[a]/np.maximum(in_cnt[a],1))
        return feat

    for j in range(len(step_starts)-1):
        lo,hi=int(step_starts[j]),int(step_starts[j+1]);now=int(steps[lo]);raw_idx=np.arange(lo,hi)
        selected=raw_idx[qpos[lo:hi]>=0]
        if len(selected):
            qq=qpos[selected];a=src_id[selected];b=dst_id[selected]
            assert (a>=0).all() and (b>=0).all()
            NX[qq,0]=get_root_feat(a);NX[qq,1]=get_root_feat(b);MASK[qq,:2]=True
            for ix,acc in [(0,a),(1,b)]:
                ages=ptr[acc]
                for kk in range(k):
                    has=ages>kk
                    if not has.any():continue
                    hrow=hist[acc[has],(ages[has]-1-kk)%k].copy()
                    age=np.log1p(np.maximum(now-hrow[:,0],1))
                    node=np.zeros((len(hrow),NODE_DIM),np.float32)
                    node[:,0]=1;node[:,1]=hrow[:,1];node[:,2]=hrow[:,2]
                    node[:,3]=hrow[:,3];node[:,4]=age;node[:,5]=hrow[:,4];node[:,6]=hrow[:,5]
                    ixnode=2+kk if ix==0 else 2+k+kk
                    NX[qq[has],ixnode]=node;MASK[qq[has],ixnode]=True
                    assert (hrow[:,0]<now).all(), 'Temporal leakage: historical event is from current or future step'
            # Current transaction features are limited to assumed pre-decision fields.
            et=np.log1p(np.maximum(amount[selected],0));es=np.log1p(np.maximum(old_src[selected],0))
            ed=np.log1p(np.maximum(old_dst[selected],0));phase=steps[selected]%24
            edge[qq,0]=et;edge[qq,1]=np.sin(phase*(2*np.pi/24));edge[qq,2]=es;edge[qq,3]=ed
            edge[qq,4:]=np.eye(len(TX_TYPES),dtype=np.float32)[tp[selected]]
        # Only *after* all queries at the same step have been captured, update history.
        # Update all previous history rows involving target accounts (labels unused).
        sr=src_id[lo:hi];dr=dst_id[lo:hi]
        valid_sr=sr>=0;valid_dr=dr>=0
        if valid_sr.any():
            ii=sr[valid_sr];np.add.at(out_cnt,ii,1);np.add.at(out_sum,ii,amount[lo:hi][valid_sr])
        if valid_dr.any():
            ii=dr[valid_dr];np.add.at(in_cnt,ii,1);np.add.at(in_sum,ii,amount[lo:hi][valid_dr])
        events=raw_idx[valid_sr|valid_dr]
        # Event snapshots are historical and refer only to pre-transaction fields.
        for row in events:
            aa=src_id[row];bb=dst_id[row];step_now=steps[row];al=np.log1p(max(0,float(amount[row])))
            if aa>=0:
                spot=ptr[aa]%k;hist[aa,spot]=[step_now,1,al,float(tp[row]),float(mer_dst[row]),np.log1p(max(0,float(old_src[row]))),0];ptr[aa]+=1
            if bb>=0:
                spot=ptr[bb]%k;hist[bb,spot]=[step_now,0,al,float(tp[row]),float(mer_src[row]),np.log1p(max(0,float(old_dst[row]))),0];ptr[bb]+=1
        if verbose and (j+1)%150==0: print(f'  steps {j+1}/{len(step_starts)-1}, elapsed {time.time()-t0:.1f}s')

    # Generate same frozen language-model input bank shared across all architectures.
    # Descriptions quantized to a fixed finite codebook; no new profiles are trained.
    vals=NX.reshape(-1,NODE_DIM);mask_flat=MASK.ravel();descr=_descriptions(vals,mask_flat)
    for i,txt in enumerate(descr):
        code=pool.get(txt)
        if code is None:code=len(texts);pool[txt]=code;texts.append(txt)
        kinds.flat[i]=code
    if verbose:print('Unique text profiles:',len(texts),'active historical event nodes:',int(MASK[:,2:].sum()), 'build_sec',round(time.time()-t0,1))
    info={'source_transactions':n_rows,'query_transactions':len(allq),'historical_event_nodes':int(MASK[:,2:].sum()),
          'fraction_with_history':float(np.mean(MASK[:,2:].any(axis=1))),
          'unique_query_endpoint_accounts':n_acc,'unique_text_profiles':len(texts),'cutoffs':qs['cuts'],
          'nodes_per_ego_graph':S,'neighbors_per_endpoint':k,
          'graph_definition':'Historical transaction-event star; NOT original full-account GGNN',
          'causality':'strictly earlier step; current step excluded; no label in feature/graph'}
    return {'node':NX,'mask':MASK,'text_id':kinds,'edge':edge,'y':qlabel,'texts':texts,'info':info}


def audit_causality_small():
    """Altering a future transaction must not change any earlier query features."""
    demo=pd.DataFrame([
        (1,'TRANSFER',10.,'A','B',100.,0.,0),
        (2,'TRANSFER',20.,'A','C',90.,0.,1),
        (3,'TRANSFER',30.,'A','D',80.,0.,0),
        (4,'TRANSFER',40.,'A','B',70.,0.,1),
        (5,'TRANSFER',50.,'A','B',60.,0.,0),
        (6,'TRANSFER',60.,'A','B',50.,0.,1)
    ],columns=['step','type','amount','nameOrig','nameDest','oldbalanceOrg','oldbalanceDest','isFraud'])
    for c in ['nameOrig','nameDest']: demo[c]=demo[c].astype('string')
    q=dict(query_ids=np.array([2,3,4,5]),cuts={'history_cut':2,'train_cut':4,'val_cut':5})
    a=build_causal_examples(demo,q,k=2,verbose=False)
    changed=demo.copy();changed.loc[5,'amount']=999999.
    b=build_causal_examples(changed,q,k=2,verbose=False)
    assert np.array_equal(a['node'][:3],b['node'][:3]);assert np.array_equal(a['mask'][:3],b['mask'][:3])
    assert np.array_equal(a['edge'][:3],b['edge'][:3]);assert np.array_equal(a['text_id'][:3],b['text_id'][:3])
    assert a['mask'][0,2] and a['mask'][1,2], 'Historical graph should not be empty'
    assert a['mask'][0,3], 'Step-3 prediction should see both steps 1 and 2 for account A'
    return True

