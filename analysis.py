import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import *
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
CFG={"main":("Base (h=4, 2+2 layers)","#1f5f9e"),"head1":("1 attention head","#d9822b"),
     "layer1":("1+1 layers","#5b8c5a"),"nope":("No positional encoding","#8a8a8a")}
R={k:[json.load(open(f)) for f in sorted(glob.glob(f"results_{k}_s*.json"))] for k in CFG}
summ={}
for k,rs in R.items():
    ex=np.array([r["test"]["exact"] for r in rs])*100; tk=np.array([r["test"]["token"] for r in rs])*100
    lo=np.array([r["test"]["loss"] for r in rs])
    summ[k]=dict(params=rs[0]["params"],exact=[ex.mean(),ex.std()],token=[tk.mean(),tk.std()],loss=[lo.mean(),lo.std()],
                 sec=float(np.mean([r["seconds"] for r in rs])),
                 val_exact=np.mean([r["hist"]["val_exact"] for r in rs],0).tolist(),
                 val_loss=np.mean([r["hist"]["val_loss"] for r in rs],0).tolist(),
                 steps=rs[0]["hist"]["step"], per_seed=ex.tolist())
json.dump(summ,open("summary.json","w"),indent=1)
for k,v in summ.items(): print(k,v["params"],"exact %.2f±%.2f tok %.2f loss %.4f"%(*v["exact"],v["token"][0],v["loss"][0]),v["per_seed"])
# curves
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
for k,(lab,c) in CFG.items():
    s=summ[k]
    ax[0].plot(s["steps"],np.array(s["val_exact"])*100,color=c,lw=1.8,label=lab)
    ax[1].plot(s["steps"],s["val_loss"],color=c,lw=1.8,label=lab)
ax[0].set_ylabel("Validation exact match (%)");ax[1].set_ylabel("Validation cross-entropy");ax[1].set_yscale("log")
for a in ax: a.set_xlabel("Training step");a.grid(alpha=.25,lw=.5)
ax[0].legend(frameon=False,fontsize=7.5,loc="center right")
plt.tight_layout();plt.savefig("fig/curves.png",dpi=200);plt.close()
# attention + samples from base seed0
rs=R["main"][0];ps=list(np.load("model_main_s0.npz").values())
m=Transformer(len(SRC_VOCAB),len(TGT_VOCAB),seed=0)
for p,d in zip(m.parameters(),ps): p.data[...]=d
train,val,test=make_splits(0)
met,preds=evaluate(m,test);print("reload test",met)
bad=[(int(n),"".join(TGT_VOCAB[i] for i in p)) for n,p in zip(test,preds) if "".join(TGT_VOCAB[i] for i in p)!=str(int(n))]
print("errors",len(bad),bad)
json.dump(bad,open("errors.json","w"))
samples=[int(n) for n in test[:8]]
print("samples",[(" ".join(to_words(n)),str(n)) for n in samples])
n=int([x for x in test if x>=1000 and x%100>20 and x%10][0]);print("attn example",n)
src,tgt=pad([encode_pair(n)[0]]),pad([encode_pair(n)[1]])
CTX["train"]=False;m.forward(src,tgt[:,:-1])
A=m.dec_layers[-1].cross_att.attn[0]   # (H,Tq,Tk)
words=to_words(n);chars=[TGT_VOCAB[i] for i in tgt[0,:-1]]
fig,ax=plt.subplots(1,4,figsize=(7.4,2.6),sharey=True)
for h in range(4):
    ax[h].imshow(A[h],cmap="Blues",vmin=0,vmax=1,aspect="auto")
    ax[h].set_xticks(range(len(words)));ax[h].set_xticklabels(words,rotation=60,ha="right",fontsize=7)
    ax[h].set_yticks(range(len(chars)));ax[h].set_yticklabels(chars,fontsize=7);ax[h].set_title(f"Head {h+1}",fontsize=8)
    for sp in ax[h].spines.values(): sp.set_visible(False)
ax[0].set_ylabel("Decoder input token")
plt.tight_layout();plt.savefig("fig/attn.png",dpi=200);plt.close()
