import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"]="serif"
fig,ax=plt.subplots(figsize=(7.2,5.0));ax.set_xlim(0,100);ax.set_ylim(0,70);ax.axis("off")
def box(x,y,w,h,t,fc="#ffffff",ec="#333",fs=7.5,bold=False,ls="-"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.2,rounding_size=1.2",fc=fc,ec=ec,lw=1,ls=ls))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal")
def arr(x1,y1,x2,y2,c="#333",ls="-"):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=1,color=c,ls=ls,shrinkA=0,shrinkB=0))
# encoder
ax.add_patch(FancyBboxPatch((3,10),38,52,boxstyle="round,pad=0.3,rounding_size=2",fc="#eef3f9",ec="#1f5f9e",lw=1.2))
ax.text(22,64.5,"ENCODER",ha="center",fontsize=9,fontweight="bold",color="#1f5f9e")
box(8,2,28,6,"Source words\n(two thousand six ...)",fc="#fff")
box(8,12,28,6,"Embedding × √d\n+ Positional encoding",fs=7.2)
ax.add_patch(FancyBboxPatch((7,22),30,34,boxstyle="round,pad=0.2,rounding_size=1.5",fc="#fff",ec="#1f5f9e",lw=1,ls="--"))
ax.text(35.5,54,"× N",fontsize=8,ha="right",color="#1f5f9e")
box(10,38,24,6,"LayerNorm → FFN\n(+ residual)",fs=7)
box(10,25,24,6,"LayerNorm → Multi-Head\nSelf-Attention (+ residual)",fs=6.8)
box(10,46,24,0.01,"",ec="none")
box(8,57,28,3.8,"Final LayerNorm → memory",fs=7)
arr(22,8.4,22,11.6);arr(22,18.4,22,24.6);arr(22,31.4,22,37.6);arr(22,44.4,22,56.6)
# decoder
ax.add_patch(FancyBboxPatch((52,10),45,52,boxstyle="round,pad=0.3,rounding_size=2",fc="#fdf3ea",ec="#d9822b",lw=1.2))
ax.text(74.5,64.5,"DECODER",ha="center",fontsize=9,fontweight="bold",color="#b8651a")
box(58,2,33,6,"Target digits shifted right\n(<bos> 2 6 5 ...)")
box(58,12,33,6,"Embedding × √d\n+ Positional encoding",fs=7.2)
ax.add_patch(FancyBboxPatch((56,21),37,28,boxstyle="round,pad=0.2,rounding_size=1.5",fc="#fff",ec="#d9822b",lw=1,ls="--"))
ax.text(92,47,"× N",fontsize=8,ha="right",color="#b8651a")
box(59,22.5,31,4.5,"LayerNorm → FFN (+ residual)",fs=6.8)
box(59,29.5,31,5.5,"LayerNorm → Cross-Attention\n(+ residual; K,V = memory)",fs=6.5)
box(59,37.5,31,5.5,"LayerNorm → Masked Multi-Head\nSelf-Attention (+ residual)",fs=6.5)
box(58,51,33,4,"Final LayerNorm → Linear (13)",fs=7)
box(58,57.5,33,3.8,"Softmax → next digit",fs=7.5,fc="#fff")
arr(74.5,8.4,74.5,11.6);arr(74.5,18.4,74.5,22.2);arr(74.5,27.2,74.5,29.2);arr(74.5,35.2,74.5,37.2);arr(74.5,43.2,74.5,50.6);arr(74.5,55.2,74.5,57.2)
arr(22,60.2,22,60.2)
ax.annotate("",xy=(58.6,32),xytext=(36,58.5),arrowprops=dict(arrowstyle="-|>",lw=1.2,color="#1f5f9e",connectionstyle="arc3,rad=-0.25"))
ax.text(44,48,"memory\n(K, V)",fontsize=7,color="#1f5f9e",ha="center")
plt.savefig("fig/arch.png",dpi=200,bbox_inches="tight");plt.close()
