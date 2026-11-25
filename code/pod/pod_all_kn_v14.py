"""POD of exact Kn folders, strict common support, nominal-state normalization.
Projection reconstruction is distinguished from parameter-only prediction.
"""
from pathlib import Path
import json,hashlib,importlib.util,math,datetime,tarfile,shutil
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,matplotlib.tri as mtri
W=Path('campaign');C=W/'pod_all_kn_v12';R=.1524
O=C/('result_v14_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));O.mkdir()
spec=importlib.util.spec_from_file_location('reader',W/'source/cylinder_step3_shock_layer_metrics.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
angles=np.linspace(100,180,65);xi=np.linspace(-1,3,301);variables=['rho','ttr','trot','u','p','mach_proxy'];provenance=[];markers=[]
def sha(p):
    with Path(p).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def smooth(q,ds,sigma):
    n=max(2,int(np.ceil(4*sigma/ds)));z=np.arange(-n,n+1)*ds;k=np.exp(-.5*(z/sigma)**2);k/=k.sum()
    return np.convolve(np.pad(q,n,mode='reflect'),k,'valid')
def outer(s,rho,sigma):
    valid=np.flatnonzero(np.isfinite(rho));
    if len(valid)<100:return None
    # Never bridge a masked solid or unsupported gap.
    cuts=np.flatnonzero(np.diff(valid)>1);parts=np.split(valid,cuts+1);ii=parts[-1]
    if len(ii)<100:return None
    ss=s[ii];ds=ss[1]-ss[0];q=smooth(rho[ii],ds,sigma);g=-np.gradient(q,ds)
    up=q[-max(20,len(q)//10):];spread=float(np.std(up)/max(abs(np.median(up)),1e-20))
    if abs(np.median(up)-1)>.10 or spread>.10:return None
    above=np.flatnonzero(q>1.2)
    if not len(above):return None
    onset=ss[above[-1]];win=max(3*sigma,.05*(ss[-1]-ss[0]));n=max(3,int(win/ds))
    candidates=[]
    for j in np.flatnonzero((g[1:-1]>g[:-2])&(g[1:-1]>=g[2:]))+1:
        if not (ss[j]>max(ss[0]+3*sigma,3*sigma) and ss[j]<onset and q[j]>1.2 and g[j]>0):continue
        left=g[max(0,j-n):j];right=g[j+1:min(len(g),j+n+1)]
        if not len(left) or not len(right):continue
        prominence=g[j]-max(left.min(),right.min())
        if prominence<.05*g[j] or g[j]*win<.04:continue
        candidates.append(j)
    if not candidates:return None
    j=max(candidates,key=lambda j:ss[j]);span=max(12*sigma,2*case_lambda)
    left=np.flatnonzero((ss>max(ss[0]+sigma,ss[j]-span))&(ss<ss[j]-2*sigma))
    if len(left)<5:return None
    valley=left[np.argmin(g[left])];dw=np.abs(ss-ss[valley])<max(sigma,.01*R)
    down=float(np.median(q[dw]));jump=down-1
    if jump<.5:return None
    delta=jump/g[j]
    if not np.isfinite(delta) or delta<=0 or delta>2*(ss[-1]-ss[0]):return None
    return dict(s=float(ss[j]),delta=float(delta),downstream_s=float(ss[valley]),upstream_spread=spread,onset=float(onset),sigma=float(sigma))
def masked_triangulation(data):
    x=data['x'];y=data['y'];tri=mtri.Triangulation(x,y);t=tri.triangles;mask=np.zeros(len(t),dtype=bool)
    for a,b in [(0,1),(1,2),(2,0)]:
        ax=x[t[:,a]]-R;ay=y[t[:,a]];vx=x[t[:,b]]-x[t[:,a]];vy=y[t[:,b]]-y[t[:,a]]
        u=np.clip(-(ax*vx+ay*vy)/np.maximum(vx*vx+vy*vy,1e-30),0,1)
        mask|=(ax+u*vx)**2+(ay+u*vy)**2 < (R*(1-1e-8))**2
    tri.set_mask(mask);return tri
class CellEvaluator:
    """Exact dyadic leaf-bound lookup, no interpolation across cells or gaps."""
    def __init__(self,d,values):
        self.groups=[]
        widths=d['xhi']-d['xlo'];base=1.45/776
        levels=np.rint(np.log2(base/widths)).astype(int)
        for level in sorted(set(levels),reverse=True):
            ids=np.flatnonzero(levels==level);dx=base/2**level;dy=(.65/400)/2**level
            ix=np.rint((d['xlo'][ids]+.6)/dx).astype(np.int64);iy=np.rint(d['ylo'][ids]/dy).astype(np.int64)
            nx=int(round(1.45/dx))+1;keys=iy*nx+ix;order=np.argsort(keys);ids=ids[order];keys=keys[order]
            assert len(np.unique(keys))==len(keys)
            assert np.allclose(widths[ids],dx,rtol=0,atol=2e-6)
            # Six-digit exported bounds: recover known dyadic grid only within
            # its decimal quantization error; never bridge an unknown gap.
            for name,exact in [('xlo',-.6+ix[order]*dx),('xhi',-.6+(ix[order]+1)*dx),('ylo',iy[order]*dy),('yhi',(iy[order]+1)*dy)]:
                assert np.max(abs(d[name][ids]-exact))<1.1e-6
                d[name][ids]=exact
            self.groups.append((dx,dy,nx,keys,ids,d,values))
    def __call__(self,x,y):
        x,y=np.broadcast_arrays(x,y);out=np.full(x.shape,np.nan)
        for dx,dy,nx,keys,ids,d,values in self.groups:
            key=np.floor(y/dy).astype(np.int64)*nx+np.floor((x+.6)/dx).astype(np.int64)
            pos=np.searchsorted(keys,key);safe=np.minimum(pos,len(keys)-1);idx=ids[safe]
            ok=(pos<len(keys))&(keys[safe]==key)&np.isnan(out)&(x>=d['xlo'][idx])&(x<d['xhi'][idx])&(y>=d['ylo'][idx])&(y<d['yhi'][idx])
            out[ok]=values[idx[ok]]
        return np.ma.masked_invalid(out)

def interp(ip,x,y):
    q=np.asarray(ip(x,y).filled(np.nan));q[(x-R)**2+y*y<=R*R]=np.nan;return q
def resample(s,q,t):
    out=np.full(len(t),np.nan);good=np.flatnonzero(np.isfinite(q))
    for part in np.split(good,np.flatnonzero(np.diff(good)>1)+1):
        if len(part)<2:continue
        mask=(t>=s[part[0]])&(t<=s[part[-1]]);out[mask]=np.interp(t[mask],s[part],q[part])
    return out
def pod(A,w,kn,label):
    support=np.all(np.isfinite(A),axis=0)&np.isfinite(w)&(w>0)
    if support.sum()<100:return dict(status='insufficient_common_support',common_points=int(support.sum()))
    X=A[:,support];weight=w[support];weight/=weight.mean();sw=np.sqrt(weight);mu=X.mean(axis=0);Z=(X-mu)*sw
    U,S,V=np.linalg.svd(Z,full_matrices=False);total=np.sum(S*S)
    if total<1e-24:return dict(status='constant',common_points=int(support.sum()))
    energy=S*S/total;rank=min(len(X)-1,len(S));recon=[];loo=[]
    assert np.allclose(V@V.T,np.eye(len(V)),atol=1e-9)
    assert np.linalg.norm(Z-(U*S)@V)/np.linalg.norm(Z)<1e-10
    for r in range(rank+1):
        pred=mu+((Z@V[:r].T)@V[:r])/sw
        recon.append(dict(r=r,relative_L2_percent=float(100*np.linalg.norm((X-pred)*sw)/np.linalg.norm(X*sw))))
    assert recon[-1]['relative_L2_percent']<1e-7
    for test in range(len(X)):
        train=np.arange(len(X))!=test;mean=X[train].mean(axis=0);zt=(X[train]-mean)*sw;u,sv,v=np.linalg.svd(zt,full_matrices=False)
        k=np.log(np.asarray(kn)[train]);order=np.argsort(k);kt=np.log(kn[test]);interior=bool(k.min()<kt<k.max())
        for r in range(1,min(5,len(X)-2)+1):
            coeff=(X[test]-mean)*sw@v[:r].T;projection=mean+(coeff@v[:r])/sw
            err=float(100*np.linalg.norm((projection-X[test])*sw)/np.linalg.norm(X[test]*sw))
            record=dict(heldout_Kn=float(kn[test]),r=r,conditional_projection_L2_percent=err,interior=interior)
            if interior:
                ct=zt@v[:r].T;cp=np.array([np.interp(kt,k[order],ct[order,j]) for j in range(r)]);prediction=mean+(cp@v[:r])/sw
                record['logKn_coefficient_interpolation_L2_percent']=float(100*np.linalg.norm((prediction-X[test])*sw)/np.linalg.norm(X[test]*sw))
            loo.append(record)
    np.savez_compressed(O/(label+'.npz'),mean=mu,modes=V/sw,S=S,coefficients=U*S,support=support,weights=weight,kn=kn)
    return dict(status='complete',snapshots=len(X),kn=list(map(float,kn)),common_points=int(support.sum()),E1=float(energy[0]),C2=float(energy[:2].sum()),N95=int(np.searchsorted(np.cumsum(energy),.95)+1),spectrum=energy.tolist(),reconstruction=recon,leave_one_out=loo,loo_mean_training_only=True,conditional_projection_is_not_prediction=True,attached_interpolation_uses_oracle_registration=('attached' in label))
# Meaningful regression: a large wall peak must not replace an outer compression.
case_lambda=.003048;syn_s=np.linspace(0,.30,1601);sig=.0015
syn=1+3/(1+np.exp(np.clip((syn_s-.095)/.004,-700,700)))+18*np.exp(-syn_s/.002)
test=outer(syn_s,syn,sig);assert test and abs(test['s']-.095)<.005,test
toy=np.array([[1.,2,4,6],[2,3,5,7],[3,4,6,8]]);assert np.linalg.matrix_rank(toy-toy.mean(axis=0))==1
rows=pd.read_csv(C/'SELECTED_EXACT_KN.csv').to_dict('records');loaded=[]
for row in rows:
    data=reader.read_case(row);loaded.append((row,data));provenance.append(dict(file=row['path'],sha256=sha(row['path']),source_kind='original_DS2V_exact_Kn',input_sha256=row['input_sha256']))
# Retain a separate historical-only result and a clearly-labelled all-Kn hybrid.
rec=W/'kn075_wide_v12/N2_Kn0075_wide';cfg=json.loads((rec/'config.json').read_text());data=None
assert (rec/'production/TARGET_COMPLETE.json').exists()
for step in range(cfg['warmup']+cfg['block'],cfg['target']+1,cfg['block']):
    f=rec/'production'/f'fields.{step}.grid'
    with f.open() as h:header=[next(h) for _ in range(9)]
    keys=header[-1].split()[2:];want=['xc','yc','c_geom','f_avg[3]','f_avg[4]','f_avg[5]','f_avg[6]','f_avg[8]','f_avg[7]','f_avg[9]','c_mfp[1]','xlo','xhi','ylo','yhi'];a=np.loadtxt(f,skiprows=9,usecols=[keys.index(k) for k in want]);a=a[np.lexsort((a[:,1],a[:,0]))]
    if data is None:data=a.copy()
    else:assert np.array_equal(a[:,:3],data[:,:3]);data[:,3:11]+=a[:,3:11];assert np.array_equal(data[:,11:],a[:,11:])
    provenance.append(dict(file=str(f),sha256=sha(f),source_kind='SPARTA_completed_recovery_block'))
data[:,3:11]/=10;fluid=data[:,2]>0;a=data[fluid];d=dict(zip(['x','y','volume','rho','u','v','w','ttr','trot','p','mfp','xlo','xhi','ylo','yhi'],a.T))
row=dict(gas='N2',kn=.075,parameter=.075,nrho_in=cfg['nrho'],T_in=200.,U_in=cfg['U'],source_kind='SPARTA_wide_refined_recovery_mean_2_4ms',path=str(rec),input_sha256=sha(rec/'config.json'));loaded.append((row,d))
physical={};attached={};metadata=[];raw_rays={}
for gas in ['Ar','N2']:
    cases=sorted([(r,d) for r,d in loaded if r['gas']==gas],key=lambda q:q[0]['kn']);assert len(cases)==9,(gas,len(cases))
    xmin=max(float(d['x'].min()) for _,d in cases);xmax=min(float(d['x'].max()) for _,d in cases);ymin=max(float(d['y'].min()) for _,d in cases);ymax=min(float(d['y'].max()) for _,d in cases)
    for row,data in cases:
        kn=float(row['kn']);mass=6.63e-26 if gas=='Ar' else 4.65e-26;gamma=5/3 if gas=='Ar' else 1.4;case_lambda=kn*2*R
        scales=dict(rho=row['nrho_in']*mass,ttr=row['T_in'],trot=row['T_in'],u=row['U_in'],p=row['nrho_in']*1.380649e-23*row['T_in'],mach_proxy=row['U_in']/np.sqrt(gamma*1.380649e-23*row['T_in']/mass))
        data['mach_proxy']=np.sqrt(data['u']**2+data['v']**2+data['w']**2)/np.sqrt(np.maximum(gamma*1.380649e-23*data['ttr']/mass,1e-30))
        if 'xlo' in data:
            ips={v:CellEvaluator(data,data[v]/scales[v]) for v in variables if v in data and np.nanmax(abs(data[v]))>0}
        else:
            tri=masked_triangulation(data);ips={v:mtri.LinearTriInterpolator(tri,data[v]/scales[v]) for v in variables if v in data and np.nanmax(abs(data[v]))>0}
        for shape in [(240,160),(480,320)]:
            xg=np.linspace(xmin,xmax,shape[0]);yg=np.linspace(ymin,ymax,shape[1]);X,Y=np.meshgrid(xg,yg);wx=np.ones(shape[0]);wy=np.ones(shape[1]);wx[[0,-1]]=.5;wy[[0,-1]]=.5;weight=np.outer(wy,wx).ravel()
            for var,ip in ips.items():physical.setdefault((gas,var,shape),[]).append((row,interp(ip,X,Y).ravel(),weight))
        # Estimate exported-point spacing, explicitly not collision-cell h.
        line=(np.abs(data['y']-data['y'].min())<1e-8)&(data['x']<0)
        xline=np.unique(data['x'][line]);spacing=float(np.median(np.diff(xline))) if len(xline)>10 else float(np.sqrt(np.ptp(data['x'])*np.ptp(data['y'])/len(data['x'])))
        for factor in [.5,1.,2.]:
            snaps={v:np.full((65,len(xi)),np.nan) for v in ips};jac=np.full((65,len(xi)),np.nan);supported=0
            for ia,angle in enumerate(angles):
                direction=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]);origin=np.array([R,0])+R*direction;limits=[]
                for k,(lo,hi) in enumerate([(data['x'].min(),data['x'].max()),(data['y'].min(),data['y'].max())]):
                    if abs(direction[k])>1e-12:
                        t=((hi if direction[k]>0 else lo)-origin[k])/direction[k]
                        if t>0:limits.append(t)
                if not limits:continue
                s=np.linspace(0,min(limits),1601);x=origin[0]+s*direction[0];y=origin[1]+s*direction[1];q={v:interp(ip,x,y) for v,ip in ips.items()};sigma=factor*max(.5*spacing,.01*case_lambda)
                if ia==56 and factor==1:raw_rays[gas,kn]=(s,q)
                marker=outer(s,q['rho'],sigma);markers.append(dict(gas=gas,kn=kn,angle=float(angle),smoothing_factor=factor,supported=marker is not None,**(marker or {})))
                if marker is None:continue
                supported+=1;t=marker['s']+xi*marker['delta'];jac[ia]=(R+t)*marker['delta']
                for var in snaps:snaps[var][ia]=resample(s,q[var],t)
            for var in snaps:attached.setdefault((gas,var,factor),[]).append((row,snaps[var].ravel(),jac.ravel(),supported))
        metadata.append(dict(gas=gas,kn=kn,source=row.get('source_kind','original_DS2V'),nominal_state=scales,output_spacing_proxy=spacing))
        print('SAMPLED',gas,kn,flush=True)
summary=[]
for (gas,var,shape),items in physical.items():
    for family in (['all_Kn_hybrid','historical_only'] if gas=='N2' else ['all_Kn_historical']):
        subset=[i for i in items if family!='historical_only' or i[0].get('source_kind')!='SPARTA_wide_refined_recovery_mean_2_4ms'];kn=[x[0]['kn'] for x in subset];label=f'{gas}_{var}_physical_{shape[0]}_{family}'
        r=pod(np.array([i[1] for i in subset]),subset[0][2],kn,label);summary.append(dict(gas=gas,variable=var,basis='physical',grid=list(shape),family=family,**r))
for gas in ['Ar','N2']:
    for var in variables:
        groups=[attached.get((gas,var,f),[]) for f in [.5,1.,2.]]
        if not all(groups):continue
        # Same cases and exact feature support across smoothing settings.
        keep=[j for j in range(len(groups[0])) if all(g[j][3]>=8 for g in groups)]
        if len(keep)<3:continue
        common=np.all(np.isfinite(np.array([[g[j][1] for j in keep] for g in groups])),axis=(0,1));ref=np.nanmean(np.array([groups[1][j][2] for j in keep]),axis=0);ref[~common]=0
        for factor,group in zip([.5,1.,2.],groups):
            r=pod(np.array([group[j][1] for j in keep]),ref,[group[j][0]['kn'] for j in keep],f'{gas}_{var}_attached_sigma{factor}')
            summary.append(dict(gas=gas,variable=var,basis='density_attached',smoothing_factor=factor,excluded_kn=[group[j][0]['kn'] for j in range(len(group)) if j not in keep],coordinate_metric='Fixed reference mean radial Jacobian across retained cases; same support/cases for all smoothing factors',**r))
for gas in ['Ar','N2']:
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for (g,kn),(s,qs) in raw_rays.items():
        if g!=gas:continue
        for ax,var in zip(axes,['rho','ttr','u']):ax.plot(s/R,qs[var],label=f'Kn={kn:g}');ax.set(xlabel='Distance outward / R',ylabel=var+' / nominal upstream',xlim=(0,8));ax.legend(fontsize=7)
    fig.suptitle(gas+' exact Kn folders; theta170; original fields and labelled recovery');fig.tight_layout();fig.savefig(O/f'{gas}_physical_profiles.png',dpi=150);plt.close(fig)
(O/'SUMMARY.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(markers).to_csv(O/'MARKERS.csv',index=False);(O/'INPUTS.json').write_text(json.dumps(metadata,indent=2));(O/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
method=dict(status='Parameter-snapshot POD reanalysis, not DSMC convergence or experimental validation',common_support='Strict all-case finite geometry mask; no70percent imputation; solid-intersecting triangles masked; no interpolation over ray gaps',centering='Training-only mean in each leave-one-out fold; weighted scalar-variable SVD',normalization='Nominal DS2VD incoming state, never heated finite-domain bulk temperature. Fixed-gamma translational Mach proxy recomputed consistently.',energy='Fraction of scalar fluctuation variance, not kinetic energy',prediction='Physical-basis interior logKn coefficient interpolation uses no heldout response amplitudes; projection is conditional reconstruction; endpoints not extrapolated. Attached results use oracle density registration and are not full predictive models.',missing='Original N2Kn.075 numeric field empty; all-nine hybrid explicitly labelled and historical-eight separately reported; independent solver recovery2–4ms. Wide-domain refined 40ppc recovery included; comparison with v13 also changes mesh/particle count and is not domain-only sensitivity.',attached='Outermost resolved compression peak; no forced peak where upstream plateau or separated peak is unsupported. Same cases/support across smoothing factors; exclusions explicit.',tests=dict(wall_peak_rejection=True,weighted_orthogonality=True,full_rank_reconstruction=True),no_global_convergence_claim=True)
(O/'METHOD.json').write_text(json.dumps(method,indent=2));shutil.copy2(__file__,O/Path(__file__).name)
dest=Path('campaign')/'pod_all_kn_v12'/O.name;dest.mkdir(parents=True);tar=dest/'analysis.tar';files=list(O.iterdir())
with tarfile.open(tar,'w') as tf:
    for f in files:tf.add(f,arcname=f.name)
with tarfile.open(tar) as tf:
    for f in files:assert hashlib.sha256(tf.extractfile(f.name).read()).hexdigest()==sha(f)
(O/'ARCHIVE_VERIFIED.json').write_text(json.dumps(dict(verified=True,archive=str(tar),sha256=sha(tar))));(C/'LATEST_RESULT.json').write_text(json.dumps(dict(path=str(O))))
print('RESULT',str(O),flush=True);print(json.dumps([dict(gas=r['gas'],variable=r['variable'],basis=r['basis'],E1=r.get('E1'),N95=r.get('N95'),snapshots=r.get('snapshots')) for r in summary]),flush=True)
