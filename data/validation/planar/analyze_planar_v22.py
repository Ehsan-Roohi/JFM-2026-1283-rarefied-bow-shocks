"""Read-only, versioned analysis of the six completed JFM planar runs."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = Path('campaign')
O = Path(sys.argv[2]) if len(sys.argv)>2 else W / 'analysis_planar_v22_20261031'
O.mkdir(exist_ok=True)
K = 1.380649e-23
SIGMAS = [.0001, .0002, .0004]
plt.rcParams.update({'font.size':10, 'axes.titlesize':11, 'figure.titlesize':13})

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name, value): (O/name).write_text(json.dumps(value, indent=2, allow_nan=False))
def gaussian_filter1d(q,sigma,mode='nearest'):
    radius=max(1,int(np.ceil(4*sigma)));k=np.arange(-radius,radius+1)
    weights=np.exp(-.5*(k/sigma)**2);weights/=weights.sum()
    return np.convolve(np.pad(q,radius,mode='edge'),weights,mode='valid')
def read(p):
    lines=p.read_text().splitlines()
    assert lines[8].startswith('ITEM: CELLS ')
    names=lines[8].split()[2:]
    a=np.loadtxt(lines[9:]); a=a[np.argsort(a[:,names.index('xc')])]
    assert len(np.unique(a[:,names.index('id')]))==len(a)
    return {n:a[:,i] for i,n in enumerate(names)}

def crossing(x,y,level,lo=None,hi=None):
    ids=np.where((y[:-1]<=level)&(y[1:]>level))[0]
    if lo is not None: ids=ids[(x[ids]>=lo)&(x[ids]<=hi)]
    if len(ids)!=1: return None
    i=ids[0];return float(x[i]+(level-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i]))

def scales(x,q,up,down,sigma,center):
    sm=gaussian_filter1d(q,sigma/np.median(np.diff(x)),mode='nearest')
    jump=down-up; z=(sm-up)/jump
    # Fixed physical front window, independent of the variable; no wall in planar runs.
    mask=abs(x-center)<.025
    grad=np.gradient(sm,x); i=np.where(mask)[0][np.argmax(abs(grad[mask]))]
    a=crossing(x,z,.1,center-.025,center+.025)
    b=crossing(x,z,.9,center-.025,center+.025)
    return {'sigma_m':sigma,'signed_jump':float(jump),
            'peak_abs_gradient':float(abs(grad[i])), 'peak_x_m':float(x[i]),
            'gradient_width_m':float(abs(jump)/abs(grad[i])),
            'width_10_90_m':float(b-a) if a is not None and b is not None else None,
            'normalized_overshoot':float(max(0,np.max(z[mask])-1)),
            'normalized_undershoot':float(max(0,-np.min(z[mask]))),
            'negative_variation_fraction':float(np.sum(np.maximum(-np.diff(z[mask]),0))/max(np.sum(abs(np.diff(z[mask]))),1e-30)),
            'interpretation':'gradient width; thermal overshoot is reported separately, not a relaxation length'}

campaign=Path(sys.argv[1])/'CAMPAIGN.json' if len(sys.argv)>1 else W/'sparta_supplemental_20261022_v1/CAMPAIGN.json'
C=json.loads(campaign.read_text()); results={}; banks={}; provenance=[]
for c in C['configs']:
    if c['geometry']!='planar': continue
    name=c['name']; r=Path(c['root'])/'production'
    complete=json.loads((r/'TARGET_COMPLETE.json').read_text()); assert complete['target']==c['target']
    assert sha(r/'in.production')==complete['input_sha256']
    steps=list(range(c['warmup']+c['block'],c['target']+1,c['block']));assert len(steps)==10
    fs=[r/f'fields.{step}.grid' for step in steps]
    data=[read(p) for p in fs];x=data[0]['xc'];assert len(x)==c['nx']
    for d in data: assert np.array_equal(x,d['xc'])
    mass=C['models'][c['model']]['mass'];n0=c['nrho'];rh=c['rankine_hugoniot']
    vars=['rho','u','Ttr']+(['Trot'] if c['gas']=='N2' else [])
    idx={'rho':'f_avg[3]','u':'f_avg[4]','Ttr':'f_avg[14]','Trot':'f_avg[7]'}
    B=np.stack([np.stack([d[idx[v]] for v in vars],axis=-1) for d in data])
    mean=B.mean(axis=0);dx=float(np.median(np.diff(x)))
    # Identical fractional windows, excluding the injection boundary itself.
    length=c['xhi']-c['xlo'];up=(x>c['xlo']+.05*length)&(x<c['xlo']+.15*length)
    down=(x>c['xhi']-.15*length)&(x<c['xhi']-.05*length)
    target={'rho':n0*mass*rh['compression'],'u':rh['U2'],'Ttr':rh['T2'],'Trot':rh['T2']}
    upstream={'rho':n0*mass,'u':c['U'],'Ttr':c.get('T1',200.),'Trot':c.get('T1',200.)}
    plateaus={}
    for j,v in enumerate(vars):
        ub=B[:,up,j].mean(axis=1);db=B[:,down,j].mean(axis=1)
        plateaus[v]={'up':float(ub.mean()),'down':float(db.mean()),
            'up_error_vs_reservoir_pct':float((ub.mean()/upstream[v]-1)*100),
            'down_error_vs_RH_pct':float((db.mean()/target[v]-1)*100),
            'up_block_SD':float(ub.std(ddof=1)),'down_block_SD':float(db.std(ddof=1)),
            'up_spatial_range_over_jump_pct':float(np.ptp(mean[up,j])/abs(target[v]-upstream[v])*100),
            'down_spatial_range_over_jump_pct':float(np.ptp(mean[down,j])/abs(target[v]-upstream[v])*100)}
    rho=mean[:,0];normalized=(rho-plateaus['rho']['up'])/(plateaus['rho']['down']-plateaus['rho']['up'])
    center=crossing(x,gaussian_filter1d(normalized,.0002/dx),.5)
    assert center is not None, name+' no unique half-density center'
    centers=[]
    for b in B:
        q=(b[:,0]-plateaus['rho']['up'])/(plateaus['rho']['down']-plateaus['rho']['up'])
        centers.append(crossing(x,gaussian_filter1d(q,.0002/dx),.5))
    assert all(z is not None for z in centers)
    times=np.array(steps)*c['dt'];drift=np.polyfit(times,centers,1)[0]
    local_metrics={v:[scales(x,mean[:,j],plateaus[v]['up'],plateaus[v]['down'],s,center) for s in SIGMAS] for j,v in enumerate(vars)}
    metrics={v:[scales(x,mean[:,j],upstream[v],target[v],s,center) for s in SIGMAS] for j,v in enumerate(vars)}
    ds=[]
    for b,cc in zip(B,centers): ds.append(scales(x,b[:,0],upstream['rho'],target['rho'],.0002,cc)['gradient_width_m'])
    # Products below are formed within each block before averaging. SPARTA ave/grid
    # pools particle moments, so rho*u and rho*usq recover sampled conserved moments.
    flux=[]
    for d in data:
        density=d['f_avg[3]'];velocity=d['f_avg[4]']; temp=d['f_avg[14]'];pressure=d['f_avg[15]']
        massflux=density*velocity;momentum=density*d['f_avg[11]']
        energy=density*velocity*(.5*velocity**2+2.5*K*temp/mass+(K*d['f_avg[7]']/mass if c['gas']=='N2' else 0))
        flux.append(np.stack([massflux,momentum,energy],axis=-1))
    F=np.array(flux);Fm=F.mean(axis=0);fluxstats={}
    for j,v in enumerate(['mass','normal_momentum','equilibrium_energy_proxy']):
        a=F[:,up,j].mean(axis=1);b=F[:,down,j].mean(axis=1)
        fluxstats[v]={'up':float(a.mean()),'down':float(b.mean()),
            'down_minus_up_pct':float((b.mean()/a.mean()-1)*100),
            'difference_block_SD_pct':float(((b-a)/a).std(ddof=1)*100)}
    mfp=np.mean([d['c_mfp[1]'] for d in data],axis=0);tau=np.mean([d['c_mfp[2]'] for d in data],axis=0)
    region=abs(x-center)<.025
    half={v:float(np.linalg.norm(B[:5,:,j].mean(axis=0)-B[5:,:,j].mean(axis=0))/np.linalg.norm(B[5:,:,j].mean(axis=0))*100) for j,v in enumerate(vars)}
    results[name]={'config':c,'blocks':10,'physical_window_s':[c['warmup']*c['dt'],c['target']*c['dt']],
        'plateau_windows_m':[[float(x[up][0]),float(x[up][-1])],[float(x[down][0]),float(x[down][-1])]],
        'plateaus':plateaus,'center_m':center,'block_centers_m':centers,'center_SD_m':float(np.std(centers,ddof=1)),
        'center_drift_m_per_s':float(drift),'drift_displacement_over_density_width':float(abs(drift)*(c['target']-c['warmup'])*c['dt']/metrics['rho'][1]['gradient_width_m']),
        'scales':metrics,'scales_normalization':'prescribed upstream and RH downstream states; observed-window audit recorded separately',
        'local_anchor_scales':local_metrics,'density_width_block_mean_m':float(np.mean(ds)),'density_width_block_SD_m':float(np.std(ds,ddof=1)),
        'half_mean_profile_L2_pct':half,'flux':fluxstats,'h_over_lambda_max':float(np.max(dx/mfp[region])),
        'dt_over_tau_max':float(np.max(c['dt']/tau[region])),
        'upstream_mfp_measured_m':float(mfp[up].mean()),'scientific_convergence':False}
    banks[name]={'x':x,'B':B,'mean':mean,'vars':vars,'centers':np.array(centers),'Fm':Fm,'mfp':mfp,'tau':tau,'times':times}
    np.savez_compressed(O/(name+'.npz'),**{k:v for k,v in banks[name].items() if k!='vars'},variables=vars)
    provenance.append({'case':name,'config_sha256':sha(Path(c['root'])/'config.json'),'input_sha256':sha(r/'in.production'),
        'target_receipt_sha256':sha(r/'TARGET_COMPLETE.json'),'fields':[{'path':str(p),'sha256':sha(p)} for p in fs]})
    print('ANALYZED',name,'center',center,'width',metrics['rho'][1]['gradient_width_m'],'drift',drift,flush=True)

comparisons={}
for gas in ['Ar','N2']:
    names={kind:next((z['name'] for z in C['configs'] if z['gas']==gas and z['name'].endswith('_'+kind)),None) for kind in ['base','fine','long']}
    if not all(names.values()): continue
    fine=banks[names['fine']];ref=results[names['fine']];cs=['#2865a2','#c64f22','#18865f'];styles=['-','--',':']
    fig,axs=plt.subplots(1,len(fine['vars']),figsize=(12,3.8),squeeze=False)
    for i,kind in enumerate(['base','fine','long']):
        name=names[kind];b=banks[name];rr=results[name]
        for j,v in enumerate(b['vars']):
            scale=rr['config']['nrho']*(6.63e-26 if gas=='Ar' else 4.65e-26) if v=='rho' else 1
            axs[0,j].plot((b['x']-rr['center_m'])*1000,b['mean'][:,j]/scale,styles[i],c=cs[i],lw=1.6,label=kind)
            axs[0,j].set(xlim=(-35,35),xlabel=r'$x-x_{\rho,1/2}$ (mm)',ylabel={'rho':r'$\rho/\rho_1$','u':'u (m/s)','Ttr':r'$T_{tr}$ (K)','Trot':r'$T_{rot}$ (K)'}[v]);axs[0,j].grid(alpha=.2)
        xx=fine['x']-ref['center_m'];select=abs(xx)<.03
        err={}
        for j,v in enumerate(fine['vars']):
            yy=np.interp(xx[select],b['x']-rr['center_m'],b['mean'][:,j]);rf=fine['mean'][select,j]
            err[v]=float(np.linalg.norm(yy-rf)/np.linalg.norm(rf)*100)
        comparisons[name+'_vs_fine_centered_L2_pct']=err
    axs[0,0].legend();fig.suptitle(gas+f" planar M={ref['config']['M']:g} | ten blocks, {ref['physical_window_s'][0]*1000:g}–{ref['physical_window_s'][1]*1000:g} ms | one density shift, no width fit")
    fig.tight_layout();
    for ext in ['png','pdf']:fig.savefig(O/f'Supplement_{gas}_planar_profiles.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for i,kind in enumerate(['base','fine','long']):
        name=names[kind];b=banks[name];rr=results[name]
        axs[0,0].plot(b['times']*1000,b['centers']*1000,'o-',c=cs[i],label=kind)
        for j in range(3):
            ax=axs.flat[j+1];f=b['Fm'][:,j];c=rr['config'];uu=(b['x']>c['xlo']+.05*(c['xhi']-c['xlo']))&(b['x']<c['xlo']+.15*(c['xhi']-c['xlo']))
            ax.plot((b['x']-rr['center_m'])*1000,f/np.mean(f[uu]),styles[i],c=cs[i],label=kind)
            ax.set(xlabel=r'$x-x_{\rho,1/2}$ (mm)',ylabel=['Mass flux / upstream','Normal momentum flux / upstream','Equilibrium energy proxy / upstream'][j],xlim=(-60,60));ax.grid(alpha=.2)
    axs[0,0].set(xlabel='Block end time (ms)',ylabel='Half-density center (mm)');axs[0,0].legend();axs[0,0].grid(alpha=.2)
    fig.suptitle(gas+' planar stationarity and flux diagnostics\nEnergy proxy omits heat flux and anisotropic-stress work')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(O/f'Supplement_{gas}_planar_drift_flux.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)

dump('REPORT.json',{'cases':results,'comparisons':comparisons,'scientific_convergence':False,
    'limitations':['Fine changes h and dt together, so their separate independence is not established.',
    'Planar-versus-cylinder differences also include wall deceleration and geometry; not a pure curvature experiment.',
    'This file alone does not compare to independent experimental data; see the validation overlay when present.',
    'Prescribed-state widths and observed-window local-anchor widths are reported separately; an upstream thermal precursor is not relabelled as a freestream plateau.',
    'Equilibrium energy proxy omits third-order heat flux and stress work, and is interpreted only on plateau windows.',
    'No CI derived from ten correlated blocks; SD, half-window differences and drift are reported.',
    'No acceptance thresholds were introduced or changed.']})
dump('PROVENANCE.json',{'campaign':str(campaign),'campaign_sha256':sha(campaign),'cases':provenance,'source_sha256':sha(Path(__file__))})
print('COMPLETE',O,flush=True)
