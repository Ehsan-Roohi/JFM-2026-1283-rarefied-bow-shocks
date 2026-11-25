from pathlib import Path
import os,json,hashlib,datetime
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
W=Path('campaign');O=W/'editorial_current_revision';F=O/'manuscript';E=O/'evidence'
root=W/'sparta_supplemental_20261022_v1/N2_M15_vibration';cfg=json.loads((root/'config.json').read_text());complete=json.loads((root/'production/TARGET_COMPLETE.json').read_text())
assert complete['target']==400000 and cfg['M']==15 and cfg['warmup']==200000 and cfg['dt']==1e-8 and cfg['block']==20000
prior={x['file']:x for x in json.loads((W/'analysis_M15_isolated_20261024T113713Z/PROVENANCE.json').read_text())}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stamp(p):
 st=p.stat();return dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
src=[];cache=E/'M15_TEMPERATURE_CONTOURS.npz';receipt=E/'M15_TEMPERATURE_CONTOURS_PROVENANCE.json'
if cache.exists() and receipt.exists():
 saved=json.loads(receipt.read_text());assert sha(cache)==saved['cache_sha256']
 for q in saved['fields']:assert stamp(Path(q['file']))==q['after']
 z=np.load(cache);x=z['x'];y=z['y'];mask=z['mask'];means=z['temperature_K'];src=saved['fields']
 print('REUSED_VERIFIED_CACHE',flush=True)
else:
 mean=None;geometry=None;ids0=None
 for it in range(220000,400001,20000):
  p=root/f'production/fields.{it}.grid';before=stamp(p);digest=sha(p);assert digest==prior[str(p)]['sha256'],(p,digest)
  with p.open() as f:hdr=[next(f).rstrip() for _ in range(9)]
  cols=hdr[-1].split()[2:];assert int(hdr[1])==it and int(hdr[3])==1241600
  wanted=['id','xc','yc','c_geom','f_avg[14]','f_avg[7]','f_avg[9]'];data=pd.read_csv(p,sep=r'\s+',skiprows=9,names=cols,usecols=wanted,dtype=np.float64)[wanted].to_numpy()
  data=data[np.argsort(data[:,0])];assert np.isfinite(data).all() and len(data)==1241600 and len(np.unique(data[:,0]))==len(data)
  if geometry is None:geometry=data[:,:4].copy();mean=data[:,4:].copy()
  else:
   assert np.array_equal(data[:,:4],geometry),(it,'geometry');mean+=data[:,4:]
  after=stamp(p);assert after==before
  src.append(dict(file=str(p),sha256=digest,before=before,after=after,timestep=it,averaging_window_ms=[(it-20000)*1e-5,it*1e-5]))
  print('FIELD',it,'SHA/STAMPS/MAPPING_PASS',flush=True)
 mean/=10
 x=np.unique(geometry[:,1]);y=np.unique(geometry[:,2]);assert len(x)==cfg['nx'] and len(y)==cfg['ny']
 ix=np.searchsorted(x,geometry[:,1]);iy=np.searchsorted(y,geometry[:,2]);assert len(np.unique(iy*len(x)+ix))==len(mean)
 means=np.empty((3,len(y),len(x)));means[:,iy,ix]=mean.T
 mask=np.zeros((len(y),len(x)),bool);mask[iy,ix]=geometry[:,3]<=0
 np.savez_compressed(cache,x=x,y=y,mask=mask,temperature_K=means)
 saved=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),case=cfg,columns=dict(Ttr='f_avg[14]',Trot='f_avg[7]',Tvib='f_avg[9]'),method='Equal arithmetic mean of the ten complete 0.2ms cell-temperature blocks from2–4ms; cell-reported temperatures, not temperatures reconstructed from mean energies. No smoothing or fill; zero vibrational excitation retained.',fields=src,config_sha256=sha(root/'config.json'),input_sha256=sha(root/'production/in.production'),completion_sha256=sha(root/'production/TARGET_COMPLETE.json'),geometry_sorted_id_sha256=hashlib.sha256(geometry.tobytes()).hexdigest(),cache_sha256=sha(cache),geometry_equal_all_fields=True)
 receipt.write_text(json.dumps(saved,indent=2))
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':11,'axes.labelsize':12,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'pdf.fonttype':42,'axes.linewidth':.75})
pal=ListedColormap(['#173b91','#2166cc','#21a6db','#20cebf','#41db74','#a5e537','#e8e526','#ffb52b','#f77721','#da3d16','#9b160b'])
# A common physical temperature scale makes colours directly comparable.
hi=float(np.max(means[:,~mask]));top=float(np.ceil(hi/500)*500);levels=np.linspace(0,top,12)
outputs=[]
for k,(key,label) in enumerate([('ttr',r'$T_{\rm tr}$ (K)'),('trot',r'$T_{\rm rot}$ (K)'),('tvib',r'$T_{\rm vib}$ (K)')]):
 a=np.ma.masked_where(mask,means[k]);fig,ax=plt.subplots(figsize=(4.4,2.8),layout='constrained')
 im=ax.pcolormesh(x,y,a,shading='nearest',cmap=pal,norm=BoundaryNorm(levels,pal.N),rasterized=True)
 ax.add_patch(plt.Circle((.1524,0),.1524,fc='white',ec='black',lw=.8))
 ax.set(xlim=(-.2,.65),ylim=(0,.4),xlabel='$x$ (m)',ylabel='$y$ (m)',aspect='equal')
 ax.tick_params(direction='in',top=True,right=True)
 cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.025);cb.set_label(label);cb.set_ticks(np.linspace(0,top,5))
 name='N2_M15_vibration_'+key
 fig.savefig(F/(name+'.pdf'),dpi=600,bbox_inches='tight',pad_inches=.06);fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.06);plt.close(fig)
 outputs.append(dict(name=name,min_K=float(a.min()),max_K=float(a.max()),zero_excitation_cells=int(np.sum((means[k]==0)&(~mask))),pdf_sha256=sha(F/(name+'.pdf')),png_sha256=sha(F/(name+'.png'))))
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_script_sha256=sha(__file__),provenance=str(receipt),cache_sha256=sha(cache),temperature_units='K',common_color_limits_K=[0,top],levels_K=list(levels),cells=1241600,fluid_cells=int(np.sum(~mask)),blocks=10,window_ms=[2,4],scattering_alpha=1.,model='SPARTA VSS implementation with alpha=1 isotropic/VHS angular limit, variable Parker rotation and discrete vibration; no chemistry; separateMach15 diagnostic, not original DS2V baseline or external validation',outputs=outputs)
(O/'M15_TEMPERATURE_FIGURES_20261107.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
