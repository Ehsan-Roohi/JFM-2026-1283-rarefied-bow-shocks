"""Decode, verify and pool immutable JFM VDF records. All densities use real-molecule weights."""
import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from pathlib import Path
import numpy as np

HEADER=struct.Struct("<8sQd7I")
U32=struct.Struct("<I")
PAIR=np.dtype([("bin","<u4"),("n","<u8"),("weight","<f8")],align=False)
MOLECULE=np.dtype([("key","<u8"),("cell","<u8"),("rank","<u4"),("index","<u4"),
                   ("particle_id","<i4"),("x","<f8",(3,)),("v","<f8",(3,)),
                   ("erot","<f8"),("weight","<f8")],align=False)
MASK=(1<<64)-1

def check(ok,msg):
    if not ok:
        raise ValueError(msg)

def close(a,b):
    return np.allclose(a,b,rtol=2e-10,atol=1e-300)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read_contract(directory):
    directory=Path(directory)
    raw=(directory/"stations.manifest").read_bytes()
    text=raw.decode("ascii").split()
    it=iter(text)
    def tag(expected):check(next(it)==expected,"unexpected manifest field: "+expected)
    def xyz():return np.array([float(next(it)) for _ in range(3)])
    tag("JFM_VDF_2");tag("case")
    c={"name":next(it),"species":next(it),"dimension":int(next(it)),"rotation":bool(int(next(it))),
       "stations_sha256":next(it),"manifest_sha256":sha(raw)}
    tag("sampling");c.update(block=int(next(it)),cap=int(next(it)),salt=int(next(it)))
    tag("frame");c.update(scale=float(next(it)),frame=xyz(),incident=xyz())
    tag("axes")
    axes=[]
    for _ in range(4):
        n=int(next(it));lo=float(next(it));hi=float(next(it))
        check(2<=n<=512 and math.isfinite(lo) and math.isfinite(hi) and lo<0<hi,"invalid velocity bins")
        axes.append((n,lo,hi))
    c["axes"]=axes
    tag("stations");count=int(next(it));check(1<=count<=16,"invalid station count");c["stations"]=[]
    for _ in range(count):
        tag("station")
        c["stations"].append(dict(id=next(it),shape=next(it),center=xyz(),normal=xyz(),tangent=xyz(),
                                  hn=float(next(it)),ht=float(next(it)),lo=xyz(),hi=xyz(),volume=float(next(it))))
    check(next(it,None) is None,"trailing manifest data")
    meta=(directory/"run.meta").read_bytes();lines=meta.decode("ascii").splitlines()
    check(lines[0]=="JFM_VDF_RUN_2","invalid run metadata")
    opts=dict(line.split() for line in lines[1:])
    c.update(first=int(opts["first"]),every=int(opts["every"]),dt=float(opts["dt"]),fnum=float(opts["fnum"]),
             mass=float(opts["mass"]),cellweight=int(opts["cellweight"]),cellint_bytes=int(opts["cellint_bytes"]),
             run_meta_sha256=sha(meta))
    a,b,jn,jt=[v[0]+2 for v in axes]
    c["offset"]=(5,5+a,5+2*a,5+2*a+b)
    c["nhist"]=5+2*a+b+jn*jt;c["extra"]=12+a*(1+int(c["rotation"]));c["ne"]=c["extra"]+23
    check(c["block"]>0 and 0<=c["cap"]<=16384 and c["every"]>0 and c["dt"]>0 and c["mass"]>0 and c["scale"]>0,"invalid run contract")
    return c

def mix64(x):
    x=(x+0x9e3779b97f4a7c15)&MASK
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK;x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)

def bin_id(values,axis):
    n,lo,hi=axis
    middle=np.clip(np.floor((values-lo)*n/(hi-lo)),0,n-1).astype(np.int64)+1
    return np.where(values<lo,0,np.where(values>=hi,n+1,middle))

def read_sample(path,c):
    path=Path(path);st=path.stat();ns=len(c["stations"])
    bound=HEADER.size+ns*(8+c["nhist"]*PAIR.itemsize+c["ne"]*8+c["cap"]*MOLECULE.itemsize)
    check(HEADER.size<=st.st_size<=bound,"sample file size exceeds its manifest bounds")
    data=path.read_bytes();after=path.stat()
    check((st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),"sample changed while reading")
    magic,step,time,n,nh,ne,snap,ranks,active,listed=HEADER.unpack_from(data)
    check(magic==b"JFMVDF02" and (n,nh,ne)==(ns,c["nhist"],c["ne"]),"native VDF header mismatch")
    check(bool(re.fullmatch("sample_%015d.vdf"%step,path.name)),"sample filename does not match its timestep")
    check(math.isfinite(time) and time>=0 and ranks>0 and 0<=listed<=active<=ranks,"invalid time or MPI contributor counts")
    check(step>=c["first"] and (step-c["first"])%c["every"]==0,"unscheduled sample")
    named=(c["cap"]>0 and ((step-c["first"])//c["every"]+1)%c["block"]==0)
    check(snap==int(named),"molecular sample phase mismatch")
    raw=np.zeros((n,nh),dtype=np.uint64);weight=np.zeros((n,nh));moments=np.zeros((n,ne));molecules=[]
    cursor=HEADER.size
    for j in range(n):
        used,=U32.unpack_from(data,cursor);cursor+=4;check(used<=nh,"excess bin records")
        entries=np.frombuffer(data,dtype=PAIR,count=used,offset=cursor);cursor+=used*PAIR.itemsize
        bins=entries["bin"];check(not used or (bins[-1]<nh and np.all(bins[1:]>bins[:-1])),"duplicate or unsorted bin index")
        check(np.all(entries["n"]>0) and np.all(entries["weight"]>0) and np.all(np.isfinite(entries["weight"])),"invalid occupied bin")
        raw[j,bins]=entries["n"];weight[j,bins]=entries["weight"]
        em=np.frombuffer(data,dtype="<f8",count=ne,offset=cursor);cursor+=ne*8
        check(np.isfinite(em).all(),"nonfinite moments");moments[j]=em
        nraw,=U32.unpack_from(data,cursor);cursor+=4
        check(nraw==(min(c["cap"],int(raw[j,0])) if named else 0),"wrong molecular retention count")
        mol=np.frombuffer(data,dtype=MOLECULE,count=nraw,offset=cursor);cursor+=nraw*MOLECULE.itemsize;molecules.append(mol)
        starts=c["offset"];sections=[(starts[k],(starts+(nh,))[k+1]) for k in range(4)]
        for b,e in sections:
            check(int(raw[j,b:e].sum())==int(raw[j,0]) and close(weight[j,b:e].sum(),weight[j,0]),"histogram lost particles or physical weight including tails")
        for which,b in [(1,starts[0]),(2,starts[1])]:
            check(int(raw[j,which])+int(raw[j,which+2])<=int(raw[j,0]),"invalid negative/zero count")
            axis=c["axes"][0];zero_pos=-axis[1]*axis[0]/(axis[2]-axis[1])
            if abs(zero_pos-round(zero_pos))<1e-11:
                end=b+1+round(zero_pos)
                check(int(raw[j,b:end].sum())==int(raw[j,which]) and close(weight[j,b:end].sum(),weight[j,which]),"backward tally and velocity bins disagree")
        normal_n=c["axes"][0][0]+2
        check(np.all(em[3:c["extra"]]>=0) and close(em[12:12+normal_n].sum(),em[3]),"inconsistent projected energies")
        if c["rotation"]:
            check(close(em[12+normal_n:c["extra"]].sum(),em[5]) and close(em[6:9].sum(),em[5]) and close(em[9:12].sum(),em[5]),"inconsistent conditional rotational energy")
        else: check(not np.any(em[5:12]),"unexpected rotational energy for inactive species")
        extra=em[c["extra"]:];w2=extra[0]
        check(np.all(extra[:7]>=0) and close(extra[1:4].sum(),w2) and close(extra[4:7].sum(),w2),"inconsistent squared weights or direction partitions")
        total_w=weight[j,0];npart=int(raw[j,0])
        check((w2>0)==bool(npart),"squared weights for empty/occupied station are inconsistent")
        if npart:
            kish=total_w*total_w/w2
            check(1-1e-9<=kish<=npart*(1+2e-9),"Kish particle-observation bound failed")
        check(np.all(extra[10:13]>=0) and close(.5*c["mass"]*extra[10:13].sum(),em[4]),"Cartesian second moments disagree with kinetic energy")
        if not c["rotation"]:check(not np.any(extra[19:22]),"unexpected rotational energy transport")
        if not nraw:continue
        station=c["stations"][j]; x=mol["x"]; vel=mol["v"]; ww=mol["weight"]
        check(np.all(np.isfinite(x)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(ww)) and np.all(ww>0) and
              np.all(mol["rank"]<ranks) and np.all(np.isfinite(mol["erot"])) and np.all(mol["erot"]>=0),"invalid retained molecule")
        if not c["cellweight"]:check(close(ww,c["fnum"]),"uniform-weight raw record differs from native fnum")
        if not c["rotation"]:check(not np.any(mol["erot"]),"raw rotational state for inactive species")
        check(np.all(mol["key"][1:]>mol["key"][:-1]),"molecular keys are not unique and sorted")
        domain=mix64(c["salt"]^mix64(step)^mix64(0xd1b54a32d192ed03+j))
        for k,r,i in zip(mol["key"],mol["rank"],mol["index"]):
            check(mix64(((int(r)<<32)|int(i))^domain)==int(k),"molecular key mismatch")
        if station["shape"]=="rect":
            d=x-station["center"]
            for axis,h in [(station["normal"],station["hn"]),(station["tangent"],station["ht"])]:
                proj=(d*axis).sum(axis=1)
                check(np.all(proj>=-h-1.e-15) and np.all(proj<=h+1.e-15),"raw molecule outside oriented station")
        else:
            dim=c["dimension"];check(np.all(x[:,:dim]>=station["lo"][:dim]) and np.all(x[:,:dim]<station["hi"][:dim]),"raw molecule outside slab")
        if nraw==int(raw[j,0]):
            u=vel-c["frame"];vi=(u*c["incident"]).sum(axis=1);vn=(u*station["normal"]).sum(axis=1);vt=(u*station["tangent"]).sum(axis=1)
            bvi=bin_id(vi/c["scale"],c["axes"][0]);bn=bin_id(vn/c["scale"],c["axes"][0]);bt=bin_id(vt/c["scale"],c["axes"][1])
            joint=bin_id(vn/c["scale"],c["axes"][2])*(c["axes"][3][0]+2)+bin_id(vt/c["scale"],c["axes"][3])
            for cells,(start,end) in zip((bvi,bn,bt,joint),sections):
                check(np.array_equal(np.bincount(cells,minlength=end-start),raw[j,start:end]) and close(np.bincount(cells,weights=ww,minlength=end-start),weight[j,start:end]),"raw velocities do not reproduce native histogram")
            check(close(np.sum(ww*.5*c["mass"]*vn*vn),em[3]) and close(np.sum(ww*mol["erot"]),em[5]),"raw energies disagree with native moments")
            speed2=(u*u).sum(axis=1)
            check(close(np.sum(ww*ww),extra[0]) and close(np.sum((ww*ww)[vi<0]),extra[1]) and close(np.sum((ww*ww)[vn<0]),extra[4]),"raw squared weights disagree")
            # Bound absolute cancellation error by a sum of absolute molecular terms for signed moments.
            def agrees(values,expected):
                return np.all(np.abs(values.sum(axis=0)-expected)<=2e-10*np.abs(values).sum(axis=0)+1e-300)
            cross=u[:,[0,0,1]]*u[:,[1,2,2]]
            check(agrees(ww[:,None]*u,extra[7:10]) and agrees(ww[:,None]*u*u,extra[10:13]) and agrees(ww[:,None]*cross,extra[13:16]),"raw Cartesian first/second moment mismatch")
            check(agrees((ww*speed2)[:,None]*u,extra[16:19]) and agrees((ww*mol["erot"])[:,None]*u,extra[19:22]) and agrees(ww*u[:,0]**3,extra[22]),"raw translational/rotational energy flux or third moment mismatch")
    check(cursor==len(data),"unparsed trailing sample bytes")
    return dict(step=step,time=time,raw=raw,weight=weight,moments=moments,molecules=molecules,
                ranks=ranks,eligible=active,listed=listed,sha256=sha(data),molecular_snapshot=named)

def div(numerator,denominator):
    a,b=np.broadcast_arrays(np.asarray(numerator,dtype=float),np.asarray(denominator,dtype=float))
    return np.divide(a,b,out=np.full(a.shape,np.nan,dtype=float),where=b>0)

def publish_npz(path,arrays):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".part.",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        try:
            if os.name=="nt":os.rename(tmp,path)  # Windows rename fails if destination exists; also works on exFAT.
            else:os.link(tmp,path)
        except FileExistsError:
            with np.load(path,allow_pickle=False) as previous:
                check(set(previous.files)==set(arrays),"different block field list already exists")
                for k,value in arrays.items():
                    a=previous[k];b=np.asarray(value)
                    equal=(np.array_equal(a,b,equal_nan=True) if b.dtype.kind in "fc" else np.array_equal(a,b))
                    check(equal,"different VDF block already exists: "+k)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def block_arrays(c,steps,times,hashes,raw,weight,mom):
    names=np.array([s["id"] for s in c["stations"]]);vol=np.array([s["volume"] for s in c["stations"]])
    out=dict(steps=np.asarray(steps,dtype="<u8"),time_s=np.asarray(times),source_sample_sha256=np.array(hashes),
             stations=names,volume_m3=vol,case=c["name"],dimensional_velocity_scale_m_s=c["scale"],
             reference_frame_m_s=c["frame"],incident_unit=c["incident"],station_normal=np.array([s["normal"] for s in c["stations"]]),
             station_tangent=np.array([s["tangent"] for s in c["stations"]]),source_manifest_sha256=c["manifest_sha256"],
             source_run_sha256=c["run_meta_sha256"],source_geometry_sha256=c["stations_sha256"],
             raw=raw,physical_number_sum=weight,moments_SI=mom,
             sample_mean_number_density_m3=weight[:,0]/(len(steps)*vol),
             backward_lab_x_fraction=div(weight[:,1],weight[:,0]),backward_normal_fraction=div(weight[:,2],weight[:,0]),
             raw_zero_lab_x=raw[:,3],raw_zero_normal=raw[:,4],retained_molecular_last_step=steps[-1],
             rotational_energy_available=bool(c["rotation"]))
    starts=c["offset"]
    for key,ax,aa in [("incident",0,starts[0]),("normal",0,starts[1]),("tangent",1,starts[2])]:
        n,lo,hi=c["axes"][ax];edges=np.linspace(lo,hi,n+1)*c["scale"]
        bins=weight[:,aa:aa+n+2];prob=div(bins,weight[:,0,None])
        out[key+"_velocity_edges_m_s"]=edges
        out[key+"_all_bin_probabilities_with_tails"]=prob
        out[key+"_interior_pdf_s_per_m"]=prob[:,1:-1]/np.diff(edges)
    n1,lo1,hi1=c["axes"][2];n2,lo2,hi2=c["axes"][3];counts=weight[:,starts[3]:].reshape(len(vol),n1+2,n2+2)
    out["joint_normal_edges_m_s"]=np.linspace(lo1,hi1,n1+1)*c["scale"];out["joint_tangent_edges_m_s"]=np.linspace(lo2,hi2,n2+1)*c["scale"]
    out["joint_all_probabilities_with_tails"]=div(counts,weight[:,0,None,None])
    a=c["axes"][0][0]+2
    out["projected_energy_J_conditional_normal_bin"]=div(mom[:,12:12+a],weight[:,starts[1]:starts[1]+a])
    if c["rotation"]:
        out["rot_energy_J_conditional_normal_bin"]=div(mom[:,12+a:c["extra"]],weight[:,starts[1]:starts[1]+a])
        for key,neg,zero,moff in [("incident",1,3,6),("normal",2,4,9)]:
            denom=np.column_stack([weight[:,neg],np.maximum(0,weight[:,0]-weight[:,neg]-weight[:,zero]),weight[:,zero]])
            out["rot_energy_J_given_"+key+"_negative_positive_zero"]=div(mom[:,moff:moff+3],denom)
    extra=mom[:,c["extra"]:];W=weight[:,0];denom=len(steps)*vol
    out["sum_physical_weight_squared"]=extra[:,0];out["sum_weight_squared_incident_negative_positive_zero"]=extra[:,1:4];out["sum_weight_squared_normal_negative_positive_zero"]=extra[:,4:7]
    out["kish_particle_observations"]=div(W*W,extra[:,0])
    for key,neg,zero,sg in [("incident",1,3,1),("normal",2,4,4)]:
        sw=np.column_stack([weight[:,neg],np.maximum(0,W-weight[:,neg]-weight[:,zero]),weight[:,zero]])
        out["kish_"+key+"_negative_positive_zero"]=div(sw*sw,extra[:,sg:sg+3])
    # Cartesian moments in the declared constant observer frame. The central
    # heat flux is invariant to a constant change of frame. Pool all real-molecule
    # weights in a block before subtracting its station mean velocity.
    s1=extra[:,7:10];u=div(s1,W[:,None]);diagonal=extra[:,10:13];cross=extra[:,13:16]
    second=np.zeros((len(W),3,3));second[:,0,0]=diagonal[:,0];second[:,1,1]=diagonal[:,1];second[:,2,2]=diagonal[:,2]
    for k,(i,j) in enumerate([(0,1),(0,2),(1,2)]):second[:,i,j]=second[:,j,i]=cross[:,k]
    contracted=extra[:,16:19];trace=diagonal.sum(axis=1)
    with np.errstate(invalid="ignore"):
        central=contracted-u*trace[:,None]-2*np.einsum("nij,nj->ni",second,u)+2*W[:,None]*u*(u*u).sum(axis=1)[:,None]
        out["translational_heat_flux_W_m2_xyz"]=.5*c["mass"]*central/denom[:,None]
        ux=u[:,0];cx3=extra[:,22]-3*ux*diagonal[:,0]+2*W*ux**3
        out["central_third_incident_velocity_m3_s3"]=div(cx3,W)
    out["sum_weighted_cartesian_velocity"]=s1;out["sum_weighted_cartesian_velocity_tensor"]=second
    out["sum_weighted_velocity_times_speed_squared"]=contracted;out["sum_weighted_vx_cubed"]=extra[:,22]
    out["mean_velocity_lab_m_s_xyz"]=u+c["frame"]
    if c["rotation"]:
        out["sum_weighted_rot_energy_times_velocity"]=extra[:,19:22]
        out["rotational_heat_flux_W_m2_xyz"]=(extra[:,19:22]-u*mom[:,5,None])/denom[:,None]
    return out

def verify(directory,first,last,block_size,block_dir=None):
    directory=Path(directory);c=read_contract(directory)
    check(block_size==c["block"] and first<=last and first>=c["first"] and
          (first-c["first"])%c["every"]==0 and (last-first)%c["every"]==0,"invalid requested interval or sampling period")
    expected=range(first,last+1,c["every"])
    check(len(expected)<=1_000_000,"unbounded sample request")
    observed=set()
    for p in directory.glob("sample_*.vdf"):
        m=re.fullmatch(r"sample_(\d{15,20})\.vdf",p.name);check(bool(m),"malformed committed sample filename")
        step=int(m.group(1))
        if first<=step<=last:
            check(step not in observed,"duplicate native step");observed.add(step)
    check(observed==set(expected),"missing or unscheduled samples inside requested interval")
    if block_dir is not None:
        check((first-c["first"])//c["every"]%block_size==0 and len(expected)%block_size==0,"incomplete configured block")
    steps=[];times=[];hashes=[];raw=weight=mom=None;prev=None;written=[];start_time=None
    ranges={s["id"]:[None,None] for s in c["stations"]}
    for step in expected:
        sample=read_sample(directory/("sample_%015d.vdf"%step),c)
        if prev is not None:check(math.isclose(sample["time"]-prev,c["every"]*c["dt"],rel_tol=1e-7,abs_tol=c["dt"]*1e-5),"native sample clock discontinuity")
        else:start_time=sample["time"]
        prev=sample["time"]
        for j,s in enumerate(c["stations"]):
            value=int(sample["raw"][j,0]);span=ranges[s["id"]]
            span[0]=value if span[0] is None else min(span[0],value);span[1]=value if span[1] is None else max(span[1],value)
        if block_dir is None:continue
        if not steps:raw=sample["raw"].copy();weight=sample["weight"].copy();mom=sample["moments"].copy()
        else:raw+=sample["raw"];weight+=sample["weight"];mom+=sample["moments"]
        steps.append(step);times.append(prev);hashes.append(sample["sha256"])
        if len(steps)==block_size:
            path=Path(block_dir)/("block_%015d_%015d.npz"%(steps[0],steps[-1]))
            publish_npz(path,block_arrays(c,steps,times,hashes,raw,weight,mom));written.append(str(path));steps=[];times=[];hashes=[]
    return dict(case=c["name"],first=first,last=last,first_time_s=start_time,last_time_s=prev,
                verified_samples=len(expected),simulator_count_range_by_station=ranges,blocks=written)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output_dir",type=Path)
    p.add_argument("--expected-first",type=int,required=True);p.add_argument("--expected-last",type=int,required=True)
    p.add_argument("--block-size",type=int,required=True);p.add_argument("--block-dir",type=Path)
    a=p.parse_args()
    print(json.dumps(verify(a.output_dir,a.expected_first,a.expected_last,a.block_size,a.block_dir),allow_nan=False))
