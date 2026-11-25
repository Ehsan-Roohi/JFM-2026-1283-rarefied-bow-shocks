"""Independent portable binary-format, aggregation, tamper, and empty-statistics tests."""
import hashlib
import math
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
import vdf_records as reader
from native_fixture import fixture_manifest

# Each sample lists distinct simulator observations. Raw velocities populate
# both out-of-range tails; finite denominators include the zero-velocity record.
OBS=[(-3,0,0,1,2), (0,-3,0,4,4), (1,0,0,1,1), (3,4,-2,4,15), (-.5,1,2,1,3)]

def pack(step):
    out=bytearray(reader.HEADER.pack(b"JFMVDF02",step,step*1e-9,2,59,47,step%2==0,4,3,3))
    tally={};mom=np.zeros(47);actual=[]
    domain=reader.mix64(1229^reader.mix64(step)^reader.mix64(0xd1b54a32d192ed03))
    def category(x):
        if x < -2:return 0
        if x >= 2:return 5
        return math.floor(x+2)+1
    for i,(vx,vy,vz,w,e) in enumerate(OBS):
        labels=[0,5+category(vx),11+category(vy),17+category(-vx),23+category(vy)*6+category(-vx)]
        if vx<0:labels.append(1)
        if vy<0:labels.append(2)
        if vx==0:labels.append(3)
        if vy==0:labels.append(4)
        for label in labels:
            n,wt=tally.get(label,(0,0));tally[label]=(n+1,wt+w)
        proj=vy*vy
        for k,val in enumerate((vx,vy,-vx,proj,vx*vx+vy*vy+vz*vz,e)):mom[k]+=w*val
        mom[6+(0 if vx<0 else (1 if vx>0 else 2))]+=w*e
        mom[9+(0 if vy<0 else (1 if vy>0 else 2))]+=w*e
        mom[12+category(vy)]+=w*proj;mom[18+category(vy)]+=w*e
        eoff=24;w2=w*w;mom[eoff]+=w2
        mom[eoff+1+(0 if vx<0 else (1 if vx>0 else 2))]+=w2
        mom[eoff+4+(0 if vy<0 else (1 if vy>0 else 2))]+=w2
        u=np.asarray([vx,vy,vz]);speed2=u.dot(u)
        mom[eoff+7:eoff+10]+=w*u;mom[eoff+10:eoff+13]+=w*u*u
        mom[eoff+13:eoff+16]+=w*np.asarray([vx*vy,vx*vz,vy*vz]);mom[eoff+16:eoff+19]+=w*speed2*u
        mom[eoff+19:eoff+22]+=w*e*u;mom[eoff+22]+=w*vx**3
        rank=1+i%2;key=reader.mix64(((rank<<32)|i)^domain)
        actual.append((key,101+i,rank,i,300+i,.5,.5,0,vx,vy,vz,e,w))
    out.extend(struct.pack("<I",len(tally)))
    for label,(n,w) in sorted(tally.items()):out.extend(struct.pack("<IQd",label,n,w))
    out.extend(mom.astype("<f8").tobytes())
    actual=sorted(actual) if step%2==0 else []
    out.extend(struct.pack("<I",len(actual)))
    for row in actual:out.extend(struct.pack("<QQIIi8d",*row))
    out.extend(struct.pack("<I",0)+bytes(47*8)+struct.pack("<I",0))
    return bytes(out)

class VDFReadback(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="fixture_",dir=Path(__file__).resolve().parent)
        self.root=Path(self.tmp.name);(self.root/"stations.manifest").write_text(fixture_manifest(32),encoding="ascii")
        (self.root/"run.meta").write_text("JFM_VDF_RUN_2\nfirst 1\nevery 1\ndt 1e-9\nfnum 4\nmass 2\ncellweight 1\ncellint_bytes 8\n",encoding="ascii")
        for step in range(1,5):self.sample(step).write_bytes(pack(step))
        self.c=reader.read_contract(self.root)
    def tearDown(self):self.tmp.cleanup()
    def sample(self,i):return self.root/f"sample_{i:015d}.vdf"

    def test_full_summary_and_rebuild(self):
        res=reader.verify(self.root,1,4,2,self.root/"block")
        self.assertEqual(res["verified_samples"],4);self.assertEqual(res["simulator_count_range_by_station"]["empty"],[0,0])
        for block in res["blocks"]:
            with np.load(block) as a:
                self.assertAlmostEqual(a["backward_lab_x_fraction"][0],2/11)
                self.assertAlmostEqual(a["backward_normal_fraction"][0],4/11)
                self.assertAlmostEqual(a["sample_mean_number_density_m3"][0],11/1.52)
                self.assertTrue(np.isnan(a["backward_lab_x_fraction"][1]))
                self.assertEqual(a["rot_energy_J_given_incident_negative_positive_zero"][0].tolist(),[2.5,12.2,4.0])
                self.assertAlmostEqual(a["incident_all_bin_probabilities_with_tails"][0].sum(),1.0)
                self.assertAlmostEqual(a["joint_all_probabilities_with_tails"][0].sum(),1.0)
                self.assertAlmostEqual(a["sum_physical_weight_squared"][0],70.)
                self.assertAlmostEqual(a["kish_particle_observations"][0],2*121/35)
                self.assertAlmostEqual(a["kish_incident_negative_positive_zero"][0,0],4.)
                u=np.array([v[:3] for v in OBS]);ww=np.array([v[3] for v in OBS]);er=np.array([v[4] for v in OBS])
                fl=u-(ww[:,None]*u).sum(axis=0)/ww.sum()
                expected=(ww[:,None]*fl*(fl*fl).sum(axis=1)[:,None]).sum(axis=0)/1.52
                expected_rot=((ww*er)[:,None]*fl).sum(axis=0)/1.52
                np.testing.assert_allclose(a["translational_heat_flux_W_m2_xyz"][0],expected,rtol=1e-12)
                np.testing.assert_allclose(a["rotational_heat_flux_W_m2_xyz"][0],expected_rot,rtol=1e-12)
        first=Path(res["blocks"][0]);before=hashlib.sha256(first.read_bytes()).hexdigest()
        reader.verify(self.root,1,2,2,self.root/"block")
        self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(),before)
        rec=reader.read_sample(self.sample(2),self.c)
        self.assertEqual(len(rec["molecules"][0]),5);self.assertEqual(len(rec["molecules"][1]),0)
        self.assertEqual(rec["raw"][0,:5].tolist(),[5,2,1,1,2])
        self.assertEqual(rec["weight"][0,:5].tolist(),[11,2,4,4,2])
        self.assertAlmostEqual(rec["moments"][0,3],101.0)

    def test_version_one_rejected(self):
        path=self.sample(2);x=bytearray(path.read_bytes());x[:8]=b"JFMVDF01";path.write_bytes(x)
        with self.assertRaises(ValueError):reader.read_sample(path,self.c)

    def test_corrupt_weight_rejected(self):
        x=bytearray(self.sample(2).read_bytes());off=reader.HEADER.size+4+4+8;struct.pack_into("<d",x,off,90)
        self.sample(2).write_bytes(x)
        with self.assertRaises(ValueError):reader.read_sample(self.sample(2),self.c)

    def test_partial_and_duplicate_not_published(self):
        with self.assertRaises(ValueError):reader.verify(self.root,2,3,2,self.root/"bad")
        with self.assertRaises(ValueError):reader.verify(self.root,1,4,3,self.root/"bad")
        self.sample(3).unlink()
        self.assertEqual(reader.verify(self.root,1,2,2)["verified_samples"],2)
        with self.assertRaises(ValueError):reader.verify(self.root,1,4,2,self.root/"bad")
        self.assertFalse((self.root/"bad").exists())

    def test_reject_tampered_block_and_raw(self):
        res=reader.verify(self.root,1,2,2,self.root/"blocks")
        where=Path(res["blocks"][0]);where.write_bytes(b"corrupt block")
        with self.assertRaises((ValueError,OSError)):reader.verify(self.root,1,2,2,self.root/"blocks")
        changed=bytearray(self.sample(2).read_bytes());changed[-(4+47*8+4+8):- (4+47*8+4)]=struct.pack("<d",777)
        self.sample(2).write_bytes(changed)
        with self.assertRaises(ValueError):reader.read_sample(self.sample(2),self.c)

if __name__=="__main__":unittest.main()
