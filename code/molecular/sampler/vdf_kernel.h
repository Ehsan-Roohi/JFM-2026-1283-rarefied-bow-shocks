// Diagnostic-only particle tally for the CPU SPARTA JFM comparison. C++11.
#ifndef JFM_VDF_KERNEL_H
#define JFM_VDF_KERNEL_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace jfmvdf {
inline void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
inline double dot(const double *a, const double *b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
inline bool finite3(const double *u) {
  return std::isfinite(u[0]) && std::isfinite(u[1]) && std::isfinite(u[2]);
}
struct Axis {
  int n;
  double lo, hi;
  int size() const { return n+2; } // 0 = underflow, n+1 = overflow
  int bin(double x) const {
    if (x<lo) return 0;
    if (x>=hi) return n+1;
    return 1+std::min(n-1,static_cast<int>((x-lo)*n/(hi-lo)));
  }
  void validate() const {
    require(n>=2 && n<=512 && std::isfinite(lo) && std::isfinite(hi) && lo<0 && hi>0,"invalid bin edges or bin count");
  }
};
struct Station {
  std::string id, shape;
  double center[3], normal[3], tangent[3], hn, ht, boxlo[3], boxhi[3], volume;
  bool overlaps(const double *lo,const double *hi,int dimension) const {
    for (int k=0;k<dimension;++k)
      if (hi[k]<boxlo[k] || lo[k]>boxhi[k]) return false;
    return true;
  }
  bool contains(const double *x,int dimension) const {
    if (shape=="box") {
      for (int k=0;k<dimension;++k) if (x[k]<boxlo[k] || x[k]>=boxhi[k]) return false;
      return true;
    }
    double d[3]={x[0]-center[0],x[1]-center[1],0.0};
    double a=dot(d,normal),b=dot(d,tangent);
    return a>=-hn && a<hn && b>=-ht && b<ht;
  }
};
struct Config {
  int dimension, samples_per_block, cap, rotation;
  uint64_t salt;
  std::string name, species, source_sha;
  double scale, frame[3], incident[3];
  Axis longitudinal,transverse,joint_n,joint_t;
  std::vector<Station> stations;
};
inline void keyword(std::istream &in,const std::string &expected) {
  std::string word; in>>word; require(in.good() && word==expected,"invalid VDF manifest token or premature end");
}
inline void read3(std::istream &in,double *v) {in>>v[0]>>v[1]>>v[2];}
inline void read_axis(std::istream &in,Axis &a) {in>>a.n>>a.lo>>a.hi; a.validate();}
inline Config parse(const std::string &text) {
  Config c{}; std::istringstream in(text);
  keyword(in,"JFM_VDF_2");keyword(in,"case");in>>c.name>>c.species>>c.dimension>>c.rotation>>c.source_sha;
  require(c.dimension==2 || c.dimension==3,"dimensionality must be two or three");
  require((c.rotation==0 || c.rotation==1) && c.source_sha.size()==64 &&
          c.source_sha.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid metadata");
  keyword(in,"sampling");in>>c.samples_per_block>>c.cap>>c.salt;
  require(c.samples_per_block>0 && c.samples_per_block<=100000 && c.cap>=0 && c.cap<=16384,"invalid block size or molecular retention cap");
  keyword(in,"frame");in>>c.scale;read3(in,c.frame);read3(in,c.incident);
  require(std::isfinite(c.scale) && c.scale>0 && finite3(c.frame) && std::abs(dot(c.incident,c.incident)-1)<1e-10,"invalid velocity frame or incident unit vector");
  keyword(in,"axes");read_axis(in,c.longitudinal);read_axis(in,c.transverse);read_axis(in,c.joint_n);read_axis(in,c.joint_t);
  require(c.joint_n.n<=128 && c.joint_t.n<=128,"joint histogram exceeds bounded size");
  keyword(in,"stations");int n=0;in>>n;require(n>0 && n<=16,"invalid number of stations");
  for(int i=0;i<n;++i) {
    Station s{};keyword(in,"station");in>>s.id>>s.shape;read3(in,s.center);read3(in,s.normal);read3(in,s.tangent);
    in>>s.hn>>s.ht;read3(in,s.boxlo);read3(in,s.boxhi);in>>s.volume;
    require(in && (s.shape=="rect" || s.shape=="box") && std::isfinite(s.volume) && s.volume>0 && finite3(s.center),"bad station geometry");
    require(finite3(s.boxlo) && finite3(s.boxhi),"nonfinite station bounds");
    for(int k=0;k<c.dimension;++k) require(s.boxlo[k]<s.boxhi[k],"station box is empty");
    require(std::abs(dot(s.normal,s.normal)-1)<1e-10 && std::abs(dot(s.tangent,s.tangent)-1)<1e-10 &&
            std::abs(dot(s.normal,s.tangent))<1e-10,"nonorthonormal station axes");
    if(s.shape=="rect") {
      require(c.dimension==2 && std::abs(s.normal[2])+std::abs(s.tangent[2])<1e-12 && s.hn>0 && s.ht>0,"invalid in-plane rectangle");
      for(int k=0;k<2;++k) {
        const double ext=s.hn*std::abs(s.normal[k])+s.ht*std::abs(s.tangent[k]);
        require(s.boxlo[k]<=s.center[k]-ext+1e-14 && s.boxhi[k]>=s.center[k]+ext-1e-14,"prefilter box excludes rectangle");
      }
      require(std::abs(s.volume/(4*s.hn*s.ht)-1)<1e-8,"incorrect two-dimensional unit-span sample volume");
    } else {
      double v=1; for(int k=0;k<c.dimension;++k) v*=s.boxhi[k]-s.boxlo[k];
      require(std::abs(v/s.volume-1)<1e-8,"incorrect sample box volume");
    }
    for(const auto &other:c.stations) require(other.id!=s.id,"duplicate station");
    c.stations.push_back(s);
  }
  std::string tail;require(!(in>>tail),"unexpected trailing manifest data");return c;
}

// Slot 0 = all observations; slots 1/2 = strictly negative incident/normal;
// slots 3/4 = exactly zero incident/normal. Scale is applied only for binning.
struct Layout {
  int incident,normal,tangent,joint,nhist,ne,eke,erot,extra;
  explicit Layout(const Config &c):incident(5) {
    normal=incident+c.longitudinal.size();tangent=normal+c.longitudinal.size();
    joint=tangent+c.transverse.size();nhist=joint+c.joint_n.size()*c.joint_t.size();
    eke=12;erot=eke+c.longitudinal.size();extra=erot+(c.rotation?c.longitudinal.size():0);ne=extra+23;
  }
};
struct Tally {
  std::vector<unsigned long long> raw;
  std::vector<double> weight,moments;
  Tally(int count,const Layout &l):raw(count*l.nhist,0),weight(count*l.nhist,0),moments(count*l.ne,0) {}
  void clear() {std::fill(raw.begin(),raw.end(),0);std::fill(weight.begin(),weight.end(),0);std::fill(moments.begin(),moments.end(),0);}
};
inline void accumulate(const Config &c,const Layout &l,Tally &t,int station,
                       const double *v,double physical_weight,double mass,double rot_energy) {
  require(finite3(v) && std::isfinite(physical_weight) && physical_weight>0 && mass>0 && std::isfinite(mass),"invalid particle velocity, weight or mass");
  if(c.rotation) require(std::isfinite(rot_energy) && rot_energy>=0,"invalid rotational energy");
  const Station &s=c.stations[station];
  const double w=physical_weight, u[3]={v[0]-c.frame[0],v[1]-c.frame[1],v[2]-c.frame[2]};
  const double vi=dot(u,c.incident),vn=dot(u,s.normal),vt=dot(u,s.tangent);
  const int ni=c.longitudinal.bin(vn/c.scale);
  const int slot=station*l.nhist, en=station*l.ne;
  auto add=[&](int k){++t.raw[slot+k];t.weight[slot+k]+=w;};
  add(0);if(vi<0) add(1);if(vn<0) add(2);if(vi==0) add(3);if(vn==0) add(4);
  add(l.incident+c.longitudinal.bin(vi/c.scale));add(l.normal+ni);add(l.tangent+c.transverse.bin(vt/c.scale));
  add(l.joint+c.joint_n.bin(vn/c.scale)*c.joint_t.size()+c.joint_t.bin(vt/c.scale));
  const double energy=.5*mass*vn*vn;
  t.moments[en]+=w*vi;t.moments[en+1]+=w*vn;t.moments[en+2]+=w*vt;
  t.moments[en+3]+=w*energy;t.moments[en+4]+=w*.5*mass*dot(u,u);
  t.moments[en+l.eke+ni]+=w*energy;
  const int a=en+l.extra;
  const double w2=w*w;
  t.moments[a]+=w2;t.moments[a+1+(vi<0?0:(vi>0?1:2))]+=w2;
  t.moments[a+4+(vn<0?0:(vn>0?1:2))]+=w2;
  const double speed2=dot(u,u);
  for(int k=0;k<3;++k){
    t.moments[a+7+k]+=w*u[k];t.moments[a+10+k]+=w*u[k]*u[k];
    t.moments[a+16+k]+=w*u[k]*speed2;
    if(c.rotation) t.moments[a+19+k]+=w*rot_energy*u[k];
  }
  t.moments[a+13]+=w*u[0]*u[1];t.moments[a+14]+=w*u[0]*u[2];t.moments[a+15]+=w*u[1]*u[2];
  t.moments[a+22]+=w*u[0]*u[0]*u[0];
  if(c.rotation){
    const double wr=w*rot_energy;
    t.moments[en+5]+=wr; t.moments[en+6+(vi<0?0:(vi>0?1:2))]+=wr;
    t.moments[en+9+(vn<0?0:(vn>0?1:2))]+=wr; t.moments[en+l.erot+ni]+=wr;
  }
}
// An independent deterministic permutation of unique (rank,local index) pairs
// for each step and station. It never accesses the simulation random stream.
inline uint64_t mix64(uint64_t x) {
  x+=UINT64_C(0x9e3779b97f4a7c15);
  x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
  x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
  return x^(x>>31);
}
inline uint64_t sample_key(uint64_t salt,uint64_t step,unsigned station,unsigned rank,unsigned local) {
  uint64_t domain=mix64(salt^mix64(step)^mix64(UINT64_C(0xd1b54a32d192ed03)+station));
  return mix64(((uint64_t(rank)<<32)|local)^domain);
}
struct Molecule {
  uint64_t key,cell; uint32_t rank,index; int32_t particle_id; uint32_t reserved;
  double x[3],v[3],erot,weight;
  bool operator<(const Molecule &b) const {return key<b.key;}
};
typedef std::priority_queue<Molecule> Heap;
inline void retain(Heap &heap,unsigned cap,const Molecule &p) {
  if(!cap) return;
  if(heap.size()<cap) heap.push(p);
  else if(p.key<heap.top().key){heap.pop();heap.push(p);}
}
inline std::vector<Molecule> take_sorted(Heap &heap) {
  std::vector<Molecule> result(heap.size());
  for(std::size_t n=result.size();n>0;--n){result[n-1]=heap.top();heap.pop();}
  return result;
}
}
#endif
