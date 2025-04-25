#include "idefix.hpp"
#include "setup.hpp"
#include "analysis.hpp"
#include "constrainedTransport.hpp"
#include "dumpImage.hpp"

real epsilonGlob;
real epsilonTopGlob;
real betaGlob;
real HidealGlob;
real AmMidGlob;
real LHMidGlob;
real gammaGlob;
real densityFloorGlob;
real trSmoothingGlob;
real diffCap;
Analysis *analysis;
bool haveHall;
real alphaGlob;
real alphaBetaGlob;
real muGlob;
real ToutGlob;

LookupTable<2> *kappapGlob;
LookupTable<2> *kapparGlob;

std::vector<real> taus;

ConstrainedTransport<DefaultPhysics> *emf;

/*********************************************/
/**
Customized random number generator
Allow one to have consistant random numbers
generators on different architectures.
**/
/*********************************************/
real randm(void) {
    const int a    =    16807;
    const int m =    2147483647;
    static int in0 = 13763 + 2417*idfx::prank;
    int q;

    /* find random number  */
    q= (int) fmod((double) a * in0, m);
    in0=q;

    return((real) ((double) q/(double)m));
}

KOKKOS_INLINE_FUNCTION real computeDensityFloor(real R, real z, real d_floor_0, real Rin, real c0){

  real  D_return ;
  if (R>Rin){
    D_return = d_floor_0 / (R*sqrt(R)) * 1.0/(z*z+1.2*(c0*R)*(c0*R));
  }
  else{
    D_return = d_floor_0 / (Rin*sqrt(Rin)) * 1.0/(z*z+1.2*(c0*Rin)*(c0*Rin));
  }
  if (D_return < 1.0e-9){
    D_return = 1e-9;
  }
  return D_return;
}


void InternalBoundary(Fluid<DefaultPhysics> *hydro, const real t) {
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray4D<real> dustVc;
  auto *data = hydro->data;
  bool haveDust = data->haveDust;
  if(haveDust) {
   dustVc= data->dust[0]->Vc;
  }
  IdefixArray4D<real> Vs = data->hydro->Vs;
  IdefixArray1D<real> x1=data->x[IDIR];
  IdefixArray1D<real> x2=data->x[JDIR];

  real vAmax = 2.0;
  real densityFloor0 = densityFloorGlob;
  real Rin = 1.0;
  real epsilon=epsilonGlob;

  auto units = idfx::units;
  real mu = muGlob;
  real gamma = gammaGlob;

  idefix_for("InternalBoundary",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real R=x1(i)*sin(x2(j));
      real z=x1(i)*cos(x2(j));
      real zh = FABS(z/R)/epsilon;

      real b2=EXPAND(Vc(BX1,k,j,i)*Vc(BX1,k,j,i) , +Vc(BX2,k,j,i)*Vc(BX2,k,j,i), +Vc(BX3,k,j,i)*Vc(BX3,k,j,i) ) ;
      real va2=b2/Vc(RHO,k,j,i);
      real myMax=vAmax;
      //if(x1(i)<1.1) myMax=myMax/50.0;
      if(va2>myMax*myMax) {
        real oldrho = Vc(RHO,k,j,i);
        real T = Vc(PRS,k,j,i)/Vc(RHO,k,j,i);
        Vc(RHO,k,j,i) = b2/(myMax*myMax);
        Vc(PRS,k,j,i) = T*Vc(RHO,k,j,i);
        Vc(VX1,k,j,i) *= oldrho/Vc(RHO,k,j,i);
        Vc(VX2,k,j,i) *= oldrho/Vc(RHO,k,j,i);
        Vc(VX3,k,j,i) *= oldrho/Vc(RHO,k,j,i);
      }
      real densityFloor = computeDensityFloor(R,z,densityFloor0,Rin,epsilon);
      if(Vc(RHO,k,j,i) < densityFloor) {
        real oldrho = Vc(RHO,k,j,i);
        real T= Vc(PRS,k,j,i)/Vc(RHO,k,j,i);
        Vc(RHO,k,j,i)=densityFloor;
        Vc(VX1,k,j,i) *= oldrho/densityFloor;
        Vc(VX2,k,j,i) *= oldrho/densityFloor;
        Vc(VX3,k,j,i) *= oldrho/densityFloor;
      }
      real Tfloor = 1.e3;
      real Tceiling = 1.e7;
      real T = Vc(PRS,k,j,i)/Vc(RHO,k,j,i)*units.GetKelvin()*mu;
      real Eint_old = Vc(PRS,k,j,i)/(gamma-1.0);
      if (T<Tfloor){
        Vc(PRS,k,j,i)=Tfloor*Vc(RHO,k,j,i)/(units.GetKelvin()*mu);
        Uc(ENG,k,j,i) -= Eint_old;
        Uc(ENG,k,j,i) += Vc(PRS,k,j,i)/(gamma-1.0);
      } else if (T>Tceiling){
        Vc(PRS,k,j,i)=Tceiling*Vc(RHO,k,j,i)/(units.GetKelvin()*mu);
        Uc(ENG,k,j,i) -= Eint_old;
        Uc(ENG,k,j,i) += Vc(PRS,k,j,i)/(gamma-1.0);
      }

    });

}
// User-defined boundaries
void UserdefBoundary(Fluid<DefaultPhysics> *hydro, int dir, BoundarySide side, real t) {

    if( (dir==IDIR) && (side == left)) {
        IdefixArray4D<real> Vc = hydro->Vc;
        IdefixArray4D<real> Vs = hydro->Vs;
        auto *data = hydro->data;
        IdefixArray1D<real> x1 = data->x[IDIR];
        IdefixArray1D<real> x2 = data->x[JDIR];

        int ighost = data->nghost[IDIR];
        real Omega=1.0;
        real Rin = 1.0;
        real csdisk = epsilonGlob/sqrt(Rin);
        real cscorona = epsilonTopGlob/sqrt(Rin);

        hydro->boundary->BoundaryFor("UserDefX1",dir,side,
            KOKKOS_LAMBDA (int k, int j, int i) {
                real R=x1(i)*sin(x2(j));
                real z=x1(i)*cos(x2(j));

                Vc(RHO,k,j,i) = Vc(RHO,k,j,ighost);

                Vc(PRS,k,j,i) = Vc(RHO,k,j,i)*csdisk*csdisk;

                if(Vc(VX1,k,j,ighost)>=ZERO_F) Vc(VX1,k,j,i) = -Vc(VX1,k,j,2*ighost-i-1);
                       else Vc(VX1,k,j,i) = Vc(VX1,k,j,ighost);
                Vc(VX2,k,j,i) = Vc(VX2,k,j,ighost);

                Vc(VX3,k,j,i) = Omega*R;
                #if DIMENSIONS < 3
                Vc(BX3,k,j,i) = - Vc(BX3,k,j,2*ighost-i-1);
                #endif

            });
      hydro->boundary->BoundaryForX2s("UserDefX2s",dir,side,
        KOKKOS_LAMBDA (int k, int j, int i) {
            Vs(BX2s,k,j,i) = Vs(BX2s,k,j,ighost);
          });
      #if DIMENSIONS == 3
      hydro->boundary->BoundaryForX3s("UserDefX3s",dir,side,
        KOKKOS_LAMBDA (int k, int j, int i) {
            Vs(BX3s,k,j,i) = -Vs(BX3s,k,j,2*ighost-i-1);
          });
      #endif
    }

    if( (dir==IDIR) && (side == right)) {
        IdefixArray4D<real> Vc = hydro->Vc;
        IdefixArray4D<real> Vs = hydro->Vs;
        auto *data = hydro->data;
        IdefixArray1D<real> x1 = data->x[IDIR];
        IdefixArray1D<real> x2 = data->x[JDIR];

        int ighost = data->end[IDIR]-1;
        real Rin = 1.0;
        real csdisk = epsilonGlob/sqrt(Rin);
        real cscorona = epsilonTopGlob/sqrt(Rin);

        hydro->boundary->BoundaryFor("UserDefX1",dir,side,
            KOKKOS_LAMBDA (int k, int j, int i) {
                real R=x1(i)*sin(x2(j));
                real z=x1(i)*cos(x2(j));

                Vc(RHO,k,j,i) = Vc(RHO,k,j,ighost);
                Vc(PRS,k,j,i) = Vc(PRS,k,j,ighost);

                if(Vc(VX1,k,j,ighost)<=ZERO_F) Vc(VX1,k,j,i) = 0.0;
                       else Vc(VX1,k,j,i) = Vc(VX1,k,j,ighost);
                Vc(VX2,k,j,i) = Vc(VX2,k,j,ighost);
                Vc(VX3,k,j,i) = Vc(VX3,k,j,ighost);
                #if DIMENSIONS < 3
                Vc(BX3,k,j,i) = - Vc(BX3,k,j,2*ighost-i+1);
                #endif

            });
      hydro->boundary->BoundaryForX2s("UserDefX2s",dir,side,
        KOKKOS_LAMBDA (int k, int j, int i) {
            Vs(BX2s,k,j,i) = Vs(BX2s,k,j,ighost);
          });
      #if DIMENSIONS == 3
      hydro->boundary->BoundaryForX3s("UserDefX3s",dir,side,
        KOKKOS_LAMBDA (int k, int j, int i) {
            Vs(BX3s,k,j,i) = 0.0;
          });
      #endif

    }


}


void EmfBoundary(Fluid<DefaultPhysics> *hydro, const real t) {
    IdefixArray3D<real> Ex1 = hydro->emf->ex;
    IdefixArray3D<real> Ex2 = hydro->emf->ey;
    IdefixArray3D<real> Ex3 = hydro->emf->ez;
    auto *data = hydro->data;

    if(data->lbound[IDIR] == userdef) {

        int ighost = data->nghost[IDIR];

        // Do not permit poloidal field to enter the seed
        idefix_for("EMFBoundary",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,ighost+1,
                    KOKKOS_LAMBDA (int k, int j, int i) {
            Ex3(k,j,i) = ZERO_F;
            #if DIMENSIONS == 3
              Ex2(k,j,i) = ZERO_F;
            #endif
        });
    }
}

void FluxBoundary(DataBlock & data, int dir, BoundarySide side, const real t) {
    IdefixArray4D<real> Flux = data.hydro->FluxRiemann[dir];
    if( dir==IDIR && side == left) {
        int iref = data.beg[IDIR];

        idefix_for("FluxBoundLeft",
                    data.beg[KDIR], data.end[KDIR],
                    data.beg[JDIR], data.end[JDIR],
          KOKKOS_LAMBDA (int k, int j) {
            if(Flux(RHO, k, j, iref) > 0.0) {
              Flux(RHO, k, j, iref) = 0.0; // Cancel incoming mass flux.
            }
          });
    }

}

void CoarsenFunction(DataBlock &data) {
  IdefixArray2D<int> coarseningLevel = data.coarseningLevel[KDIR];
  IdefixArray1D<real> th = data.x[JDIR];
  idefix_for("set_coarsening", 0, data.np_tot[JDIR], 0, data.np_tot[IDIR],
      KOKKOS_LAMBDA(int j,int i) {
        int c = 1.0/fabs(sin(th(j)));
        if(c>5) c = 5;
        coarseningLevel(j,i) = c;

      });
}


void InternalBoundaryRad(Fluid<RadiationPhysics> *radiation, const real t) {
  IdefixArray4D<real> Vc = radiation->Vc;
  auto *data = radiation->data;
  auto units = idfx::units;

  idefix_for("InternalBoundaryRad",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
        real Tceiling = 1.e7;
        if (Vc(ER,k,j,i) > units.ar*std::pow(Tceiling,4)/units.GetEnergy()){
          Vc(ER,k,j,i) = units.ar*std::pow(Tceiling,4)/units.GetEnergy();
          real Fnorm = EXPAND(Vc(FR1,k,j,i)*Vc(FR1,k,j,i),+Vc(FR2,k,j,i)*Vc(FR2,k,j,i),+Vc(FR3,k,j,i)*Vc(FR3,k,j,i));
          if (Fnorm > Vc(ER,k,j,i)){
            EXPAND(Vc(FR1,k,j,i) *= Vc(ER,k,j,i)/Fnorm;,
                   Vc(FR2,k,j,i) *= Vc(ER,k,j,i)/Fnorm;,
                   Vc(FR3,k,j,i) *= Vc(ER,k,j,i)/Fnorm;)
          }
        }
        //std::printf("Er=%e in InternalBoundary\n",Vc(ER,k,j,i));
    });
}

void UserdefBoundaryRad(Fluid<RadiationPhysics> *radiation, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = radiation->Vc;
  auto *data = radiation->data;
  auto units=idfx::units;

  real Tout = ToutGlob;

  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  if(dir==IDIR) {
    int ighost,nxi,iend,ibeg;
    if(side == left) {
      ighost = data->nghost[IDIR];
      ibeg = 0;
      iend = data->beg[IDIR];
      idefix_for("UserDefBoundaryRad",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) {
          Vc(ER,k,j,i) = units.ar*std::pow(Tout,4.)/units.GetEnergy();    
          if (Vc(FR1,k,j,ighost) >=ZERO_F){
            Vc(FR1,k,j,i) = ZERO_F;
          } else {
            Vc(FR1,k,j,i) = Vc(FR1,k,j,ighost);
          }
          Vc(FR2,k,j,i) = Vc(FR2,k,j,ighost);
          Vc(FR3,k,j,i) = Vc(FR3,k,j,ighost);
        });
    } else if (side==right){
      ighost = data->nghost[IDIR];
      nxi = data->np_int[IDIR];
      ibeg = data->end[IDIR];
      iend =data->np_tot[IDIR];
      idefix_for("UserDefBoundaryRad",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) {
          Vc(ER,k,j,i) = units.ar*std::pow(Tout,4.)/units.GetEnergy();
          if (Vc(FR1,k,j,ighost+nxi-1) <=ZERO_F){
            Vc(FR1,k,j,i) = ZERO_F;
          } else {
            Vc(FR1,k,j,i) = Vc(FR1,k,j,ighost+nxi-1);
          }
          Vc(FR2,k,j,i) = Vc(FR2,k,j,ighost+nxi-1);
          Vc(FR3,k,j,i) = Vc(FR3,k,j,ighost+nxi-1);
        });
    }
  }
}

void ComputeUserVars(DataBlock & data, UserDefVariablesContainer &variables) {

  // custom scratch array
  IdefixArray3D<real> scrh("Scratch", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);

  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  auto units = idfx::units;

  // Make references to the user-defined arrays (variables is a container of IdefixHostArray3D)
  // Note that the labels should match the variable names in the input file
  IdefixHostArray3D<real> InvDt  = variables["InvDt"];

  IdefixHostArray3D<real> rhovr = variables["rhovr"];
  IdefixHostArray3D<real> rhovt = variables["rhovt"];
  
  IdefixHostArray3D<real> rhovrvr = variables["rhovrvr"];
  IdefixHostArray3D<real> rhovrvt = variables["rhovrvt"];
  IdefixHostArray3D<real> rhovrvp = variables["rhovrvp"];
  IdefixHostArray3D<real> rhovtvt = variables["rhovtvt"];
  IdefixHostArray3D<real> rhovtvp = variables["rhovtvp"];
  IdefixHostArray3D<real> rhovpvp = variables["rhovpvp"];
  
  IdefixHostArray3D<real> BrBr = variables["BrBr"];
  IdefixHostArray3D<real> BrBt = variables["BrBt"];
  IdefixHostArray3D<real> BrBp = variables["BrBp"];
  IdefixHostArray3D<real> BtBt = variables["BtBt"];
  IdefixHostArray3D<real> BtBp = variables["BtBp"];
  IdefixHostArray3D<real> BpBp = variables["BpBp"];
  
  IdefixHostArray3D<real> Pvr = variables["Pvr"];
  IdefixHostArray3D<real> Pvt = variables["Pvt"];
  IdefixHostArray3D<real> rhov2vr = variables["rhov2vr"];
  IdefixHostArray3D<real> rhov2vt = variables["rhov2vt"];
  IdefixHostArray3D<real> B2vr = variables["B2vr"];
  IdefixHostArray3D<real> B2vt = variables["B2vt"];
  IdefixHostArray3D<real> BVBr = variables["BVBr"];
  IdefixHostArray3D<real> BVBt = variables["BVBt"];
  
  IdefixHostArray3D<real> Fluxrhor = variables["Fluxrhor"];
  IdefixHostArray3D<real> Fluxrhot = variables["Fluxrhot"];
  IdefixHostArray3D<real> Fluxmrr = variables["Fluxmrr"];
  IdefixHostArray3D<real> Fluxmrt = variables["Fluxmrt"];
  IdefixHostArray3D<real> Fluxmtr = variables["Fluxmtr"];
  IdefixHostArray3D<real> Fluxmtt = variables["Fluxmtt"];
  IdefixHostArray3D<real> FluxEngr = variables["FluxEngr"];
  IdefixHostArray3D<real> FluxEngt = variables["FluxEngt"];
  
  IdefixHostArray3D<real> Emfr = variables["Emfr"];
  IdefixHostArray3D<real> Emft = variables["Emft"];
  IdefixHostArray3D<real> Emfp = variables["Emfp"];

  IdefixHostArray3D<real> kappap = variables["kappap"];
  IdefixHostArray3D<real> kappar = variables["kappar"];
  IdefixHostArray3D<real> rhokappapEr = variables["rhokappapEr"];
  IdefixHostArray3D<real> rhokapparFr = variables["rhokapparFr"];
  IdefixHostArray3D<real> rhokapparFt = variables["rhokapparFt"];
  IdefixHostArray3D<real> rhokappaparT4 = variables["rhokappaparT4"];
  
  IdefixHostArray3D<real> FluxRadErr = variables["FluxRadErr"];
  IdefixHostArray3D<real> FluxRadErt = variables["FluxRadErt"];
  IdefixHostArray3D<real> FluxRadFrr = variables["FluxRadFrr"];
  IdefixHostArray3D<real> FluxRadFrt = variables["FluxRadFrt"];
  IdefixHostArray3D<real> FluxRadFtr = variables["FluxRadFtr"];
  IdefixHostArray3D<real> FluxRadFtt = variables["FluxRadFtt"];
  IdefixHostArray3D<real> FluxRadFtr = variables["FluxRadFtr"];

  IdefixHostArray1D<real> x1=d.x[IDIR];
  IdefixHostArray1D<real> x2=d.x[JDIR];
  IdefixHostArray4D<real> Vc=d.Vc;
  IdefixArray3D<real>::HostMirror scrhHost = Kokkos::create_mirror_view(scrh);
  
  Kokkos::deep_copy(scrhHost,scrh);
  IdefixArray3D<real>::HostMirror xHHost;

  for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
    for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
      for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {
        real z=x1(i)*cos(x2(j));
        real R=FMAX(FABS(x1(i)*sin(x2(j))),ONE_F);
        real H=R*epsilonGlob;
        real Omega=pow(R,-1.5);
        InvDt(k,j,i) = d.InvDt(k,j,i);
        
        rhovr(k,j,i) = d.Vc(RHO,k,j,i)*d.Vc(VX1,k,j,i);
        rhovt(k,j,i) = d.Vc(RHO,k,j,i)*d.Vc(VX2,k,j,i);

        rhovrvr(k,j,i) = d.Vc(RHO,k,j,i)*d.Vc(VX1,k,j,i)*d.Vc(VX1,k,j,i);
        rhovrvt(k,j,i) = d.Vc(RHO,k,j,i)*d.Vc(VX1,k,j,i)*d.Vc(VX2,k,j,i);
        rhovrvp(k,j,i) = d.Vc(RHO,k,j,i)*d.Vc(VX1,k,j,i)*d.Vc(VX3,k,j,i);
        rhovtvt(k,j,i) = d.Vc(RHO,k,j,i)*d.Vc(VX2,k,j,i)*d.Vc(VX2,k,j,i);
        rhovtvp(k,j,i) = d.Vc(RHO,k,j,i)*d.Vc(VX2,k,j,i)*d.Vc(VX3,k,j,i);
        rhovpvp(k,j,i) = d.Vc(RHO,k,j,i)*d.Vc(VX3,k,j,i)*d.Vc(VX3,k,j,i);
        
        BrBr(k,j,i) = d.Vc(BX1,k,j,i)*d.Vc(BX1,k,j,i);
        BrBt(k,j,i) = d.Vc(BX1,k,j,i)*d.Vc(BX2,k,j,i);
        BrBp(k,j,i) = d.Vc(BX1,k,j,i)*d.Vc(BX3,k,j,i);
        BtBt(k,j,i) = d.Vc(BX2,k,j,i)*d.Vc(BX2,k,j,i);
        BtBp(k,j,i) = d.Vc(BX2,k,j,i)*d.Vc(BX3,k,j,i);
        BpBp(k,j,i) = d.Vc(BX3,k,j,i)*d.Vc(BX3,k,j,i);
        
        Pvr(k,j,i) = d.Vc(PRS,k,j,i)*d.Vc(VX1,k,j,i);
        Pvt(k,j,i) = d.Vc(PRS,k,j,i)*d.Vc(VX2,k,j,i);
        real rhov2 = d.Vc(RHO)*(d.Vc(VX1,k,j,i)*d.Vc(VX1,k,j,i)+d.Vc(VX2,k,j,i)*d.Vc(VX2,k,j,i)+d.Vc(VX3,k,j,i)*d.Vc(VX3,k,j,i));
        rhov2vr(k,j,i) = rhov2*d.Vc(VX1,k,j,i);
        rhov2vt(k,j,i) = rhov2*d.Vc(VX2,k,j,i);
        real B2 = d.Vc(BX1,k,j,i)*d.Vc(BX1,k,j,i)+d.Vc(BX2,k,j,i)*d.Vc(BX2,k,j,i)+d.Vc(BX3,k,j,i)*d.Vc(BX3,k,j,i);
        B2vr(k,j,i) = B2*d.Vc(VX1,k,j,i);
        B2vt(k,j,i) = B2*d.Vc(VX2,k,j,i);
        real BV = d.Vc(BX1,k,j,i)*d.Vc(VX1,k,j,i)+d.Vc(BX2,k,j,i)*d.Vc(VX2,k,j,i)+d.Vc(BX3,k,j,i)*d.Vc(VX3,k,j,i);
        BVBr(k,j,i) = BV*d.Vc(BX1,k,j,i);
        BVBt(k,j,i) = BV*d.Vc(BX2,k,j,i);

        Fluxrhor(k,j,i) = data.hydro->FluxRiemann[IDIR](RHO,k,j,i);
        Fluxrhot(k,j,i) = data.hydro->FluxRiemann[JDIR](RHO,k,j,i);
        Fluxmrr(k,j,i) = data.hydro->FluxRiemann[IDIR](VX1,k,j,i);
        Fluxmrt(k,j,i) = data.hydro->FluxRiemann[JDIR](VX1,k,j,i);
        Fluxmtr(k,j,i) = data.hydro->FluxRiemann[IDIR](VX2,k,j,i);
        Fluxmtt(k,j,i) = data.hydro->FluxRiemann[JDIR](VX2,k,j,i);
        FluxEngr(k,j,i) = data.hydro->FluxRiemann[IDIR](ENG,k,j,i);
        FluxEngt(k,j,i) = data.hydro->FluxRiemann[JDIR](ENG,k,j,i);

        Emfr(k,j,i) = data.hydro->emf->ex(k,j,i);
        Emft(k,j,i) = data.hydro->emf->ey(k,j,i);
        Emfp(k,j,i) = data.hydro->emf->ez(k,j,i);

       real T = Vc(PRS,k,j,i)/Vc(RHO,k,j,i)*units.GetKelvin()*muGlob;
       real logrho = std::log10(Vc(RHO,k,j,i)*units.GetDensity());
       real x[2];
       x[0] = FMIN(-4.05,FMAX(-14.,logrho));
       x[1] = FMIN(FMAX(std::log10(T),2.5),5.98);
       kappap(k,j,i) = std::pow(10.,kappapGlob->GetHost(x));
       kappar(k,j,i) = std::pow(10.,kapparGlob->GetHost(x));
       rhokappapEr(k,j,i) = d.Vc(RHO,k,j,i)*units.GetDensity()*kappap(k,j,i)*d.RadVc[0](ER,k,j,i)*units.GetEnergy();
       rhokapparFr(k,j,i) = d.Vc(RHO,k,j,i)*units.GetDensity()*kappar(k,j,i)*d.RadVc[0](FR1,k,j,i)*units.GetEnergy();
       rhokapparFt(k,j,i) = d.Vc(RHO,k,j,i)*units.GetDensity()*kappar(k,j,i)*d.RadVc[0](FR2,k,j,i)*units.GetEnergy();
       rhokappaparT4(k,j,i) = d.Vc(RHO,k,j,i)*units.GetDensity()*kappap(k,j,i)*units.ar*std::pow(T,4.);

       FluxRadErr(k,j,i) = data.radiation[0]->FluxRiemann[IDIR](ER,k,j,i);
       FluxRadErt(k,j,i) = data.radiation[0]->FluxRiemann[JDIR](ER,k,j,i);
       FluxRadFrr(k,j,i) = data.radiation[0]->FluxRiemann[IDIR](FR1,k,j,i);
       FluxRadFrt(k,j,i) = data.radiation[0]->FluxRiemann[JDIR](FR1,k,j,i);
       FluxRadFtr(k,j,i) = data.radiation[0]->FluxRiemann[IDIR](FR2,k,j,i);
       FluxRadFtt(k,j,i) = data.radiation[0]->FluxRiemann[JDIR](FR2,k,j,i);

      }
    }
  }

  Kokkos::deep_copy(variables["InvDt"], InvDt);

  Kokkos::deep_copy(variables["rhovr"], rhovr);
  Kokkos::deep_copy(variables["rhovt"], rhovt);
  
  Kokkos::deep_copy(variables["rhovrvr"], rhovrvr);
  Kokkos::deep_copy(variables["rhovrvt"], rhovrvt);
  Kokkos::deep_copy(variables["rhovrvp"], rhovrvp);
  Kokkos::deep_copy(variables["rhovtvt"], rhovtvt);
  Kokkos::deep_copy(variables["rhovtvp"], rhovtvp);
  Kokkos::deep_copy(variables["rhovpvp"], rhovpvp);

  Kokkos::deep_copy(variables["BrBr"], BrBr);
  Kokkos::deep_copy(variables["BrBt"], BrBt);
  Kokkos::deep_copy(variables["BrBp"], BrBp);
  Kokkos::deep_copy(variables["BtBt"], BtBt);
  Kokkos::deep_copy(variables["BtBp"], BtBp);
  Kokkos::deep_copy(variables["BpBp"], BpBp);

  Kokkos::deep_copy(variables["Pvr"], Pvr);
  Kokkos::deep_copy(variables["Pvt"], Pvt);
  Kokkos::deep_copy(variables["rhov2vr"], rhov2vr);
  Kokkos::deep_copy(variables["rhov2vt"], rhov2vt);
  Kokkos::deep_copy(variables["B2vr"], B2vr);
  Kokkos::deep_copy(variables["B2vt"], B2vt);
  Kokkos::deep_copy(variables["BVBr"], BVBr);
  Kokkos::deep_copy(variables["BVBt"], BVBt);

  Kokkos::deep_copy(variables["Fluxrhor"], Fluxrhor);
  Kokkos::deep_copy(variables["Fluxrhot"], Fluxrhot);
  Kokkos::deep_copy(variables["Fluxmrr"], Fluxmrr);
  Kokkos::deep_copy(variables["Fluxmrt"], Fluxmrt);
  Kokkos::deep_copy(variables["Fluxmtr"], Fluxmtr);
  Kokkos::deep_copy(variables["Fluxmtt"], Fluxmtt);
  Kokkos::deep_copy(variables["FluxEngr"], FluxEngr);
  Kokkos::deep_copy(variables["FluxEngt"], FluxEngt);

  Kokkos::deep_copy(variables["Emfr"], Emfr);
  Kokkos::deep_copy(variables["Emft"], Emft);
  Kokkos::deep_copy(variables["Emfp"], Emfp);

  Kokkos::deep_copy(variables["kappap"], kappap);
  Kokkos::deep_copy(variables["kappar"], kappar);
  Kokkos::deep_copy(variables["rhokappapEr"], rhokappapEr);
  Kokkos::deep_copy(variables["rhokapparFr"], rhokapparFr);
  Kokkos::deep_copy(variables["rhokapparFt"], rhokapparFt);
  Kokkos::deep_copy(variables["rhokappaparT4"], rhokappaparT4);

  Kokkos::deep_copy(variables["FluxRadErr"], FluxRadErr);
  Kokkos::deep_copy(variables["FluxRadErt"], FluxRadErt);
  Kokkos::deep_copy(variables["FluxRadFrr"], FluxRadFrr);
  Kokkos::deep_copy(variables["FluxRadFrt"], FluxRadFrt);
  Kokkos::deep_copy(variables["FluxRadFtr"], FluxRadFtr);
  Kokkos::deep_copy(variables["FluxRadFtt"], FluxRadFtt);

}

void analysisFunction(DataBlock& data) {
  analysis->PerformAnalysis(data);
}


// Default constructor


// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Copy the pointer to the emf function for future usage
  emf = data.hydro->emf.get();
  // Set the function for userdefboundary
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  if(data.haveGridCoarsening) {
    data.EnrollGridCoarseningLevels(&CoarsenFunction);
  }
  data.hydro->EnrollInternalBoundary(&InternalBoundary);
  //data.hydro->EnrollEmfBoundary(&EmfBoundary);
  //data.hydro->EnrollFluxBoundary(&FluxBoundary);

  if(data.haveRadiation) {
    int nFrequencies = data.radiation.size();
    data.radiation[0]->EnrollUserDefBoundary(&UserdefBoundaryRad);
    for(int n = 0 ; n < nFrequencies ; n++) {
      data.radiation[n]->EnrollInternalBoundary(&InternalBoundaryRad);
    }
  }

  gammaGlob=data.hydro->eos->GetGamma();
  muGlob = input.Get<real>("Hydro","mu",0);
  alphaGlob = input.GetOrSet<real>("Setup","alpha",0,1.5);  // density power law
  alphaBetaGlob = input.GetOrSet<real>("Setup","alphaBeta",0,0.0); // beta power law
  epsilonGlob = input.Get<real>("Setup","epsilon",0);
  epsilonTopGlob = input.Get<real>("Setup","epsilonTop",0);
  betaGlob = input.Get<real>("Setup","beta",0);
  HidealGlob = input.Get<real>("Setup","Hideal",0);
  ToutGlob = input.Get<real>("Setup","Tout",0);

  densityFloorGlob = input.Get<real>("Setup","densityFloor",0);
  trSmoothingGlob = input.Get<real>("Setup","transitionSmoothing",0);

  output.EnrollUserDefVariables(&ComputeUserVars);
  // assume disc surface at 6.5 h
  analysis = new Analysis(grid, data,std::string("profile.dat"), HidealGlob*epsilonGlob);
  output.EnrollAnalysis(&analysisFunction);

  std::string kappapfile = input.Get<std::string>("Rad","kappa",2);
  std::string kapparfile = input.Get<std::string>("Rad","kappa",3);
  kappapGlob = new LookupTable<2>(kappapfile,',');
  kapparGlob = new LookupTable<2>(kapparfile,',');

}


// Flow initialisation, read directly from the DumpImage
void Setup::InitFlow(DataBlock &data) {

  // Create a host copy
  DataBlockHost d(data);

  std::printf("hello3\n");

  DumpImage image("dump.0141.dmp", &data);
  
  std::printf("hello\n");

  for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
    for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
      for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {

        // Note that the restart dump array only contains the full (global) active domain
        // (i.e. it excludes the boundaries, but it is not decomposed accross MPI procs)
        int iglob=i-2*d.beg[IDIR]+d.gbeg[IDIR];
        int jglob=j-2*d.beg[JDIR]+d.gbeg[JDIR];
        int kglob=k-2*d.beg[KDIR]+d.gbeg[KDIR];

        d.Vc(RHO,k,j,i) = image.arrays["Vc-RHO"](kglob,jglob,iglob);
        d.Vc(PRS,k,j,i) = image.arrays["Vc-PRS"](kglob,jglob,iglob);
        real T = d.Vc(PRS,k,j,i)/d.Vc(RHO,k,j,i)*idfx::units.GetKelvin()*muGlob;
        d.RadVc[0](ER,k,j,i) = idfx::units.ar*std::pow(T,4)/idfx::units.GetEnergy();
        d.RadVc[0](FR1,k,j,i) = 0.;
        d.RadVc[0](FR2,k,j,i) = 0.;
        d.RadVc[0](FR3,k,j,i) = 0.;
        d.Vc(VX1,k,j,i) = image.arrays["Vc-VX1"](kglob,jglob,iglob);
        d.Vc(VX2,k,j,i) = image.arrays["Vc-VX2"](kglob,jglob,iglob);
        d.Vc(VX3,k,j,i) = image.arrays["Vc-VX3"](kglob,jglob,iglob);
}}}

  std::printf("hello2\n");
  // For magnetic variable, we should fill the entire active domain, hence an additional
  // point in the field direction
  for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
    for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
        for(int i = d.beg[IDIR]; i < d.end[IDIR]+IOFFSET ; i++) {
          int iglob=i-2*d.beg[IDIR]+d.gbeg[IDIR];
          int jglob=j-2*d.beg[JDIR]+d.gbeg[JDIR];
          int kglob=k-2*d.beg[KDIR]+d.gbeg[KDIR];
          d.Vs(BX1s,k,j,i) = image.arrays["Vs-BX1s"](kglob,jglob,iglob);
          d.Vs(BX2s,k,j,i) = image.arrays["Vs-BX2s"](kglob,jglob,iglob);
          d.Vs(BX3s,k,j,i) = image.arrays["Vs-BX3s"](kglob,jglob,iglob);
  }}}


  // Send our datablock to the device
  d.SyncToDevice();
}




Setup::~Setup() {
  delete analysis;
}


