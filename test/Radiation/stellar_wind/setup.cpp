#include "idefix.hpp"
#include "setup.hpp"
#include "column.hpp"
#include "lookupTable.hpp"

real rho0Glob;
real rhominGlob;
real muGlob;
real rsGlob;
real TsGlob;
std::string kappatypeGlob;
std::string xitypeGlob;

real kGlob;
real alphaGlob;
real MmaxGlob;
real sigmaeGlob;

Column *columnGlob;

void MyKappa(DataBlock &data, IdefixArray3D<real> &kappap, IdefixArray3D<real> &kappar) {
  IdefixArray1D<real> dr = data.dx[IDIR];
  IdefixArray4D<real> Vc=data.hydro->Vc;
  auto units = idfx::units;

  real k = kGlob;
  real alpha = alphaGlob;
  real Mmax = MmaxGlob;
  real sigmae = sigmaeGlob;

  idefix_for("MyKappa",data.beg[KDIR],data.end[KDIR],data.beg[JDIR],data.end[JDIR],data.beg[IDIR],data.end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                real dvdr = Kokkos::fabs(Vc(VX1,k,j,i)-Vc(VX1,k,j,i-1))/dr(i);
                real cs = std::sqrt(Vc(PRS,k,j,i)/Vc(RHO,k,j,i));
                real t = sigmae*units.GetDensity()*units.GetLength()*Vc(RHO,k,j,i)*cs/dvdr;
                if (t > 1.e10) t = 1.e10;
                real M = Kokkos::min(k*std::pow(t,alpha),Mmax);
                std::printf("VX1[i]=%e, VX1[i]=%e, dvdr=%e t=%e, M=%e at i=%i\n",Vc(VX1,k,j,i),Vc(VX1,k,j,i-1),dvdr,t,M,i);

                kappap(k,j,i) = (1.+M)*sigmae;
                kappar(k,j,i) = (1.+M)*sigmae;
              });
}

void MyXi(DataBlock &data, IdefixArray3D<real> &xi) {
  IdefixArray1D<real> dr = data.dx[IDIR];
  IdefixArray4D<real> Vc=data.hydro->Vc;
  auto units = idfx::units;

  real k = kGlob;
  real alpha = alphaGlob;
  real Mmax = MmaxGlob;
  real sigmae = sigmaeGlob;

  idefix_for("MyXi",data.beg[KDIR],data.end[KDIR],data.beg[JDIR],data.end[JDIR],data.beg[IDIR],data.end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                real dvdr = Kokkos::fabs(Vc(VX1,k,j,i)-Vc(VX1,k,j,i-1))/dr(i);
                real cs = std::sqrt(Vc(PRS,k,j,i)/Vc(RHO,k,j,i));
                real t = sigmae*units.GetDensity()*units.GetLength()*Vc(RHO,k,j,i)*cs/dvdr;
                if (t > 1.e10) t = 1.e10;
                real M = Kokkos::min(k*std::pow(t,alpha),Mmax);

                xi(k,j,i) = (1.+M)*sigmae;
              });
}


void UserdefBoundary(Fluid<DefaultPhysics> *hydro, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = hydro->Vc;
  auto *data = hydro->data;

  real rho0 = rho0Glob/idfx::units.GetDensity();
  real P0 = rho0Glob*idfx::units.k_B*TsGlob/(idfx::units.m_p*muGlob)/idfx::units.GetEnergy();

  if(dir==IDIR) {
    int ighost,nxi,iend,ibeg;
    if(side == left) {
      ighost = data->nghost[IDIR];
      ibeg = 0;
      iend = data->beg[IDIR];
      idefix_for("UserDefBoundary",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) {
          Vc(RHO,k,j,i) = rho0;
          Vc(PRS,k,j,i) = P0;
          Vc(VX1,k,j,i) = Vc(VX1,k,j,ighost);
          Vc(VX2,k,j,i) = 0.;
        });
    } else if (side ==right) {
      ighost = data->nghost[IDIR];
      nxi = data->np_int[IDIR];
      ibeg = data->end[IDIR];
      iend = data->np_tot[IDIR];
      idefix_for("UserDefBoundary",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) {
          Vc(RHO,k,j,i) = Vc(RHO,k,j,ighost+nxi-1);
          Vc(PRS,k,j,i) = Vc(PRS,k,j,ighost+nxi-1);
          Vc(VX1,k,j,i) = Vc(VX1,k,j,ighost+nxi-1);
          Vc(VX2,k,j,i) = Vc(VX2,k,j,ighost+nxi-1);
        });
    }
  }
}


void UserdefBoundaryRad(Fluid<RadiationPhysics> *radiation, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = radiation->Vc;
  auto *data = radiation->data;

  real Er0 = idfx::units.ar*std::pow(TsGlob,4.);

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
          Vc(ER,k,j,i) = Er0;
          Vc(FR1,k,j,i) = Er0;
          Vc(FR2,k,j,i) = 0.;
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
          Vc(ER,k,j,i) = Vc(ER,k,j,ighost+nxi-1);
          Vc(FR1,k,j,i) = Vc(FR1,k,j,ighost+nxi-1);
          Vc(FR2,k,j,i) = Vc(FR2,k,j,ighost+nxi-1);
        });
    }
  }
}



// Compute user variables which will be written in vtk files
void ComputeUserVars(DataBlock & data, UserDefVariablesContainer &variables) {
  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  IdefixHostArray1D<real> x1=d.x[IDIR];
  IdefixHostArray1D<real> x1l=d.xl[IDIR];
  IdefixHostArray1D<real> x2=d.x[JDIR];
  IdefixHostArray1D<real> x2l=d.xl[JDIR];
  IdefixHostArray3D<real> A1=d.A[IDIR];
  IdefixHostArray3D<real> A2=d.A[JDIR];
  IdefixHostArray3D<real> dV=d.dV;


}


// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)
{
  DataBlockHost d(data);
  auto Vc = data.hydro->Vc;

  // Mirror data on Host
  rho0Glob = input.Get<real>("Setup","rho0",0);
  rhominGlob = input.Get<real>("Setup","rhomin",0);
  muGlob = input.Get<real>("Hydro","mu",0);
  rsGlob=input.Get<real>("Setup","rs",0);
  TsGlob=input.Get<real>("Setup","Ts",0);

  MmaxGlob = input.Get<real>("Setup","Mmax",0);
  kGlob = input.Get<real>("Setup","k",0);
  alphaGlob = input.Get<real>("Setup","alpha",0);
  sigmaeGlob = input.Get<real>("Setup","sigmae",0);

  kappatypeGlob = input.Get<std::string>("Rad","kappa",0);
  xitypeGlob = input.Get<std::string>("Rad","xi",0);

  columnGlob = new Column(IDIR,1,&data);

  if (kappatypeGlob == "userfunc") {
    data.radiation[0]->EnrollKappa(&MyKappa);
  }
  if (xitypeGlob == "userfunc") {
    data.radiation[0]->EnrollXi(&MyXi);
  }

  // Set the function for userdefboundary
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  if(data.haveRadiation) {
    int nFrequencies = data.radiation.size();
    for(int n = 0 ; n < nFrequencies ; n++) {
      data.radiation[n]->EnrollUserDefBoundary(&UserdefBoundaryRad);
    }
  }  //output.EnrollUserDefVariables(&ComputeUserVars);

  // Compute tau in dumps to check error with or without MPI
  auto temp_array = columnGlob->GetColumn();
}

Setup::~Setup() {
  delete columnGlob;
}
// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);

    real rho0 = rho0Glob;
    real rhomin = rhominGlob;
    real Ts = TsGlob;
    real mu = muGlob;


    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {


              d.Vc(RHO,k,j,i) = rho0/idfx::units.GetDensity()/d.x[IDIR](i);
              d.Vc(PRS,k,j,i) = rho0*Ts/idfx::units.GetKelvin();
              d.Vc(VX1,k,j,i) = 0.;
              d.Vc(VX2,k,j,i) = 0.;
              d.Vc(VX3,k,j,i) = 0.;
              d.RadVc[0](ER,k,j,i) = 0.;
              d.RadVc[0](FR1,k,j,i) = ZERO_F;
              d.RadVc[0](FR2,k,j,i) = ZERO_F;
              d.RadVc[0](FR3,k,j,i) = ZERO_F;
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}
