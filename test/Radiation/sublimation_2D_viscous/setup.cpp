#include "idefix.hpp"
#include "setup.hpp"
#include "column.hpp"
#include "lookupTable.hpp"

real R0Glob;
real rho0Glob;
real rhominGlob;
real T0Glob;
real muGlob;
real gammaGlob;
real rsGlob;
real TsGlob;
real kappaGlob;
real kappairrGlob;
std::string kappatypeGlob;
real alphaMRIGlob;
real alphaDZGlob;
real epsilonGlob;
real TsubGlob;
real rhosGlob;
real rhosindexGlob;
real TwidthGlob;
real f0Glob;
real kappastarGlob;
real kappagasGlob;
real MGlob;
real GGlob;
real densityFloorGlob;

Column *columnGlob;
LookupTable<1> *kappatableGlob;


void MyKappa(DataBlock &data, IdefixArray3D<real> &kappap, IdefixArray3D<real> &kappar) {
  IdefixArray4D<real> Vc=data.hydro->Vc;
  auto units = idfx::units;
  
  IdefixArray3D<real> tau;
  IdefixArray3D<real> kappa=(&data)->radiation[0]->radsource->kappapArr;
  IdefixArray3D<real> kapparho("rho",data.np_tot[KDIR],data.np_tot[JDIR],data.np_tot[IDIR]);
  idefix_for("init rho",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {
      kapparho(k,j,i) = Vc(RHO,k,j,i)*kappa(k,j,i)*units.GetDensity()*units.GetLength();
    });

  columnGlob->ComputeColumn(kapparho);
  tau = columnGlob->GetColumn();
  
  IdefixArray1D<real> dr = data.dx[IDIR];

  real Tsub = TsubGlob;
  real rhos = rhosGlob;
  real rhos_index = rhosindexGlob;
  real Twidth = TwidthGlob;
  real f0 = f0Glob;
  real kappa_star = kappastarGlob;
  real kappa_gas = kappagasGlob;
  real mu = muGlob;

  idefix_for("MyKappa",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                real Tsublim = Tsub*std::pow(Vc(RHO,k,j,i)*units.GetDensity()/rhos,rhos_index);
                real T = Vc(PRS,k,j,i)/Vc(RHO,k,j,i)*units.GetKelvin()*mu;
                real f_delta = 0.2/(Vc(RHO,k,j,i)*units.GetDensity()*kappa_star*dr(i)*units.GetLength())-kappa_gas/kappa_star;
                real f_gtod;
                if ((T < Tsublim) || (tau(k,j,i) > 3.)){
                  f_gtod = f0;
                } else {
                  f_gtod = f_delta*0.25*(1.-std::tanh(std::pow((T-Tsublim)/Twidth,3.)));
                }
                f_gtod *= 1.-std::tanh(2./3.-tau(k,j,i));
                kappap(k,j,i) = kappa_star*f_gtod+kappa_gas;
                kappar(k,j,i) = kappa_star*f_gtod+kappa_gas;
              });
}


void MyViscosity(DataBlock &data, const real t, IdefixArray3D<real> &eta1, IdefixArray3D<real> &eta2) {
  
  IdefixArray3D<real> InvDt = data.hydro->InvDt;
  IdefixArray4D<real> Vc=data.hydro->Vc;
  IdefixArray1D<real> r=data.x[IDIR];
  IdefixArray1D<real> th=data.x[JDIR];
  real alpha = alphaDZGlob;
  real CG = MGlob*GGlob;
  real R0 = R0Glob;
  idefix_for("MyViscosity",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                real R = FMAX(r(i)*sin(th(j)),R0);
                real cs2 = Vc(PRS,k,j,i)/Vc(RHO,k,j,i);
                real Omega = std::sqrt(CG)*std::pow(R,-1.5);
                eta1(k,j,i) = alpha*cs2*Vc(RHO,k,j,i)/Omega;
                eta2(k,j,i) = 0.;
              });

}

// Compute user variables which will be written in vtk files
void ComputeUserVars(DataBlock & data, UserDefVariablesContainer &variables) {
  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();
  
  auto units = idfx::units;

  IdefixHostArray1D<real> x1=d.x[IDIR];
  IdefixHostArray1D<real> x1l=d.xl[IDIR];
  IdefixHostArray1D<real> x2=d.x[JDIR];
  IdefixHostArray1D<real> x2l=d.xl[JDIR];
  IdefixHostArray3D<real> A1=d.A[IDIR];
  IdefixHostArray3D<real> A2=d.A[JDIR];
  IdefixHostArray3D<real> dV=d.dV;
 
  IdefixArray3D<real> tau;
  IdefixArray4D<real> Vc=(&data)->hydro->Vc;
  IdefixArray3D<real> kappa=(&data)->radiation[0]->radsource->kappapArr;

  IdefixHostArray3D<real> divF  = variables["divF"];
  IdefixHostArray3D<real> A1_out  = variables["A1"];

  real rs = rsGlob;
  real Ts = TsGlob;
  real kappa_irr = kappairrGlob;

  real kirr = kappa_irr*idfx::units.GetDensity()*idfx::units.GetLength(); 
  real flux_pre = std::pow(rs/idfx::units.GetLength(),2.)*idfx::units.sigma_sb*std::pow(Ts,4.)/idfx::units.GetLength();


  IdefixArray3D<real> kapparho("rho",d.np_tot[KDIR],d.np_tot[JDIR],d.np_tot[IDIR]);
  idefix_for("init rho",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {
      kapparho(k,j,i) = Vc(RHO,k,j,i)*kappa(k,j,i)*units.GetDensity();
    });

  columnGlob->ComputeColumn(kapparho);
  tau = columnGlob->GetColumn();
  
  

  Kokkos::deep_copy(variables["tau"], tau);
  Kokkos::deep_copy(variables["dV"], dV);

  if(kappatypeGlob=="constant") {

    for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
      for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
        for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {
          A1_out(k,j,i) = A1(k,j,i);

          real Fim = std::exp(-kirr*variables["tau"](k,j,i-1))/std::pow(x1l(i),2.);
          real Fip = std::exp(-kirr*variables["tau"](k,j,i))/std::pow(x1l(i+1),2.);

          divF(k,j,i) = (Fip*A1(k,j,i+1)-Fim*A1(k,j,i));
          divF(k,j,i) *= flux_pre;
          divF(k,j,i) /= dV(k,j,i);

        }
      }
    }
  } else if (kappatypeGlob=="usertable") {
    
    for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
      for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
        for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {
          A1_out(k,j,i) = A1(k,j,i);

          real logtaum = std::log10(FMAX(variables["tau"](k,j,i-1)*idfx::units.GetDensity()*idfx::units.GetLength(),1.e-15));
          real Fim = pow(10.,kappatableGlob->GetHost(&logtaum))*A1(k,j,i)/std::pow(x1l(i),2.);
          real logtaup = std::log10(FMAX(variables["tau"](k,j,i)*idfx::units.GetDensity()*idfx::units.GetLength(),1.e-15));
          real Fip = pow(10.,kappatableGlob->GetHost(&logtaup))*A1(k,j,i+1)/std::pow(x1l(i+1),2.);
          divF(k,j,i)  = flux_pre*(Fip-Fim)/dV(k,j,i);

        }
      }
    }
  }
   
  Kokkos::deep_copy(variables["divF"], divF);
  Kokkos::deep_copy(variables["A1"], A1_out);

}

void InternalBoundary(Hydro *hydro, const real t) {
  auto *data = hydro->data;
  auto units = idfx::units;

  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Vs = hydro->Vs;
  IdefixArray1D<real> x1=data->x[IDIR];
  IdefixArray1D<real> x2=data->x[JDIR];

  real densityFloor = densityFloorGlob;
  idefix_for("InternalBoundary",
    0, data->np_tot[KDIR],
    0, data->np_tot[JDIR],
    0, data->np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                if(Vc(RHO,k,j,i)*units.GetDensity() < densityFloor) {
                  real T= Vc(PRS,k,j,i)/Vc(RHO,k,j,i);
                  Vc(RHO,k,j,i)=densityFloor/units.GetDensity();
                  Vc(PRS,k,j,i)=T*Vc(RHO,k,j,i);
                }
              });
}

// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)
{ 
  auto Vc = data.hydro->Vc;

  // Mirror data on Host
  alphaMRIGlob = input.Get<real>("Setup","alphaMRI",0);
  alphaDZGlob = input.Get<real>("Setup","alphaDZ",0);
  epsilonGlob = input.Get<real>("Setup","epsilon",0);
  R0Glob = input.Get<real>("Setup","R0",0);
  rho0Glob = input.Get<real>("Setup","rho0",0);
  rhominGlob = input.Get<real>("Setup","rhomin",0);
  T0Glob = input.Get<real>("Setup","T0",0);
  TsubGlob = input.Get<real>("Setup","Tsub",0);
  rhosGlob = input.Get<real>("Setup","rhos",0);
  rhosindexGlob = input.Get<real>("Setup","rhos_index",0);
  TwidthGlob = input.Get<real>("Setup","Twidth",0);
  f0Glob = input.Get<real>("Setup","f0",0);
  kappastarGlob = input.Get<real>("Setup","kappa_star",0);
  kappagasGlob = input.Get<real>("Setup","kappa_gas",0);
  densityFloorGlob = input.Get<real>("Setup","density_floor",0);

  GGlob = input.Get<real>("Gravity","gravCst",0);
  MGlob = input.Get<real>("Gravity","Mcentral",0);
  
  muGlob = input.Get<real>("Hydro","mu",0);
  gammaGlob=data.hydro->eos->GetGamma();
  rsGlob=input.Get<real>("Rad","irr",1);
  TsGlob=input.Get<real>("Rad","irr",2);
  kappatypeGlob = input.Get<std::string>("Rad","irr",0);
  if (kappatypeGlob == "constant") {
    kappairrGlob = input.Get<real>("Rad","irr",3);
  } else if (kappatypeGlob == "usertable") {
    std::string irr_file = input.Get<std::string>("Rad","irr",4);
    kappatableGlob = new LookupTable<1>(irr_file,',');
  }

  columnGlob = new Column(IDIR,1,&data);
  columnGlob->ComputeColumn(Vc,RHO);
  auto temp_array = columnGlob->GetColumn();
  data.dump->RegisterVariable(temp_array,"tau");

  data.radiation[0]->EnrollKappa(&MyKappa); 
  data.hydro->EnrollInternalBoundary(&InternalBoundary);
  data.hydro->viscosity->EnrollViscousDiffusivity(&MyViscosity);
  //output.EnrollUserDefVariables(&ComputeUserVars);
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

    auto units=idfx::units;

    real R0 = R0Glob;
    real rho0 = rho0Glob;
    real rhomin = rhominGlob;
    real T0 = T0Glob;
    real mu = muGlob;
    real epsilon = epsilonGlob;
    real CG = MGlob*GGlob;

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
              
              real R = FMAX(d.x[IDIR](i)*std::sin(d.x[JDIR](j)),R0);
              real z2 = std::pow(d.x[IDIR](i)*std::cos(d.x[JDIR](j)),2.);
              real H = epsilon*R;
              real Omega = std::sqrt(CG)*std::pow(R,-1.5);
              real cs = H*Omega;

              d.Vc(RHO,k,j,i) = (rho0*(R0/R)*std::exp(-0.25*M_PI*z2/(H*H))+rhomin)/idfx::units.GetDensity();
              d.Vc(PRS,k,j,i) = d.Vc(RHO,k,j,i)*T0/units.GetKelvin()/mu;
              d.Vc(VX1,k,j,i) = 0.;
              d.Vc(VX2,k,j,i) = 0.;
              d.Vc(VX3,k,j,i) =  Omega*R;

              real T = d.Vc(PRS,k,j,i)/d.Vc(RHO,k,j,i)*units.GetKelvin()*mu;
              d.RadVc[0](ER,k,j,i) = idfx::units.ar*std::pow(T,4.)/idfx::units.GetEnergy();
              d.RadVc[0](FR1,k,j,i) = ZERO_F;
              d.RadVc[0](FR2,k,j,i) = ZERO_F;
              d.RadVc[0](FR3,k,j,i) = ZERO_F;

            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

