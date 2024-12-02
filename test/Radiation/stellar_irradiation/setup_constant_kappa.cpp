#include "idefix.hpp"
#include "setup.hpp"
#include "column.hpp"
#include "lookupTable.hpp"

real R0Glob;
real h0Glob;
real hpowGlob;
real rho0Glob;
real rhominGlob;
real T0Glob;
real muGlob;
real gammaGlob;
real rsGlob;
real TsGlob;
real kappaGlob;
real kappairrGlob;

Column *columnGlob;

void MySourceTerm(Hydro *hydro, const real t, const real dtin) {
  auto *data = hydro->data;

  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray3D<real> InvDt = hydro->InvDt;
  IdefixArray1D<real> x1=data->x[IDIR];
  IdefixArray1D<real> x1l=data->xl[IDIR];
  IdefixArray1D<real> x2l=data->xl[JDIR];
  IdefixArray1D<real> x2=data->x[JDIR];
  IdefixArray3D<real> A1=data->A[IDIR];
  IdefixArray3D<real> A2=data->A[JDIR];
  IdefixArray3D<real> dV=data->dV;

  IdefixArray3D<real> tau;

  real C_kb = idfx::units.k_B;
  real C_amu = idfx::units.u;

  real unit_density = idfx::units.density;
  real unit_velocity = idfx::units.velocity;
  real unit_length = idfx::units.length;
  real unit_time = unit_length/unit_velocity;
  real unit_energy = unit_density*unit_velocity*unit_velocity;

  real mu = muGlob;
  real T0 = T0Glob;
  real dt=dtin;
  real csiso = std::sqrt(C_kb*T0/(mu*C_amu));
  real rs = rsGlob;
  real Ts = TsGlob;
  real kappa_irr = kappairrGlob*unit_density*unit_length;

  //Constant kappa
  real flux_pre = std::pow(rs/unit_length,2.)*idfx::units.sigma_sb*std::pow(Ts,4.)/unit_energy/unit_velocity;
  
  columnGlob->ComputeColumn(hydro->Vc);
  tau = columnGlob->GetColumn();

  idefix_for("MySourceTerm",
    data->beg[KDIR], data->end[KDIR],
    data->beg[JDIR], data->end[JDIR],
    data->beg[IDIR], data->end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                // Constant kappa
                real Fim = std::exp(-kappa_irr*tau(k,j,i-1))*A1(k,j,i)/std::pow(x1l(i),2.);
                real Fip = std::exp(-kappa_irr*tau(k,j,i))*A1(k,j,i+1)/std::pow(x1l(i+1),2.);
                real divF = flux_pre*(Fip-Fim)/dV(k,j,i);
                //printf("divF=%e at i=%i and j=%i in MyUserDef\n",divF,i,j);

                //printf("divF/Uc(ENG,k,j,i)=%e Uc(ENG,k,j,i)=%e divF=%e at i %i j %i k %i\n",divF/Uc(ENG,k,j,i),Uc(ENG,k,j,i),divF,i,j,k);
                //if (FABS(divF/Uc(ENG,k,j,i)) > InvDt(k,j,i)) {
                //  printf("Add timestep constraint at i %i j %i k %i\n",i,j,k);
               // }
                //InvDt(k,j,i) = InvDt(k,j,i) + FABS(divF/Uc(ENG,k,j,i));
                
                //Uc(ENG,k,j,i) -= dt*divF;
  });
}

void FluxBoundary(Fluid<DefaultPhysics> *hydro, int dir, BoundarySide side, const real t) {
    idfx::pushRegion("FluxInternal");

    auto *data = hydro->data;

    IdefixArray4D<real> Flux = data->hydro->FluxRiemann;
    
    if( dir== IDIR ) {
      idefix_for("FluxInternal",
                  0, data->np_tot[KDIR],
                  0, data->np_tot[JDIR],
                  0, data->np_tot[IDIR],
         KOKKOS_LAMBDA (int k, int j, int i) {
           Flux(RHO, k, j, i) = 0.0; 
           Flux(MX1, k, j, i) = 0.0; 
           Flux(MX2, k, j, i) = 0.0; 
           Flux(MX3, k, j, i) = 0.0; 
           Flux(ENG, k, j, i) = 0.0; 
      });
    }

    if( dir== JDIR ) {
      idefix_for("FluxInternal",
                  0, data->np_tot[KDIR],
                  0, data->np_tot[JDIR],
                  0, data->np_tot[IDIR],
         KOKKOS_LAMBDA (int k, int j, int i) {
           Flux(RHO, k, j, i) = 0.0; 
           Flux(MX1, k, j, i) = 0.0; 
           Flux(MX2, k, j, i) = 0.0; 
           Flux(MX3, k, j, i) = 0.0; 
           Flux(ENG, k, j, i) = 0.0; 
      });
    }

    if( dir== KDIR ) {
      idefix_for("FluxInternal",
                  0, data->np_tot[KDIR],
                  0, data->np_tot[JDIR],
                  0, data->np_tot[IDIR],
         KOKKOS_LAMBDA (int k, int j, int i) {
           Flux(RHO, k, j, i) = 0.0; 
           Flux(MX1, k, j, i) = 0.0; 
           Flux(MX2, k, j, i) = 0.0; 
           Flux(MX3, k, j, i) = 0.0; 
           Flux(ENG, k, j, i) = 0.0; 
      });
    }

    idfx::popRegion();

}

void InternalBoundary(Hydro *hydro, const real t) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray1D<real> x1=data->x[IDIR];
  IdefixArray1D<real> x2=data->x[JDIR];

  real R0 = R0Glob;
  real h0 = h0Glob;
  real hpow = hpowGlob;
  real rho0 = rho0Glob;
  real rhomin = rhominGlob;
  real unit_density = idfx::units.density;


    idefix_for("InternalBoundary",
      0, data->np_tot[KDIR],
      0, data->np_tot[JDIR],
      0, data->np_tot[IDIR],
            KOKKOS_LAMBDA (int k, int j, int i) {
              
              real R = FMAX(x1(i)*std::sin(x2(j)),1.);
              real z2 = std::pow(x1(i)*std::cos(x2(j)),2.);
              real H = h0*std::pow(R/R0,hpow);

              Vc(RHO,k,j,i) = (rho0*(R0/R)*std::exp(-0.25*M_PI*z2/(H*H))+rhomin)/unit_density;
              Vc(VX1,k,j,i) = 0.;
              Vc(VX2,k,j,i) = 0.;
              Vc(VX3,k,j,i) = 0.;
            });

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
 
  IdefixArray3D<real> tau;
  IdefixArray4D<real> Vc=(&data)->hydro->Vc;

  IdefixHostArray3D<real> divF  = variables["divF"];
  IdefixHostArray3D<real> A1_out  = variables["A1"];
 
  real unit_density = idfx::units.density;
  real unit_velocity = idfx::units.velocity;
  real unit_length = idfx::units.length;
  real unit_energy = unit_density*unit_velocity*unit_velocity;

  real rs = rsGlob;
  real Ts = TsGlob;
  real kappa_irr = kappairrGlob;

  real kappa_rad = kappa_irr*unit_density*unit_length;
  real flux_pre = std::pow(rs/unit_length,2.)*idfx::units.sigma_sb*std::pow(Ts,4.)/unit_energy/unit_velocity;

  // Make references to the user-defined arrays (variables is a container of IdefixHostArray3D)
  // Note that the labels should match the variable names in the input file

  columnGlob->ComputeColumn(Vc);
  tau = columnGlob->GetColumn();
  Kokkos::deep_copy(variables["tau"], tau);
  Kokkos::deep_copy(variables["dV"], dV);

  for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
    for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
      for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {
        A1_out(k,j,i) = A1(k,j,i);

        real Fim = std::exp(-kappa_rad*variables["tau"](k,j,i-1))/std::pow(x1l(i),2.);
        real Fip = std::exp(-kappa_rad*variables["tau"](k,j,i))/std::pow(x1l(i+1),2.);

        divF(k,j,i) = (Fip*A1(k,j,i+1)-Fim*A1(k,j,i));
        divF(k,j,i) *= flux_pre;
        divF(k,j,i) /= dV(k,j,i);

      }
    }
  }
  Kokkos::deep_copy(variables["divF"], divF);
  Kokkos::deep_copy(variables["A1"], A1_out);

}


// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)
{ 
  // Mirror data on Host
  R0Glob = input.Get<real>("Setup","R0",0);
  h0Glob = input.Get<real>("Setup","h0",0);
  hpowGlob = input.Get<real>("Setup","hpow",0);
  rho0Glob = input.Get<real>("Setup","rho0",0);
  rhominGlob = input.Get<real>("Setup","rhomin",0);
  T0Glob = input.Get<real>("Setup","T0",0);
  muGlob = input.Get<real>("Hydro","mu",0);
  gammaGlob=data.hydro->eos->GetGamma();
  rsGlob=input.Get<real>("Setup","rs",0);
  TsGlob=input.Get<real>("Setup","Ts",0);
  kappairrGlob = input.Get<real>("Setup","kappa_irr",0);

  columnGlob = new Column(IDIR,1,RHO,&data);

  data.hydro->EnrollInternalBoundary(&InternalBoundary);
  //data.hydro->EnrollUserSourceTerm(&MySourceTerm);
  data.hydro->EnrollFluxBoundary(&FluxBoundary);
  output.EnrollUserDefVariables(&ComputeUserVars);
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

    real C_kb = idfx::units.k_B;
    real C_ar = idfx::units.ar;
    real C_amu = idfx::units.u;

    real unit_density = idfx::units.density;
    real unit_velocity = idfx::units.velocity;
    real unit_length = idfx::units.length;
    real unit_time = unit_length/unit_velocity;
    real unit_energy = unit_density*unit_velocity*unit_velocity;

    real R0 = R0Glob;
    real h0 = h0Glob;
    real hpow = hpowGlob;
    real rho0 = rho0Glob;
    real rhomin = rhominGlob;
    real T0 = T0Glob;
    real mu = muGlob;
    

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
              
              real R = FMAX(d.x[IDIR](i)*std::sin(d.x[JDIR](j)),1.);
              real z2 = std::pow(d.x[IDIR](i)*std::cos(d.x[JDIR](j)),2.);
              real H = h0*std::pow(R/R0,hpow);
              real T = T0;

              d.Vc(RHO,k,j,i) = (rho0*(R0/R)*std::exp(-0.25*M_PI*z2/(H*H))+rhomin)/unit_density;
              d.Vc(PRS,k,j,i) = d.Vc(RHO,k,j,i)*unit_density*C_kb*T/(mu*C_amu)/unit_energy;
              d.Vc(VX1,k,j,i) = 0.;
              d.Vc(VX2,k,j,i) = 0.;
              d.Vc(VX3,k,j,i) = 0.;
              d.RadVc[0](ER,k,j,i) = C_ar*std::pow(T,4.)/unit_energy;
              d.RadVc[0](FR1,k,j,i) = ZERO_F;
              d.RadVc[0](FR2,k,j,i) = ZERO_F;
              d.RadVc[0](FR3,k,j,i) = ZERO_F;

              d.RadUc[0](ER,k,j,i) = C_ar*std::pow(T,4.)/unit_energy;
              d.RadUc[0](FR1,k,j,i) = ZERO_F;
              d.RadUc[0](FR2,k,j,i) = ZERO_F;
              d.RadUc[0](FR3,k,j,i) = ZERO_F;
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

