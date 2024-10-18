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

  real C_G = idfx::units.G;
  real C_Msol = idfx::units.M_sun;
  real C_au = idfx::units.au;
  real C_kb = idfx::units.k_B;
  real C_amu = idfx::units.u;
  real C_h = idfx::units.h;
  real C_c = idfx::units.c;

  real unit_density = idfx::units.density;
  real unit_velocity = idfx::units.velocity;
  real unit_length = idfx::units.length;
  real unit_time = unit_length/unit_velocity;
  real unit_energy = unit_density*unit_velocity*unit_velocity;

  real mu = muGlob;
  real T0 = T0Glob;
  real gamma = gammaGlob;
  real dt=dtin;
  real csiso = std::sqrt(C_kb*T0/(mu*C_amu));
  real rs = rsGlob;
  real Ts = TsGlob;
  real R0 = R0Glob;
  real h0 = h0Glob;
  real hpow = hpowGlob;

  
  // Usertable kappa
  real flux_pre = std::pow(rs/unit_length,2.)*idfx::units.sigma_sb*std::pow(Ts,4.)/unit_energy/unit_velocity;
  auto irr_flux = LookupTable<1>("irr_flux.dat",',');

  columnGlob->ComputeColumn(hydro->Vc);
  tau = columnGlob->GetColumn();

  idefix_for("MySourceTerm",
    data->beg[KDIR], data->end[KDIR],
    data->beg[JDIR], data->end[JDIR],
    data->beg[IDIR], data->end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                // Usertable kappa
                real logtaum = std::log10(FMAX(tau(k,j,i-1)*unit_density*unit_length,1.e-15));
                real Fim = pow(10.,irr_flux.Get(&logtaum))*A1(k,j,i)/std::pow(x1l(i),2.);
                real logtaup = std::log10(FMAX(tau(k,j,i)*unit_density*unit_length,1.e-15));
                real Fip = pow(10.,irr_flux.Get(&logtaup))*A1(k,j,i+1)/std::pow(x1l(i+1),2.);
                real divF = flux_pre*(Fip-Fim)/dV(k,j,i);

                //printf("divF/Uc(ENG,k,j,i)=%e Uc(ENG,k,j,i)=%e divF=%e at i %i j %i k %i\n",divF/Uc(ENG,k,j,i),Uc(ENG,k,j,i),divF,i,j,k);
                InvDt(k,j,i) = InvDt(k,j,i) + FABS(divF/Uc(ENG,k,j,i));
                
                Uc(ENG,k,j,i) -= dt*divF;
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

void UserdefBoundaryRad(Fluid<RadiationPhysics> *radiation, int dir, BoundarySide side, real t) {
  auto *data = radiation->data;

  
  if( (dir==IDIR) && (side == left)) {
        IdefixArray4D<real> Vc = radiation->Vc;

        int ighost = data->nghost[IDIR];
        radiation->boundary->BoundaryFor("UserDefX1Rad",dir,side,
            KOKKOS_LAMBDA (int k, int j, int i) {
                Vc(ER,k,j,i) = Vc(ER,k,j,ighost);
                if (Vc(FR1,k,j,i) > ZERO_F){
                  Vc(FR1,k,j,i) = ZERO_F;
                } else {
                  Vc(FR1,k,j,i) = Vc(FR1,k,j,ighost);
                }
                Vc(FR2,k,j,i) = Vc(FR2,k,j,ighost);
                Vc(FR3,k,j,i) = Vc(FR3,k,j,ighost);

            });
    }

    if( (dir==IDIR) && (side == right)) {
        IdefixArray4D<real> Vc = radiation->Vc;

        int ighost = data->end[IDIR]-1;
        radiation->boundary->BoundaryFor("UserDefX1Rad",dir,side,
            KOKKOS_LAMBDA (int k, int j, int i) {
                Vc(ER,k,j,i) = Vc(ER,k,j,ighost);
                if (Vc(FR1,k,j,i) < ZERO_F){
                  Vc(FR1,k,j,i) = ZERO_F;
                } else {
                  Vc(FR1,k,j,i) = Vc(FR1,k,j,ighost);
                }                
                Vc(FR2,k,j,i) = Vc(FR2,k,j,ighost);
                Vc(FR3,k,j,i) = Vc(FR3,k,j,ighost);

            });
    }

    if( (dir==JDIR) && (side == left)) {
        IdefixArray4D<real> Vc = radiation->Vc;

        int jref = data->beg[JDIR];
        int offset = -1;
        int jghost = data->nghost[JDIR];
        real unit_energy = idfx::units.density*idfx::units.velocity*idfx::units.velocity;
        real Tmin = T0Glob;

        radiation->boundary->BoundaryFor("UserDefX2Rad",dir,side,
            KOKKOS_LAMBDA (int k, int j, int i) {
                Vc(ER,k,j,i) = idfx::units.ar*std::pow(Tmin,4.)/unit_energy;
                Vc(FR1,k,j,i) = Vc(FR1,k,2*jref-j+offset,i);
                Vc(FR2,k,j,i) = -Vc(FR2,k,2*jref-j+offset,i);        
                Vc(FR3,k,j,i) = -Vc(FR3,k,2*jref-j+offset,i);

            });
    }

    if( (dir==JDIR) && (side == right)) {
        IdefixArray4D<real> Vc = radiation->Vc;

        int jref = data->end[JDIR]-1;
        int offset = 1;
        int jghost = data->end[JDIR]-1;
        real unit_energy = idfx::units.density*idfx::units.velocity*idfx::units.velocity;
        real Tmin = T0Glob;

        radiation->boundary->BoundaryFor("UserDefX2Rad",dir,side,
            KOKKOS_LAMBDA (int k, int j, int i) {
                Vc(ER,k,j,i) = idfx::units.ar*std::pow(Tmin,4.)/unit_energy;
                Vc(FR1,k,j,i) = Vc(FR1,k,2*jref-j,i);
                Vc(FR2,k,j,i) = -Vc(FR2,k,2*jref-j,i);                 
                Vc(FR3,k,j,i) = -Vc(FR3,k,2*jref-j,i);
            });
    }

}

// User-defined boundaries hydro
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
    auto *data = hydro->data;
    if( (dir==IDIR) && (side == left)) {
        IdefixArray4D<real> Vc = hydro->Vc;

        int ighost = data->nghost[IDIR];

        hydro->boundary->BoundaryFor("UserDefX1",dir,side,
            KOKKOS_LAMBDA (int k, int j, int i) {
                Vc(RHO,k,j,i) = Vc(RHO,k,j,ighost);
                Vc(PRS,k,j,i) = Vc(PRS,k,j,ighost);
                Vc(VX1,k,j,i) = ZERO_F;
                Vc(VX2,k,j,i) = ZERO_F;
                Vc(VX3,k,j,i) = ZERO_F;

            });
    }

    if( (dir==IDIR) && (side == right)) {
        IdefixArray4D<real> Vc = hydro->Vc;

        int ighost = data->end[IDIR]-1;

        hydro->boundary->BoundaryFor("UserDefX1",dir,side,
            KOKKOS_LAMBDA (int k, int j, int i) {
                Vc(RHO,k,j,i) = Vc(RHO,k,j,ighost);
                Vc(PRS,k,j,i) = Vc(PRS,k,j,ighost);
                Vc(VX1,k,j,i) = ZERO_F;
                Vc(VX2,k,j,i) = ZERO_F;
                Vc(VX3,k,j,i) = ZERO_F;

            });
    }

    if( (dir==JDIR) && (side == left)) {
        IdefixArray4D<real> Vc = hydro->Vc;

        int jghost = data->nghost[JDIR];

        hydro->boundary->BoundaryFor("UserDefX2",dir,side,
            KOKKOS_LAMBDA (int k, int j, int i) {
                Vc(RHO,k,j,i) = Vc(RHO,k,jghost,i);
                Vc(PRS,k,j,i) = Vc(PRS,k,jghost,i);
                Vc(VX1,k,j,i) = ZERO_F;
                Vc(VX2,k,j,i) = ZERO_F;
                Vc(VX3,k,j,i) = ZERO_F;

            });
    }

    if( (dir==JDIR) && (side == right)) {
        IdefixArray4D<real> Vc = hydro->Vc;

        int jghost = data->end[JDIR]-1;

        hydro->boundary->BoundaryFor("UserDefX2",dir,side,
            KOKKOS_LAMBDA (int k, int j, int i) {
                Vc(RHO,k,j,i) = Vc(RHO,k,jghost,i);
                Vc(PRS,k,j,i) = Vc(PRS,k,jghost,i);
                Vc(VX1,k,j,i) = ZERO_F;
                Vc(VX2,k,j,i) = ZERO_F;
                Vc(VX3,k,j,i) = ZERO_F;
            });
    }
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
 
  IdefixHostArray3D<real> tau;
  IdefixHostArray3D<real> divF  = variables["divF"];
  IdefixHostArray3D<real> A1_out  = variables["A1"];
  IdefixHostArray3D<real> irr  = variables["irr_flux"];
  IdefixHostArray3D<real> kappa_p  = variables["kappa_p"];
  IdefixHostArray3D<real> kappa_r  = variables["kappa_r"];

  real unit_density = idfx::units.density;
  real unit_velocity = idfx::units.velocity;
  real unit_length = idfx::units.length;
  real unit_energy = unit_density*unit_velocity*unit_velocity;
  real KELVIN = idfx::units.Kelvin;

  real mu = muGlob;
  real rs = rsGlob;
  real Ts = TsGlob;
  real R0 = R0Glob;
  real h0 = h0Glob;
  real hpow = hpowGlob;

  real flux_pre = std::pow(rs/unit_length,2.)*idfx::units.sigma_sb*std::pow(Ts,4.)/unit_energy/unit_velocity;
  auto irr_flux = LookupTable<1>("irr_flux.dat",',');
  auto kappa_planck = LookupTable<1>("kappa_p.dat",',');
  auto kappa_rosseland = LookupTable<1>("kappa_r.dat",',');

  // Make references to the user-defined arrays (variables is a container of IdefixHostArray3D)
  // Note that the labels should match the variable names in the input file

  columnGlob->ComputeColumn(data.hydro->Vc);
  Kokkos::deep_copy(variables["tau"], columnGlob->GetColumn());
  Kokkos::deep_copy(variables["dV"], dV);

  tau = columnGlob->GetColumn();
  for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
    for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
      for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {
        A1_out(k,j,i) = A1(k,j,i);

        real logtaum = std::log10(FMAX(tau(k,j,i-1)*unit_density*unit_length,1.e-15));
        real Fim = std::pow(10.,irr_flux.Get(&logtaum))/std::pow(x1l(i),2.);
        //printf("Fim=%e at i %i j %i k %i\n",Fim,i,j,k);
        real logtaup = std::log10(FMAX(tau(k,j,i)*unit_density*unit_length,1.e-15));
        real Fip = std::pow(10.,irr_flux.Get(&logtaup))/std::pow(x1l(i+1),2.);
        //printf("Fip=%e at i %i j %i k %i\n",Fip,i,j,k);

        divF(k,j,i) = (Fip*A1(k,j,i+1)-Fim*A1(k,j,i));
        divF(k,j,i) *= flux_pre;
        divF(k,j,i) /= dV(k,j,i);

        irr(k,j,i) = irr_flux.Get(&logtaum);
        
        real T = data.hydro->Vc(PRS,k,j,i)/(data.hydro->Vc(RHO,k,j,i))*KELVIN*mu;
        real logT = std::log10(T);

        kappa_p(k,j,i) = kappa_planck.Get(&logT);
        kappa_r(k,j,i) = kappa_rosseland.Get(&logT);
      }
    }
  }
  Kokkos::deep_copy(variables["divF"], divF);
  Kokkos::deep_copy(variables["A1"], A1_out);
  Kokkos::deep_copy(variables["irr_flux"], irr);
  Kokkos::deep_copy(variables["kappa_p"], kappa_p);
  Kokkos::deep_copy(variables["kappa_r"], kappa_r);
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
  //kappaGlob = input.Get<real>("Rad","kappa",1);

  columnGlob = new Column(IDIR,1,RHO,&data);

  //data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  data.hydro->EnrollInternalBoundary(&InternalBoundary);
  data.hydro->EnrollUserSourceTerm(&MySourceTerm);
  data.hydro->EnrollFluxBoundary(&FluxBoundary);
  output.EnrollUserDefVariables(&ComputeUserVars);

  // Set the function for userdefboundary
  if(data.haveRadiation) {
    int nFrequencies = data.radiation.size();
    for(int n = 0 ; n < nFrequencies ; n++) {
      //data.radiation[n]->EnrollUserDefBoundary(&UserdefBoundaryRad);
    }
  }
  //data.hydro->haveSourceTerms=false;
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

    int direction = IDIR;
    int sign = 0;
    int variable = RHO;
    
    real C_au = idfx::units.au;
    real C_G = idfx::units.G;
    real C_Msol = idfx::units.M_sun;
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
    real gamma = gammaGlob;
    

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

