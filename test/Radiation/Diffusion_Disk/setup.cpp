#include "idefix.hpp"
#include "setup.hpp"

real epsilonGlob;
real alphaGlob;
real RGlob;
real rho0Glob;
real rhominGlob;
real T0Glob;
real muGlob;
real gammaGlob;

void MySourceTerm(Hydro *hydro, const real t, const real dtin) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray1D<real> x1=data->x[IDIR];
  IdefixArray1D<real> x2=data->x[JDIR];

  real C_G = idfx::units.G;
  real C_Msol = idfx::units.M_sun;
  real C_au = idfx::units.au;
  real C_kb = idfx::units.k_B;
  real C_amu = idfx::units.u;

  real unit_density = idfx::units.density;
  real unit_velocity = idfx::units.velocity;
  real unit_length = idfx::units.length;
  real unit_time = unit_length/unit_velocity;
  real unit_energy = unit_density*unit_velocity*unit_velocity;

  real epsilon = epsilonGlob;
  real alpha = alphaGlob;
  real R = RGlob*C_au;
  real mu = muGlob;
  real T0 = T0Glob;
  real gamma = gammaGlob;
  real dt=dtin;
  real Omega_K = std::sqrt(C_G*C_Msol/(R*R*R));
  //real csiso = epsilon*R*Omega_K;
  real csiso = std::sqrt(C_kb*T0/(mu*C_amu));

  idefix_for("MySourceTerm",
    0, data->np_tot[KDIR],
    0, data->np_tot[JDIR],
    0, data->np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
              
                real SE = 2.25*alpha*Omega_K*csiso*csiso*Vc(RHO,k,j,i)*unit_density/(unit_energy/unit_time);

                Uc(ENG,k,j,i) += dt*SE;

});
}


void UserdefBoundaryRad(Fluid<RadiationPhysics> *radiation, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = radiation->Vc;
  auto *data = radiation->data;

  real C_c = idfx::units.c;
  real C_ar = idfx::units.ar;
  real T0 = T0Glob;

  real unit_density = idfx::units.density;
  real unit_velocity = idfx::units.velocity;
  real unit_length = idfx::units.length;
  real unit_time = unit_length/unit_velocity;
  real unit_energy = unit_density*unit_velocity*unit_velocity;


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
          Vc(ER,k,j,i) = C_ar*std::pow(T0,4.)/unit_energy;
          Vc(FR1,k,j,i) = Vc(FR1,k,j,ighost);
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
          Vc(ER,k,j,i) = C_ar*std::pow(T0,4.)/unit_energy;
          Vc(FR1,k,j,i) = Vc(FR1,k,j,ighost+nxi-1);
        });
    }
  }
}

void UserdefBoundary(Fluid<DefaultPhysics> *hydro, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = hydro->Vc;
  auto *data = hydro->data;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];

  real rhomin = rhominGlob;

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
          Vc(RHO,k,j,i) = Vc(RHO,k,j,ighost);
          Vc(PRS,k,j,i) = Vc(PRS,k,j,ighost);
          Vc(VX1,k,j,i) = Vc(VX1,k,j,ighost);
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
        });
    }
  }
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
           Flux(ENG, k, j, i) = 0.0; 
      });
    }
    idfx::popRegion();

}

void FluxBoundaryOld(DataBlock & data, int dir, BoundarySide side, const real t) {
    idfx::pushRegion("FluxInternal");

    IdefixArray4D<real> Flux = data.hydro->FluxRiemann;
    
    if( dir== IDIR ) {
      idefix_for("FluxInternal",
                  0, data.np_tot[KDIR],
                  0, data.np_tot[JDIR],
                  0, data.np_tot[IDIR],
         KOKKOS_LAMBDA (int k, int j, int i) {
           Flux(RHO, k, j, i) = 0.0; 
           Flux(MX1, k, j, i) = 0.0; 
           Flux(ENG, k, j, i) = 0.0; 
      });
    }
    idfx::popRegion();

}

// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)
{ 
  alphaGlob = input.Get<real>("Setup","alpha",0);
  epsilonGlob = input.Get<real>("Setup","epsilon",0);
  RGlob = input.Get<real>("Setup","R",0);
  rho0Glob = input.Get<real>("Setup","rho0",0);
  rhominGlob = input.Get<real>("Setup","rhomin",0);
  T0Glob = input.Get<real>("Setup","T0",0);
  gammaGlob=data.hydro->eos->GetGamma();
  muGlob=data.hydro->eos->GetMu();

  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  data.hydro->EnrollUserSourceTerm(&MySourceTerm);
  data.hydro->EnrollFluxBoundary(&FluxBoundary);

  // Set the function for userdefboundary
  if(data.haveRadiation) {
    int nFrequencies = data.radiation.size();
    for(int n = 0 ; n < nFrequencies ; n++) {
      data.radiation[n]->EnrollUserDefBoundary(&UserdefBoundaryRad);
    }
  }
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);
    
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

    real epsilon = epsilonGlob;
    real R = RGlob*C_au;
    real rho0 = rho0Glob;
    real rhomin = rhominGlob;
    real T0 = T0Glob;
    real mu = muGlob;
    real gamma = gammaGlob;
    real Omega_K = std::pow(C_G*C_Msol/(R*R*R),0.5);
    real csiso = epsilon*R*Omega_K;
    real H2 = std::pow(epsilon*R,2.);

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {

              real x2 = std::pow(d.x[IDIR](i)*unit_length,2.);

              d.Vc(RHO,k,j,i) = (rho0*std::exp(-0.5*x2/H2)+rhomin)/unit_density;
              d.Vc(PRS,k,j,i) = d.Vc(RHO,k,j,i)*unit_density*C_kb*T0/(mu*C_amu)/unit_energy;
              d.Vc(VX1,k,j,i) = 0.;
              d.Vc(VX2,k,j,i) = 0.;
              d.Vc(VX3,k,j,i) = 0.;
              d.RadVc[0](ER,k,j,i) = C_ar*std::pow(T0,4.)/unit_energy;
              d.RadVc[0](FR1,k,j,i) = ZERO_F;
              d.RadVc[0](FR2,k,j,i) = ZERO_F;
              d.RadVc[0](FR3,k,j,i) = ZERO_F;
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}


// Compute user variables which will be written in vtk files
void ComputeUserVars(DataBlock & data, UserDefVariablesContainer &variables) {
  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  // Make references to the user-defined arrays (variables is a container of IdefixHostArray3D)
  // Note that the labels should match the variable names in the input file

  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        
      }
    }
  }
}






