#include "idefix.hpp"
#include "setup.hpp"

real unit_velocity;
real unit_length;
real unit_density;
real rho_0;
real T_0;


void UserdefBoundaryRad(Fluid<RadiationPhysics> *radiation, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = radiation->Vc;
  auto *data = radiation->data;

  real C_c = 29979245800.0;
  real C_ar = 7.5646e-15;
  real T = 1740.;

  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  if(dir==IDIR) {
    int ighost,ibeg,iend;
    if(side == left) {
      ighost = data->beg[IDIR];
      ibeg = 0;
      iend = data->beg[IDIR];
      idefix_for("UserDefBoundaryRad",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) {
          Vc(ER,k,j,i) = C_ar*std::pow(T,4);
          Vc(FR1,k,j,i) = Vc(ER,k,j,i);
          Vc(FR2,k,j,i) = 0.;
        });
    }
  }
}

void UserdefBoundary(Fluid<DefaultPhysics> *hydro, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = hydro->Vc;
  auto *data = hydro->data;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  if(dir==IDIR) {
    int ighost,ibeg,iend;
    if(side == left) {
      ighost = data->beg[IDIR];
      ibeg = 0;
      iend = data->beg[IDIR];
      idefix_for("UserDefBoundary",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) {
          Vc(RHO,k,j,i) = Vc(RHO,k,j,2*ighost-i-1);
          Vc(PRS,k,j,i) = Vc(PRS,k,j,2*ighost-i-1);
          Vc(VX1,k,j,i) = -Vc(VX1,k,j,2*ighost-i-1);
          Vc(VX2,k,j,i) = -Vc(VX2,k,j,2*ighost-i-1);
        });
    }
  }
}

// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)
{

  unit_velocity = input.Get<real>("Units","velocity",0);
  unit_length = input.Get<real>("Units","length",0);
  unit_density = input.Get<real>("Units","density",0);
  rho_0 = input.Get<real>("Rad","kappa",2);
  T_0 = input.Get<real>("Rad","kappa",3);

  //output.EnrollUserDefVariables(&ComputeUserVars);
  // Set the function for userdefboundary
  if(data.haveRadiation) {
    int nFrequencies = data.radiation.size();
    for(int n = 0 ; n < nFrequencies ; n++) {
      data.radiation[n]->EnrollUserDefBoundary(&UserdefBoundaryRad);
    }
    data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
    //data.hydro->EnrollInternalBoundary(&InternalBoundary);

  }
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);
    real C_kb = 1.38e-16;
    real C_mp = 1.6726e-24;
    real C_ar = 7.5646e-15;
    real mu = 1.;
    real KELVIN = C_kb/(C_mp*mu);
    real x02 = 0.01/(unit_length*unit_length);
    real y02 = 0.0036/(unit_length*unit_length);
    real rho_1 = 1.e3;


    real unit_time = unit_length/unit_velocity;
    real unit_energy = unit_density*unit_velocity*unit_velocity;

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
              real delta = 10.*(d.x[IDIR](i)*d.x[IDIR](i)/x02+d.x[JDIR](j)*d.x[JDIR](j)/y02-1.);
              d.Vc(RHO,k,j,i) = rho_0+(rho_1-rho_0)/(1.+std::exp(delta));
              d.Vc(PRS,k,j,i) = d.Vc(RHO,k,j,i)*unit_density*T_0*KELVIN/(unit_energy);
              d.Vc(VX1,k,j,i) = 0.;
              d.Vc(VX2,k,j,i) = 0.;
              d.RadVc[0](ER,k,j,i) = 4.*C_kb*std::pow(T_0,4)/unit_energy;
              d.RadVc[0](FR1,k,j,i) = ZERO_F;
              d.RadVc[0](FR2,k,j,i) = ZERO_F;
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



