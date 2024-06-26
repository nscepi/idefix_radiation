#include "idefix.hpp"
#include "setup.hpp"

real unit_velocity;
real unit_length;
real unit_density;


// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)
{

  unit_velocity = input.Get<real>("Units","velocity",0);
  unit_length = input.Get<real>("Units","length",0);
  unit_density = input.Get<real>("Units","density",0);

  //output.EnrollUserDefVariables(&ComputeUserVars);
  // Set the function for userdefboundary
  if(data.haveRadiation) {
    int nFrequencies = data.radiation.size();
    //for(int n = 0 ; n < nFrequencies ; n++) {
    //  data.radiation[n]->EnrollUserDefBoundary(&UserdefBoundaryRad);
    //}
    //data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
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
    real T0 = 1.e4;
    real T;
    real w = 5.;

    real unit_time = unit_length/unit_velocity;
    real unit_energy = unit_density*unit_velocity*unit_velocity;

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
              real r2 = d.x[IDIR](i)*d.x[IDIR](i)+d.x[JDIR](j)*d.x[JDIR](j)+d.x[KDIR](k)*d.x[KDIR](k);
              d.Vc(RHO,k,j,i) = 1.;
              d.Vc(PRS,k,j,i) = d.Vc(RHO,k,j,i)*unit_density*T0*KELVIN/(unit_energy);
              d.Vc(VX1,k,j,i) = 0.;
              d.Vc(VX2,k,j,i) = 0.;
              d.Vc(VX3,k,j,i) = 0.;
              T = T0*(1.+100.*std::exp(-r2/(w*w)));
              d.RadVc[0](ER,k,j,i) = 4.*C_kb*std::pow(T,4)/unit_energy;
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

void UserdefBoundaryRad(Fluid<RadiationPhysics> *radiation, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = radiation->Vc;
  auto *data = radiation->data;
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
          Vc(ER,k,j,i) = 1.e6;
          Vc(FR1,k,j,i) = 1.e6;
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
          Vc(RHO,k,j,i) = 1.;
          Vc(VX1,k,j,i) = 0.;
          Vc(VX2,k,j,i) = 0.;
        });
    }
  }
}


