#include "idefix.hpp"
#include "setup.hpp"

real rho0Glob,vx1Glob,T0Glob,muGlob;

// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)
{
  rho0Glob = input.Get<real>("Setup","rho0",0);
  vx1Glob = input.Get<real>("Setup","vx1",0);
  T0Glob = input.Get<real>("Setup","T0",0);
  muGlob = input.Get<real>("Hydro","mu",0);


}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);

    real T = T0Glob;
    real rho0 = rho0Glob;
    real vx1 = vx1Glob;
    real mu = muGlob;

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
              d.Vc(RHO,k,j,i) = rho0/idfx::units.GetDensity();
              d.Vc(PRS,k,j,i) = d.Vc(RHO,k,j,i)*idfx::units.GetDensity()*idfx::units.k_B*T/(mu*idfx::units.u*idfx::units.GetEnergy());
              d.Vc(VX1,k,j,i) = vx1/idfx::units.GetVelocity();
              d.RadVc[0](ER,k,j,i) = idfx::units.ar*std::pow(T,4)/(idfx::units.GetEnergy());
              d.RadVc[0](FR1,k,j,i) = ZERO_F;
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
