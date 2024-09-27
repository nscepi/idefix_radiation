#include "idefix.hpp"
#include "setup.hpp"


real T0Glob;
real muGlob;
real rho0Glob;
real wGlob;

// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)
{
  T0Glob = input.Get<real>("Setup","T0",0);
  wGlob = input.Get<real>("Setup","w",0);
  rho0Glob = input.Get<real>("Setup","rho0",0);
  muGlob = input.Get<real>("Hydro","mu",0);
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
    real C_ar = idfx::units.ar;
    real mu = muGlob;
    real KELVIN = idfx::units.Kelvin*mu;
    real T0 = T0Glob;
    real rho0 = rho0Glob;
    real w = wGlob;
    real r2,T;

    real unit_density = idfx::units.density;
    real unit_velocity = idfx::units.velocity;
    real unit_energy = unit_density*unit_velocity*unit_velocity;

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
              if (GEOMETRY==CARTESIAN){
                r2 = d.x[IDIR](i)*d.x[IDIR](i)+d.x[JDIR](j)*d.x[JDIR](j)+d.x[KDIR](k)*d.x[KDIR](k);
              } else if (GEOMETRY==SPHERICAL) {
                r2 = d.x[IDIR](i)*d.x[IDIR](i);
              }
              d.Vc(RHO,k,j,i) = rho0;
              d.Vc(PRS,k,j,i) = d.Vc(RHO,k,j,i)*T0/KELVIN;
              d.Vc(VX1,k,j,i) = ZERO_F;
              d.Vc(VX2,k,j,i) = ZERO_F;
              d.Vc(VX3,k,j,i) = ZERO_F;
              T = T0*(1.+100.*std::exp(-r2/(w*w)));
              d.RadVc[0](ER,k,j,i) = C_ar*std::pow(T,4)/unit_energy;
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


