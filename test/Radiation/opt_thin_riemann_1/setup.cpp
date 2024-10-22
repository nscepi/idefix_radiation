#include "idefix.hpp"
#include "setup.hpp"


real csisoGlob;
real ERLGlob, ERRGlob,FR1LGlob,FR1RGlob,FR2LGlob,FR2RGlob,RhoGlob,VX1Glob,VX2Glob;

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)
{
  csisoGlob = input.Get<real>("Hydro","csiso",1);
  ERLGlob = input.Get<real>("Setup","ERL",0);
  ERRGlob = input.Get<real>("Setup","ERR",0);
  FR1LGlob = input.Get<real>("Setup","FR1L",0);
  FR1RGlob = input.Get<real>("Setup","FR1R",0);
  FR2LGlob = input.Get<real>("Setup","FR2L",0);
  FR2RGlob = input.Get<real>("Setup","FR2R",0);
  RhoGlob = input.Get<real>("Setup","RHO",0);
  VX1Glob = input.Get<real>("Setup","VX1",0);
  VX2Glob = input.Get<real>("Setup","VX2",0);

}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);

    real ERL = ERLGlob;
    real ERR= ERRGlob;
    real FR1L = FR1LGlob;
    real FR1R= FR1RGlob;
    real FR2L = FR2LGlob;
    real FR2R= FR2RGlob;
    real csiso = csisoGlob;
    real rho0 = RhoGlob;
    real vx1_0 = VX1Glob;
    real vx2_0 = VX2Glob;
    
    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
              
              d.Vc(RHO,k,j,i) = rho0;
              d.Vc(VX1,k,j,i) = vx1_0;
              d.Vc(VX2,k,j,i) = vx2_0;
              d.Vc(PRS,k,j,i) = rho0*csiso*csiso;
        
              if (d.x[IDIR](i) < 0.){
                  d.RadVc[0](ER,k,j,i) = ERL;
                  d.RadVc[0](FR1,k,j,i) = FR1L;
                  d.RadVc[0](FR2,k,j,i) = FR2L;
              } else {
                d.RadVc[0](ER,k,j,i) = ERR;
                d.RadVc[0](FR1,k,j,i) = FR1R;
                d.RadVc[0](FR2,k,j,i) = FR2R;
              }
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



