#include "idefix.hpp"
#include "setup.hpp"


// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {

              if (d.x[IDIR](i) < -5.){
                d.Vc(RHO,k,j,i) = 5.99924;
                d.Vc(VX1,k,j,i) = 19.5975;
                d.Vc(VX2,k,j,i) = 0.;
                d.Vc(PRS,k,j,i) = 460.894;
              } else {
                d.Vc(RHO,k,j,i) = 5.99242;
                d.Vc(VX1,k,j,i) = -6.19633;
                d.Vc(VX2,k,j,i) = 0.;
                d.Vc(PRS,k,j,i) = 46.0950;

              }
        
              if (d.x[IDIR](i) <= 0.){
                  d.RadVc[0](ER,k,j,i) = 0.1;
                  d.RadVc[0](FR1,k,j,i) = 0.1;
                  d.RadVc[0](FR2,k,j,i) = ZERO_F;
              } else {
                d.RadVc[0](ER,k,j,i) = 1.;
                d.RadVc[0](FR1,k,j,i) = ZERO_F;
                d.RadVc[0](FR2,k,j,i) = 1.;
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
  IdefixHostArray3D<real> flag = variables["flag"];

  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
          //flag = d.shockFlattening->flagArray(k,j,i);
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

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)
{
  //output.EnrollUserDefVariables(&ComputeUserVars);
  // Set the function for userdefboundary
  if(data.haveRadiation) {
    int nFrequencies = data.radiation.size();
    //for(int n = 0 ; n < nFrequencies ; n++) {
    //  data.radiation[n]->EnrollUserDefBoundary(&UserdefBoundaryRad);
    //}
    //data.hydro->EnrollUserDefBoundary(&UserdefBoundary);

  }
}