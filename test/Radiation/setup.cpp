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

                const real r0 = 5.;
                const real center = pow(d.x[IDIR](i)-r0,2)+pow(d.x[JDIR](j)-r0,2);

                d.Vc(RHO,k,j,i) = 1.;
                d.Vc(VX1,k,j,i) = 2.;
                d.Vc(VX2,k,j,i) = 3.;
                d.RadVc[0](ER,k,j,i) = 4.*exp(-center/2.)+0.1;
                d.RadVc[0](FR1,k,j,i) = 5.*exp(-center/5.)+0.1;
                d.RadVc[0](FR2,k,j,i) = 6.;
                d.RadUc[0](ER,k,j,i) = 10.*exp(-center/2.)+0.1;
                d.RadUc[0](FR1,k,j,i) = 11.*exp(-center/5.)+0.1;
                d.RadUc[0](FR2,k,j,i) = 12.;
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
  IdefixHostArray3D<real> Er = variables["ER"];
  IdefixHostArray3D<real> FrX = variables["FRX"];
  IdefixHostArray3D<real> FrY = variables["FRY"];

  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        Er(k,j,i) = d.RadUc[0](ER,k,j,i);
        FrX(k,j,i) = d.RadUc[0](FR1,k,j,i);
        FrY(k,j,i) = d.RadUc[0](FR2,k,j,i);
      }
    }
  }
}


Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output)// : m_planet(0)//, Planet &planet)
{
  output.EnrollUserDefVariables(&ComputeUserVars);
}