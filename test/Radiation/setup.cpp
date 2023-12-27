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
                //d.RadVc[0](ER,k,j,i) = 4.*exp(-center/2.)+0.1;
                //d.RadVc[0](FR1,k,j,i) =  2.*exp(-center/2.)+0.1;
                //d.RadVc[0](FR2,k,j,i) = 2.*exp(-center/2.)+0.1;

                if (std::abs(d.x[IDIR](i)-3.) < 0.5 && std::abs(d.x[JDIR](j)-3.) < 0.5){
                  d.RadVc[0](ER,k,j,i) = 1.e6;
                  d.RadVc[0](FR1,k,j,i) = d.RadVc[0](ER,k,j,i)*cos(M_PI/4.);
                  d.RadVc[0](FR2,k,j,i) = d.RadVc[0](ER,k,j,i)*sin(M_PI/4.);
                  //d.RadVc[0](FR1,k,j,i) = 1.;
                  //d.RadVc[0](FR2,k,j,i) = 1.;
                  //printf("cos(M_PI_4) - sin(M_PI_4)**2=%.25f\n",cos(M_PI_4)-sin(M_PI_4));
                  //real Fnorm = std::sqrt(d.RadVc[0](FR1,k,j,i)*d.RadVc[0](FR1,k,j,i) + d.RadVc[0](FR2,k,j,i)*d.RadVc[0](FR2,k,j,i));
                  //printf("Fnorm=%.20f, FR1=%.20f FR2=%.20f\n",Fnorm,d.RadVc[0](FR1,k,j,i),d.RadVc[0](FR2,k,j,i));
                  // if (Fnorm >= d.RadVc[0](ER,k,j,i)){
                  //   d.RadVc[0](FR1,k,j,i) = 0.;
                  //   d.RadVc[0](FR2,k,j,i) *= 0.9999999*d.RadVc[0](ER,k,j,i)/Fnorm;
                  //   Fnorm = std::sqrt(d.RadVc[0](FR1,k,j,i)*d.RadVc[0](FR1,k,j,i) + d.RadVc[0](FR2,k,j,i)*d.RadVc[0](FR2,k,j,i));
                  //   printf("Fnorm_new=%.20f, FR1_new =%.20f FR2_new=%.20f\n",Fnorm,d.RadVc[0](FR1,k,j,i),d.RadVc[0](FR2,k,j,i));
                  // }
                  //if (d.RadVc[0](FR1,k,j,i) > d.RadVc[0](FR2,k,j,i)){

                  //}
                } else {
                  d.RadVc[0](ER,k,j,i) = 1.e2;
                  d.RadVc[0](FR1,k,j,i) = 1.e-10;
                  d.RadVc[0](FR2,k,j,i) = 1.e-10;
                }
                // d.RadVc[0](ER,k,j,i) = 1.e1;
                // d.RadVc[0](FR1,k,j,i) = 0.0;
                // d.RadVc[0](FR2,k,j,i) = 0.0;
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
        Er(k,j,i) = d.RadFlux[0](ER,k,j,i);
        FrX(k,j,i) = d.RadFlux[0](FR1,k,j,i);
        FrY(k,j,i) = d.RadFlux[0](FR2,k,j,i);
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
          //if (std::abs(x2(j)-3.) < 0.5){
            Vc(ER,k,j,i) = 1.e6;
            Vc(FR1,k,j,i) = 1.e6;
            Vc(FR2,k,j,i) = 0.;
          //} else {
          //  Vc(ER,k,j,i) = 1.e4;
          //  Vc(FR1,k,j,i) = 0.;
          //  Vc(FR2,k,j,i) = 0.;
          //}
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
  output.EnrollUserDefVariables(&ComputeUserVars);
  // Set the function for userdefboundary
  if(data.haveRadiation) {
    int nFrequencies = data.radiation.size();
    for(int n = 0 ; n < nFrequencies ; n++) {
      data.radiation[n]->EnrollUserDefBoundary(&UserdefBoundaryRad);
    }
    data.hydro->EnrollUserDefBoundary(&UserdefBoundary);

  }
}