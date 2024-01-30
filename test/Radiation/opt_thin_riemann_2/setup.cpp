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

              if (d.x[IDIR](i) < 0.){
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
        
              if (d.x[IDIR](i) < 0.){
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
  IdefixHostArray3D<real> lambda1 = variables["lambda1"];
  IdefixHostArray3D<real> lambda2 = variables["lambda2"];
  IdefixHostArray3D<real> lambda3 = variables["lambda3"];
  IdefixHostArray3D<real> f = variables["f"];
  IdefixHostArray3D<real> cos = variables["cos"];

  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        
        real v[4];
        real Fnorm = std::sqrt(EXPAND(d.RadVc[0](FR1,k,j,i)*d.RadVc[0](FR1,k,j,i) , + d.RadVc[0](FR2,k,j,i)*d.RadVc[0](FR2,k,j,i), + d.RadVc[0](FR3,k,j,i)*d.RadVc[0](FR3,k,j,i)));

        //if (Fnorm>=d.RadVc[0](ER,k,j,i)) {
        //  v[FR1] *= (Fnorm == ZERO_F ? ZERO_F : d.RadVc[0](ER,k,j,i)/Fnorm); 
        //  v[FR2] *= (Fnorm == ZERO_F ? ZERO_F : d.RadVc[0](ER,k,j,i)/Fnorm);
        //  Fnorm = std::sqrt(EXPAND(v[FR1]*v[FR1] , + v[FR2]*v[FR2], + v[FR3]*v[FR3]));
        //}

        // 2-- Get the wave speed
        real f_param = Fnorm/d.RadVc[0](ER,k,j,i);
        real f2_param = f_param*f_param;
      
        real cos_theta = (Fnorm <= 1.e-5 ? ZERO_F : d.RadVc[0](FR1,k,j,i) / Fnorm);
        real lambda[3];

        K_speeds_Rad(lambda,f_param, f2_param, cos_theta);


        f(k,j,i) = f_param;
        cos(k,j,i) = cos_theta;
        lambda1(k,j,i) = lambda[0];
        lambda2(k,j,i) = (Fnorm <= 1.e-5 ? ZERO_F : lambda[1]);
        lambda3(k,j,i) = lambda[2];
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
  output.EnrollUserDefVariables(&ComputeUserVars);
  // Set the function for userdefboundary
  if(data.haveRadiation) {
    int nFrequencies = data.radiation.size();
    //for(int n = 0 ; n < nFrequencies ; n++) {
    //  data.radiation[n]->EnrollUserDefBoundary(&UserdefBoundaryRad);
    //}
    //data.hydro->EnrollUserDefBoundary(&UserdefBoundary);

  }
}