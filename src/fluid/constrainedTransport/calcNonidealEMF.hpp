// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef FLUID_CONSTRAINEDTRANSPORT_CALCNONIDEALEMF_HPP_
#define FLUID_CONSTRAINEDTRANSPORT_CALCNONIDEALEMF_HPP_

#include "fluid.hpp"
#include "dataBlock.hpp"

// Compute Corner EMFs from nonideal MHD
template<typename Phys>
void ConstrainedTransport<Phys>::CalcNonidealEMF(real t) {
  idfx::pushRegion("ConstrainedTransport::CalcNonidealEMF");

#if MHD == YES
  // Corned EMFs
  IdefixArray3D<real> ex = this->ex;
  IdefixArray3D<real> ey = this->ey;
  IdefixArray3D<real> ez = this->ez;
  IdefixArray4D<real> J = hydro->J;
  IdefixArray4D<real> Jperp = hydro->Jperp;
  IdefixArray4D<real> Vs = hydro->Vs;
  IdefixArray4D<real> Vc = hydro->Vc;

  // These arrays have been previously computed in calcParabolicFlux
  IdefixArray3D<real> etaArr = hydro->etaOhmic;
  IdefixArray3D<real> xAmbiArr = hydro->xAmbipolar;

  // these two are required to ensure that the type is captured by KOKKOS_LAMBDA
  HydroModuleStatus resistivity = hydro->resistivityStatus.status;
  HydroModuleStatus ambipolar = hydro->ambipolarStatus.status;

  bool haveResistivity{false};
  bool haveAmbipolar{false};

  if(data->rklCycle) {
    haveResistivity = hydro->resistivityStatus.isRKL;
    haveAmbipolar = hydro->ambipolarStatus.isRKL;
  } else {
    haveResistivity = hydro->resistivityStatus.isExplicit;
    haveAmbipolar = hydro->ambipolarStatus.isExplicit;
  }

  real etaConstant = hydro->etaO;
  real xAConstant = hydro->xA;

  idefix_for("CalcNIEMF",
             data->beg[KDIR],data->end[KDIR]+KOFFSET,
             data->beg[JDIR],data->end[JDIR]+JOFFSET,
             data->beg[IDIR],data->end[IDIR]+IOFFSET,
    KOKKOS_LAMBDA (int k, int j, int i) {
      real Bx1, Bx2, Bx3;
      real Jx1, Jx2, Jx3;
      real eta, xA;
      // CT_EMF_ArithmeticAverage (emf, 0.25);

      if(resistivity == Constant)
        eta = etaConstant;
      if(ambipolar == Constant)
        xA = xAConstant;

  #if DIMENSIONS == 3
      // -----------------------
      // X1 EMF Component
      // -----------------------
      Jx1 = J(IDIR,k,j,i);

      // Ohmic resistivity

      if(haveResistivity) {
        if(resistivity == UserDefFunction) eta = AVERAGE_3D_YZ(etaArr,k,j,i);
        ex(k,j,i) += eta * Jx1;
      }

      // Ambipolar diffusion
      if(haveAmbipolar) {
        if(ambipolar == UserDefFunction) xA = AVERAGE_3D_YZ(xAmbiArr,k,j,i);

        ex(k,j,i) += xA * Jperp(IDIR,k,j,i);
      }

      // -----------------------
      // X2 EMF Component
      // -----------------------
      Jx2 = J(JDIR,k,j,i);

      // Ohmic resistivity
      if(haveResistivity) {
        if(resistivity == UserDefFunction) eta = AVERAGE_3D_XZ(etaArr,k,j,i);
        ey(k,j,i) += eta * Jx2;
      }

      // Ambipolar diffusion
      if(haveAmbipolar) {
        if(ambipolar == UserDefFunction) xA = AVERAGE_3D_XZ(xAmbiArr,k,j,i);

        ey(k,j,i) += xA * Jperp(JDIR,k,j,i);
      }
  #endif
      // -----------------------
      // X3 EMF Component
      // -----------------------
      Jx3 = J(KDIR,k,j,i);

      // Ohmic resistivity
      if(haveResistivity) {
        if(resistivity == UserDefFunction) eta = AVERAGE_3D_XY(etaArr,k,j,i);
        ez(k,j,i) += eta * Jx3;
      }

      // Ambipolar diffusion
      if(haveAmbipolar) {
        if(ambipolar == UserDefFunction) xA = AVERAGE_3D_XY(xAmbiArr,k,j,i);

        ez(k,j,i) += xA * Jperp(KDIR,k,j,i);
      }
    }
  );
#endif

  idfx::popRegion();
}

#endif //FLUID_CONSTRAINEDTRANSPORT_CALCNONIDEALEMF_HPP_
