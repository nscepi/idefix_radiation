// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef FLUID_CONVERTCONSTOPRIM_HPP_
#define FLUID_CONVERTCONSTOPRIM_HPP_

#include "fluid.hpp"
#include "dataBlock.hpp"
#include "tracer.hpp"

template <typename Phys>
KOKKOS_INLINE_FUNCTION void K_ConsToPrim(real Vc[], real Uc[], const EquationOfState *eos) {
  Vc[RHO] = Uc[RHO];

  if constexpr(Phys::radiation) {
      real reduced_c = 1.;

      real Fnorm2 = EXPAND(Uc[FR1]*Uc[FR1] , + Uc[FR2]*Uc[FR2], + Uc[FR3]*Uc[FR3]);
      real inv_Fnorm2 = (Fnorm2 <= 1.e-40 ? ZERO_F : ONE_F / Fnorm2);
      real Er2 = Uc[ER]*Uc[ER];
      real f_param2 = (Er2 < 1.e-40 ? Fnorm2/(reduced_c*reduced_c*1.e-40) : Fnorm2/(reduced_c*reduced_c*Er2));
      if (f_param2 > ONE_F) {
        //printf("f_param2=%e\n",f_param2);
        f_param2 = ONE_F;
      }
      real xi  = 3.+4.*f_param2;
      xi /= 5.+2.*std::sqrt(4.-3.*f_param2);
  
      EXPAND ( Vc[FR1] = HALF_F*(3.*xi-1.)*Uc[ER]*Uc[FR1]*inv_Fnorm2;  ,
               Vc[FR2] = HALF_F*(3.*xi-1.)*Uc[ER]*Uc[FR2]*inv_Fnorm2;  ,
               Vc[FR3] = HALF_F*(3.*xi-1.)*Uc[ER]*Uc[FR3]*inv_Fnorm2;  )

  } else {
      EXPAND( Vc[VX1] = Uc[MX1]/Uc[RHO];  ,
              Vc[VX2] = Uc[MX2]/Uc[RHO];  ,
              Vc[VX3] = Uc[MX3]/Uc[RHO];  )
  }

  if constexpr(Phys::mhd) {
    EXPAND( Vc[BX1] = Uc[BX1];  ,
            Vc[BX2] = Uc[BX2];  ,
            Vc[BX3] = Uc[BX3];  )
  }


  if constexpr(Phys::pressure) {
    real kin = HALF_F / Uc[RHO] * (EXPAND( Uc[MX1]*Uc[MX1]   ,
                                    + Uc[MX2]*Uc[MX2]  ,
                                    + Uc[MX3]*Uc[MX3]  ));

    if constexpr(Phys::mhd) {
      real mag = HALF_F * (EXPAND( Uc[BX1]*Uc[BX1]   ,
                          + Uc[BX2]*Uc[BX2]  ,
                          + Uc[BX3]*Uc[BX3]  ));

      Vc[PRS] = eos->GetPressure(Uc[ENG] - kin - mag, Uc[RHO]);

      // Check pressure positivity
      if(Vc[PRS]<= ZERO_F) {
        #ifdef SMALL_PRESSURE_TEMPERATURE
          Vc[PRS] = SMALL_PRESSURE_TEMPERATURE*Vc[RHO];
        #else
          Vc[PRS] = SMALL_PRESSURE_FIX;
        #endif

          Uc[ENG] = eos->GetInternalEnergy(Vc[PRS],Vc[RHO]) + kin + mag;
      }

    } else { // Hydro case
      Vc[PRS] = eos->GetPressure(Uc[ENG] - kin, Uc[RHO]);
      // Check pressure positivity
      if(Vc[PRS]<= ZERO_F) {
        #ifdef SMALL_PRESSURE_TEMPERATURE
          Vc[PRS] = SMALL_PRESSURE_TEMPERATURE*Vc[RHO];
        #else
          Vc[PRS] = SMALL_PRESSURE_FIX;
        #endif

          Uc[ENG] = eos->GetInternalEnergy(Vc[PRS],Vc[RHO]) + kin;
      }
    } // MHD
  } // Have Energy
}

template <typename Phys>
KOKKOS_INLINE_FUNCTION void K_PrimToCons(real Uc[], real Vc[], const EquationOfState *eos) {
  Uc[RHO] = Vc[RHO];

  if constexpr(Phys::radiation) {
      int iter;
      int max_iter = 200;
      real tol = 1.e-8;
      real func, dfunc, dfnorm, sqrt;

      real beta_norm =  std::sqrt(EXPAND(Vc[FR1]*Vc[FR1] , + Vc[FR2]*Vc[FR2], + Vc[FR3]*Vc[FR3]));

      real fnorm;
      if (beta_norm <= 1.e-10){
        fnorm = 0.;
        //printf("beta_norm = 0\n");
        //if (fnorm == fnorm) printf("fnorm=%e for beta_norm=%e\n",fnorm,beta_norm);

        EXPAND( Uc[FR1] = 1.e-10;  ,
                Uc[FR2] = 1.e-10;  ,
                Uc[FR3] = 1.e-10;  )
      } else if (beta_norm >= 1.) {
        fnorm = 1.;

        EXPAND( Uc[FR1] = Vc[ER]*Vc[FR1];  ,
                Uc[FR2] = Vc[ER]*Vc[FR2];  ,
                Uc[FR3] = Vc[ER]*Vc[FR3];  )       
        //if (fnorm == fnorm) printf("fnorm=%e for beta_norm=%e\n",fnorm,beta_norm);
 
      } else {
        fnorm = 1.0;
        for (iter = 0; iter < max_iter ; iter++) {

          sqrt  = std::sqrt(4.-3*fnorm*fnorm);
          func = beta_norm*(5.*fnorm+2.*fnorm*sqrt)-2.-6*fnorm*fnorm+sqrt;
          dfunc = beta_norm*(5.+2.*sqrt-6.*fnorm*fnorm/sqrt)-12.*fnorm-3.*fnorm/sqrt;
          dfnorm  = func/dfunc;
          fnorm  -= dfnorm;

          if (fnorm < 0.) {
            fnorm = 0.;
            //printf("fnorm_min reached at iter=%i\n",iter);
            break;
          }
          if (fnorm >= 1.) {
            fnorm = 1.;
            //printf("fnorm_max reached at iter=%i\n",iter);
            break;
          } 
          if (fabs (dfnorm) < tol*fnorm) break;
        }
        
        if ((fnorm != fnorm) && (beta_norm == beta_norm)) printf("fnorm=%e at iter=%i for beta_norm=%e\n",fnorm,iter,beta_norm);

        real fnorm2 = fnorm*fnorm;
        real xi  = 3.+4.*fnorm2;
        xi /= 5.+2.*std::sqrt(4.-3.*fnorm2);

        EXPAND( Uc[FR1] = HALF_F*(3.-xi)*Vc[ER]*Vc[FR1];  ,
                Uc[FR2] = HALF_F*(3.-xi)*Vc[ER]*Vc[FR2];  ,
                Uc[FR3] = HALF_F*(3.-xi)*Vc[ER]*Vc[FR3];  )
      }

  } else {
      EXPAND( Uc[MX1] = Vc[VX1]*Vc[RHO];  ,
              Uc[MX2] = Vc[VX2]*Vc[RHO];  ,
              Uc[MX3] = Vc[VX3]*Vc[RHO];  )  
  }

  if constexpr(Phys::mhd) {
    EXPAND( Uc[BX1] = Vc[BX1];  ,
            Uc[BX2] = Vc[BX2];  ,
            Uc[BX3] = Vc[BX3];  )
  }


  if constexpr(Phys::pressure) {
    if constexpr(Phys::mhd) {
      Uc[ENG] = eos->GetInternalEnergy(Vc[PRS],Vc[RHO])
                + HALF_F * Vc[RHO] * (EXPAND( Vc[VX1]*Vc[VX1]  ,
                                            + Vc[VX2]*Vc[VX2]  ,
                                            + Vc[VX3]*Vc[VX3]  ))
                + HALF_F * (EXPAND( Uc[BX1]*Uc[BX1]  ,
                                  + Uc[BX2]*Uc[BX2]  ,
                                  + Uc[BX3]*Uc[BX3]  ));
    } else {
      Uc[ENG] = eos->GetInternalEnergy(Vc[PRS],Vc[RHO])
                + HALF_F * Vc[RHO] * (EXPAND( Vc[VX1]*Vc[VX1]  ,
                                            + Vc[VX2]*Vc[VX2]  ,
                                            + Vc[VX3]*Vc[VX3]  ));
    } //MHD
  } // Energy
}



// Convect Conservative to Primitive variable
template<typename Phys>
void Fluid<Phys>::ConvertConsToPrim() {
  idfx::pushRegion("Fluid::ConvertConsToPrim");

  IdefixArray4D<real> Vc = this->Vc;
  IdefixArray4D<real> Uc = this->Uc;
  EquationOfState eos;
  if constexpr(Phys::eos) {
    eos = *(this->eos.get());
  }

  if constexpr(Phys::mhd) {
    #ifdef EVOLVE_VECTOR_POTENTIAL
      emf->ComputeMagFieldFromA(Ve,Vs);
    #endif
    boundary->ReconstructVcField(Uc);
  }

  idefix_for("ConsToPrim",
             0,data->np_tot[KDIR],
             0,data->np_tot[JDIR],
             0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real U[Phys::nvar];
      real V[Phys::nvar];

#pragma unroll
      for(int nv = 0 ; nv < Phys::nvar; nv++) {
        U[nv] = Uc(nv,k,j,i);
      }

      K_ConsToPrim<Phys>(V,U,&eos);

#pragma unroll
      for(int nv = 0 ; nv<Phys::nvar; nv++) {
        Vc(nv,k,j,i) = V[nv];
      }
  });

  if(haveTracer) {
    tracer->ConvertConsToPrim();
  }

  idfx::popRegion();
}

// Convert Primitive to conservative variables
template<typename Phys>
void Fluid<Phys>::ConvertPrimToCons() {
  idfx::pushRegion("Fluid::ConvertPrimToCons");

  IdefixArray4D<real> Vc = this->Vc;
  IdefixArray4D<real> Uc = this->Uc;
  EquationOfState eos;
  if constexpr(Phys::eos) {
    eos = *(this->eos.get());
  }

  idefix_for("ConvertPrimToCons",
             0,data->np_tot[KDIR],
             0,data->np_tot[JDIR],
             0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real U[Phys::nvar];
      real V[Phys::nvar];

#pragma unroll
      for(int nv = 0 ; nv < Phys::nvar; nv++) {
        V[nv] = Vc(nv,k,j,i);
      }

      K_PrimToCons<Phys>(U,V,&eos);

#pragma unroll
      for(int nv = 0 ; nv<Phys::nvar; nv++) {
        Uc(nv,k,j,i) = U[nv];
      }
  });

  if(haveTracer) {
    tracer->ConvertPrimToCons();
  }

  idfx::popRegion();
}

#endif //FLUID_CONVERTCONSTOPRIM_HPP_
