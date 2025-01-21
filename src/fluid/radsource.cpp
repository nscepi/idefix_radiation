// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************
#include "../idefix.hpp"
#include "radsource.hpp"
#include "physics.hpp"
#include "units.hpp"
#include "lookupTable.hpp"
#include "column.hpp"

void RadSource::SourceFullImplicit(const real dt) {
  idfx::pushRegion("RadSource::Source_full_implicit");

  // Ensure that radiation cannot be run with isothermal eos
  #ifndef ISOTHERMAL 

  auto UcGas = this->UcGas;
  auto VcGas = this->VcGas;
  auto UcRad = this->UcRad;
  auto VcRad = this->VcRad;
  auto InvDt = this->InvDt;

  auto units = idfx::units;

  auto kp1D = this->kappa_planck_1D;
  auto kr1D = this->kappa_ross_1D;
  auto xi1D = this->xi_1D;

  EquationOfState eos = this->eos;

  real reduced_c = this->reduced_c;

  // Irradiation source
  bool irr_flag=false;
  IdefixArray3D<real> divF = this->divF;
  if (haveIrradiation){
    IrrFlux(divF);
    irr_flag=true;
  }

  // Local copy of opacity parameters
  const Type_opac kappa_type = this->kappa_type;
  IdefixArray3D<real> kappapArr;
  IdefixArray3D<real> kapparArr;
  IdefixArray3D<real> xiArr;
  real kappa_0,rho_0,T_0,xi_0;
  if (kappa_type == Type_opac::constant) {
    kappa_0 = this->kappa_0;
  } else if (kappa_type == Type_opac::kramers) {
    kappa_0 = this->kappa_0;
    T_0 = this->T_0;
    rho_0 = this->rho_0;
  } else if (kappa_type == Type_opac::userfunc) {
    kappapArr = this->kappapArr;
    kapparArr = this->kapparArr;
  }

  const Type_opac xi_type = this->xi_type;
  if (xi_type == Type_opac::constant) {
    xi_0 = this->xi_0;
  } else if (xi_type == Type_opac::userfunc) {
    xiArr = this->xiArr;
  }


  idefix_for("RadSourceFullImplicit",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      real URad[RadiationPhysics::nvar];
      real UGas[DefaultPhysics::nvar];
      real VGas[DefaultPhysics::nvar];

      real kappa_p, kappa_r, xi;

      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        URad[nv] = UcRad(nv,k,j,i);
      }
      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UGas[nv] = UcGas(nv,k,j,i);
        VGas[nv] = VcGas(nv,k,j,i);
      }

      // Compute total modified energy
      real Etot = UGas[ENG]+URad[ER]*units.c/(reduced_c*units.GetVelocity());
      
      // Add irradiation heating if needed
      if (irr_flag){  
        Etot -= divF(k,j,i)*dt*units.GetTime()/units.GetEnergy();
      }
      
      // Compute total modified momentum
      EXPAND(real m1tot = UGas[MX1]+URad[FR1]/reduced_c;,
             real m2tot = UGas[MX2]+URad[FR2]/reduced_c;,
             real m3tot = UGas[MX3]+URad[FR3]/reduced_c;)
      
      // Store conserved variables after hyperbolic step
      real Er_hyp = URad[ER];
      EXPAND(real Fr1_hyp = URad[FR1];,
             real Fr2_hyp = URad[FR2];,
             real Fr3_hyp = URad[FR3];)

      real Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));
        
      real mu = eos.GetMu(VGas[PRS],VGas[RHO]);
      real cv = units.k_B/(units.u*mu);

      real T = VGas[PRS]/(VGas[RHO])*units.GetKelvin()*mu;
      real T3 = std::pow(T,3);

      // Compute opacities
      if (kappa_type == Type_opac::constant) {
        kappa_p = kappa_0;
        kappa_r = kappa_0;
      } else if (kappa_type == Type_opac::kramers) {
        kappa_p = kappa_0*std::pow(VGas[RHO]*units.GetDensity()/rho_0,2.)*std::pow(T/T_0,-3.5);
        kappa_r = kappa_p;
      } else if (kappa_type == Type_opac::usertable) {
        real logT = std::log10(T);
        kappa_p = kp1D.Get(&logT);
        kappa_r = kr1D.Get(&logT);
      } else if (kappa_type == Type_opac::userfunc) {
        kappa_p = kappapArr(k,j,i);
        kappa_r = kapparArr(k,j,i);
      }

      if (xi_type == Type_opac::constant) {
        xi = xi_0;
      } else if (xi_type == Type_opac::usertable) {
        real logT = std::log10(T);
        xi = xi1D.Get(&logT);
      } else if (xi_type== Type_opac::userfunc) {
        xi = xiArr(k,j,i);
      }


      real kk_red = reduced_c * units.GetVelocity() * dt * units.GetTime() * kappa_p * VGas[RHO]*units.GetDensity();
      real kk = units.c * dt * units.GetTime() * kappa_p * VGas[RHO]*units.GetDensity();
      real xx_red = reduced_c * units.GetVelocity() * dt * units.GetTime() * (xi + kappa_r) * VGas[RHO]*units.GetDensity();

      // Define matrix to invert
      real gamma = eos.GetGamma(VGas[PRS],VGas[RHO]);

      real M00 = ONE_F + kk_red;
      real M11 = VGas[RHO]*units.GetDensity()*cv/(gamma-1.) + 4.*kk*units.ar*T3;
      real M01 = -4.*kk_red*units.ar*T3;
      real M10 = -kk;

      // Define right-hand side of system
      real S0 = Er_hyp*units.GetEnergy() - 3.*kk_red*units.ar*T3*T;
      real S1 = VGas[RHO]*units.GetDensity()*cv*T/(gamma-1.) + 3.*kk*units.ar*T3*T;

      // Add irradiation heating to RHS if needed
      if (irr_flag){
        S1 -= divF(k,j,i)*dt*units.GetTime();
      }
 
      // Invert system
      real det = M00*M11 - M01*M10;

      real Minv00 = M11/det;
      //real Minv11 = M00/det;
      real Minv01 = -M01/det;
      //real Minv10 = -M10/det;

      real Er_new = Minv00*S0 + Minv01*S1;
      //real T_new = Minv10*S0 + Minv11*S1;

      // Update conservative variables
      URad[ER] = Er_new/units.GetEnergy();
      EXPAND( URad[FR1] = Fr1_hyp/(1.+xx_red);,
              URad[FR2] = Fr2_hyp/(1.+xx_red);,
              URad[FR3] = Fr3_hyp/(1.+xx_red);)
      Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

      if ((Etot - URad[ER]*units.c/(reduced_c*units.GetVelocity()))<=ZERO_F) {
        Kokkos::abort("ENG=0 in Radsource");
      } else {
        UGas[ENG] = Etot - URad[ER]*units.c/(reduced_c*units.GetVelocity());
      }

      EXPAND( UGas[MX1] = m1tot - URad[FR1]/reduced_c;,
              UGas[MX2] = m2tot - URad[FR2]/reduced_c;,
              UGas[MX3] = m3tot - URad[FR3]/reduced_c;)

      // Update primitive  variables
      K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);

      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        UcRad(nv,k,j,i) = URad[nv];
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UcGas(nv,k,j,i) = UGas[nv];
      }
    });

    #endif
  idfx::popRegion();
}


void RadSource::SourceFixedPointRad(const real dt) {
  idfx::pushRegion("RadSource::SourceFixedPointRad");

  // Ensure that radiation cannot be run with isothermal eos
  #ifndef ISOTHERMAL 

  auto UcGas = this->UcGas;
  auto VcGas = this->VcGas;
  auto UcRad = this->UcRad;
  auto VcRad = this->VcRad;
  auto InvDt = this->InvDt;
  
  auto units=idfx::units;

  auto kp1D = this->kappa_planck_1D;
  auto kr1D = this->kappa_ross_1D;
  auto xi1D = this->xi_1D;

  real reduced_c = this->reduced_c;

  // Max iteration for fixed-point solver
  int MAX_ITER = 200;
  // Tolerance on ER and ENG for fixed-point solver
  real tol = 1.e-3;

  EquationOfState eos = this->eos;

  // Irradiation source
  bool irr_flag=false;
  IdefixArray3D<real> divF = this->divF;
  if (haveIrradiation){
    IrrFlux(divF);
    irr_flag=true;
  }

  // Local copy of opacity parameters
  const Type_opac kappa_type = this->kappa_type;
  real kappa_0,rho_0,T_0,xi_0;
  IdefixArray3D<real> kappapArr;
  IdefixArray3D<real> kapparArr;
  IdefixArray3D<real> xiArr;
  if (kappa_type == Type_opac::constant) {
    kappa_0 = this->kappa_0;
  } else if (kappa_type == Type_opac::kramers) {
    kappa_0 = this->kappa_0;
    T_0 = this->T_0;
    rho_0 = this->rho_0;
  } else if (kappa_type == Type_opac::userfunc) {
    kappapArr = this->kappapArr;
    kapparArr = this->kapparArr;
  }

  const Type_opac xi_type = this->xi_type;
  if (xi_type == Type_opac::constant) {
    xi_0 = this->xi_0;
  } else if (xi_type == Type_opac::userfunc) {
    xiArr = this->xiArr;
  }

  idefix_for("RadSourceFixedPointRad",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      // Add heating due to irradiation flux if needed
      if (irr_flag){
        // Limit time step relative to characteristic time of irradiation heating
        InvDt(k,j,i) += FABS(units.GetTime()*divF(k,j,i)/units.GetEnergy()/UcGas(ENG,k,j,i));
        UcGas(ENG,k,j,i) -= dt*units.GetTime()*divF(k,j,i)/units.GetEnergy();
      }

      real UGas[DefaultPhysics::nvar];
      real VGas[DefaultPhysics::nvar];
      real URad[RadiationPhysics::nvar];

      real kappa_p, kappa_r, xi;

      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        URad[nv] = UcRad(nv,k,j,i);
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UGas[nv] = UcGas(nv,k,j,i); 
        VGas[nv] = VcGas(nv,k,j,i);
      }
      
      // Compute total modified energy and momentum
      real Etot = UGas[ENG]+URad[ER]*units.c/(reduced_c*units.GetVelocity());
      EXPAND(real m1tot = UGas[MX1]+URad[FR1]/reduced_c;,
             real m2tot = UGas[MX2]+URad[FR2]/reduced_c;,
             real m3tot = UGas[MX3]+URad[FR3]/reduced_c;)
      
      // Store conservative variables after hydro step
      real Er_hyp = URad[ER];
      EXPAND(real Fr1_hyp = URad[FR1];,
             real Fr2_hyp = URad[FR2];,
             real Fr3_hyp = URad[FR3];)

      real Er_old, Fnorm_old, Egas_old, Mnorm_old;      
    
      real err1= 1.;
      real err2= 1.;
      real err3= 1.;
      real err4= 1.;
      int count = 0;

      real Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));
      real Mnorm = std::sqrt(EXPAND(UGas[MX1]*UGas[MX1] , + UGas[MX2]*UGas[MX2], + UGas[MX3]*UGas[MX3]));

      // Assume mu is constant during iteration (to check)
      real mu = eos.GetMu(VGas[PRS],VGas[RHO]);

      real T = VGas[PRS]/(VGas[RHO])*units.GetKelvin()*mu;

      // Assume kappa and xi are constant during iteration (to check)
      if (kappa_type == Type_opac::constant) {
        kappa_p = kappa_0;
        kappa_r = kappa_0;
      } else if (kappa_type == Type_opac::kramers) {
        kappa_p = kappa_0*std::pow(VGas[RHO]*units.GetDensity()/rho_0,2.)*std::pow(T/T_0,-3.5);
        kappa_r = kappa_p;
      } else if (kappa_type == Type_opac::usertable) {
        real logT = std::log10(T);
        kappa_p = kp1D.Get(&logT);
        kappa_r = kr1D.Get(&logT);
      } else if (kappa_type == Type_opac::userfunc) {
        kappa_p = kappapArr(k,j,i);
        kappa_r = kapparArr(k,j,i);
      }
      
      if (xi_type == Type_opac::constant) {
        xi = xi_0;
      } else if (xi_type == Type_opac::usertable) {
        real logT = std::log10(T);
        xi = xi1D.Get(&logT);
      } else if (xi_type== Type_opac::userfunc) {
        xi = xiArr(k,j,i);
      }

      // Iterate on radiative variables
      while (((err1>tol) || (err2>tol) || (err3>tol) || (err4>tol)) && (count < MAX_ITER)){

        Er_old = URad[ER];
        Fnorm_old = Fnorm;
        Egas_old = UGas[ENG];
        Mnorm_old = Mnorm;
        
        real kk_red = reduced_c * units.GetVelocity() * dt * units.GetTime() * kappa_p * VGas[RHO]*units.GetDensity();
        real xx_red = reduced_c * units.GetVelocity() * dt * units.GetTime() * (xi + kappa_r) * VGas[RHO]*units.GetDensity();

        // "Implicit" step on radiation conservative variables
        URad[ER] = Er_hyp +  kk_red*units.ar*std::pow(T,4.)/units.GetEnergy();
        URad[ER] /= 1. + kk_red;
        EXPAND( URad[FR1] = Fr1_hyp/(1.+xx_red);,
                URad[FR2] = Fr2_hyp/(1.+xx_red);,
                URad[FR3] = Fr3_hyp/(1.+xx_red);)
        Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

        // Update gas conservative variables
        if ((Etot - URad[ER]*units.c/(reduced_c*units.GetVelocity()))<=ZERO_F) {
          Kokkos::abort("ENG=0 in Radsource");
        } else {
          UGas[ENG] = Etot - URad[ER]*units.c/(reduced_c*units.GetVelocity());
        }
        EXPAND( UGas[MX1] = m1tot - URad[FR1]/reduced_c;,
                UGas[MX2] = m2tot - URad[FR2]/reduced_c;,
                UGas[MX3] = m3tot - URad[FR3]/reduced_c;)
        Mnorm = std::sqrt(EXPAND(UGas[MX1]*UGas[MX1] , + UGas[MX2]*UGas[MX2], + UGas[MX3]*UGas[MX3]));

        // Update gas primitive variables
        K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);
        
        // Compute new temperature
        T = VGas[PRS]/(VGas[RHO])*units.GetKelvin()*mu;
        
        // Compute errors and number of cycles
        err1 = std::abs(1.-URad[ER]/Er_old);
        err2 = std::abs(1.-Fnorm/Fnorm_old);
        err3 = std::abs(1.-UGas[ENG]/Egas_old);
        err4 = std::abs(1.-Mnorm/Mnorm_old);
        count += 1;
      } 

      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        UcRad(nv,k,j,i) = URad[nv];
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UcGas(nv,k,j,i) = UGas[nv];
      }
  });

  #endif

  idfx::popRegion();
}


void RadSource::SourceFixedPointGas(const real dt) {
  idfx::pushRegion("RadSource::SourceFixedPointGas");

  // Ensure that radiation cannot be run with isothermal eos
  #ifndef ISOTHERMAL 

  auto UcGas = this->UcGas;
  auto VcGas = this->VcGas;
  auto UcRad = this->UcRad;
  auto VcRad = this->VcRad;
  auto InvDt = this->InvDt;
  
  real reduced_c = this->reduced_c;

  auto units=idfx::units;

  auto kp1D = this->kappa_planck_1D;
  auto kr1D = this->kappa_ross_1D;
  auto xi1D = this->xi_1D;

  // Max iteration for fixed-point solver
  int MAX_ITER = 200;
  // Tolerance on ER and ENG for fixed-point solver
  real tol = 1.e-3;

  EquationOfState eos = this->eos;

  // Local copy of opacity parameters
  const Type_opac kappa_type = this->kappa_type;
  real kappa_0,rho_0,T_0,xi_0;
  IdefixArray3D<real> kappapArr;
  IdefixArray3D<real> kapparArr;
  IdefixArray3D<real> xiArr;
  if (kappa_type == Type_opac::constant) {
    kappa_0 = this->kappa_0;
  } else if (kappa_type == Type_opac::kramers) {
    kappa_0 = this->kappa_0;
    T_0 = this->T_0;
    rho_0 = this->rho_0;
  } else if (kappa_type == Type_opac::userfunc) {
    kappapArr = this->kappapArr;
    kapparArr = this->kapparArr;
  }

  const Type_opac xi_type = this->xi_type;
  if (xi_type == Type_opac::constant) {
    xi_0 = this->xi_0;
  } else if (xi_type == Type_opac::userfunc) {
    xiArr = this->xiArr;
  }
  
  idefix_for("RadSource",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      real Etot = UcGas(ENG,k,j,i)+UcRad(ER,k,j,i)*units.c/(reduced_c*units.GetVelocity());
      real m1tot = UcGas(MX1,k,j,i)+UcRad(FR1,k,j,i)/reduced_c;
      real m2tot = UcGas(MX2,k,j,i)+UcRad(FR2,k,j,i)/reduced_c;
      real m3tot = UcGas(MX3,k,j,i)+UcRad(FR3,k,j,i)/reduced_c;
      
      real Er_hyp = UcRad(ER,k,j,i);
      real Fr1_hyp = UcRad(FR1,k,j,i);
      real Fr2_hyp = UcRad(FR2,k,j,i);
      real Fr3_hyp = UcRad(FR3,k,j,i);

      real Egas_hyp = UcGas(ENG,k,j,i);
      real m1gas_hyp = UcGas(MX1,k,j,i);
      real m2gas_hyp = UcGas(MX2,k,j,i);
      real m3gas_hyp = UcGas(MX3,k,j,i);

      real URad[RadiationPhysics::nvar];
      real Er_old, Fnorm_old, Egas_old, Mnorm_old;

      real UGas[DefaultPhysics::nvar];
      real VGas[DefaultPhysics::nvar];

      real kappa_p, kappa_r, xi;

      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        URad[nv] = UcRad(nv,k,j,i);
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UGas[nv] = UcGas(nv,k,j,i);
        VGas[nv] = VcGas(nv,k,j,i);
      }
    
      real err1= 1.;
      real err2= 1.;
      real err3= 1.;
      real err4= 1.;
      int count = 0;

      real Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));
      real Mnorm = std::sqrt(EXPAND(UGas[MX1]*UGas[MX1] , + UGas[MX2]*UGas[MX2], + UGas[MX3]*UGas[MX3]));

      // Assume mu is constant during iteration (to check)
      real mu = eos.GetMu(VGas[PRS],VGas[RHO]);
      real T = VGas[PRS]/(VGas[RHO])*units.GetKelvin()*mu;

      // Compute opacities (out of while loop so that opacity is contant throughout the fixed_point iteration)
      if (kappa_type == Type_opac::constant) {
        kappa_p = kappa_0;
        kappa_r = kappa_0;
      } else if (kappa_type == Type_opac::kramers) {
        kappa_p = kappa_0*std::pow(VGas[RHO]*units.GetDensity()/rho_0,2.)*std::pow(T/T_0,-3.5);
        kappa_r = kappa_p;
      } else if (kappa_type == Type_opac::usertable) {
        real logT = std::log10(T);
        kappa_p = kp1D.Get(&logT);
        kappa_r = kr1D.Get(&logT);
      } else if (kappa_type == Type_opac::userfunc) {
        kappa_p = kappapArr(k,j,i);
        kappa_r = kapparArr(k,j,i);
      }

      if (xi_type == Type_opac::constant) {
        xi = xi_0;
      } else if (xi_type == Type_opac::usertable) {
        real logT = std::log10(T);
        xi = xi1D.Get(&logT);
      } else if (xi_type== Type_opac::userfunc) {
        xi = xiArr(k,j,i);
      }

      while (((err1>tol) || (err2>tol) || (err3>tol) || (err4>tol)) && (count < MAX_ITER)){

        Er_old = URad[ER];
        Fnorm_old = Fnorm;
        Egas_old = UGas[ENG];
        Mnorm_old = Mnorm;
        
        real kk =  units.c * dt * units.GetTime() * kappa_p * VGas[RHO]*units.GetDensity();
        real xx =  dt * units.GetTime() * (xi + kappa_r) * VGas[RHO]*units.GetDensity();

        // Stop if UGas <= 0
        if ((Egas_hyp +  kk*(URad[ER]-units.ar*std::pow(T,4)/units.GetEnergy()))<=ZERO_F) {
          Kokkos::abort("EGas=0 in Radsource");
          UGas[ENG] = 1.e-6;
        } else {
          UGas[ENG] = Egas_hyp +  kk*(URad[ER]-units.ar*std::pow(T,4)/units.GetEnergy());
        }

        EXPAND( UGas[MX1] = m1gas_hyp + URad[FR1]*xx;,
                UGas[MX2] = m2gas_hyp + URad[FR2]*xx;,
                UGas[MX3] = m3gas_hyp + URad[FR3]*xx;)
        Mnorm = std::sqrt(EXPAND(UGas[MX1]*UGas[MX1] , + UGas[MX2]*UGas[MX2], + UGas[MX3]*UGas[MX3]));

        URad[ER] = (Etot - UGas[ENG])*reduced_c*units.GetVelocity()/units.c;

        // Stop if URad <= 0
        if (URad[ER]<=ZERO_F) {
          Kokkos::abort("ERad=0 in Radsource");
        }

        EXPAND( URad[FR1] = (m1tot - UGas[MX1])*reduced_c;,
                URad[FR2] = (m2tot - UGas[MX2])*reduced_c;,
                URad[FR3] = (m3tot - UGas[MX3])*reduced_c;)

        Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

        K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);
        T = VGas[PRS]/(VGas[RHO])*units.GetKelvin()*mu;

        err1 = std::abs(1.-UGas[ENG]/Egas_old);
        err2 = std::abs(1.-Mnorm/Mnorm_old);
        err3 = std::abs(1.-URad[ER]/Er_old);
        err4 = std::abs(1.-Fnorm/Fnorm_old);
        count += 1;
        
      }
      
      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        UcRad(nv,k,j,i) = URad[nv];
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UcGas(nv,k,j,i) = UGas[nv];
      }
  });

  #endif

  idfx::popRegion();
}

void RadSource::ShowConfig() {

  // Ensure that radiation cannot be run with isothermal eos
  #ifndef ISOTHERMAL 

  idfx::cout << "RadSource: kappa is ";
  switch(kappa_type) {
    case Type_opac::constant:
      idfx::cout << "constant." << std::endl;
      break;
    case Type_opac::kramers:
      idfx::cout << "from kramers' law." << std::endl;
      break;
    case Type_opac::usertable:
      idfx::cout << "from a user table." << std::endl;
      break;
    case Type_opac::userfunc:
      idfx::cout << "from a user-defined function."
                     << std::endl;
      if(!data->radiation[0]->kappaFunc) {
        IDEFIX_ERROR("No opacity function has been enrolled for kappa");
      }
      break;
  }
  idfx::cout << "RadSource: xi is ";
  switch(xi_type) {
    case Type_opac::constant:
      idfx::cout << "constant." << std::endl;
      break;
    case Type_opac::kramers:
      idfx::cout << "!!! from kramers' law, which is not allowed !!!" << std::endl;
      break;
    case Type_opac::usertable:
      idfx::cout << "from a user table." << std::endl;
      break;
    case Type_opac::userfunc:
      idfx::cout << "from a user-defined function."
                     << std::endl;
      if(!data->radiation[0]->xiFunc) {
        IDEFIX_ERROR("No opacity function has been enrolled for xi");
      }
      break;
  }
  idfx::cout << "Source term solver is ";
  switch(source_solver) {
    case Type_isolver::full_implicit:
      idfx::cout << "full_implicit." << std::endl;
      break;
    case Type_isolver::fixed_point_rad:
      idfx::cout << "fixed_point_rad." << std::endl;
      break;
    case Type_isolver::fixed_point_gas:
      idfx::cout << "fixed_point_gas (to test)." << std::endl;
      break;
  }
 
 #else 
    IDEFIX_ERROR("Isothermal EOS is not compatible with radiative transfer.");
 #endif 

}

void RadSource::AddRadSource(const real dt) {
  idfx::pushRegion("RadSource::AddRadSource");

  switch(source_solver) {
    case Type_isolver::full_implicit:
      RadSource::SourceFullImplicit(dt);
      break;
    case Type_isolver::fixed_point_rad:
      RadSource::SourceFixedPointRad(dt);
      break;
    case Type_isolver::fixed_point_gas:
      RadSource::SourceFixedPointGas(dt);
      break;
  }

  idfx::popRegion();
}



void RadSource::IrrFlux(IdefixArray3D<real> divFin) {
  idfx::pushRegion("RadSource::IrrFlux");
  
  auto VcGas = this->VcGas;
  IdefixArray3D<real>  dV = this->data->dV;
  IdefixArray3D<real>  A1 = this->data->A[IDIR];
  IdefixArray1D<real>  x1l = this->data->xl[IDIR];
  auto units=idfx::units;
  auto irr_type = this->irr_type;
  auto irr1D = this->irr_1D;
  IdefixArray3D<real> divFlux = divFin;
  IdefixArray3D<real> kapparho = this->kapparhoArr;
  IdefixArray3D<real> kappapArr = this->kappapArr;
  IdefixArray3D<real> tau("tau",this->data->np_tot[KDIR],this->data->np_tot[JDIR],this->data->np_tot[IDIR]);
  real kappa_irr = this->kappa_irr;

  real flux_pre = std::pow(rs/units.GetLength(),2.)*units.sigma_sb*std::pow(Ts,4.)/units.GetLength();
  
  if (irr_type==Type_irr::constant){
    column_rho->ComputeColumn(this->VcGas,RHO);
    tau = column_rho->GetColumn();
  } else if (irr_type==Type_irr::usertable){
    column_rho->ComputeColumn(this->VcGas,RHO);
    tau = column_rho->GetColumn();
  } else if (irr_type==Type_irr::userfunc){
    idefix_for("RadSourceInitKapparho",
    data->beg[KDIR], data->end[KDIR],
    data->beg[JDIR], data->end[JDIR],
    data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
                kapparho(k,j,i) = kappapArr(k,j,i)*units.GetDensity()*units.GetLength()*VcGas(RHO,k,j,i);
    });
    column_rho->ComputeColumn(kapparho);
    tau = column_rho->GetColumn();
  }

  idefix_for("RadSourceIrrFlux",
  data->beg[KDIR], data->end[KDIR],
  data->beg[JDIR], data->end[JDIR],
  data->beg[IDIR], data->end[IDIR],
  KOKKOS_LAMBDA (int k, int j, int i) {

              real Fip,Fim;

              // Constant kappa
              if(irr_type==Type_irr::constant) {
                real kirr = kappa_irr*units.GetDensity()*units.GetLength();
                Fim = std::exp(-kirr*tau(k,j,i-1))*A1(k,j,i)/std::pow(x1l(i),2.);
                Fip = std::exp(-kirr*tau(k,j,i))*A1(k,j,i+1)/std::pow(x1l(i+1),2.);
                
              // Usertable kappa
              } else if (irr_type==Type_irr::usertable) {  
                real logtaum = std::log10(FMAX(tau(k,j,i-1)*units.GetDensity()*units.GetLength(),1.e-15));
                Fim = pow(10.,irr1D.Get(&logtaum))*A1(k,j,i)/std::pow(x1l(i),2.);
                real logtaup = std::log10(FMAX(tau(k,j,i)*units.GetDensity()*units.GetLength(),1.e-15));
                Fip = pow(10.,irr1D.Get(&logtaup))*A1(k,j,i+1)/std::pow(x1l(i+1),2.);
              
              // Userfunc kappa
              } else if (irr_type==Type_irr::userfunc){
                Fim = std::exp(-tau(k,j,i-1))*A1(k,j,i)/std::pow(x1l(i),2.);
                Fip = std::exp(-tau(k,j,i))*A1(k,j,i+1)/std::pow(x1l(i+1),2.);
              }

              divFlux(k,j,i) = flux_pre*(Fip-Fim)/dV(k,j,i);
    });


  idfx::popRegion();

} 
