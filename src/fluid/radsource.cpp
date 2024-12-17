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
  Type_opac kappa_type = this->kappa_type;
  real kappa_0,rho_0,T_0,xi_0;
  if (kappa_type == Type_opac::constant) {
    kappa_0 = this->kappa_0;
  } else if (kappa_type == Type_opac::kramers) {
    kappa_0 = this->kappa_0;
    T_0 = this->T_0;
    rho_0 = this->rho_0;
  }
  Type_opac xi_type = this->xi_type;
  if (xi_type == Type_opac::constant) {
    xi_0 = this->xi_0;
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
      real Etot = UGas[ENG]+URad[ER]*units.c/(reduced_c*units.velocity);
      
      // Add irradiation heating if needed
      if (irr_flag){  
        Etot -= divF(k,j,i)*dt*units.time/units.energy;
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

      real T = VGas[PRS]/(VGas[RHO])*units.Kelvin*mu;
      real T3 = std::pow(T,3);

      // Compute opacities
      if (kappa_type == Type_opac::constant) {
        kappa_p = kappa_0;
        kappa_r = kappa_0;
      } else if (kappa_type == Type_opac::kramers) {
        kappa_p = kappa_0*std::pow(VGas[RHO]*units.density/rho_0,2.)*std::pow(T/T_0,-3.5);
        kappa_r = kappa_p;
      } else if (kappa_type == Type_opac::usertable) {
        real logT = std::log10(T);
        kappa_p = kp1D.Get(&logT);
        kappa_r = kr1D.Get(&logT);
      }
      if (xi_type == Type_opac::constant) {
        xi = xi_0;
      } else if (xi_type == Type_opac::usertable) {
        real logT = std::log10(T);
        xi = xi1D.Get(&logT);
      }

      real kk_red = reduced_c * units.velocity * dt * units.time * kappa_p * VGas[RHO]*units.density;
      real kk = units.c * dt * units.time * kappa_p * VGas[RHO]*units.density;
      real xx_red = reduced_c * units.velocity * dt * units.time * (xi + kappa_r) * VGas[RHO]*units.density;

      // Define matrix to invert
      real gamma = eos.GetGamma(VGas[PRS],VGas[RHO]);

      real M00 = ONE_F + kk_red;
      real M11 = VGas[RHO]*units.density*cv/(gamma-1.) + 4.*kk*units.ar*T3;
      real M01 = -4.*kk_red*units.ar*T3;
      real M10 = -kk;

      // Define right-hand side of system
      real S0 = Er_hyp*units.energy - 3.*kk_red*units.ar*T3*T;
      real S1 = VGas[RHO]*units.density*cv*T/(gamma-1.) + 3.*kk*units.ar*T3*T;

      // Add irradiation heating to RHS if needed
      if (irr_flag){
        S1 -= divF(k,j,i)*dt*units.time;
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
      URad[ER] = Er_new/units.energy;
      EXPAND( URad[FR1] = Fr1_hyp/(1.+xx_red);,
              URad[FR2] = Fr2_hyp/(1.+xx_red);,
              URad[FR3] = Fr3_hyp/(1.+xx_red);)
      Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

      if ((Etot - URad[ER]*units.c/(reduced_c*units.velocity))<=ZERO_F) {
        Kokkos::abort("ENG=0 in Radsource");
      } else {
        UGas[ENG] = Etot - URad[ER]*units.c/(reduced_c*units.velocity);
      }

      EXPAND( UGas[MX1] = m1tot - URad[FR1]/reduced_c;,
              UGas[MX2] = m2tot - URad[FR2]/reduced_c;,
              UGas[MX3] = m3tot - URad[FR3]/reduced_c;)

      // Update primitive  variables
      K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);

      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        UcRad(nv,k,j,i) = URad[nv];
        VcRad(nv,k,j,i) = URad[nv];
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UcGas(nv,k,j,i) = UGas[nv];
        VcGas(nv,k,j,i) = VGas[nv];
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
  Type_opac kappa_type = this->kappa_type;
  real kappa_0,rho_0,T_0,xi_0;
  if (kappa_type == Type_opac::constant) {
    kappa_0 = this->kappa_0;
  } else if (kappa_type == Type_opac::kramers) {
    kappa_0 = this->kappa_0;
    T_0 = this->T_0;
    rho_0 = this->rho_0;
  }
  Type_opac xi_type = this->xi_type;
  if (xi_type == Type_opac::constant) {
    xi_0 = this->xi_0;
  }

  idefix_for("RadSourceFixedPointRad",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      // Add heating due to irradiation flux if needed
      if (irr_flag){
        // Limit time step relative to characteristic time of irradiation heating
        InvDt(k,j,i) += FABS(units.time*divF(k,j,i)/units.energy/UcGas(ENG,k,j,i));
        UcGas(ENG,k,j,i) -= dt*units.time*divF(k,j,i)/units.energy;
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
      real Etot = UGas[ENG]+URad[ER]*units.c/(reduced_c*units.velocity);
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

      real T = VGas[PRS]/(VGas[RHO])*units.Kelvin*mu;

      // Assume kappa and xi are constant during iteration (to check)
      if (kappa_type == Type_opac::constant) {
        kappa_p = kappa_0;
        kappa_r = kappa_0;
      } else if (kappa_type == Type_opac::kramers) {
        kappa_p = kappa_0*std::pow(VGas[RHO]*units.density/rho_0,2.)*std::pow(T/T_0,-3.5);
        kappa_r = kappa_p;
      } else if (kappa_type == Type_opac::usertable) {
        real logT = std::log10(T);
        kappa_p = kp1D.Get(&logT);
        kappa_r = kr1D.Get(&logT);
      }
      if (xi_type == Type_opac::constant) {
        xi = xi_0;
      } else if (xi_type == Type_opac::usertable) {
        real logT = std::log10(T);
        xi = xi1D.Get(&logT);
      }

      // Iterate on radiative variables
      while (((err1>tol) || (err2>tol) || (err3>tol) || (err4>tol)) && (count < MAX_ITER)){

        Er_old = URad[ER];
        Fnorm_old = Fnorm;
        Egas_old = UGas[ENG];
        Mnorm_old = Mnorm;
        
        real kk_red = reduced_c * units.velocity * dt * units.time * kappa_p * VGas[RHO]*units.density;
        real xx_red = reduced_c * units.velocity * dt * units.time * (xi + kappa_r) * VGas[RHO]*units.density;

        // "Implicit" step on radiation conservative variables
        URad[ER] = Er_hyp +  kk_red*units.ar*std::pow(T,4.)/units.energy;
        URad[ER] /= 1. + kk_red;
        EXPAND( URad[FR1] = Fr1_hyp/(1.+xx_red);,
                URad[FR2] = Fr2_hyp/(1.+xx_red);,
                URad[FR3] = Fr3_hyp/(1.+xx_red);)
        Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

        // Update gas conservative variables
        if ((Etot - URad[ER]*units.c/(reduced_c*units.velocity))<=ZERO_F) {
          Kokkos::abort("ENG=0 in Radsource");
        } else {
          UGas[ENG] = Etot - URad[ER]*units.c/(reduced_c*units.velocity);
        }
        EXPAND( UGas[MX1] = m1tot - URad[FR1]/reduced_c;,
                UGas[MX2] = m2tot - URad[FR2]/reduced_c;,
                UGas[MX3] = m3tot - URad[FR3]/reduced_c;)
        Mnorm = std::sqrt(EXPAND(UGas[MX1]*UGas[MX1] , + UGas[MX2]*UGas[MX2], + UGas[MX3]*UGas[MX3]));

        // Update gas primitive variables
        K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);
        
        // Compute new temperature
        T = VGas[PRS]/(VGas[RHO])*units.Kelvin*mu;
        
        // Compute errors and number of cycles
        err1 = std::abs(1.-URad[ER]/Er_old);
        err2 = std::abs(1.-Fnorm/Fnorm_old);
        err3 = std::abs(1.-UGas[ENG]/Egas_old);
        err4 = std::abs(1.-Mnorm/Mnorm_old);
        count += 1;
      } 

      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        UcRad(nv,k,j,i) = URad[nv];
        VcRad(nv,k,j,i) = URad[nv];
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UcGas(nv,k,j,i) = UGas[nv];
        VcGas(nv,k,j,i) = VGas[nv];
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
  Type_opac kappa_type = this->kappa_type;
  real kappa_0,rho_0,T_0,xi_0;
  if (kappa_type == Type_opac::constant) {
    kappa_0 = this->kappa_0;
  } else if (kappa_type == Type_opac::kramers) {
    kappa_0 = this->kappa_0;
    T_0 = this->T_0;
    rho_0 = this->rho_0;
  }
  Type_opac xi_type = this->xi_type;
  if (xi_type == Type_opac::constant) {
    xi_0 = this->xi_0;
  }
  
  idefix_for("RadSource",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      real Etot = UcGas(ENG,k,j,i)+UcRad(ER,k,j,i)*units.c/(reduced_c*units.velocity);
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
      real T = VGas[PRS]/(VGas[RHO])*units.Kelvin*mu;

      // Compute opacities (out of while loop so that opacity is contant throughout the fixed_point iteration)
      if (kappa_type == Type_opac::constant) {
        kappa_p = kappa_0;
        kappa_r = kappa_0;
      } else if (kappa_type == Type_opac::kramers) {
        kappa_p = kappa_0*std::pow(VGas[RHO]*units.density/rho_0,2.)*std::pow(T/T_0,-3.5);
        kappa_r = kappa_p;
      } else if (kappa_type == Type_opac::usertable) {
        real logT = std::log10(T);
        kappa_p = kp1D.Get(&logT);
        kappa_r = kr1D.Get(&logT);
      }

      if (xi_type == Type_opac::constant) {
        xi = xi_0;
      } else if (xi_type == Type_opac::usertable) {
        real logT = std::log10(T);
        xi = xi1D.Get(&logT);
      }

      while (((err1>tol) || (err2>tol) || (err3>tol) || (err4>tol)) && (count < MAX_ITER)){

        Er_old = URad[ER];
        Fnorm_old = Fnorm;
        Egas_old = UGas[ENG];
        Mnorm_old = Mnorm;
        
        real kk =  units.c * dt * units.time * kappa_p * VGas[RHO]*units.density;
        real xx =  dt * units.time * (xi + kappa_r) * VGas[RHO]*units.density;

        // Stop if UGas <= 0
        if ((Egas_hyp +  kk*(URad[ER]-units.ar*std::pow(T,4)/units.energy))<=ZERO_F) {
          Kokkos::abort("EGas=0 in Radsource");
          UGas[ENG] = 1.e-6;
        } else {
          UGas[ENG] = Egas_hyp +  kk*(URad[ER]-units.ar*std::pow(T,4)/units.energy);
        }

        EXPAND( UGas[MX1] = m1gas_hyp + URad[FR1]*xx;,
                UGas[MX2] = m2gas_hyp + URad[FR2]*xx;,
                UGas[MX3] = m3gas_hyp + URad[FR3]*xx;)
        Mnorm = std::sqrt(EXPAND(UGas[MX1]*UGas[MX1] , + UGas[MX2]*UGas[MX2], + UGas[MX3]*UGas[MX3]));

        URad[ER] = (Etot - UGas[ENG])*reduced_c*units.velocity/units.c;

        // Stop if URad <= 0
        if (URad[ER]<=ZERO_F) {
          Kokkos::abort("ERad=0 in Radsource");
        }

        EXPAND( URad[FR1] = (m1tot - UGas[MX1])*reduced_c;,
                URad[FR2] = (m2tot - UGas[MX2])*reduced_c;,
                URad[FR3] = (m3tot - UGas[MX3])*reduced_c;)

        Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

        K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);
        T = VGas[PRS]/(VGas[RHO])*units.Kelvin*mu;

        err1 = std::abs(1.-UGas[ENG]/Egas_old);
        err2 = std::abs(1.-Mnorm/Mnorm_old);
        err3 = std::abs(1.-URad[ER]/Er_old);
        err4 = std::abs(1.-Fnorm/Fnorm_old);
        count += 1;
        
      }
      
      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        UcRad(nv,k,j,i) = URad[nv];
        VcRad(nv,k,j,i) = URad[nv];
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UcGas(nv,k,j,i) = UGas[nv];
        VcGas(nv,k,j,i) = VGas[nv];
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
  
  IdefixArray3D<real>  dV = this->data->dV;
  IdefixArray3D<real>  A1 = this->data->A[IDIR];
  IdefixArray1D<real>  x1l = this->data->xl[IDIR];
  auto units=idfx::units;
  auto irr1D = this->irr_1D;
  IdefixArray3D<real> divFlux = divFin;

  column_rho->ComputeColumn(this->VcGas);
  IdefixArray3D<real> tau = column_rho->GetColumn();
  real kirr = kappa_irr*units.density*units.length; 
  real flux_pre = std::pow(rs/units.length,2.)*units.sigma_sb*std::pow(Ts,4.)/units.length;

  // Constant kappa
  if(irr_type==Type_irr::constant) {
    
    idefix_for("constant_irr_source",
    data->beg[KDIR], data->end[KDIR],
    data->beg[JDIR], data->end[JDIR],
    data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {

                real Fim = std::exp(-kirr*tau(k,j,i-1))*A1(k,j,i)/std::pow(x1l(i),2.);
                real Fip = std::exp(-kirr*tau(k,j,i))*A1(k,j,i+1)/std::pow(x1l(i+1),2.);
                divFlux(k,j,i) = flux_pre*(Fip-Fim)/dV(k,j,i);
    });

  // Usertable kappa
  } else if (irr_type==Type_irr::usertable) {
    
    idefix_for("usertable_irr_source",
    data->beg[KDIR], data->end[KDIR],
    data->beg[JDIR], data->end[JDIR],
    data->beg[IDIR], data->end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                real logtaum = std::log10(FMAX(tau(k,j,i-1)*units.density*units.length,1.e-15));
                real Fim = pow(10.,irr1D.Get(&logtaum))*A1(k,j,i)/std::pow(x1l(i),2.);
                real logtaup = std::log10(FMAX(tau(k,j,i)*units.density*units.length,1.e-15));
                real Fip = pow(10.,irr1D.Get(&logtaup))*A1(k,j,i+1)/std::pow(x1l(i+1),2.);
                divFlux(k,j,i)  = flux_pre*(Fip-Fim)/dV(k,j,i);
    });
  }

  idfx::popRegion();

} 
