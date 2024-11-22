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


void RadSource::Source_full_implicit(const real dt) {
  idfx::pushRegion("RadSource::Source_full_implicit");

  auto UcGas = this->UcGas;
  auto VcGas = this->VcGas;
  auto UcRad = this->UcRad;
  auto VcRad = this->VcRad;
  auto InvDt = this->InvDt;

  real rho_0 = this->rho_0;
  real T_0 = this->T_0;
  real reduced_c = this->reduced_c;
  real mu = this->mu;
  real gamma = this->gamma;

  const real C_c = idfx::units.c;
  real C_ar = idfx::units.ar;
  real C_cv = idfx::units.k_B/(idfx::units.u*mu);

  real unit_velocity = idfx::units.velocity;
  real unit_length = idfx::units.length;
  real unit_density = idfx::units.density;
  real KELVIN = idfx::units.Kelvin;
  real unit_time = unit_length/unit_velocity;
  real unit_energy = unit_density*unit_velocity*unit_velocity;

  EquationOfState eos = *(this->eos);

  idefix_for("RadSource",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      real Etot = UcGas(ENG,k,j,i)+UcRad(ER,k,j,i)*C_c/(reduced_c*unit_velocity);
      real m1tot = UcGas(MX1,k,j,i)+UcRad(FR1,k,j,i)/reduced_c;
      real m2tot = UcGas(MX2,k,j,i)+UcRad(FR2,k,j,i)/reduced_c;
      real m3tot = UcGas(MX3,k,j,i)+UcRad(FR3,k,j,i)/reduced_c;
      
      real Er_hyp = UcRad(ER,k,j,i);
      real Fr1_hyp = UcRad(FR1,k,j,i);
      real Fr2_hyp = UcRad(FR2,k,j,i);
      real Fr3_hyp = UcRad(FR3,k,j,i);

      real URad[RadiationPhysics::nvar];

      real UGas[DefaultPhysics::nvar];
      real VGas[DefaultPhysics::nvar];
      
      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        URad[nv] = UcRad(nv,k,j,i);
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UGas[nv] = UcGas(nv,k,j,i);
        VGas[nv] = VcGas(nv,k,j,i);
      }

      real Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));
        
      real T = VGas[PRS]/(VGas[RHO])*KELVIN*mu;
      real T3 = std::pow(T,3);

      real kappa_p,kappa_r,xi;
      K_kappa_p(i,j,k,&kappa_p);
      K_kappa_r(i,j,k,&kappa_r);
      K_xi(i,j,k,&xi);

      real kk_red = reduced_c * unit_velocity * dt * unit_time * kappa_p * VGas[RHO]*unit_density;
      real kk = C_c * dt * unit_time * kappa_p * VGas[RHO]*unit_density;
      real xx_red = reduced_c * unit_velocity * dt * unit_time * (xi + kappa_r) * VGas[RHO]*unit_density;
      real xx = C_c * dt * unit_time * (xi + kappa_r) * VGas[RHO]*unit_density;

      real M00 = ONE_F + kk_red;
      real M11 = VGas[RHO]*unit_density*C_cv/(gamma-1.) + 4.*kk*C_ar*T3;
      real M01 = -4.*kk_red*C_ar*T3;
      real M10 = -kk;

      real S0 = Er_hyp*unit_energy - 3.*kk_red*C_ar*T3*T;
      real S1 = VGas[RHO]*unit_density*C_cv*T/(gamma-1.) + 3.*kk*C_ar*T3*T;

      real det = M00*M11 - M01*M10;

      real Minv00 = M11/det;
      real Minv11 = M00/det;
      real Minv01 = -M01/det;
      real Minv10 = -M10/det;

      real Er_new = Minv00*S0 + Minv01*S1;
      real T_new = Minv10*S0 + Minv11*S1;

      URad[ER] = Er_new/unit_energy;
      EXPAND( URad[FR1] = Fr1_hyp/(1.+xx_red);,
              URad[FR2] = Fr2_hyp/(1.+xx_red);,
              URad[FR3] = Fr3_hyp/(1.+xx_red);)
      Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

      if ((Etot - URad[ER]*C_c/(reduced_c*unit_velocity))<=ZERO_F) {
        Kokkos::abort("ENG=0 in Radsource");
      } else {
        UGas[ENG] = Etot - URad[ER]*C_c/(reduced_c*unit_velocity);
      }

      EXPAND( UGas[MX1] = m1tot - URad[FR1]/reduced_c;,
              UGas[MX2] = m2tot - URad[FR2]/reduced_c;,
              UGas[MX3] = m3tot - URad[FR3]/reduced_c;)

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

  idfx::popRegion();
}


void RadSource::Source_fixed_point_rad(const real dt) {
  idfx::pushRegion("RadSource::Source_Fixed_point_rad");

  auto UcGas = this->UcGas;
  auto VcGas = this->VcGas;
  auto UcRad = this->UcRad;
  auto VcRad = this->VcRad;
  auto InvDt = this->InvDt;
  
  const Type_opac kappa_type = this->kappa_type;
  const Type_opac xi_type = this->xi_type;
  real kappa_0 = this->kappa_0;
  real rho_0 = this->rho_0;
  real T_0 = this->T_0;
  real xi_0 = this->xi_0;
  real reduced_c = this->reduced_c;
  real mu = this->mu;

  const real C_c = idfx::units.c;
  real C_ar = idfx::units.ar;

  real unit_velocity = idfx::units.velocity;
  real unit_length = idfx::units.length;
  real unit_density = idfx::units.density;
  real KELVIN = idfx::units.Kelvin;
  real unit_time = unit_length/unit_velocity;
  real unit_energy = unit_density*unit_velocity*unit_velocity;
  // Max iteration for fixed-point solver
  int MAX_ITER = 200;
  // Tolerance on ER and ENG for fixed-point solver
  real tol = 1.e-3;

  EquationOfState eos = *(this->eos);

  idefix_for("RadSource",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      real Etot = UcGas(ENG,k,j,i)+UcRad(ER,k,j,i)*C_c/(reduced_c*unit_velocity);
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

      while (((err1>tol) || (err2>tol) || (err3>tol) || (err4>tol)) && (count < MAX_ITER)){

        Er_old = URad[ER];
        Fnorm_old = Fnorm;
        Egas_old = UGas[ENG];
        Mnorm_old = Mnorm;

        real kappa_p,kappa_r,xi;
        K_kappa_p(i,j,k,&kappa_p);
        K_kappa_r(i,j,k,&kappa_r);
        K_xi(i,j,k,&xi);


        real kk_red = reduced_c * unit_velocity * dt * unit_time * kappa_p * VGas[RHO]*unit_density;
        real xx_red = reduced_c * unit_velocity * dt * unit_time * (xi + kappa_r) * VGas[RHO]*unit_density;

        real T = VGas[PRS]/(VGas[RHO])*KELVIN*mu;
        URad[ER] = Er_hyp +  kk_red*C_ar*std::pow(T,4.)/unit_energy;
        URad[ER] /= 1. + kk_red;
        EXPAND( URad[FR1] = Fr1_hyp/(1.+xx_red);,
                URad[FR2] = Fr2_hyp/(1.+xx_red);,
                URad[FR3] = Fr3_hyp/(1.+xx_red);)
        Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

        if ((Etot - URad[ER]*C_c/(reduced_c*unit_velocity))<=ZERO_F) {
          Kokkos::abort("ENG=0 in Radsource");
        } else {
          UGas[ENG] = Etot - URad[ER]*C_c/(reduced_c*unit_velocity);
        }

        EXPAND( UGas[MX1] = m1tot - URad[FR1]/reduced_c;,
                UGas[MX2] = m2tot - URad[FR2]/reduced_c;,
                UGas[MX3] = m3tot - URad[FR3]/reduced_c;)
        Mnorm = std::sqrt(EXPAND(UGas[MX1]*UGas[MX1] , + UGas[MX2]*UGas[MX2], + UGas[MX3]*UGas[MX3]));

        K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);

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

  idfx::popRegion();
}


void RadSource::Source_fixed_point_gas(const real dt) {
  idfx::pushRegion("RadSource::Source_Fixed_point_gas");

  auto UcGas = this->UcGas;
  auto VcGas = this->VcGas;
  auto UcRad = this->UcRad;
  auto VcRad = this->VcRad;
  auto InvDt = this->InvDt;
  
  const Type_opac kappa_type = this->kappa_type;
  const Type_opac xi_type = this->xi_type;
  real kappa_0 = this->kappa_0;
  real rho_0 = this->rho_0;
  real T_0 = this->T_0;
  real xi_0 = this->xi_0;
  real reduced_c = this->reduced_c;
  real mu = this->mu;

  const real C_c = idfx::units.c;
  real C_ar = idfx::units.ar;

  real unit_velocity = idfx::units.velocity;
  real unit_length = idfx::units.length;
  real unit_density = idfx::units.density;
  real KELVIN = idfx::units.Kelvin;
  real unit_time = unit_length/unit_velocity;
  real unit_energy = unit_density*unit_velocity*unit_velocity;
  // Max iteration for fixed-point solver
  int MAX_ITER = 200;
  // Tolerance on ER and ENG for fixed-point solver
  real tol = 1.e-3;

  EquationOfState eos = *(this->eos);

  idefix_for("RadSource",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      real Etot = UcGas(ENG,k,j,i)+UcRad(ER,k,j,i)*C_c/(reduced_c*unit_velocity);
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

      while (((err1>tol) || (err2>tol) || (err3>tol) || (err4>tol)) && (count < MAX_ITER)){

        Er_old = URad[ER];
        Fnorm_old = Fnorm;
        Egas_old = UGas[ENG];
        Mnorm_old = Mnorm;

        real kappa_p,kappa_r,xi;
        K_kappa_p(i,j,k,&kappa_p);
        K_kappa_r(i,j,k,&kappa_r);
        K_xi(i,j,k,&xi);
        
        real kk =  C_c * dt * unit_time * kappa_p * VGas[RHO]*unit_density;
        real xx =  dt * unit_time * (xi + kappa_r) * VGas[RHO]*unit_density;

        real T = VGas[PRS]/(VGas[RHO])*KELVIN*mu;

        // Stop if UGas <= 0
        if ((Egas_hyp +  kk*(URad[ER]-C_ar*std::pow(T,4)/unit_energy))<=ZERO_F) {
          Kokkos::abort("EGas=0 in Radsource");
          UGas[ENG] = 1.e-6;
        } else {
          UGas[ENG] = Egas_hyp +  kk*(URad[ER]-C_ar*std::pow(T,4)/unit_energy);
        }

        EXPAND( UGas[MX1] = m1gas_hyp + URad[FR1]*xx;,
                UGas[MX2] = m2gas_hyp + URad[FR2]*xx;,
                UGas[MX3] = m3gas_hyp + URad[FR3]*xx;)
        Mnorm = std::sqrt(EXPAND(UGas[MX1]*UGas[MX1] , + UGas[MX2]*UGas[MX2], + UGas[MX3]*UGas[MX3]));

        URad[ER] = (Etot - UGas[ENG])*reduced_c*unit_velocity/C_c;

        // Stop if URad <= 0
        if (URad[ER]<=ZERO_F) {
          Kokkos::abort("ERad=0 in Radsource");
        }

        EXPAND( URad[FR1] = (m1tot - UGas[MX1])*reduced_c;,
                URad[FR2] = (m2tot - UGas[MX2])*reduced_c;,
                URad[FR3] = (m3tot - UGas[MX3])*reduced_c;)

        Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

        K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);

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

  idfx::popRegion();
}

void RadSource::ShowConfig() {
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
}

void RadSource::AddRadSource(const real dt) {
  idfx::pushRegion("RadSource::AddRadSource");

  switch(source_solver) {
    case Type_isolver::full_implicit:
      RadSource::Source_full_implicit(dt);
      break;
    case Type_isolver::fixed_point_rad:
      RadSource::Source_fixed_point_rad(dt);
      break;
    case Type_isolver::fixed_point_gas:
      RadSource::Source_fixed_point_gas(dt);
      break;
  }

  idfx::popRegion();
}
