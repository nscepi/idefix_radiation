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


void RadSource::AddRadSource(const real dt) {
  idfx::pushRegion("RadSource::AddRadSource");

  auto UcGas = this->UcGas;
  auto VcGas = this->VcGas;
  auto UcRad = this->UcRad;
  auto VcRad = this->VcRad;
  auto InvDt = this->InvDt;
  
  const Type kappa_type = this->kappa_type;
  real kappa_0 = this->kappa_0;
  real rho_0 = this->rho_0;
  real T_0 = this->T_0;
  real xi_0 = this->xi_0;
  real reduced_c = this->reduced_c;
  real gamma = this->gamma;
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

  auto kp1D = this->kappa_planck_1D;
  auto kr1D = this->kappa_ross_1D;
  auto xi1D = this->xi_1D;
  
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
      real VRad[RadiationPhysics::nvar];
      real Er_old, Fnorm_old, Egas_old, Mnorm_old;

      real UGas[DefaultPhysics::nvar];
      real VGas[DefaultPhysics::nvar];
      
      real kappa_p,kappa_r,xi;

      if (kappa_type == Type::constant){
        kappa_p = kappa_0;
        kappa_r = kappa_0;
      }

      if (xi_type == Type::constant){
        xi = xi_0;
      }

      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        URad[nv] = UcRad(nv,k,j,i);
        VRad[nv] = VcRad(nv,k,j,i);
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

      // If E_gas > E_rad iterate on radiative field
      //if (UGas[ENG] > URad[ER]/reduced_c){
      if (true){
        while (((err1>tol) || (err2>tol) || (err3>tol) || (err4>tol)) && (count < MAX_ITER)){

          Er_old = URad[ER];
          Fnorm_old = Fnorm;
          Egas_old = UGas[ENG];
          Mnorm_old = Mnorm;
        
          real T = VGas[PRS]/(VGas[RHO])*KELVIN*mu;
          if (kappa_type == Type::kramers){
            kappa_p = kappa_0*(VGas[RHO]*unit_density/rho_0)*std::pow(T/T_0,-3.5);
            kappa_r = kappa_p;
          } else if (kappa_type == Type::usertable){
            real logT = std::log10(T);
            kappa_p = kp1D.Get(&logT);
            kappa_r = kr1D.Get(&logT);
          }
          if (xi_type == Type::usertable){
            real logT = std::log10(T);
            xi = xi1D.Get(&logT);
          }

          real kk_red = reduced_c * unit_velocity * dt * unit_time * kappa_p * VGas[RHO]*unit_density;
          real xx_red = reduced_c * unit_velocity * dt * unit_time * (xi + kappa_r) * VGas[RHO]*unit_density;

          URad[ER] = Er_hyp +  kk_red*C_ar*std::pow(T,4.)/unit_energy;
          URad[ER] /= 1. + kk_red;
          EXPAND( URad[FR1] = Fr1_hyp/(1.+xx_red);,
                  URad[FR2] = Fr2_hyp/(1.+xx_red);,
                  URad[FR3] = Fr3_hyp/(1.+xx_red);)
          Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

          // Change value of s if UGas <= 0
          if ((Etot - URad[ER]*C_c/(reduced_c*unit_velocity))<=ZERO_F) {
            printf("UGas[ENG]=%e URad[ER]*c/c_red=%e Etot=%e at i=%i j=%i and k=%i at iteration %i with Egas=%e and Erad*c/c_red=%e at iteration 0\n",UGas[ENG],URad[ER]*C_c/(reduced_c*unit_velocity),Etot,i,j,k,count,UcGas(ENG,k,j,i),UcRad(ER,k,j,i)*C_c/(reduced_c*unit_velocity));
            throw std::runtime_error("ENG=0 in Radsource");
            //URad[ER] = (Etot -UGas[ENG])*reduced_c;
            //s *= 0.1;
            //continue;
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
          if (count == MAX_ITER - 1) printf("MAX_ITER reached at i %i j %i k %i\n",i,j,k);
        } 
      
      // If E_rad > E_gas iterate on hydro field
      } else {
        while (((err1>tol) || (err2>tol) || (err3>tol) || (err4>tol)) && (count < MAX_ITER)){

          Er_old = URad[ER];
          Fnorm_old = Fnorm;
          Egas_old = UGas[ENG];
          Mnorm_old = Mnorm;
        
          real T = VGas[PRS]/(VGas[RHO])*KELVIN*mu;
          real logT = std::log10(T);
          if (kappa_type == Type::kramers){
            kappa_p = kappa_0*(VGas[RHO]*unit_density/rho_0)*std::pow(T/T_0,-3.5);
            kappa_r = kappa_p;
          } else if (kappa_type == Type::usertable){
            kappa_p = kp1D.Get(&logT);
            kappa_r = kr1D.Get(&logT);
          }
          if (xi_type == Type::usertable){
            real logT = std::log10(T);
            xi = xi1D.Get(&logT);
          }

          real kk =  C_c * dt * unit_time * kappa_p * VGas[RHO]*unit_density;
          real xx =  dt * unit_time * (xi + kappa_r) * VGas[RHO]*unit_density;

          // Stop if UGas <= 0
          if ((Egas_hyp +  kk*(URad[ER]-C_ar*std::pow(T,4)/unit_energy))<=ZERO_F) {
            printf("UGas[ENG]=%e URad[ER]*c/c_red=%e Etot=%e at i=%i j=%i and k=%i at iteration %i with Egas=%e and Erad*c/c_red=%e at iteration 0\n",UGas[ENG],URad[ER]*C_c/(reduced_c*unit_velocity),Etot,i,j,k,count,UcGas(ENG,k,j,i),UcRad(ER,k,j,i)*C_c/(reduced_c*unit_velocity));
            throw std::runtime_error("EGas=0 in Radsource");
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
            printf("UGas[ENG]=%e URad[ER]*c/c_red=%e Etot=%e  at i=%i j=%i and k=%i at iteration %i with Egas=%e and Erad*c/c_red=%e at iteration 0\n",UGas[ENG],URad[ER]/reduced_c,Etot,i,j,k,count,UcGas(ENG,k,j,i),UcRad(ER,k,j,i)/reduced_c);
            throw std::runtime_error("ERad=0 in Radsource");
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
          if (count == MAX_ITER - 1) printf("MAX_ITER reached at i %i j %i k %i\n",i,j,k);
        }
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

real RadSource::Limit_speeds_Rad(int i, int j, int k, real dx) const{
  auto VcGas = this->VcGas;
  real kappa_0 = this->kappa_0;
  auto kappa_type = this->kappa_type;
  real xi_0 = this->xi_0;
  real mu =this->mu;
  real KELVIN = idfx::units.Kelvin;
  real kappa,xi;

  // Compute kappa
  if (kappa_type == Type::constant){
    kappa = kappa_0;
  } else if (kappa_type == Type::kramers){
    real T = VcGas(PRS,k,j,i)/(VcGas(RHO,k,j,i))*KELVIN*mu;
    kappa = kappa_0*std::pow(VcGas(RHO,k,j,i)*unit_density/rho_0,2.)*std::pow(T/T_0,-3.5);
  } else if (kappa_type == Type::usertable){
    real T = VcGas(PRS,k,j,i)/(VcGas(RHO,k,j,i))*KELVIN*mu;
    real logT = std::log10(T);
    auto k_p = this->kappa_planck_1D;
    kappa = k_p.Get(&logT);
  }

  // Compute xi
  if (xi_type == Type::constant){
    xi = xi_0;
  } else if (xi_type == Type::usertable){
    real T = VcGas(PRS,k,j,i)/(VcGas(RHO,k,j,i))*KELVIN*mu;
    real logT = std::log10(T);
    auto xi1D = this->xi_1D;
    xi = xi1D.Get(&logT);
  }

  // Compute optical depth across one cell
  real tau = VcGas(RHO,k,j,i)*idfx::units.density*(kappa+xi)*dx*idfx::units.length;

  // return characteristic velocity of radiative diffusion 
  return 4./(3.*tau)*this->reduced_c;
}

void RadSource::ShowConfig() {
  idfx::cout << "RadSource: kappa is ";
  switch(kappa_type) {
    case Type::constant:
      idfx::cout << "constant." << std::endl;
      break;
    case Type::kramers:
      idfx::cout << "from kramers' law." << std::endl;
      break;
    case Type::usertable:
      idfx::cout << "from a user table." << std::endl;
      break;
  }
  idfx::cout << "RadSource: xi is ";
  switch(xi_type) {
    case Type::constant:
      idfx::cout << "constant." << std::endl;
      break;
    case Type::kramers:
      idfx::cout << "!!! from kramers' law, which is not allowed !!!" << std::endl;
      break;
    case Type::usertable:
      idfx::cout << "from a user table." << std::endl;
      break;
  }
}
