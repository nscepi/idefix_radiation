// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************
#ifndef FLUID_RADSOURCE_HPP_
#define FLUID_RADSOURCE_HPP_

#include <string>
#include "idefix.hpp"
#include "input.hpp"
#include "fluid_defs.hpp"
#include "eos.hpp"
#include "units.hpp"
#include "lookupTable.hpp"

class RadSource {
 public:
  enum class Type{constant,kramers,usertable};
  // Different types of implementation for the radiation source terms.
  template <typename Phys>
  RadSource(Input &, Fluid<Phys> *);
  void ShowConfig();                    // print configuration
  void AddRadSource(const real);
  
  KOKKOS_INLINE_FUNCTION real Limit_speeds_Rad(int i, int j, int k, real dx) const {
    auto VcGas = this->VcGas;
    real kappa_0 = this->kappa_0;
    auto kappa_type = this->kappa_type;
    real xi_0 = this->xi_0;
    real mu =this->mu;
    real kappa,xi;

    real unit_density = this->unit_density;
    real unit_length = this->unit_length;
    real KELVIN = this->Kelvin;
    
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
    real tau = VcGas(RHO,k,j,i)*unit_density*(kappa+xi)*dx*unit_length;

    // return characteristic velocity of radiative diffusion 
    return 4./(3.*tau)*this->reduced_c;
  };

  IdefixArray4D<real> UcRad;  // Radiation conservative quantities
  IdefixArray4D<real> UcGas;  // Gas conservative quantities
  IdefixArray4D<real> VcRad;  // Radiation primitive quantities
  IdefixArray4D<real> VcGas;  // Gas primitive quantities
  IdefixArray3D<real> InvDt;  // The InvDt of current radiation multigroup
  Type kappa_type;
  Type xi_type;
  
 private:
  DataBlock* data;
  real kappa_0;
  real xi_0;
  real rho_0;
  real T_0;
  real kappa_irr;
  real reduced_c;
  real gamma;
  real mu;
  int count_max;

  real C_c;
  real C_ar;

  // Units
  real unit_length;
  real unit_velocity;
  real unit_density;
  real Kelvin;

  // Sound speed computation
  EquationOfState *eos;

  // Planck and Rosseland opacities 
  int kappa_ndim;
  int xi_ndim;
  
  LookupTable<1> kappa_planck_1D;
  LookupTable<1> kappa_ross_1D;
  LookupTable<1> xi_1D;

};

#include "fluid.hpp"

template<typename Phys>
RadSource::RadSource(Input &input, Fluid<Phys> *hydroin):
                      UcRad{hydroin->Uc},
                      UcGas{hydroin->data->hydro->Uc},
                      VcRad{hydroin->Vc},
                      VcGas{hydroin->data->hydro->Vc},
                      InvDt{hydroin->InvDt} {
  idfx::pushRegion("RadSource::RadSource");
  // Save the parent hydro object

  this->data = hydroin->data;
  this->eos = hydroin->data->hydro->eos.get();

  // Check in which block we should fetch our information
  std::string BlockName;
  if(Phys::radiation) {
    BlockName = "Rad";
  } else {
    IDEFIX_ERROR("Fluid is not radiative");
  }

  // Reduced velocity of light 
  this->reduced_c =  hydroin->reduced_c;
  
  // Adiabatic index
  this->gamma = this->eos->GetGamma();
  
  // Mean molecular weight
  this->mu = this->eos->GetMu();

  // Units
  this->unit_length = idfx::units.length;
  this->unit_velocity = idfx::units.velocity;
  this->unit_density = idfx::units.density;
  this->Kelvin = idfx::units.Kelvin ;

  if(input.CheckEntry(BlockName,"xi")>=0) {
    // Fetch the opacity coefficient for the current radiation group.
    const int n = hydroin->instanceNumber;

    std::string xiType = input.Get<std::string>(BlockName,"xi",0);
    if(xiType.compare("constant") == 0) {
      this->xi_type = Type::constant;
      this->xi_0 = input.Get<real>(BlockName,"xi",n+1);
    } else if(xiType.compare("usertable") == 0) {
      this->xi_type = Type::usertable;
      this->xi_ndim = input.Get<int>(BlockName,"xi",n+1);
      std::string xi_file = input.Get<std::string>(BlockName,"xi",n+2);
      if (input.Get<int>(BlockName,"xi",n+1) == 1){
        this->xi_1D = LookupTable<1>(xi_file,',');
      } else {
        std::stringstream msg;
        msg << "Only 1 dimension for scattering opacity tables are currently accepted." << std::endl;
        IDEFIX_ERROR(msg);
      }
    } else {
      std::stringstream msg;
      msg << "Unknown xi type \"" <<  xiType
          << "\" in your input file." << std::endl
          << "Allowed values are: constant, usertable." << std::endl;

      IDEFIX_ERROR(msg);
    }
  }

  if(input.CheckEntry(BlockName,"kappa_irr")>=0) {
    const int n = hydroin->instanceNumber;
    this->kappa_irr = input.Get<real>(BlockName,"kappa_irr",n);
  }

  if(input.CheckEntry(BlockName,"kappa")>=0) {
    // Fetch the opacity coefficient for the current radiation group.
    const int n = hydroin->instanceNumber;

    std::string kappaType = input.Get<std::string>(BlockName,"kappa",0);
    if(kappaType.compare("constant") == 0) {
      this->kappa_type = Type::constant;
      this->kappa_0 = input.Get<real>(BlockName,"kappa",n+1);
    } else if(kappaType.compare("kramers") == 0) {
      this->kappa_0 = input.Get<real>(BlockName,"kappa",n+1);
      this->kappa_type = Type::kramers;
      this->rho_0 = input.Get<real>(BlockName,"kappa",n+2);
      this->T_0 = input.Get<real>(BlockName,"kappa",n+3);
    } else if(kappaType.compare("usertable") == 0) {
      this->kappa_type = Type::usertable;
      this->kappa_ndim = input.Get<int>(BlockName,"kappa",n+1);
      std::string kappap_file = input.Get<std::string>(BlockName,"kappa",n+2);
      std::string kappar_file = input.Get<std::string>(BlockName,"kappa",n+3);
      if (input.Get<int>(BlockName,"kappa",n+1) == 1){
        this->kappa_planck_1D = LookupTable<1>(kappap_file,',');
        this->kappa_ross_1D = LookupTable<1>(kappar_file,',');
      } else {
        std::stringstream msg;
        msg << "Only 1 dimension for absorption opacity tables are currently accepted." << std::endl;
        IDEFIX_ERROR(msg);
      }
      
    } else {
      std::stringstream msg;
      msg << "Unknown kappa type \"" <<  kappaType
          << "\" in your input file." << std::endl
          << "Allowed values are: constant, kramers, usertable." << std::endl;

      IDEFIX_ERROR(msg);
    }
    

  } else {
    IDEFIX_ERROR("A [Rad] block is required in your input file to define the radiation source terms.");
  }

  idfx::popRegion();
}

#endif // FLUID_RADSOURCE_HPP_
