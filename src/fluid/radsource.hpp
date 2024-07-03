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

class RadSource {
 public:
  enum class Type{constant,kramers};
  // Different types of implementation for the radiation source terms.
  template <typename Phys>
  RadSource(Input &, Fluid<Phys> *);
  void ShowConfig();                    // print configuration
  void AddRadSource(const real);
  real Limit_speeds_Rad(int,int,int,real);

  IdefixArray4D<real> UcRad;  // Radiation conservative quantities
  IdefixArray4D<real> UcGas;  // Gas conservative quantities
  IdefixArray4D<real> VcRad;  // Radiation primitive quantities
  IdefixArray4D<real> VcGas;  // Gas primitive quantities
  IdefixArray3D<real> InvDt;  // The InvDt of current radiation multigroup
  Type kappa_type;
  
 private:
  DataBlock* data;
  real kappa_0;
  real rho_0;
  real T_0;
  real xi_rad;
  real reduced_c;
  real gamma;
  int count_max;

  real C_c;
  real C_ar;

  real unit_length;
  real unit_velocity;
  real unit_density;
  real Kelvin;

  // Sound speed computation
  EquationOfState *eos;
  
  // Instance of Unit Class
  idfx::Units *units;

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

  this->C_c = idfx::units.c;
  this->C_ar = idfx::units.ar;

  this->unit_length = idfx::units.length;
  this->unit_velocity = idfx::units.velocity;
  this->unit_density = idfx::units.density;
  this->Kelvin = idfx::units.Kelvin;
  
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
  if(input.CheckEntry("Hydro","gamma")>=0){
      this->gamma =  input.Get<real>("Hydro","gamma",0);
      printf("gamma found!\n");
  } else {
      printf("gamma not found!\n");
  }
 
  if(input.CheckEntry(BlockName,"kappa")>=0) {
    // Fetch the opacity coefficient for the current radiation group.
    const int n = hydroin->instanceNumber;
    this->xi_rad = input.Get<real>(BlockName,"xi",n);

    std::string KappaType = input.Get<std::string>(BlockName,"kappa",0);
    if(KappaType.compare("constant") == 0) {
      this->kappa_type = Type::constant;
      this->kappa_0 = input.Get<real>(BlockName,"kappa",n+1);
    } else if(KappaType.compare("kramers") == 0) {
      this->kappa_0 = input.Get<real>(BlockName,"kappa",n+1);
      this->kappa_type = Type::kramers;
      this->rho_0 = input.Get<real>(BlockName,"kappa",n+2);
      this->T_0 = input.Get<real>(BlockName,"kappa",n+3);
    } else {
      std::stringstream msg;
      msg << "Unknown kappa type \"" <<  KappaType
          << "\" in your input file." << std::endl
          << "Allowed values are: constant, kramers." << std::endl;

      IDEFIX_ERROR(msg);
    }
    

  } else {
    IDEFIX_ERROR("A [Rad] block is required in your input file to define the radiation source terms.");
  }

  idfx::popRegion();
}
#endif // FLUID_RADSOURCE_HPP_
