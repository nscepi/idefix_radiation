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

class RadSource {
 public:
  enum class Type{Tconst,Tvar};
  // Different types of implementation for the radiation source terms.
  template <typename Phys>
  RadSource(Input &, Fluid<Phys> *);
  void ShowConfig();                    // print configuration
  void AddRadSource(const real);

  IdefixArray4D<real> UcRad;  // Radiation conservative quantities
  IdefixArray4D<real> UcGas;  // Gas conservative quantities
  IdefixArray4D<real> VcRad;  // Radiation primitive quantities
  IdefixArray4D<real> VcGas;  // Gas primitive quantities
  IdefixArray3D<real> InvDt;  // The InvDt of current radiation multigroup
  Type type;

 private:
  DataBlock* data;
  real kappa_rad;
  real xi_rad;
  real reduced_c;
  real gamma;
  real unit_velocity;
  real unit_length;
  real unit_mass;
  int count_max;

  // Sound speed computation
  EquationOfState *eos;

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

  // Physical units 
  if(input.CheckEntry("Setup","unit_velocity")>=0){
      this->unit_velocity =  input.Get<real>("Setup","unit_velocity",0);
  }
  if(input.CheckEntry("Setup","unit_length")>=0){
      this->unit_length =  input.Get<real>("Setup","unit_length",0);
  }
  if(input.CheckEntry("Setup","unit_mass")>=0){
      this->unit_mass =  input.Get<real>("Setup","unit_mass",0);
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
 
  if(input.CheckEntry(BlockName,"radsource")>=0) {
    std::string RadType = input.Get<std::string>(BlockName,"radsource",0);
    if(RadType.compare("Tconst") == 0) {
      this->type = Type::Tconst;
    } else if(RadType.compare("Tvar") == 0) {
      this->type = Type::Tvar;
    } else {
      std::stringstream msg;
      msg << "Unknown radsource type \"" <<  RadType
          << "\" in your input file." << std::endl
          << "Allowed values are: Tconst, Tvar." << std::endl;

      IDEFIX_ERROR(msg);
    }
    // Fetch the opacity coefficient for the current radiation group.
    const int n = hydroin->instanceNumber;
    this->kappa_rad = input.Get<real>(BlockName,"kappa",n);
    this->xi_rad = input.Get<real>(BlockName,"xi",n);

  } else {
    IDEFIX_ERROR("A [Rad] block is required in your input file to define the radiation source terms.");
  }


  idfx::popRegion();
}
#endif // FLUID_RADSOURCE_HPP_
