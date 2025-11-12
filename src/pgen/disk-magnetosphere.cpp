//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file disk-magnetosphere.cpp
//! \brief Initializes disk interacting with magnetosphere of central star.

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt(), pow()
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <cstdio>     // snprintf()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

// CFHERE .....

// prototypes for functions used internally to this pgen
namespace {


KOKKOS_INLINE_FUNCTION
static void GetCylCoord(Real &rad,Real &phi,Real &z,Real &x1,Real &x2,Real &x3)

KOKKOS_INLINE_FUNCTION
static Real VelProfileCyl(struct warp_pgen pgen, const Real rad, const Real phi, const Real z);

// CALLUM: HydroDiffusion Class is not implemented in this version of the code
// void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
//                     const AthenaArray<Real> &bcc, int is, int ie, int js, int je,
//                     int ks, int ke);

// KOKKOS_INLINE_FUNCTION
// static void CartToCylCoord(struct warp_pgen pgen, Real &rad, Real &phi, Real &z, const Real xcart, const Real ycart, const Real zcart);

// KOKKOS_INLINE_FUNCTION
// static void CylToCartCoord(struct warp_pgen pgen, Real &xcart, Real &ycart, Real &zcart, const Real rad, const Real phi, const Real z);

// KOKKOS_INLINE_FUNCTION
// static void CylToSphCoord(struct warp_pgen pgen, Real &r, Real &theta, Real &phi, const Real rad_cyl, const Real phi_cyl, const Real z_cyl);

// KOKKOS_INLINE_FUNCTION
// static void SphToCartCoord(struct warp_pgen pgen, Real &xcart, Real &ycart, Real &zcart, const Real r, const Real theta, const Real phi);

// KOKKOS_INLINE_FUNCTION
// static void CartToSphCoord(struct warp_pgen pgen, Real &r, Real &theta, Real &phi, const Real xcart, const Real ycart, const Real zcart);

// KOKKOS_INLINE_FUNCTION
// static void SphToCartVec(struct warp_pgen pgen, Real &vec_x,Real &vec_y,Real &vec_z, const Real vec_r, const Real vec_theta, const Real vec_phi, const Real theta, const Real phi);

// KOKKOS_INLINE_FUNCTION
// static void CartToSphVec(struct warp_pgen pgen, Real &vec_r,Real &vec_theta,Real &vec_phi, const Real vec_x, const Real vec_y, const Real vec_z, const Real theta, const Real phi);

// KOKKOS_INLINE_FUNCTION
// static void CylToCartVec(struct warp_pgen pgen, Real &vec_x,Real &vec_y,Real &vec_z, const Real vec_cr, const Real vec_cphi, const Real vec_cz, const Real phi);

// KOKKOS_INLINE_FUNCTION
// static void CartToCylVec(struct warp_pgen pgen, Real &vec_cr,Real &vec_cphi,Real &vec_cz, const Real vec_x, const Real vec_y, const Real vec_z, const Real phi);

// KOKKOS_INLINE_FUNCTION
// static void GetWarpBeta(struct warp_pgen pgen, const Real rad, Real &warp_beta);

// KOKKOS_INLINE_FUNCTION
// static void GetWarpGamma(struct warp_pgen pgen, const Real rad, Real &warp_gamma);

// KOKKOS_INLINE_FUNCTION
// static void InverseWarpRot(struct warp_pgen pgen, const Real warp_beta, const Real warp_gamma, Real &xflat, Real &yflat, Real &zflat, const Real xwarp, const Real ywarp, const Real zwarp);

// KOKKOS_INLINE_FUNCTION
// static void WarpRot(struct warp_pgen pgen, const Real warp_beta, const Real warp_gamma, Real &xwarp, Real &ywarp, Real &zwarp, const Real xflat, const Real yflat, const Real zflat);

// KOKKOS_INLINE_FUNCTION
// static void GetUnwarpedCylCoord(struct warp_pgen pgen, Real &rad,Real &phi,Real &z,const Real rwarp,const Real thetawarp,const Real phiwarp);

// KOKKOS_INLINE_FUNCTION
// static void ComputePrimitives(struct warp_pgen pgen, Real xwarp, Real ywarp, Real zwarp, Real &rho, Real &pgas, Real &ux, Real &uy, Real &uz);

// Useful container for physical parameters of the warped disc
struct warp_pgen {
    Real gm0, r0, rminmask, rmaxmask,thetaminmask,thetamaxmask,softening_len, rho0, d_slope, c0sq, s_slope, gamma_gas, beta, tau, alpha;
    int relax_cs, damp_switch;
    Real r_warp_mid, warp_beta0;
    Real dfloor;
    enum eos_enum {ideal, isothermal} eos_flag;
    eos_enum eos_isothermal=isothermal;
    eos_enum eos_ideal=ideal;
    int error=0;
    };

// Now store porblem parameters in a struct to be passed to functions for
// more controlled access.
warp_pgen disc_params;

} // End of namespace

// prototypes for user-defined BCs and source functions
void FixedDiscBC(Mesh *pm);
void MySourceTerms(Mesh* pm, const Real bdt);

//----------------------------------------------------------------------------------------
//! \fn
//! \brief Problem Generator for warped disc experiments. Sets initial conditions for an equilibrium
//! which is then rotated in spherical shells to introduce a radially dependent tilt and twait profile.
//! Compile with '-D PROBLEM=warp_disc' to enroll as a user-specific problem generator.
//----------------------------------------------------------------------------------------

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

    // Now enroll user source terms and boundary conditions if specified
    if (user_srcs) {
        user_srcs_func = MySourceTerms;
     }

    if (user_bcs) {
        user_bcs_func = FixedDiscBC;
    }

    // If restarting then end initialisation here
    if (restart) return;

    // Read problem parameters from input file
    // Get parameters for gravitatonal potential of softened central point mass
    disc_params.gm0 = pin->GetOrAddReal("problem","GM_soft",0.0);
    disc_params.r0 = pin->GetOrAddReal("problem","r0",1.0);
    disc_params.softening_len = pin->GetOrAddReal("problem","softening_len",disc_params.r0/100.0);
    disc_params.rminmask = pin->GetOrAddReal("problem","rminmask",0.0);
    disc_params.rmaxmask = pin->GetOrAddReal("problem","rmaxmask",0.0);
    disc_params.thetaminmask = pin->GetOrAddReal("problem","thetaminmask",0.0);
    disc_params.thetamaxmask = pin->GetOrAddReal("problem","thetamaxmask",0.0);

    // Get parameters for initial density
    disc_params.rho0 = pin->GetReal("problem","rho0");
    disc_params.d_slope = pin->GetOrAddReal("problem","d_slope",0.0);

    // Get parameters of initial pressure and cooling parameters

    // First extract the EOS - currently the can choose from 'isothermal' (globally) or 'ideal' (adiabatic)
    disc_params.eos_flag = disc_params.eos_isothermal;
    if (pin->GetString("hydro","eos").compare("isothermal") != 0) {
        disc_params.eos_flag=disc_params.eos_ideal;
        disc_params.c0sq = pin->GetOrAddReal("problem","c0sq",0.0025);
        disc_params.s_slope = pin->GetOrAddReal("problem","s_slope",0.0);
        disc_params.gamma_gas = pin->GetReal("hydro","gamma");
        disc_params.relax_cs = pin->GetOrAddInteger("problem","relax_cs",0);
        disc_params.beta = pin->GetReal("problem","beta");
    } else {
        disc_params.c0sq=SQR(pin->GetReal("hydro","iso_sound_speed"));
    }

    // Get parameters for velocity damping
    disc_params.damp_switch = pin->GetOrAddInteger("problem","damp_switch",0);
    disc_params.tau = pin->GetOrAddReal("problem","tau",0.1);

    // Get parameters for alpha viscosity
    disc_params.alpha =  pin->GetOrAddReal("problem","nu_iso",0.0);

    // Get parameters for the warp profile
    disc_params.r_warp_mid =  pin->GetOrAddReal("problem","r_warp_mid",0.0);
    disc_params.warp_beta0 =  pin->GetOrAddReal("problem","warp_beta0",0.0);

    // Set density floor
    Real float_min = std::numeric_limits<float>::min();
    disc_params.dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

    // Capture variables for kernel - e.g. indices for looping over the meshblocks and the size of the meshblocks.
    auto &indcs = pmy_mesh_->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
    auto &size = pmbp->pmb->mb_size;

    // Initialise a pointer to the disc parameter structure   
    auto disc_params_ = disc_params;

    // Select either Hydro or MHD and extract the arrays - set on the device specifically since this is where the calculations
    // are going to be done anyway. 
    DvceArray5D<Real> u0_, w0_;
    if (pmbp->phydro != nullptr) {
        u0_ = pmbp->phydro->u0;
        w0_ = pmbp->phydro->w0;
    }

    // Loop over array and assign the quantities.
    par_for("pgen_UserProblem",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real xwarp = CellCenterX(i-is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real ywarp = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        int nx3 = indcs.nx3;
        Real zwarp = CellCenterX(k-ks, nx3, x3min, x3max);

        // Declare the primtive variables
        Real den(0.0), pgas(0.0), ux(0.0), uy(0.0), uz(0.0);

        // Now compute the warped primitive variables at this location
        ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

        // Now set the conserved variables using the primitive variables
        u0_(m,IDN,k,j,i) = den;
        u0_(m,IM1,k,j,i) = den*ux;
        u0_(m,IM2,k,j,i) = den*uy;
        u0_(m,IM3,k,j,i) = den*uz;
        if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
            u0_(m,IEN,k,j,i) = pgas/(disc_params_.gamma_gas - 1.0)+0.5*(SQR(ux)+SQR(uy)+ SQR(uz))/den;
        }
    });

    // Check that no errors were flagged during the initialisation
    if (disc_params_.error == 1) {
        std::cout << "Error: Negative azimuthal velocity detected. Check your input parameters." << std::endl;
        exit(1);
    }

    return; // END OF ProblemGenerator::UserProblem()
}

//----------------------------------------------------------------------------------------
//! Now we define a variety of functions for use in the problem generation within the local 
//! namespace. Compatible with their declaration in the preamble.
//----------------------------------------------------------------------------------------

namespace {

    //----------------------------------------------------------------------------------------
    //! Transform from cartesian to cylindrical coordinates
    KOKKOS_INLINE_FUNCTION
    static void CartToCylCoord(struct warp_pgen pgen, Real &rad, Real &phi, Real &z, const Real xcart, const Real ycart, const Real zcart) {
    
    rad = std::sqrt(xcart*xcart+ycart*ycart);
    phi = std::atan2(ycart,xcart);
    z = zcart;
    
    return;
    }
        
    // //----------------------------------------------------------------------------------------
    // //! Transform from cartesian to spherical from cylindrical coordinates
    // KOKKOS_INLINE_FUNCTION
    // static void CylToSphCoord(struct warp_pgen pgen, Real &r, Real &theta, Real &phi, const Real rad_cyl, const Real phi_cyl, const Real z_cyl) {
    
    // r = std::sqrt(rad_cyl*rad_cyl+z_cyl*z_cyl);
    // theta = std::atan2(rad_cyl,z_cyl);
    // phi = phi_cyl;
    
    // return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Transform from cylindrical to cartesian coordinates
    // KOKKOS_INLINE_FUNCTION
    // static void CylToCartCoord(struct warp_pgen pgen, Real &xcart, Real &ycart, Real &zcart, const Real rad, const Real phi, const Real z) {
    
    // xcart = rad*std::cos(phi);
    // ycart = rad*std::sin(phi);
    // zcart = z;

    // return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Transform from spherical to cartesian coordinates
    // KOKKOS_INLINE_FUNCTION
    // static void SphToCartCoord(struct warp_pgen pgen, Real &xcart, Real &ycart, Real &zcart, const Real r, const Real theta, const Real phi) {

    // xcart=r*std::sin(theta)*std::cos(phi);
    // ycart=r*std::sin(theta)*std::sin(phi);
    // zcart=r*std::cos(theta);

    // return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Transform from cartesian to spherical coordinates
    // KOKKOS_INLINE_FUNCTION
    // static void CartToSphCoord(struct warp_pgen pgen, Real &r, Real &theta, Real &phi, const Real xcart, const Real ycart, const Real zcart) {

    // r=std::sqrt(xcart*xcart+ycart*ycart+zcart*zcart);
    // theta=std::atan2(std::sqrt(xcart*xcart+ycart*ycart),zcart);
    // phi=std::atan2(ycart,xcart);

    // return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Transform the local vector defined in terms of spherical basis to the same vector in terms of cartesian basis. 
    // KOKKOS_INLINE_FUNCTION
    // static void SphToCartVec(struct warp_pgen pgen, Real &vec_x,Real &vec_y,Real &vec_z, const Real vec_r, const Real vec_theta, const Real vec_phi, const Real theta, const Real phi) {

    // Real ctheta = std::cos(theta);
    // Real stheta = std::sin(theta);
    // Real cphi = std::cos(phi);
    // Real sphi = std::sin(phi);

    // vec_x = stheta*cphi*vec_r+ctheta*cphi*vec_theta-sphi*vec_phi;
    // vec_y = stheta*sphi*vec_r+ctheta*sphi*vec_theta+cphi*vec_phi;
    // vec_z   = ctheta*vec_r-stheta*vec_theta;

    // return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Transform the local vector defined in terms of cartesian basis to the same vector in terms of spherical basis. 
    // KOKKOS_INLINE_FUNCTION
    // static void CartToSphVec(struct warp_pgen pgen, Real &vec_r,Real &vec_theta,Real &vec_phi, const Real vec_x, const Real vec_y, const Real vec_z, const Real theta, const Real phi) {

    // Real ctheta = std::cos(theta);
    // Real stheta = std::sin(theta);
    // Real cphi = std::cos(phi);
    // Real sphi = std::sin(phi);

    // vec_r     = stheta*cphi*vec_x+stheta*sphi*vec_y+ctheta*vec_z;
    // vec_theta = ctheta*cphi*vec_x+ctheta*sphi*vec_y-stheta*vec_z;
    // vec_phi   = -sphi*vec_x+cphi*vec_y;

    // return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Transform the local vector defined in terms of cylindrical basis to the same vector in terms of cartesian basis. 
    // KOKKOS_INLINE_FUNCTION
    // static void CylToCartVec(struct warp_pgen pgen, Real &vec_x,Real &vec_y,Real &vec_z, const Real vec_cr, const Real vec_cphi, const Real vec_cz, const Real phi) {

    // Real cphi = std::cos(phi);
    // Real sphi = std::sin(phi);

    // vec_x = cphi*vec_cr-sphi*vec_cphi;
    // vec_y = sphi*vec_cr+cphi*vec_cphi;
    // vec_z = vec_cz;

    // return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Transform the local vector defined in terms of cartesian basis to the same vector in terms of cylindrical basis. 
    // KOKKOS_INLINE_FUNCTION
    // static void CartToCylVec(struct warp_pgen pgen, Real &vec_cr,Real &vec_cphi,Real &vec_cz, const Real vec_x, const Real vec_y, const Real vec_z, const Real phi) {

    // Real cphi = std::cos(phi);
    // Real sphi = std::sin(phi);

    // vec_cr = cphi*vec_x+sphi*vec_y;
    // vec_cphi = -sphi*vec_x+cphi*vec_y;
    // vec_cz = vec_z;

    // return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Warp profiles for beta (tilt) and gamma (twist)
    // KOKKOS_INLINE_FUNCTION
    // static void GetWarpBeta(struct warp_pgen pgen, const Real rad, Real &warp_beta){

    //     warp_beta = pgen.warp_beta0*std::exp(-SQR(std::abs(rad-pgen.r_warp_mid)));
    //     return;
    // }

    // KOKKOS_INLINE_FUNCTION
    // static void GetWarpGamma(struct warp_pgen pgen, const Real rad, Real &warp_gamma){
    //     warp_gamma = 0.0;
    //     return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Tilt rotation on cartesian coordinate
    // KOKKOS_INLINE_FUNCTION
    // static void InverseWarpRot(struct warp_pgen pgen, const Real warp_beta, const Real warp_gamma, Real &xflat, Real &yflat, Real &zflat, const Real xwarp, const Real ywarp, const Real zwarp){

    //     // Initialise the relevant trig angles.
    //     Real cb = std::cos(warp_beta);
    //     Real sb = std::sin(warp_beta);
    //     Real cg = std::cos(warp_gamma);
    //     Real sg = std::sin(warp_gamma);

    //     // This function performs the inverse warping rotation and returns the unwarped cartesian coordinates
    //     xflat = cb*cg*xwarp+cb*sg*ywarp-sb*zwarp;
    //     yflat = -sg*xwarp+cg*ywarp;
    //     zflat = sb*cg*xwarp+sb*sg*ywarp+cb*zwarp;

    //     return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Tilt rotation on cartesian cooridnates
    // KOKKOS_INLINE_FUNCTION  
    // static void WarpRot(struct warp_pgen pgen, const Real warp_beta, const Real warp_gamma, Real &xwarp, Real &ywarp, Real &zwarp, const Real xflat, const Real yflat, const Real zflat){

    //     // Initialise the relevant trig angles.
    //     Real cb = std::cos(warp_beta);
    //     Real sb = std::sin(warp_beta);
    //     Real cg = std::cos(warp_gamma);
    //     Real sg = std::sin(warp_gamma);

    //     // This function performs the inverse warping rotation and returns the unwarped cartesian coordinates
    //     xwarp = cg*cb*xflat+cg*sb*zflat-sg*yflat;
    //     ywarp = sg*cb*xflat+sg*sb*zflat+cg*yflat;
    //     zwarp = -sb*xflat+cb*zflat;

    //     return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Get unwarped cylindrical coordinates given input warped cartesian grid.
    // // This function takes us from the actual grid point and then subtracts off the desired, initial warp profile.
    // KOKKOS_INLINE_FUNCTION  
    // static void GetUnwarpedCylCoord(struct warp_pgen pgen, Real &rad,Real &phi,Real &z,const Real xwarp,const Real ywarp,const Real zwarp){

    //     // Exract the spherical radius from the cartesian coordinates
    //     Real rwarp = std::sqrt(SQR(xwarp)+SQR(ywarp)+SQR(zwarp));

    //     // Get the warping angles at this position from the chosen warp profile
    //     Real warp_beta(0.0), warp_gamma(0.0);
    //     GetWarpBeta(pgen,rwarp,warp_beta);
    //     GetWarpGamma(pgen,rwarp,warp_gamma);

    //     // Now rotate these to the unwarped reference frame and modify the flat cartesian coordinates.
    //     Real xflat(0.0), yflat(0.0), zflat(0.0);
    //     InverseWarpRot(pgen,warp_beta, warp_gamma, xflat, yflat, zflat, xwarp, ywarp, zwarp);

    //     // Finally convert these to the desired unwarped cylindrical coordinates
    //     CartToCylCoord(pgen,rad,phi,z,xflat,yflat,zflat);

    //     return;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Computes density in cylindrical coordinates
    // KOKKOS_INLINE_FUNCTION  
    // static Real DenProfileCyl(struct warp_pgen pgen, const Real rad, const Real phi, const Real z) {
    //     Real den;
    //     Real csq = pgen.c0sq;
    //     enum Masking {nomask,innermask,outermask} inner=innermask;

    //     if (pgen.eos_flag != pgen.eos_isothermal) csq = CSoundSqCyl(pgen,rad, phi, z);

    //     if (MaskingCylCoords(pgen,rad,phi,z)==inner){
    //         den = 1.0;
    //     } else {
    //         Real denmid = pgen.rho0*pow(rad/pgen.r0,-pgen.d_slope);
    //         Real dentem = denmid*std::exp(pgen.gm0/csq*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
    //         den = dentem;
    //     }
    //     return fmax(den,pgen.dfloor);
    // }

    // //----------------------------------------------------------------------------------------
    // //! Target cylindrical sound speed squared = pressure/density
    // // Constant on cylinders of constant R
    // KOKKOS_INLINE_FUNCTION  
    // static Real CSoundSqCyl(struct warp_pgen pgen, const Real rad, const Real phi, const Real z) {
    // Real csq;
    // enum Masking {nomask,innermask,outermask};

    // if (MaskingCylCoords(pgen,rad,phi,z)==innermask){
    //     csq = 0.1;
    // } else {
    //     csq = pgen.c0sq*pow(rad/pgen.r0, -pgen.s_slope);
    // }
    // return csq;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Target spherical sound speed squared 
    // // Constant on shells of constant r
    // KOKKOS_INLINE_FUNCTION  
    // static Real CSoundSqSph(struct warp_pgen pgen, const Real rad, const Real theta, const Real phi) {
    // Real csq;
    // enum Masking {nomask,innermask,outermask};

    // if (MaskingSphCoords(pgen,rad,theta,phi)==innermask){
    //     csq = 0.1;
    // } else {
    //     csq = pgen.c0sq*pow(rad/pgen.r0, -pgen.s_slope);
    // }
    // return csq;
    // }

    // //----------------------------------------------------------------------------------------
    // //! Computes rotational velocity in cylindrical coordinates
    // KOKKOS_INLINE_FUNCTION  
    // static Real VelProfileCyl(struct warp_pgen pgen, const Real rad, const Real phi, const Real z) {
    //     enum Masking {nomask,innermask,outermask};

    //     if (MaskingCylCoords(pgen,rad,phi,z)==innermask){
    //         Real vel = 0.01;
    //         return vel;
            
    //     } else {
    //         Real csq = CSoundSqCyl(pgen, rad, phi, z);
    //         Real vel = (-pgen.d_slope-pgen.s_slope)*csq/(pgen.gm0/rad)+(1.0-pgen.s_slope)+pgen.s_slope*rad/std::sqrt(rad*rad+z*z);
            
    //         if (vel < 0.0){
    //             pgen.error=1;
    //         }
    //         vel = std::sqrt(pgen.gm0/rad)*std::sqrt(vel);
    //         return vel;
    //     }
    // }

    // //----------------------------------------------------------------------------------------
    // // Masked location where we reset the background values in different coordinates
    // // Include an outer boundary region which interfaces with the physical domain and 
    // // can be used to effectively set boundary conditions. Also include an inner region which
    // // can simply be set to whatever to avoid singularities or limitations on the CFL.

    // KOKKOS_INLINE_FUNCTION  
    // static int MaskingCartCoords(struct warp_pgen pgen,const Real x, const Real y, const Real z){
    //     Real r(0.0), theta(0.0), phi(0.0);
    //     enum Masking {nomask,innermask,outermask} region;

    //     // Convert to spherical coordinates     
    //     CartToSphCoord(pgen,r, theta, phi, x, y, z);

    //     if ((r < pgen.rminmask) || (r > pgen.rmaxmask) || (theta < pgen.thetaminmask) || (theta > pgen.thetamaxmask)) {
    //         region=outermask;
    //         if ((r < 0.5*pgen.rminmask) || (r > 1.5*pgen.rmaxmask) || (theta < pgen.thetaminmask-0.3) || (theta > pgen.thetamaxmask+0.3))
    //             region=innermask;
    //     } else {
    //         region=nomask;
    //     }
    //     return region;
    // }

    // KOKKOS_INLINE_FUNCTION
    // static int MaskingCylCoords(struct warp_pgen pgen,const Real rad, const Real phi, const Real z){
    // Real r(0.0), theta(0.0), phi_sph(0.0);
    // enum Masking {nomask,innermask,outermask} region;

    // // Convert to spherical coordinates     
    // CylToSphCoord(pgen, r, theta, phi_sph, rad, phi, z);

    // if ((r < pgen.rminmask) || (r > pgen.rmaxmask) || (theta < pgen.thetaminmask) || (theta > pgen.thetamaxmask)) {
    //     region=outermask;
    //     if ((r < 0.5*pgen.rminmask) || (r > 1.5*pgen.rmaxmask) || (theta < pgen.thetaminmask-0.3) || (theta > pgen.thetamaxmask+0.3))
    //         region=innermask;
    // } else {
    //     region=nomask;
    // }
    // return region;
    // }
    
    // KOKKOS_INLINE_FUNCTION
    // static int MaskingSphCoords(struct warp_pgen pgen,const Real r, const Real theta, const Real phi){  
    // enum Masking {nomask,innermask,outermask} region;
    // if ((r < pgen.rminmask) || (r > pgen.rmaxmask) || (theta < pgen.thetaminmask) || (theta > pgen.thetamaxmask)) {
    //     region=outermask;
    //     if ((r < 0.5*pgen.rminmask) || (r > 1.5*pgen.rmaxmask) || (theta < pgen.thetaminmask-0.3) || (theta > pgen.thetamaxmask+0.3))
    //         region=innermask;
    // } else {
    //     region=nomask;
    // }
    // return region;
    // }

    // //----------------------------------------------------------------------------------------
    // // Function to calculate the initial condition primitive variables at any specified cartesian 
    // // location in the disc.

    // KOKKOS_INLINE_FUNCTION
    // static void ComputePrimitives(struct warp_pgen pgen,
    //                                    Real xwarp, Real ywarp, Real zwarp,
    //                                    Real& rho, Real& pgas,
    //                                    Real& ux, Real& uy, Real& uz) {

    //     // Extract the spherical radius
    //     Real rwarp = std::sqrt(SQR(xwarp)+SQR(ywarp)+SQR(zwarp));
        
    //     // Start by converting the system to the unwarped cylindrical coordinates
    //     Real rad(0.0), phi(0.0), z(0.0);
    //     GetUnwarpedCylCoord(pgen,rad,phi,z,xwarp,ywarp,zwarp);
    //     // Now initialise the density scalar as read from the unwarped equilibrium profile
    //     Real den = DenProfileCyl(pgen,rad,phi,z);
    //     // Read in the purely cylindrical azimuthal velocity at this location
    //     Real vel = VelProfileCyl(pgen,rad,phi,z);

    //     // Now convert this into the cartesian velocity components at this location
    //     Real velx_flat(0.0), vely_flat(0.0), velz_flat(0.0);
    //     CylToCartVec(pgen,velx_flat,vely_flat,velz_flat, 0.0, vel, 0.0, phi);

    //     // Now rotate the velocity components according to the warping rotation
    //     Real velx_warp(0.0), vely_warp(0.0), velz_warp(0.0);
    //     Real warp_beta(0.0), warp_gamma(0.0);
    //     // Extract the warp angles
    //     GetWarpBeta(pgen,rwarp,warp_beta);
    //     GetWarpGamma(pgen,rwarp,warp_gamma);
    //     // Rotate the cartesian vector according to the warp
    //     WarpRot(pgen,warp_beta, warp_gamma, velx_warp, vely_warp, velz_warp, velx_flat, vely_flat, velz_flat);

    //     // Now set the primitive variables
    //     rho = den;
    //     ux = velx_warp;
    //     uy = vely_warp;
    //     uz = velz_warp;
    //     if (pgen.eos_flag != pgen.eos_isothermal) {
    //         pgas = CSoundSqCyl(pgen,rad,phi,z)*den;
    //         }  
        
    //     return;      
    }

} // End of namespace

//----------------------------------------------------------------------------------------
//! Below we will define the custom user defined source terms and boundary conditions.
//----------------------------------------------------------------------------------------
//! Write the user source term function. Note that all source terms should be included in
//! the same function and enrolled together. Currently I have written a cooling function
//! and a velocity damping function.

void MySourceTerms(Mesh* pm, const Real bdt) {

    // Capture variables for kernel - e.g. indices for looping over the meshblocks and the size of the meshblocks.
    auto &indcs = pm->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    MeshBlockPack *pmbp = pm->pmb_pack;
    auto &size = pmbp->pmb->mb_size;

    // Now set a pointer to the disc parameter struct for use in this function
    auto disc_params_ = disc_params;

    DvceArray5D<Real> u0_, w0_;
    if (pmbp->phydro != nullptr) {
        u0_ = pmbp->phydro->u0;
        w0_ = pmbp->phydro->w0;
    }

    //  Initialize density and momenta - need to specify ICs in terms of the conserved quantities.
    par_for("pgen_UserProblem",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x = CellCenterX(i-is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real y = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        int nx3 = indcs.nx3;
        Real z = CellCenterX(k-ks, nx3, x3min, x3max);

        // Loop over the array elements 
        Real sph_rad (0.0), sph_theta (0.0), sph_phi(0.0); // Initialise a spherical coordinates
        Real cyl_rad (0.0); // Initialise a cylindrical radial coordinate
        Real csq_target (0.0); // Initialise a target sound speed sqr
        Real csq (0.0); // Initialise the actual sound speed sqr

        // Now perform the action of the source terms in the loop.
        CartToSphCoord(disc_params_,sph_rad,sph_theta,sph_phi,x,y,z);  

        cyl_rad = std::sqrt(SQR(x)+SQR(y));

        // Softened gravitational potential source for the central body -->
        if (disc_params_.gm0 > 0.0){
            Real ax(0.0), ay(0.0), az(0.0);
            
            // Plummer softening
            Real acc = -disc_params_.gm0/pow(SQR(sph_rad)+SQR(disc_params_.softening_len),1.5);
            ax = acc*x;
            ay = acc*y;
            az = acc*z;

            // Update the momentum source terms
            u0_(m,IM1,k,j,i) += bdt*w0_(m,IDN,k,j,i)*ax;
            u0_(m,IM2,k,j,i) += bdt*w0_(m,IDN,k,j,i)*ay;
            u0_(m,IM3,k,j,i) += bdt*w0_(m,IDN,k,j,i)*az;

            // Update the total energy source terms
            if (disc_params_.eos_flag != disc_params_.eos_isothermal){
                u0_(m,IEN,k,j,i) += bdt*w0_(m,IDN,k,j,i)*(w0_(m,IVX,k,j,i)*ax+w0_(m,IVY,k,j,i)*ay+w0_(m,IVZ,k,j,i)*az);
                } 
            }
            
        // Cooling source terms -->
        if (disc_params_.relax_cs == 1){
        csq_target = CSoundSqSph(disc_params_,sph_rad,sph_theta,sph_phi);
        // Compute the local sound speed squared
        csq = (disc_params_.gamma_gas-1.0)*w0_(m,IEN,k,j,i)/w0_(m,IDN,k,j,i);
        if (csq != csq_target) {
            u0_(m,IEN,k,j,i) -= bdt/(disc_params_.beta*pow(sph_rad/disc_params_.r0,3.0/2.0))*w0_(m,IDN,k,j,i)*(csq - csq_target)/(disc_params_.gamma_gas - 1.0);
            }
        } else if (disc_params_.relax_cs == 2){
        csq_target = CSoundSqCyl(disc_params_,cyl_rad, 0.0, 0.0);
        // Compute the local sound speed squared
        csq = (disc_params_.gamma_gas-1.0)*w0_(m,IEN,k,j,i)/w0_(m,IDN,k,j,i);
        if (csq != csq_target) {
            u0_(m,IEN,k,j,i) -= bdt/(disc_params_.beta*pow(cyl_rad/disc_params_.r0,3.0/2.0))*w0_(m,IDN,k,j,i)*(csq - csq_target)/(disc_params_.gamma_gas - 1.0);
            }
        }
        
        // Cylindrical equilibrium velocity damping source terms -->
        if (disc_params_.damp_switch == 1) {
        //  Now damp the z vertical velocity
        u0_(m,IM3,k,j,i) -= bdt/(disc_params_.tau*pow(cyl_rad/disc_params_.r0,3.0/2.0))*(w0_(m,IDN,k,j,i)*w0_(m,IVZ,k,j,i)-0.0);
        }

        // Finally if in the masked region set the conserved and primitive variables to the initial conditions
        enum Masking {nomask,innermask,outermask};
        if ((MaskingCartCoords(disc_params_,x,y,z)==innermask) || (MaskingCartCoords(disc_params_,x,y,z)==outermask)){

            // Get the primitive variables
            Real den(0.0), pgas(0.0), ux(0.0), uy(0.0), uz(0.0);
            ComputePrimitives(disc_params_,x,y,z,den,pgas,ux,uy,uz);

            // Now set the conserved variables using the primitive variables
            u0_(m,IDN,k,j,i) = den;
            u0_(m,IM1,k,j,i) = den*ux;
            u0_(m,IM2,k,j,i) = den*uy;
            u0_(m,IM3,k,j,i) = den*uz;
            if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
                u0_(m,IEN,k,j,i) = pgas/(disc_params_.gamma_gas - 1.0)+0.5*(SQR(ux)+SQR(uy)+ SQR(uz))/den;
            }

            // Also set the primitive variables
            w0_(m,IDN,k,j,i) = den;
            w0_(m,IVX,k,j,i) = ux;
            w0_(m,IVY,k,j,i) = uy;
            w0_(m,IVZ,k,j,i) = uz;
            if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
                w0_(m,IEN,k,j,i) = pgas/(disc_params_.gamma_gas - 1.0);
            }
        }

    });

    return;
}

//----------------------------------------------------------------------------------------
//! \fn FixedDiscBC
//  \brief Sets boundary condition on surfaces of computational domain
// Note quantities at boundaries are held fixed to initial condition values

void FixedDiscBC(Mesh *pm) {

  // Start by extracting the mesh block and cell information 
  auto &indcs = pm->mb_indcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &coord = pm->pmb_pack->pcoord->coord_data;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  // Initialise a pointer to the disc parameter structure   
  auto disc_params_ = disc_params;

  int nmb = pm->pmb_pack->nmb_thispack;

  // Select either Hydro or MHD arrays
  DvceArray5D<Real> u0_, w0_;
  if (pm->pmb_pack->phydro != nullptr) {
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
  }

  // X1 BOUNDARY CONDITIONS ---------------> 

  // Start off by converting all the conservative variables to primitives so everything is synchronised
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is-1,0,(n2-1),0,(n3-1));
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  par_for("fixed_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(ng-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {

    // Extract coordinates at inner x1 boundary on each meshblock in the pack
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real xwarp = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real ywarp = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real zwarp = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real den, pgas, ux, uy, uz;
    // Inner x1 boundary
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        // Compute the warped primitive variables at this location
        ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,k,j,i) = den;
        w0_(m,IVX,k,j,i) = ux;
        w0_(m,IVY,k,j,i) = uy;
        w0_(m,IVZ,k,j,i) = uz;
        if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
            w0_(m,IEN,k,j,i) = pgas/(disc_params_.gamma_gas - 1.0);
        }
    }

    // Outer x1 boundary
    xwarp = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        // Compute the warped primitive variables at this location
        ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,k,j,(ie+i+1)) = den;
        w0_(m,IVX,k,j,(ie+i+1)) = ux;
        w0_(m,IVY,k,j,(ie+i+1)) = uy;
        w0_(m,IVZ,k,j,(ie+i+1)) = uz;
        if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
            w0_(m,IEN,k,j,(ie+i+1)) = pgas/(disc_params_.gamma_gas - 1.0);
        }
    }
  });
  // Now synchronise PrimToCons on X1 physical boundary ghost zones
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));


  // X2 BOUNDARY CONDITIONS ---------------> 

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js-1,0,(n3-1));
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je+1,je+ng,0,(n3-1));
  par_for("fixed_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x2 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real xwarp = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real ywarp = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real zwarp = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real den, pgas, ux, uy, uz;
    // Inner x2 boundary
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        // Compute the warped primitive variables at this location
        ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,k,j,i) = den;
        w0_(m,IVX,k,j,i) = ux;
        w0_(m,IVY,k,j,i) = uy;
        w0_(m,IVZ,k,j,i) = uz;
        if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
            w0_(m,IEN,k,j,i) = pgas/(disc_params_.gamma_gas - 1.0);
        }
    }

    // Outer x2 boundary
    xwarp = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        // Compute the warped primitive variables at this location
        ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,k,(je+j+1),i) = den;
        w0_(m,IVX,k,(je+j+1),i) = ux;
        w0_(m,IVY,k,(je+j+1),i) = uy;
        w0_(m,IVZ,k,(je+j+1),i) = uz;
        if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
            w0_(m,IEN,k,(je+j+1),i) = pgas/(disc_params_.gamma_gas - 1.0);
        }
    }
  });
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));

  // X3 BOUNDARY CONDITIONS ---------------> 

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks-1);
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  par_for("fixed_ix3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x3 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real xwarp = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real ywarp = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real zwarp = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // Initialise primitive variables
    Real den, pgas, ux, uy, uz;

    // Inner x3 boundary
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        // Compute the warped primitive variables at this location
        ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,k,j,i) = den;
        w0_(m,IVX,k,j,i) = ux;
        w0_(m,IVY,k,j,i) = uy;
        w0_(m,IVZ,k,j,i) = uz;
        if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
            w0_(m,IEN,k,j,i) = pgas/(disc_params_.gamma_gas - 1.0);
        }
    }

    // Outer x3 boundary
    xwarp = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        // Compute the warped primitive variables at this location
        ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,(ke+k+1),j,i) = den;
        w0_(m,IVX,(ke+k+1),j,i) = ux;
        w0_(m,IVY,(ke+k+1),j,i) = uy;
        w0_(m,IVZ,(ke+k+1),j,i) = uz;
        if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
            w0_(m,IEN,(ke+k+1),j,i) = pgas/(disc_params_.gamma_gas - 1.0);
        }
    }
  });
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);

  return;
}

    //----------------------------------------------------------------------------------------

    // CALLUM: FIGURE OUT ALPHA VISCOSITY INCORPORATION AT A LATER DATE

    // //! Introduce an alpha disc prescription
    // KOKKOS_INLINE_FUNCTION  
    // void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    //                     const AthenaArray<Real> &bcc, int is, int ie, int js, int je,
    //                     int ks, int ke) {
    // // Need to set this to some value greater than 0.0 in the pgen file
    // // Treat this as the constant alpha parameter 

    // Real x(0.0), y(0.0), z(0.0), visc(0.0);

    // if (phdif->nu_iso > 0.0) {
    //     for (int k = pmb->ks; k <= pmb->ke; ++k) {
    //     z = pmb->pcoord->x3v(k);
    //     for (int j = pmb->js; j <= pmb->je; ++j) {
    //     y = pmb->pcoord->x2v(j);
    //     for (int i = pmb->is; i <= pmb->ie; ++i) {
    //         x = pmb->pcoord->x1v(i);
            
    //         // Create a local variable to store the angular velocity at the disc midplane
    //         Real Omega (0.0);

    //         // Extract the spherical coordinates
    //         Real sph_rad (0.0), sph_theta (0.0), sph_phi(0.0); // Initialise a spherical coordinates
    //         CartToSphCoord(sph_rad,sph_theta,sph_phi,x,y,z);
            
    //         // Set the target sound speed
    //         Real csq_target(0.0);
    //         if (NON_BAROTROPIC_EOS){
    //             csq_target = CSoundSqSph(sph_rad,sph_theta,sph_phi);
    //         } else {
    //             csq_target = c0sq; 
    //         }
    //         // Set the target orbital velocity
    //         Omega = VelProfileCyl(sph_rad,0.0,0.0)/sph_rad;
    //         visc = alpha*csq_target/Omega;
    
    //         phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = visc;
    //         }
    //     }
    //     }
    // }
    // return;
    // }   