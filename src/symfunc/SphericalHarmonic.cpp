/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2012-2017 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "function/Function.h"
#include "multicolvar/MultiColvarBase.h"
#include "core/ActionRegister.h"

#include <complex>

using namespace std;

namespace PLMD {
namespace symfunc {

//+PLUMEDOC MCOLVAR SPHERICAL_HARMONIC
/*

\par Examples


*/
//+ENDPLUMEDOC


class SphericalHarmonic : public function::Function {
private:
  int tmom;
  std::vector<double> coeff_poly;
  std::vector<double> normaliz;
  unsigned factorial( const unsigned& n ) const ;
  double deriv_poly( const unsigned& m, const double& val, double& df ) const ;
  void addVectorDerivatives( const unsigned& ival, const Vector& der, MultiValue& myvals ) const ;
public:
  static void registerKeywords( Keywords& keys );
  explicit SphericalHarmonic(const ActionOptions&);
  void calculateFunction( const std::vector<double>& args, MultiValue& myvals ) const override;
};

PLUMED_REGISTER_ACTION(SphericalHarmonic,"SPHERICAL_HARMONIC")

void SphericalHarmonic::registerKeywords( Keywords& keys ) {
  function::Function::registerKeywords( keys ); keys.use("ARG");
  keys.add("compulsory","L","the value of the angular momentum");
  keys.addOutputComponent("rm","default","the real parts of the spherical harmonic values with the m value given");
  keys.addOutputComponent("im","default","the real parts of the spherical harmonic values with the m value given");
}

unsigned SphericalHarmonic::factorial( const unsigned& n ) const {
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

SphericalHarmonic::SphericalHarmonic(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{
  parse("L",tmom);
  log.printf("  calculating %dth order spherical harmonic with %s, %s and %s as input \n", tmom, getPntrToArgument(0)->getName().c_str(), getPntrToArgument(1)->getName().c_str(), getPntrToArgument(2)->getName().c_str() );
  if( getNumberOfArguments()==4 ) log.printf("  multiplying cylindrical harmonic by weight from %s \n", getPntrToArgument(3)->getName().c_str() ); 
  for(int i=-tmom; i<=tmom; ++i) {
    std::string num; Tools::convert(i,num);
    addComponentWithDerivatives( "rm-[" + num + "]" );
    componentIsNotPeriodic( "rm-[" + num + "]" );
  }
  for(int i=-tmom; i<=tmom; ++i) {
    std::string num; Tools::convert(i,num);
    addComponentWithDerivatives( "im-[" + num + "]" );
    componentIsNotPeriodic( "im-[" + num + "]" );
  }
  normaliz.resize( tmom+1 );
  for(unsigned i=0; i<=tmom; ++i) {
    normaliz[i] = sqrt( (2*tmom+1)*factorial(tmom-i)/(4*pi*factorial(tmom+i)) );
    if( i%2==1 ) normaliz[i]*=-1;
  }

  coeff_poly.resize( tmom+1 );
  if( tmom==1 ) {
    // Legendre polynomial coefficients of order one
    coeff_poly[0]=0; coeff_poly[1]=1.0;
  } else if( tmom==2 ) {
    // Legendre polynomial coefficients of order two
    coeff_poly[0]=-0.5; coeff_poly[1]=0.0;
    coeff_poly[2]=1.5;
  } else if( tmom==3 ) {
    // Legendre polynomial coefficients of order three
    coeff_poly[0]=0.0; coeff_poly[1]=-1.5;
    coeff_poly[2]=0.0; coeff_poly[3]=2.5;
  } else if( tmom==4 ) {
    // Legendre polynomial coefficients of order four
    coeff_poly[0]=0.375; coeff_poly[1]=0.0;
    coeff_poly[2]=-3.75; coeff_poly[3]=0.0;
    coeff_poly[4]=4.375;
  } else if( tmom==5 ) {
    // Legendre polynomial coefficients of order five
    coeff_poly[0]=0.0; coeff_poly[1]=1.875;
    coeff_poly[2]=0.0; coeff_poly[3]=-8.75;
    coeff_poly[4]=0.0; coeff_poly[5]=7.875;
  } else if( tmom==6 ) {
    // Legendre polynomial coefficients of order six
    coeff_poly[0]=-0.3125; coeff_poly[1]=0.0;
    coeff_poly[2]=6.5625; coeff_poly[3]=0.0;
    coeff_poly[4]=-19.6875; coeff_poly[5]=0.0;
    coeff_poly[6]=14.4375;
  } else {
    error("Insert Legendre polynomial coefficients into SphericalHarmonics code");
  }
  checkRead();
}

void SphericalHarmonic::calculateFunction( const std::vector<double>& args, MultiValue& myvals ) const {
  double weight=1; if( args.size()==4 ) weight = args[3];
  if( weight<epsilon ) return; 

  double dlen2 = args[0]*args[0]+args[1]*args[1]+args[2]*args[2]; 
  double dlen = sqrt( dlen2 ); double dlen3 = dlen2*dlen;
  double tq6, itq6, dpoly_ass, poly_ass=deriv_poly( 0, args[2]/dlen, dpoly_ass );
  // Derivatives of z/r wrt x, y, z
  Vector dz; 
  dz[0] = -( args[2] / dlen3 )*args[0]; 
  dz[1] = -( args[2] / dlen3 )*args[1]; 
  dz[2] = -( args[2] / dlen3 )*args[2] + (1.0 / dlen);
  // Accumulate for m=0
  addValue( tmom, weight*poly_ass, myvals );
  addVectorDerivatives( tmom, weight*dpoly_ass*dz, myvals );
  if( args.size()==4 ) addDerivative( tmom, 3, poly_ass, myvals );

  // The complex number of which we have to take powers
  std::complex<double> com1( args[0]/dlen,args[1]/dlen ), dp_x, dp_y, dp_z; double md, real_z, imag_z;
  std::complex<double> powered = std::complex<double>(1.0,0.0); std::complex<double> ii( 0.0, 1.0 );
  Vector myrealvec, myimagvec, real_dz, imag_dz;
  // Do stuff for all other m values
  for(unsigned m=1; m<=tmom; ++m) {
    // Calculate Legendre Polynomial
    poly_ass=deriv_poly( m, args[2]/dlen, dpoly_ass );
    // Real and imaginary parts of z
    real_z = real(com1*powered); imag_z = imag(com1*powered);

    // Calculate steinhardt parameter
    tq6=poly_ass*real_z;   // Real part of steinhardt parameter
    itq6=poly_ass*imag_z;  // Imaginary part of steinhardt parameter

    // Derivatives wrt ( x/r + iy )^m
    md=static_cast<double>(m);
    dp_x = md*powered*( (1.0/dlen)-(args[0]*args[0])/dlen3-ii*(args[0]*args[1])/dlen3 );
    dp_y = md*powered*( ii*(1.0/dlen)-(args[0]*args[1])/dlen3-ii*(args[1]*args[1])/dlen3 );
    dp_z = md*powered*( -(args[0]*args[2])/dlen3-ii*(args[1]*args[2])/dlen3 );

    // Derivatives of real and imaginary parts of above
    real_dz[0] = real( dp_x ); real_dz[1] = real( dp_y ); real_dz[2] = real( dp_z );
    imag_dz[0] = imag( dp_x ); imag_dz[1] = imag( dp_y ); imag_dz[2] = imag( dp_z );

    // Complete derivative of steinhardt parameter
    myrealvec = weight*dpoly_ass*real_z*dz + weight*poly_ass*real_dz;
    myimagvec = weight*dpoly_ass*imag_z*dz + weight*poly_ass*imag_dz;

    // Real part
    addValue( tmom+m, weight*tq6, myvals );
    addVectorDerivatives( tmom+m, myrealvec, myvals );
    // Imaginary part
    addValue( 3*tmom+1+m, weight*itq6, myvals );
    addVectorDerivatives( 3*tmom+1+m, myimagvec, myvals );
    // Store -m part of vector
    double pref=pow(-1.0,m);
    // -m part of vector is just +m part multiplied by (-1.0)**m and multiplied by complex
    // conjugate of Legendre polynomial
    // Real part
    addValue( tmom-m, pref*weight*tq6, myvals );
    addVectorDerivatives( tmom-m, pref*myrealvec, myvals );
    // Imaginary part
    addValue( 3*tmom+1-m, -pref*weight*itq6, myvals );
    addVectorDerivatives( 3*tmom+1-m, -pref*myimagvec, myvals );
    if( args.size()==4 ) {
        addDerivative( tmom+m, 3, tq6, myvals ); addDerivative( 3*tmom+1+m, 3, itq6, myvals );
        addDerivative( tmom-m, 3, pref*tq6, myvals ); addDerivative( 3*tmom+1-m, 3, -pref*itq6, myvals );
    }
    // Calculate next power of complex number
    powered *= com1;
  }
}

double SphericalHarmonic::deriv_poly( const unsigned& m, const double& val, double& df ) const {
  double fact=1.0;
  for(unsigned j=1; j<=m; ++j) fact=fact*j;
  double res=coeff_poly[m]*fact;

  double pow=1.0, xi=val, dxi=1.0; df=0.0;
  for(int i=m+1; i<=tmom; ++i) {
    double fact=1.0;
    for(unsigned j=i-m+1; j<=i; ++j) fact=fact*j;
    res=res+coeff_poly[i]*fact*xi;
    df = df + pow*coeff_poly[i]*fact*dxi;
    xi=xi*val; dxi=dxi*val; pow+=1.0;
  }
  df = df*normaliz[m];
  return normaliz[m]*res;
}

void SphericalHarmonic::addVectorDerivatives( const unsigned& ival, const Vector& der, MultiValue& myvals ) const {
  for(unsigned j=0;j<3;++j) addDerivative( ival, j, der[j], myvals );
}

}
}

