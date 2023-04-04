#include "mtanh_mpfit_lin_core.h"
/*
 * Test of MPFIT for a modified TANH fit
 * for DIII-D Real-time Thomson scattering
 * fits
 * The testmpfit.c was used as a template.
 *
 *
 */

#include "mpfit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* This is the private data structure which contains the data points
   and their uncertainties */
struct vars_struct {
  double *x;
  double *y;
  double *ey;
};


/* This is the "model" function, which is a modified
 * hyperbolic tangent.
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters 
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success) */
int mtanh(int m, int n, double *p, double *dy, double **dvec, void *vars)
{
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey, z, mt, f;

  x = v->x;
  y = v->y;
  ey = v->ey;

  /* This is the model equation
   * y = a*MTANH(alpha,z) + b
   *
   * z = (xsym-x)/hwid
   *
   * MTANH = ( (1 + alpha*z)*EXP(z) - EXP(-z) )
   *     /( EXP(z) + EXP(-z) )
   *
   * the parameters are
   * p[0] = a
   * p[1] = b
   * p[2] = alpha
   * p[3] = xsym
   * p[4] = hwid
   */

  for (i=0; i<m; i++){
    z = (p[3] - x[i])/p[4];
    // Handle the exp(large number) problem for
    // points very close to the symmetry solution
    if (z < 500.0) {
      mt = ((1.0 + p[2]*z)*exp(z) - exp(-z))/(exp(z)+exp(-z));
      f = p[0]*mt + p[1];
      dy[i] = (y[i] - f)/ey[i];
    } else {
      dy[i] = 0.0;
    }
  }

  return 0;
}

/* This is the "model" function, which is a modified
 * hyperbolic tangent.
 *
 * rho - x-axis value [0.0, 1.2]
 * p - array of fit parameters 
 *
 * RETURNS: function evaluated at rho */
double mtanh_eval(double rho, double *p)
{
  double z, mt, f;

  /* This is the model equation
   * y = a*MTANH(alpha,z) + b
   *
   * z = (xsym-x)/hwid
   *
   * MTANH = ( (1 + alpha*z)*EXP(z) - EXP(-z) )
   *     /( EXP(z) + EXP(-z) )
   *
   * the parameters are
   * p[0] = a
   * p[1] = b
   * p[2] = alpha
   * p[3] = xsym
   * p[4] = hwid
   */

  z = (p[3] - rho)/p[4];
  //  if (z < 100.0){
    mt = ((1.0 + p[2]*z)*exp(z) - exp(-z))/(exp(z)+exp(-z));
    f = p[0]*mt + p[1];
    //} else {
    //f=0.0;
    //}

  return f;
}

/* This is the function that recieves the data, the number
 * of parameters in the fit, and the guess.
 * Then calls MPFIT
 *
 */
/*
  int test_mtanh_mpfit(int nx, double *x, double *y, double *ey, 
    double *presult, double *prho, double *pfit)
*/
int mtanh_mpfit(int nx, double *x, double *y, double *ey,
		double *presult, double *prho, double *pfit)
{

  // The guess for fit parameters
  double p[5]={1.0, 3.0, 0.01, 1.0, 0.01};

  // Array to hold error in fit parameters
  double perror[5];

  // Integers
  int i,np;
  int status;

  // Variables to fit (x,y,ey)
  struct vars_struct v;

  //  int status;
  mp_result result;

  // Add constraint arrays
  mp_par pars5[5];
  
  // Zero results structure
  memset(&result,0,sizeof(result));

  // Zero constraint structure for either 5 or six parameter fit
  memset(&pars5,0,sizeof(pars5));

  // Set the error for return value
  result.xerror = perror;

  // Quality-control on input data
  for (i=0; i<nx; i++){
    // If y=0 then make error huge
    if (y[i] == 0.0){
      ey[i] = 1.0e30;
    }
    // If ey=0 then make error huge
    if (ey[i] == 0.0){
      ey[i] = 1.0e30;
    }
  }
    
  v.x = x;
  v.y = y;
  v.ey = ey;

  // Five parameters
  np=5;
  
  // Anything fixed ?
  pars5[0].fixed=0;
  pars5[1].fixed=0;
  pars5[2].fixed=0;
  pars5[3].fixed=0;
  pars5[4].fixed=0;
  
  // Make pedestal height positive
  pars5[0].limited[0]=1;
  pars5[0].limited[1]=0;
  pars5[0].limits[0]=0.1;
  pars5[0].limits[1]=0.0; //
  
  // Make offset positive
  pars5[1].limited[0]=1;
  pars5[1].limited[1]=0;
  pars5[1].limits[0]=0.001;
  pars5[1].limits[1]=0.0; //
  
  // Make core slope positive
  pars5[2].limited[0]=1;
  pars5[2].limited[1]=0;
  pars5[2].limits[0]=0.0001;
  pars5[2].limits[1]=0.0; //
  
  // Make symmetry point positive petween rho=[0.85,1.15]
  pars5[3].limited[0]=1;
  pars5[3].limited[1]=1;
  pars5[3].limits[0]=0.85;
  pars5[3].limits[1]=1.15;
  
  // Make width positive and >=0.01 or else we get a Inf
  // in the mpfit due to exp(z) where z is like 1.e3
  pars5[4].limited[0]=1;
  pars5[4].limited[1]=0;
  pars5[4].limits[0]=0.01;
  pars5[4].limits[1]=0.0;
  
  /* Call fitting function for 5 parameters */
  status = mpfit(mtanh, nx, np, p, pars5, 0, (void *) &v, &result);

  // Put the result in the "presult" array for 
  // returning to calling routine.
  for (i=0; i<np; i++){
    presult[i] = p[i];
  }

  //  printf("*** fit status (>0 is good) = %d\n", status);
  
  // Calculate the final function
  for (i=0; i<121; i++){
    prho[i] = ((double) i) /100.0;
    pfit[i] = mtanh_eval(prho[i],presult);
  }
  
  return status;
}

