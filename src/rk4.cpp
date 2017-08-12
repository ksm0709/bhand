# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <time.h>

# include "rk4.h"

/******************************************************************************/
double rk4(double t0, double u0, double dt, double (*f)(double t, double u))
{
	double f0;
	double f1;
	double f2;
	double f3;
	double t1;
	double t2;
	double t3;
	double u;
	double u1;
	double u2;
	double u3;
	/*
	Get four sample values of the derivative.
	*/
	f0 = f(t0, u0);

	t1 = t0 + dt * f0 / 2.0;
	u1 = u0;
	f1 = f(t1, u1);

	t2 = t0 + dt * f1 / 2.0;
	u2 = u0;
	f2 = f(t2, u2);

	t3 = t0 + dt * f2;
	u3 = u0;
	f3 = f(t3, u3);
	/*
	Combine them to estimate the solution.
	*/
	return t0 + dt * (f0 + 2.0 * f1 + 2.0 * f2 + f3) / 6.0;

}
/******************************************************************************/

double *rk4vec(double t0, int m, double u0[], double dt, double *f(double t, int m, double u[]))
{
	double *f0;
	double *f1;
	double *f2;
	double *f3;
	int i;
	double t1;
	double t2;
	double t3;
	double *u;
	double *u1;
	double *u2;
	double *u3;

	/*
	Get four sample values of the derivative.
	*/
	f0 = f(t0, m, u0);

	t1 = t0 + dt / 2.0;
	u1 = (double *)malloc(m * sizeof(double));
	for (i = 0; i < m; i++)
	{
		u1[i] = u0[i] + dt * f0[i] / 2.0;
	}
	f1 = f(t1, m, u1);

	t2 = t0 + dt / 2.0;
	u2 = (double *)malloc(m * sizeof(double));
	for (i = 0; i < m; i++)
	{
		u2[i] = u0[i] + dt * f1[i] / 2.0;
	}
	f2 = f(t2, m, u2);

	t3 = t0 + dt;
	u3 = (double *)malloc(m * sizeof(double));
	for (i = 0; i < m; i++)
	{
		u3[i] = u0[i] + dt * f2[i];
	}
	f3 = f(t3, m, u3);
	/*
	Combine them to estimate the solution.
	*/
	u = (double *)malloc(m * sizeof(double));
	for (i = 0; i < m; i++)
	{
		u[i] = u0[i] + dt * (f0[i] + 2.0 * f1[i] + 2.0 * f2[i] + f3[i]) / 6.0;
	}
	/*
	Free memory.
	*/
	free(f0);
	free(f1);
	free(f2);
	free(f3);
	free(u1);
	free(u2);
	free(u3);

	return u;
}
/******************************************************************************/
