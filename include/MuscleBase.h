#pragma once

/*

*** Ref : Zajac89 - Muscle Activation Dynamics

	da/dt + [ (1/T_act)( beta + ( 1 - beta )u(t) ]a(t) = (1/T_act)u(t)

==> da/dt = f( a, u )	
==>	a(t + dt) = a(t) + dt * f(a(t),u(t))


*** Dependency : rk4.h, rk4.c

*/

#include "rk4.h"

#define T_act	0.012
#define T_deact 0.024

class MuscleBase
{
private:

	static double f(double a, double u)
	{
		double beta = T_act / T_deact;
		return (1 / T_act)*(u - (beta + (1 - beta)*u)*a);
	}
public:
	double get_next_activation(double a, double u, double dt)
	{
		//return a + dt * this->f(a, u);
		return rk4(a, u, dt, this->f);
	}

};
