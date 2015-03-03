#ifndef OPTIMIZABLE_HPP
#define OPTIMIZABLE_HPP

#include <cstdlib>
#include <armadillo>
class Optimizable {
public:
	virtual size_t numParameters()=0;
	virtual void setParameters(const double* parameters)=0;
	virtual const double* getParameters()=0;
	virtual double objective()=0;
	virtual void gradient(double* out)=0;
	virtual void hessianVectorProduct(const double* vector, double* out)=0;
};

class HessianFreeOptimizer {
	unsigned int miniters;
	unsigned int maxiters;
	unsigned int mingap;
	double gapratio;
	double tolerance;
	
public:
HessianFreeOptimizer(unsigned int miniters=1,	unsigned int maxiters=250, unsigned int mingap=10,	double gapratio=0.1, double tolerance=0.0005);
	
void HessianFreeIteration(Optimizable& optim, arma::vec& x0);
};

#endif
