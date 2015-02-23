#include <cstdlib>
#include <armadillo>
#include <algorithm>

#include "Optimizable.hpp"

HessianFreeOptimizer::HessianFreeOptimizer(unsigned int miniters, unsigned int maxiters, unsigned int mingap, double gapratio, double tolerance)
													:miniters(miniters),maxiters(maxiters),mingap(mingap),gapratio(gapratio),tolerance(tolerance) {
}
void HessianFreeOptimizer::HessianFreeIteration(Optimizable& optim, arma::vec& x0) {
	unsigned int maxTestGap = std::max((unsigned int)(maxiters * gapratio), mingap);
	
	std::vector<double> valueHistory(maxTestGap+1,0.0);
	
	//We store x as an offset from x0, as it makes calculations easier.
	arma::vec x(optim.numParameters(),arma::fill::zeros);
	
	arma::vec b(optim.numParameters(),arma::fill::none);
	optim.gradient(&b[0]);
	b=-b; //b is the negative gradient
	
	arma::vec residual(optim.numParameters(),arma::fill::none);
	residual=b;
	arma::vec direction = residual;
	residual.print();
	double residualnorm=dot(residual,residual);
	
	arma::vec Hd(optim.numParameters());

	for(unsigned int i=0; i<maxiters; i++) {
		optim.hessianVectorProduct(&direction[0],&Hd[0]);
		direction.print();
		Hd.print();
		double dHd=dot(direction,Hd);
		
		if(dHd<=0) {
			std::cout << "Hessian not positive definite" <<std::endl;
		}
		double alpha=residualnorm/dHd;
		std::cout << "alpha "<< alpha << std::endl; 
		x+=direction*alpha;
		
		arma::vec residualnew = residual- Hd*alpha;
		
		double residualnormnew=dot(residualnew,residualnew);
		double beta=residualnormnew/residualnorm;
		direction*=beta;
		direction+=residualnew;
		residual=residualnew;
		residualnorm=residualnormnew;
		double val = 0.5 * dot(residual-b,x);
		x.print();
		printf("after iteration val=%f\n",val);
		valueHistory[i%maxTestGap]=val;
		unsigned int testGap = std::max((unsigned int)( i * gapratio ), mingap);
		double prevVal= valueHistory[(i-testGap+maxTestGap)%maxTestGap];
		if(i>testGap && (val - prevVal)/val < tolerance*testGap && i >= miniters) {
			break;
		}
		//Added for the case of exact solutions.
		if(residualnorm==0.0) break;
	}
	x0+=x;
	optim.setParameters(&x0[0]);
}

