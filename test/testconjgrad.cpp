#include <armadillo>
#include "Optimizable.hpp"
class MatrixInversionOptimizable:public Optimizable {
public:
	const double* parameters;
	arma::mat A;
	arma::vec b;
	virtual size_t numParameters() {
		return A.n_rows;
	}
	virtual const double* getParameters() {
		return parameters;
	}
	virtual void setParameters(const double* parameters) {
		this->parameters=parameters;
	}
	virtual double objective() {
		const arma::vec x((double*)parameters,A.n_rows,false);
		return 0.5*dot(x,A*x)-dot(x,b);
	}
	
	virtual void gradient( double* out) {
		const arma::vec x((double*)parameters,A.n_rows,false);
		arma::vec o(out,A.n_rows,false);
		o=A*x-b;
	}
	virtual void hessianVectorProduct(const double* vector, double* out) {
		const arma::vec v((double*)vector,A.n_rows,false);
		arma::vec o(out,A.n_rows,false);
		o=A*v;
	}
};

int main(int argc, char** argv) {
	MatrixInversionOptimizable optim;
	optim.A << 2 << -1 << 0 << arma::endr
			<< -1 << 2 << -1 << arma::endr
			<< 0 << -1 << 2 << arma::endr;
		
	optim.b<< 1 <<2 <<3 ;
	arma::vec x0(optim.A.n_rows,arma::fill::ones);
	optim.setParameters(&x0[0]);
	HessianFreeOptimizer o;
	o.HessianFreeIteration(optim,x0);
}