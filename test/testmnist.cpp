
#include "mnist_reader.hpp"
#include <cmath>
#include <random>
#include <adolc/adouble.h>
#include <adolc/taping.h>
#include <adolc/drivers/drivers.h>

class ThreeLayerClassifier {
public:
	ThreeLayerClassifier(uint8_t numclasses, uint32_t imagesize):numclasses(numclasses), imagesize(imagesize),weights(imagesize*imagesize*2+imagesize*numclasses) {
		
	}
	bool trace_created=false;
	uint8_t numclasses;
	uint32_t imagesize;
	std::vector<double> weights; //flattened list of neuron weights, with inputs grouped together
	
	inline double normalize_raw_image(uint8_t pix) {
		return ((double)(pix))/UINT8_MAX;
	}
	
	inline std::vector<double> normalize_raw_image(const std::vector<uint8_t>& img) {
		std::vector<double> o(imagesize);
		for(uint32_t i=0; i<imagesize; i++) {
			o[i]=normalize_raw_image(img[i]);
		}
		return std::move(o);
	}

	template <typename ftype>
	inline ftype activation_function (ftype x) {
		return x/(1+exp(-x));
	}
	void randomly_initialize() {
		std::default_random_engine engine;
		engine.seed(5);
		std::normal_distribution<double> dist(0.0,1.0/sqrt(imagesize));
		for(unsigned i=0; i<weights.size(); i++) {
			weights[i]=dist(engine);
		}
	}
	void getNNOutputs(std::vector<double>& out,const std::vector<uint8_t>& in) {
		getNNOutputs(weights,out,normalize_raw_image(in));
	}
	
	//Use adol-c to automatically calculate gradient
	void getNNOutputAndGrad(std::vector<double>& out, std::vector<double>& gradout, const std::vector<uint8_t>& in) {
		std::vector<double> inputs=normalize_raw_image(in);
		
		short tag=1;
		
		if(!trace_created) {
			trace_on(tag);
			std::vector<adouble> aweights(weights.size());
			std::vector<adouble> aout(weights.size());
			adouble a;
			for(uint32_t i=0; i<weights.size(); i++) {
				aweights[i]<<=weights[i];
			}
			//getNNOutputs(aweights,aout,inputs);
			for(uint32_t i=0; i<out.size(); i++) {
				aout[i]>>=out[i];
			}
			trace_off();
			std::cout <<"trace created\n";
			trace_created=true;
		} else {
			getNNOutputs(weights,out,inputs);
		}
	}
	private: 
	template <typename ftype> 
	void getNNOutputs(std::vector<ftype>& weights, std::vector<ftype>& out,const std::vector<double>& inputs) {
		std::vector<ftype> layer1(imagesize);
		for(uint32_t l1=0; l1<imagesize; l1++) {
			for(uint32_t l0=0; l0<imagesize; l0++) {
				layer1[l1]+=inputs[l0]*weights[l1*imagesize+l0];
			}
			layer1[l1]=activation_function(layer1[l1]);
		}
		int32_t weightsind=imagesize*imagesize; //The start of the weights for this layer
		std::vector<ftype> layer2(imagesize);
		for(uint32_t l2=0; l2<imagesize; l2++) {
			for(uint32_t l1=0; l1<imagesize; l1++) {
				layer2[l2]+=layer1[l1]*weights[weightsind+l2*imagesize+l1];
			}
			layer2[l2]=activation_function(layer2[l2]);
		}
		
		ftype sumout=0;
		weightsind=2*imagesize*imagesize;
		for(uint32_t l3=0; l3<numclasses; l3++) {
			for(uint32_t l2=0; l2<imagesize; l2++) {
				out[l3]+=layer2[l2]*weights[weightsind+l3*imagesize+l2];
			}
			out[l3]=exp(out[l3]);
			sumout+=out[l3];
		}
		for(uint32_t l3=0; l3<numclasses; l3++) {
			out[l3]/=sumout;
		}
	} 
};
int main(int argc, char** argv) {
	mnist::MNIST_dataset<> dataset=mnist::read_dataset();
	ThreeLayerClassifier classifier(10,dataset.rows*dataset.columns);
	classifier.randomly_initialize();
	std::vector<double> output(10);
	std::vector<double> grad(dataset.rows*dataset.columns);
	classifier.getNNOutputAndGrad(output,grad,dataset.training_images[0]);
	for(int i=0; i<10; i++){
		std::cout <<  output[i] <<", ";
	}
	std::cout << "\n";
		for(uint32_t i=0; i<grad.size(); i++){
		std::cout <<  grad[i] <<", ";
	}
	std::cout << "\n";
}
