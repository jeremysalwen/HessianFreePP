
#include "mnist_reader.hpp"
#include <cmath>
#include <random>
#include <numeric>
#include <armadillo>
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
    return 1/(1+exp(-x))-0.5;
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
  void getNNOutputAndGrad(double& out, std::vector<double>& gradout, uint8_t correct_label, const std::vector<uint8_t>& in) {
    std::vector<double> inputs=normalize_raw_image(in);
		
    short tag=1;
		
    if(!trace_created) {
      trace_on(tag);
      std::vector<adouble> aweights(weights.size());
      std::vector<adouble> aout(numclasses);
      adouble a;
      for(uint32_t i=0; i<weights.size(); i++) {
	aweights[i]<<=weights[i];
      }
      getNNOutputs(aweights,aout,inputs);
      aout[correct_label] >>=out;
      trace_off();
      size_t tape_stats[STAT_SIZE];
      tapestats(tag, tape_stats);
 
      std::cout <<"trace created "<<tape_stats[NUM_INDEPENDENTS]<< ", "<<tape_stats[NUM_DEPENDENTS]<<"\n";
      trace_created=true;
    } else {
      std::vector<double> outvec(numclasses);
      getNNOutputs(weights,outvec,inputs);
      out=outvec[correct_label];
    } 
    gradient(tag,  weights.size(), weights.data(), gradout.data());
		
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
    std::cout<<"\n";
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
  double output=0;
  std::vector<double> grad(classifier.weights.size());

  double log_prob=0;
  int TRAIN_COUNT=100;
  int BATCH_SIZE=100;
  std::cout <<"TRAIN SIZE "<< dataset.training_images.size()<<"\n";

  std::default_random_engine engine;
  std::uniform_int_distribution<uint32_t> uint_dist10(0,dataset.training_images.size()-1);
  
  for(int i=0; i<TRAIN_COUNT; i++) {
    
    std::cout<<"\nTrain iteration "<<i<<"\n";

    double total_prob=0;
    std::vector<double> total_grad(classifier.weights.size());
    
    for(int batch=0; batch<BATCH_SIZE; batch++) {
      std::cout <<"\nBatch entry "<<batch<<"\n";
      uint32_t sample=uint_dist10(engine);
      std::cout<<"Image "<<sample<<"\n";
      classifier.getNNOutputAndGrad(output,grad,dataset.training_labels[sample], dataset.training_images[sample]);
      
      std::cout <<"Correct answer " << (int)dataset.training_labels[sample] <<"\n";
      std::cout <<"Output Prob " << output<<"\n";

      log_prob+=log(output);
      total_prob+=output;
      std::transform(total_grad.begin(), total_grad.end(), grad.begin(), total_grad.begin(),
		     std::plus<double>());
    }
    std::cout <<"Total Prob " << total_prob <<"\n";
    double gradtotal = sqrt(inner_product(total_grad.begin(), total_grad.end(), total_grad.begin(), 0.0));
    std::cout <<"gradtotal "<<gradtotal <<"\n";
    for(uint32_t j=0; j<grad.size(); j++){
      //std::cout << total_grad[j] <<"\n";
      classifier.weights[j]+=total_grad[j]/gradtotal*0.001;
    }
  }
  std::cout <<"Total log prob is "<< log_prob << "\n";
  double expected_log_prob=log(0.1)*TRAIN_COUNT;
  std::cout <<"Would expect if random " << expected_log_prob << "\n";
}
