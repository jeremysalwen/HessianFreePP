#include "mnist_reader.hpp"
#include <cmath>
#include <random>
#include <numeric>
#include <armadillo>
#include <adolc/adouble.h>
#include <adolc/taping.h>
#include <adolc/drivers/drivers.h>

#include <SDL/SDL.h>

#define VIENNACL_WITH_OPENC
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

//#include "viennacl/linalg/matrix_operations.hpp"
//#include "viennacl/linalg/vector_operations.hpp"
#include "viennacl/linalg/reduce.hpp"
#include "viennacl/linalg/prod.hpp"

using namespace viennacl;

// Usable AlmostEqual function    
bool AlmostEqual2sComplement(float A, float B, int maxUlps)    
{    
    // Make sure maxUlps is non-negative and small enough that the    
    // default NAN won't compare as equal to anything.    
  //assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);    
    int aInt = *(int*)&A;    
    // Make aInt lexicographically ordered as a twos-complement int    
    if (aInt < 0)    
        aInt = 0x80000000 - aInt;    
    // Make bInt lexicographically ordered as a twos-complement int    
    int bInt = *(int*)&B;    
    if (bInt < 0)    
        bInt = 0x80000000 - bInt;    
    int intDiff = abs(aInt - bInt);    
    if (intDiff <= maxUlps)    
        return true;    
    return false;    
}
inline double normalize_raw_image(uint8_t pix) {
  return ((double)(pix))/UINT8_MAX;
}

inline std::vector<double> normalize_raw_image(const std::vector<uint8_t>& img) {
  std::vector<double> o(img.size());
  for(uint32_t i=0; i<img.size(); i++) {
    o[i]=normalize_raw_image(img[i]);
  }

  return std::move(o);
}

void draw_digit(SDL_Surface* screen, int x, int y, const std::vector<uint8_t>& in);

class ThreeLayerClassifier {
public:
  
  ThreeLayerClassifier(uint8_t numclasses, uint32_t imagesize):
    numclasses(numclasses),
    imagesize(imagesize),
    weights((imagesize+1)*imagesize*2+(imagesize+1)*numclasses),
    dropout_mask(weights.size()),
    layer1weights(weights.memptr(), imagesize, imagesize, false),
    layer1mask(dropout_mask.memptr(), imagesize, imagesize, false),
    layer2weights(weights.memptr() + imagesize*imagesize, imagesize, imagesize, false),
    layer2mask(dropout_mask.memptr() +imagesize*imagesize, imagesize, imagesize, false),
    layer3weights(weights.memptr() + 2*imagesize*imagesize, numclasses, imagesize, false),
    layer3mask(dropout_mask.memptr() + 2*imagesize*imagesize, numclasses, imagesize, false),
    layer1biases(weights.memptr()+(2*imagesize+numclasses)*imagesize,imagesize,false),
    layer1bmask(dropout_mask.memptr()+(2*imagesize+numclasses)*imagesize,imagesize,false),
    layer2biases(weights.memptr()+(2*imagesize+numclasses+1)*imagesize,imagesize,false),
    layer2bmask(dropout_mask.memptr()+(2*imagesize+numclasses+1)*imagesize,imagesize,false),
    layer3biases(weights.memptr()+(2*imagesize+numclasses+2)*imagesize,numclasses,false),
    layer3bmask(dropout_mask.memptr()+(2*imagesize+numclasses+2)*imagesize,numclasses,false)
    
  {
    clear_dropout_mask();
  }

  

  SDL_Surface* screen;
  std::default_random_engine engine;
  bool trace_created=false;
  uint8_t numclasses;
  uint32_t imagesize;
  arma::vec weights; //flattened list of neuron weights, with inputs grouped together
  arma::uvec dropout_mask;

  double regularization_strength=0;

  arma::mat layer1weights;
  arma::umat layer1mask;
  arma::mat layer2weights;
  arma::umat layer2mask;
  arma::mat layer3weights;
  arma::umat layer3mask;
  
  arma::vec layer1biases;
  arma::uvec layer1bmask;
  arma::vec layer2biases;
  arma::uvec layer2bmask;
  arma::vec layer3biases;
  arma::uvec layer3bmask;

  typedef double ScalarType;

  viennacl::matrix<ScalarType,column_major> cl_layer1weights;
  viennacl::matrix<ScalarType,column_major> cl_layer1mask;
  viennacl::matrix<ScalarType,column_major> cl_layer2weights;
  viennacl::matrix<ScalarType,column_major> cl_layer2mask;
  viennacl::matrix<ScalarType,column_major> cl_layer3weights;
  viennacl::matrix<ScalarType,column_major> cl_layer3mask;
  
  viennacl::vector<ScalarType> cl_layer1biases;
  viennacl::vector<ScalarType> cl_layer1bmask;
  viennacl::vector<ScalarType> cl_layer2biases;
  viennacl::vector<ScalarType> cl_layer2bmask;
  viennacl::vector<ScalarType> cl_layer3biases;
  viennacl::vector<ScalarType> cl_layer3bmask;


  void generate_new_dropout_mask() {
    std::bernoulli_distribution dropout_dist(0.5);
    
    auto generator = std::bind(dropout_dist, std::ref(engine));
    std::generate(dropout_mask.begin(), dropout_mask.end(), generator); 

    cl_copy_dropout_masks();
   }

  void cl_copy_dropout_masks() {

    //  viennacl::copy(layer1mask.begin(),layer1mask.end(),cl_layer1mask);
    //  viennacl::copy(layer2mask.begin(),layer2mask.end(),cl_layer2mask);
    //  viennacl::copy(layer3mask.begin(),layer3mask.end(),cl_layer3mask);

    // viennacl::copy(layer1bmask.begin(),layer1bmask.end(),cl_layer1bmask);
    // viennacl::copy(layer2bmask.begin(),layer2bmask.end(),cl_layer2bmask);
    // viennacl::copy(layer3bmask.begin(),layer3bmask.end(),cl_layer3bmask); 

  }
  void clear_dropout_mask() {
    std::fill(dropout_mask.begin(),dropout_mask.end(),true);
    cl_copy_dropout_masks();
  }
  
  template <typename ftype>
  inline ftype activation_function (ftype x) {
    return 1/(1+exp(-x))-0.5;
  }

  inline double activation_derivative(double x) {
    double t=exp(-x);
    double tmp=1+t;
    return t/(tmp*tmp);
  }
  inline arma::vec activation_derivative(arma::vec v) {
    arma::vec tmp=exp(-v);
    return tmp / ((1+tmp) %(1+tmp));
  }
  void randomly_initialize() {
    engine.seed(100);
    double NUMCONNECT=15.0;
    std::bernoulli_distribution selection_dist(NUMCONNECT/imagesize);
    std::normal_distribution<double> weight_dist(0.0,1.0/sqrt(NUMCONNECT));
    std::normal_distribution<double> bias_dist(0.0,1.0);

    for(unsigned i=0; i<imagesize*imagesize*2+imagesize*numclasses; i++) {
      if(selection_dist(engine)) {
	weights[i]=weight_dist(engine);
      } else {
	weights[i]=0;
      }
    }
    for(unsigned i=imagesize*(imagesize*2+numclasses); i<weights.size(); i++) {
	weights[i]=bias_dist(engine);
    }
    copy_weights_to_opencl();  
  }

  void copy_weights_to_opencl() {
    // viennacl::copy(layer1weights,cl_layer1weights);
    //viennacl::copy(layer2weights,cl_layer2weights);
    //viennacl::copy(layer3weights,cl_layer3weights);
    
    //viennacl::copy(layer1biases,cl_layer1biases);
    //viennacl::copy(layer2biases,cl_layer2biases);
    //viennacl::copy(layer3biases,cl_layer3biases);
  }


  void addRegularization(double& out, arma::vec& gradout, double strength) {
    for(uint32_t i=0; i<weights.size(); i++) {
      out-=weights[i]*weights[i]*strength;
      gradout[i]-=2*weights[i]*strength;
    }
  }
	
  //Use adol-c to automatically calculate gradient
  void getNNOutputAndGrad(double& out, arma::vec& gradout, uint8_t correct_label, const std::vector<uint8_t>& in) {
    std::vector<double> inputs=normalize_raw_image(in);
   
    short tag=1;
		
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

    gradient(tag,  weights.size(), weights.memptr(), gradout.memptr());
    addRegularization(out, gradout, regularization_strength);
  }

  
  void opencl_get_output_and_grad(double& out, arma::vec& gradout, uint8_t correct_lable, const std::vector<uint8_t>& in) {
    copy_weights_to_opencl();

    viennacl::vector<ScalarType> cl_in(in.size());
    viennacl::copy(in,cl_in);

    cl_in/=UINT8_MAX;

    viennacl::matrix<ScalarType> cl_layer1weightsmasked=linalg::prod(cl_layer1weights,cl_layer1mask);
    viennacl::vector<ScalarType> cl_layer1pre_activation= linalg::prod(cl_layer1weightsmasked, cl_in)
      + linalg::element_prod(cl_layer1biases,cl_layer1bmask);
    viennacl::vector<ScalarType> cl_layer1=linalg::element_tanh(cl_layer1pre_activation);

    viennacl::matrix<ScalarType> cl_layer2weightsmasked=linalg::prod(cl_layer2weights, cl_layer2mask);
    viennacl::vector<ScalarType> cl_layer2pre_activation=linalg::prod(cl_layer2weightsmasked, cl_layer1)
      + linalg::element_prod(cl_layer2biases,cl_layer2bmask);
    viennacl::vector<ScalarType> cl_layer2=linalg::element_tanh(cl_layer2pre_activation);

    viennacl::matrix<ScalarType> cl_layer3weightsmasked=linalg::prod(cl_layer3weights, cl_layer3mask);
    viennacl::vector<ScalarType> cl_layer3pre_activation=linalg::prod(cl_layer3weightsmasked, cl_layer2)
      + linalg::element_prod(cl_layer1biases,cl_layer1bmask);
    viennacl::vector<ScalarType> cl_layer3=linalg::element_exp(cl_layer3pre_activation);
    
    //    viennacl::scalar<ScalarType> sumout=linalg::reduce<op_add>(cl_layer3);
    
  }

  
  void armadilloOutputAndGrad(double& out, arma::vec& gradout, uint8_t correct_label, const std::vector<uint8_t>& in) {

    arma::vec inputs=normalize_raw_image(in);

    arma::mat layer1gradout(gradout.memptr(),imagesize,imagesize,false);
    arma::mat layer2gradout(gradout.memptr()+imagesize*imagesize,imagesize,imagesize,false);
    arma::mat layer3gradout(gradout.memptr()+2*imagesize*imagesize,numclasses,imagesize,false);

    arma::vec layer1bgradout(gradout.memptr()+(2*imagesize+numclasses)*imagesize,imagesize,false);
    arma::vec layer2bgradout(gradout.memptr()+(2*imagesize+numclasses+1)*imagesize,imagesize,false);
    arma::vec layer3bgradout(gradout.memptr()+(2*imagesize+numclasses+2)*imagesize,numclasses,false);
    
    arma::vec layer1pre_activation=(layer1weights % layer1mask)*inputs + (layer1biases % layer1bmask);
    arma::vec layer1=tanh(layer1pre_activation/2)/2;
   
    arma::vec layer2pre_activation=(layer2weights % layer2mask)*layer1+ (layer2biases % layer2bmask);
    arma::vec layer2=tanh(layer2pre_activation/2)/2;

    arma::vec layer3= (layer3weights % layer3mask)*layer2+(layer3biases % layer3bmask);
     
    double sumout=arma::accu(exp(layer3)); 
    double logsumout=log(sumout);
    
    out=layer3[correct_label]-logsumout;
    
    std::fill(gradout.begin(),gradout.end(),0.0f);

    arma::vec l3grad=-exp(layer3)/sumout;
    l3grad[correct_label]+=1;

    layer3gradout=(l3grad*layer2.t()) % layer3mask;
    layer3bgradout=l3grad % layer3bmask;

    arma::vec l2grad=(layer3weights % layer3mask).t() * l3grad;
    arma::vec l2pre_grad=l2grad % activation_derivative(layer2pre_activation);
    arma::vec l1grad=(layer2weights % layer2mask).t()*l2pre_grad;
    layer2gradout=(l2pre_grad*layer1.t()) % layer2mask;
    layer2bgradout=l2pre_grad% layer2bmask;

    arma::vec l1pre_grad=l1grad % activation_derivative(layer1pre_activation);
    layer1gradout=(l1pre_grad*inputs.t()) %layer1mask;
    layer1bgradout=l1pre_grad %layer1bmask;
 
    //std::cout << "L3arma"<<layer3gradout[1]<<"\n";



    addRegularization(out, gradout, regularization_strength);
  }

  void handCodedOutputAndGrad(double& out, arma::vec& gradout, uint8_t correct_label, const std::vector<uint8_t>& in) {


    std::vector<double> inputs=normalize_raw_image(in);

    uint32_t bias_ind=imagesize*(imagesize*2+numclasses);
    std::vector<double> layer1(imagesize);
    std::vector<double> layer1pre_activation(imagesize);

    for(uint32_t l1=0; l1<imagesize; l1++) {      
      for(uint32_t l0=0; l0<imagesize; l0++) {
	uint32_t wi=l1*imagesize+l0;
	if(dropout_mask[wi]) {
	  layer1pre_activation[l1]+=inputs[l0]*weights[wi];
	}
      }
      if(dropout_mask[bias_ind+l1]) {
	layer1pre_activation[l1]+=weights[bias_ind+l1];
      }
      layer1[l1]=activation_function(layer1pre_activation[l1]);
    }
   
    /*
    std::vector<uint8_t> to_display(imagesize);
    for(unsigned int i=0; i<imagesize; i++) {
      to_display[i]=layer1[i]*50+127.0;
    }
    draw_digit(screen,0,28,to_display);
    */
    uint32_t l2weightsind=imagesize*imagesize; //The start of the weights for this layer
    std::vector<double> layer2(imagesize);
    std::vector<double> layer2pre_activation(imagesize);

    for(uint32_t l2=0; l2<imagesize; l2++) {
      for(uint32_t l1=0; l1<imagesize; l1++) {
	uint32_t wi=l2weightsind+l2*imagesize+l1;

	if(dropout_mask[wi]) {
	  layer2pre_activation[l2]+=layer1[l1]*weights[wi];
	}
      }
      if(dropout_mask[bias_ind+imagesize+l2]) {
	layer2pre_activation[l2]+=weights[bias_ind+imagesize+l2];
      }
      layer2[l2]=activation_function(layer2pre_activation[l2]);
    }

    /*
    for(unsigned int i=0; i<imagesize; i++) {
      to_display[i]=layer2[i]*1000+127.0;
    }
    draw_digit(screen,28,28,to_display);
    */
    std::vector<double> layer3(numclasses);
    double sumout=0;
    uint32_t l3weightsind=2*imagesize*imagesize;


   for(uint32_t l3=0; l3<numclasses; l3++) {
      for(uint32_t l2=0; l2<imagesize; l2++) {
	uint32_t wi=l3weightsind+l3*imagesize+l2;
	if(dropout_mask[wi]) {
	  layer3[l3]+=layer2[l2]*weights[wi];
	}
      }
      if(dropout_mask[bias_ind+2*imagesize+l3]) {
	layer3[l3]+=weights[bias_ind+2*imagesize+l3];
      }
      sumout+=exp(layer3[l3]);
    }

    double logsumout=log(sumout);
    /*
    for(uint32_t l3=0; l3<numclasses; l3++) {
      double prob=layer3[l3]-logsumout;
      uint8_t val=exp(prob)*255;
      for(unsigned int i=0; i<imagesize; i++) {
	to_display[i]=val;
      }
      draw_digit(screen, 56+l3*28, 28, to_display);
    }
    */

    out=layer3[correct_label]-logsumout;

    std::fill(gradout.begin(),gradout.end(),0.0f);
    
    std::vector<double> l3grad(numclasses);
    for(uint32_t l3=0; l3<numclasses; l3++) {
      l3grad[l3]=-exp(layer3[l3])/sumout;
    }
    l3grad[correct_label]+=1;



    std::vector<double> l2grad(imagesize);
    for(uint32_t l3=0; l3<numclasses; l3++) {
      for(uint32_t l2=0; l2<imagesize; l2++) {
	uint32_t wi=l3weightsind+l3*imagesize+l2;
	if(dropout_mask[wi]) {
	  l2grad[l2]+=l3grad[l3]*weights[wi];
	  gradout[wi]+=l3grad[l3]*layer2[l2];
	}
      }
      if(dropout_mask[bias_ind+imagesize*2+l3]) {
	gradout[bias_ind+imagesize*2+l3]=l3grad[l3];
      }
    }
    // std::cout << "L3hand"<<gradout[l3weightsind+1]<<"\n";    
 
    std::vector<double> l1grad(imagesize);
    for(uint32_t l2=0; l2<imagesize; l2++) {
      double l2activation_grad=l2grad[l2]*activation_derivative(layer2pre_activation[l2]);
      for(uint32_t l1=0; l1<imagesize; l1++) {
	uint32_t wi=l2weightsind+l2*imagesize+l1;
	if(dropout_mask[wi]) {
	  l1grad[l1]+= l2activation_grad*weights[wi];
	  gradout[wi]+=l2activation_grad*layer1[l1];
	}
      }
      if(dropout_mask[bias_ind+imagesize+l2]) {
	gradout[bias_ind+imagesize+l2]=l2activation_grad;
      }
    }
     for(uint32_t l1=0; l1<imagesize; l1++) {
      double l1activation_grad=l1grad[l1]*activation_derivative(layer1pre_activation[l1]);
      for(uint32_t l0=0; l0<imagesize; l0++) {
	uint32_t wi=l1*imagesize+l0;
	
	if(dropout_mask[wi]) {
	  gradout[wi]+=l1activation_grad*inputs[l0];
       	}
      }
      if(dropout_mask[bias_ind+l1]) {
	gradout[bias_ind+l1]=l1activation_grad;
      }
    }
    


 
     addRegularization(out, gradout, regularization_strength);
  }
  void check_grad(std::vector<uint8_t>& image, int correct_label, arma::vec& dx) {
    double val;
    arma::vec grad(weights.size());
    armadilloOutputAndGrad(val, grad, correct_label, image);

    double expected=arma::dot(grad,dx);

    for(unsigned int i=0; i<weights.size(); i++) {
      weights[i]+=dx[i];
    }
    double newval;
    armadilloOutputAndGrad(newval, grad, correct_label, image);


    double diff=newval-val; 
    std::cout <<"CHANGE "<<diff<<"\n";
    if(fabs(diff-expected)>0.01*fabs(expected)) {
      std::cout <<"Gradient mismatch "<<expected<<", "<<(newval-val)<<"\n";
    }
  }
private: 
  template <typename ftype> 
  void getNNOutputs(std::vector<ftype>& w, std::vector<ftype>& out,const std::vector<double>& inputs) {

    uint32_t bias_ind=imagesize*(imagesize*2+numclasses);
      
    std::vector<ftype> layer1(imagesize);
    for(uint32_t l1=0; l1<imagesize; l1++) {
      for(uint32_t l0=0; l0<imagesize; l0++) {
	if(dropout_mask[l1*imagesize+l0]) {
	  layer1[l1]+=inputs[l0]*w[l1*imagesize+l0];
	}
      }
      if(dropout_mask[bias_ind+l1]) {
	layer1[l1]+=w[bias_ind+l1];
      }
      layer1[l1]=activation_function(layer1[l1]);
    }

    int32_t weightsind=imagesize*imagesize; //The start of the weights for this layer
    std::vector<ftype> layer2(imagesize);
    for(uint32_t l2=0; l2<imagesize; l2++) {
      for(uint32_t l1=0; l1<imagesize; l1++) {
	if(dropout_mask[weightsind+l2*imagesize+l1]) {
	  layer2[l2]+=layer1[l1]*w[weightsind+l2*imagesize+l1];
	}
      }
      if(dropout_mask[bias_ind+imagesize+l2]) {
	layer2[l2]+=w[bias_ind+imagesize+l2];
      }
      layer2[l2]=activation_function(layer2[l2]);
    }

    ftype sumout=0;
    weightsind=2*imagesize*imagesize;
    for(uint32_t l3=0; l3<numclasses; l3++) {
      out[l3]=0;
      for(uint32_t l2=0; l2<imagesize; l2++) {
	if(dropout_mask[weightsind+l3*imagesize+l2]) {
	  out[l3]+=layer2[l2]*w[weightsind+l3*imagesize+l2];
	}
      }
      if(dropout_mask[bias_ind+imagesize*2+l3]) {
	out[l3]+=w[bias_ind+imagesize*2+l3];
      }
      sumout+=exp(out[l3]);
    }

    sumout=log(sumout);

    for(uint32_t l3=0; l3<numclasses; l3++) {
      out[l3]-=sumout;
    }
 }

};

void draw_weights(SDL_Surface* screen, int y, arma::vec& weights) {
  Uint32 *pixels = (Uint32 *)screen->pixels;
  Uint32 col;
  if( SDL_MUSTLOCK(screen) )
    SDL_LockSurface(screen);

  int nwide=screen->w/28;
  int ntall=(screen->h-y)/28;


  int slot=0;
  int maxslot=28*28*2+10;
  for(int bw=0; bw<nwide; bw++) {
    for(int bh=0; bh<ntall; bh++) {
      for(int px=0; px<28; px++) {
	for(int py=0; py<28; py++) {
	  double w=weights[slot*28*28+py*28+px];
	  uint8_t l=w*200.0+128.0;
	  col=SDL_MapRGB(screen->format,l,l,l);
    
	  pixels[bh*28*screen->w+bw*28+(y+py)*screen->w+px]=col;

	}
      }
      if(++slot>=maxslot) {
	goto out;
      }
    }
  }
  out:
  
  if( SDL_MUSTLOCK(screen) )
    SDL_LockSurface(screen);
  SDL_Flip(screen);

}

void draw_digit(SDL_Surface* screen, int x, int y, const std::vector<uint8_t>& in) {
  
    Uint32 *pixels = (Uint32 *)screen->pixels;
     Uint32 col;
     if( SDL_MUSTLOCK(screen) )
        SDL_LockSurface(screen);
    
     for(uint32_t i=0; i<28; i++) {
       for (uint32_t j=0; j<28; j++) {
	 
       uint8_t l=in[i*28+j];
       col=SDL_MapRGB(screen->format,l,l,l);

       pixels[(y+i)*screen->w+ x+j]=col;
       }
     }
     
    if( SDL_MUSTLOCK(screen) )
        SDL_LockSurface(screen);
    SDL_Flip(screen);
      
}

void draw_error(SDL_Surface* screen, int x, int y, const std::vector<double>& in, double ratio) {
  
    Uint32 *pixels = (Uint32 *)screen->pixels;

     if( SDL_MUSTLOCK(screen) )
        SDL_LockSurface(screen);
     uint32_t black=SDL_MapRGB(screen->format,0,0,0);
     uint32_t white=SDL_MapRGB(screen->format,255,255,255);
     SDL_Rect rect = {(int16_t)x,(int16_t)y,(uint16_t)in.size(),56};
     SDL_FillRect(screen, &rect, black);
     for(uint32_t i=0; i<in.size(); i++) {
       unsigned int height=(in[i]*ratio);
       pixels[(y+height)*screen->w+ x+i]=white;
     }
     
    if( SDL_MUSTLOCK(screen) )
        SDL_LockSurface(screen);
    SDL_Flip(screen);
      
}

void test_gradients(ThreeLayerClassifier& classifier, mnist::MNIST_dataset<>& dataset, double epsilon) {
  std::default_random_engine rand;
  std::uniform_int_distribution<> sample_dist(0,dataset.training_images.size()-1);
  std::uniform_real_distribution<> dx_dist(-1,1);
  arma::vec dx(classifier.weights.size());
  for(int i=0; i<100; i++){
    classifier.randomly_initialize();
    classifier.generate_new_dropout_mask();
    std::fill(dx.begin(),dx.end(),0.0f);

    double norm=0;
    for(unsigned int j=0; j<dx.size(); j++) {
      dx[j]=dx_dist(rand);
      norm+=dx[j]*dx[j];
    }
    norm=sqrt(norm);
    std::cout <<"norm "<<norm<<"\n";
    for(unsigned int j=0; j<dx.size(); j++) {
      dx[j]/=norm;
      dx[j]*=epsilon;
    }

    int sample=sample_dist(rand);
    std::cout <<"sample "<<sample<<"\n";
    classifier.check_grad(dataset.training_images[sample], dataset.training_labels[sample], dx);
  }

  for(unsigned int i=28*28*28*28*2+7840+784*2; i<classifier.weights.size(); i++) {
    classifier.randomly_initialize();
    classifier.generate_new_dropout_mask();
    std::fill(dx.begin(),dx.end(),0.0f);
    std::cout <<"Dimension "<<i<<"\n";
    dx[i]=epsilon;
    int sample=sample_dist(rand);
    std::cout <<"sample "<<sample<<"\n";
    classifier.check_grad(dataset.training_images[sample], dataset.training_labels[sample], dx);
  
  }
    
}
int main(int argc, char** argv) {


  SDL_Init(SDL_INIT_VIDEO);
  SDL_Surface* screen = SDL_SetVideoMode( 1700, 900, 32, SDL_SWSURFACE ); 
   

  mnist::MNIST_dataset<> dataset=mnist::read_dataset();
  ThreeLayerClassifier classifier(10,dataset.rows*dataset.columns);
  classifier.screen=screen;
  classifier.randomly_initialize();

  //   test_gradients(classifier, dataset, 0.000001);

  double output=0;
  arma::vec grad(classifier.weights.size());

  double rho=0.95;
  double epsilon=0.000001;
  arma::vec EGradSqr(classifier.weights.size(),arma::fill::zeros);
  arma::vec EStepsSizeSqr(classifier.weights.size(),arma::fill::zeros);

  std::vector<double> error_history(700);
  double log_prob=0;
  int TRAIN_COUNT=100000;
  int BATCH_SIZE=10;
  std::cout <<"TRAIN SIZE "<< dataset.training_images.size()<<"\n";

  std::default_random_engine engine;
  engine.seed(5);
  std::uniform_int_distribution<uint32_t> uint_dist10(0,dataset.training_images.size()-1);
  
  for(int i=0; i<TRAIN_COUNT; i++) {

    std::cout<<"\nTrain iteration "<<i<<"\n";


    if(0==0) {
      draw_weights(screen, 56, classifier.weights);
    }
    double total_prob=0;
    std::vector<double> total_grad(classifier.weights.size());

    double batch_error=0;
    for(int batch=0; batch<BATCH_SIZE; batch++) {
      //std::cout <<"\nBatch entry "<<batch<<"\n";
      uint32_t sample=uint_dist10(engine);
      SDL_Rect rect = {0,0,28*12,28};
      if(batch%50==0) {
	SDL_FillRect(screen, &rect, 0);
	draw_digit(screen, 56+28*dataset.training_labels[sample], 0, dataset.training_images[sample]);
       }
      std::cout<<"Image "<<sample<<"\n";
      classifier.generate_new_dropout_mask();
      classifier.handCodedOutputAndGrad(output,grad, dataset.training_labels[sample],dataset.training_images[sample]);
      arma::vec og(grad.size());
      double oout;
      classifier.getNNOutputAndGrad(oout, og, dataset.training_labels[sample],dataset.training_images[sample]);
      std::cout <<"COMPAROO "<<output<<", "<<oout<<"\n";
      for(int j=28*28*28*28*2+7840+784*2; j<28*28*28*28*2+7840+784*2+10; j++) {
	if(fabs(grad[j]-og[j])>fabs(og[j])*0.00001){
	//if(!AlmostEqual2sComplement(grad[j], og[j], 4)) {
	   std::cout <<j<<", "<<"COMPARE "<<grad[j]<<", "<<og[j]<<"\n";
	}
	}
      
      //std::cout <<"Correct answer " << (int)dataset.training_labels[sample] <<"\n";
      //std::cout <<"Output Prob " << output<<"\n";

      log_prob+=output;
      batch_error+=output;
      total_prob+=exp(output);
      std::transform(total_grad.begin(), total_grad.end(), grad.begin(), total_grad.begin(),
		     std::plus<double>());
      if(1==0) {
	draw_weights(screen, 56, grad);
      }
    }

    std::rotate(error_history.begin(), error_history.begin()+1, error_history.end());
    error_history.back()=batch_error;
    draw_error(screen, 336, 0, error_history,-2);
    
    std::vector<double> lrhistl1(100);
    std::vector<double> lrhistl2(100);
    std::vector<double> lrhistl3(100);
    
    std::vector<double> dxhistl1(100);
    std::vector<double> dxhistl2(100);
    std::vector<double> dxhistl3(100);
    double total_step_size=0;
    for(uint32_t j=0; j<grad.size(); j++){
      //std::cout << total_grad[j] <<"\n";
      EGradSqr[j]=EGradSqr[j]*rho+(1.0-rho)*grad[j]*grad[j];
      double RMSStep=sqrt(EStepsSizeSqr[j]+epsilon);
      double RMSGrad=sqrt(EGradSqr[j]+epsilon);
      double learningRate=RMSStep/RMSGrad;
      double dx=learningRate*grad[j];
      if(0<=learningRate && learningRate<=1) {
	if(j<28*28*28*28){
	  lrhistl1[(int)(learningRate*99)]++;
	} else if(j<28*28*28*28*2) {
	  lrhistl2[(int)(learningRate*99)]++;
	} else if(j<28*28*28*28*2+28*28*10) {
	  lrhistl3[(int)(learningRate*99)]++;
	}
      }
      if(-0.01<=dx && dx<=0.01) {
	if(j<28*28*28*28){
	  dxhistl1[(int)(50+dx*4900)]++;
	} else if(j<28*28*28*28*2) {
	  dxhistl2[(int)(50+dx*4900)]++;
	} else if(j<28*28*28*28*2+28*28*10) {
	  dxhistl3[(int)(50+dx*4900)]++;
	}
      }  
      EStepsSizeSqr[j]=EStepsSizeSqr[j]*rho+(1.0-rho)*dx*dx;

      classifier.weights[j]+=dx;

      total_step_size+=dx*dx;
    }
    for_each(begin(lrhistl1),end(lrhistl1),[](double& x) {x=log(x+1);});
    for_each(begin(lrhistl2),end(lrhistl2),[](double& x) {x=log(x+1);});
    for_each(begin(lrhistl3),end(lrhistl3),[](double& x) {x=log(x+1);});

    for_each(begin(dxhistl1),end(dxhistl1),[](double& x) {x=log(x+1);});
    for_each(begin(dxhistl2),end(dxhistl2),[](double& x) {x=log(x+1);});
    for_each(begin(dxhistl3),end(dxhistl3),[](double& x) {x=log(x+1);});
    
    draw_error(screen, 1100, 0, lrhistl1,1);
    draw_error(screen, 1200, 0, lrhistl2,1);
    draw_error(screen, 1300, 0, lrhistl3,1);

    draw_error(screen, 1400, 0, dxhistl1,1);
    draw_error(screen, 1500, 0, dxhistl2,1);
    draw_error(screen, 1600, 0, dxhistl3,1);

    std::cout <<"DX "<< sqrt(total_step_size);

  }
  std::cout <<"Total log prob is "<< log_prob << "\n";
  double expected_log_prob=log(0.1)*TRAIN_COUNT;
  std::cout <<"Would expect if random " << expected_log_prob << "\n";
  SDL_Quit();
  
}
