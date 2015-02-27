#include "mnist_reader.hpp"
#include <cmath>
#include <random>
#include <numeric>
#include <armadillo>
#include <adolc/adouble.h>
#include <adolc/taping.h>
#include <adolc/drivers/drivers.h>

#include <SDL/SDL.h>


class ThreeLayerClassifier {
public:
  ThreeLayerClassifier(uint8_t numclasses, uint32_t imagesize):numclasses(numclasses), imagesize(imagesize),weights(imagesize*imagesize*2+imagesize*numclasses) {
		
  }
  std::mt19937 engine;
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
    engine.seed(100);
    double NUMCONNECT=15.0;
    std::bernoulli_distribution selection_dist(NUMCONNECT/imagesize);
    std::normal_distribution<double> weight_dist(0.0,1.0/sqrt(NUMCONNECT));
    for(unsigned i=0; i<weights.size(); i++) {
      if(selection_dist(engine)) {
	weights[i]=weight_dist(engine);
      } else {
	weights[i]=0;
      }
    }
  }
  void getNNOutputs(std::vector<double>& out,const std::vector<uint8_t>& in, bool dropout) {
    getNNOutputs(weights,out,normalize_raw_image(in), dropout);
  }

  void addRegularization(double& out, std::vector<double>& gradout, double strength) {
    for(uint32_t i=0; i<weights.size(); i++) {
      out-=weights[i]*weights[i]*strength;
      gradout[i]-=2*weights[i]*strength;
    }
  }
	
  //Use adol-c to automatically calculate gradient
  void getNNOutputAndGrad(double& out, std::vector<double>& gradout, uint8_t correct_label, const std::vector<uint8_t>& in, bool dropout) {
    std::vector<double> inputs=normalize_raw_image(in);
		
    short tag=1;
		
    if(true) {
      trace_on(tag);
      std::vector<adouble> aweights(weights.size());
      std::vector<adouble> aout(numclasses);
      adouble a;
      for(uint32_t i=0; i<weights.size(); i++) {
	aweights[i]<<=weights[i];
      }
      getNNOutputs(aweights,aout,inputs,dropout);
      aout[correct_label] >>=out;
      trace_off();
      size_t tape_stats[STAT_SIZE];
      tapestats(tag, tape_stats);
 
      std::cout <<"trace created "<<tape_stats[NUM_INDEPENDENTS]<< ", "<<tape_stats[NUM_DEPENDENTS]<<"\n";
      trace_created=true;
    } else {
      std::vector<double> outvec(numclasses);
      getNNOutputs(weights,outvec,inputs, dropout);
      out=outvec[correct_label];
    } 
    gradient(tag,  weights.size(), weights.data(), gradout.data());
    addRegularization(out, gradout, 0.000);
  }
private: 
  template <typename ftype> 
  void getNNOutputs(std::vector<ftype>& weights, std::vector<ftype>& out,const std::vector<double>& inputs,  bool dropout) {
    std::bernoulli_distribution dropout_dist(dropout?0.5:0.0);
    std::vector<ftype> layer1(imagesize);
    for(uint32_t l1=0; l1<imagesize; l1++) {
      for(uint32_t l0=0; l0<imagesize; l0++) {
	if(!dropout_dist(engine)) {
	  layer1[l1]+=inputs[l0]*weights[l1*imagesize+l0];
	}
      }
      layer1[l1]=activation_function(layer1[l1]);
    }

    int32_t weightsind=imagesize*imagesize; //The start of the weights for this layer
    std::vector<ftype> layer2(imagesize);
    for(uint32_t l2=0; l2<imagesize; l2++) {
      for(uint32_t l1=0; l1<imagesize; l1++) {
	if(!dropout_dist(engine)) {
	  layer2[l2]+=layer1[l1]*weights[weightsind+l2*imagesize+l1];
	}
      }	
      layer2[l2]=activation_function(layer2[l2]);
    }
		
    ftype sumout=0;
    weightsind=2*imagesize*imagesize;
    for(uint32_t l3=0; l3<numclasses; l3++) {
      for(uint32_t l2=0; l2<imagesize; l2++) {
	if(!dropout_dist(engine)) {
	  out[l3]+=layer2[l2]*weights[weightsind+l3*imagesize+l2];
	}
      }
      sumout+=exp(out[l3]);
    }
    sumout=log(sumout);
    for(uint32_t l3=0; l3<numclasses; l3++) {
      out[l3]-=sumout;
    }
  } 
};

void draw_weights(SDL_Surface* screen, int y, std::vector<double>& weights) {
  Uint32 *pixels = (Uint32 *)screen->pixels;
  Uint32 col;
  if( SDL_MUSTLOCK(screen) )
    SDL_LockSurface(screen);

  int nwide=screen->w/28;
  int ntall=(screen->h-y)/28;


  int slot=0;
  int maxslot=28*28*2;
  for(int bw=0; bw<nwide; bw++) {
    for(int bh=0; bh<ntall; bh++) {
      for(int px=0; px<28; px++) {
	for(int py=0; py<28; py++) {
	  double w=weights[slot*28*28+py*28+px];
	  uint8_t l=w*1000.0+128.0;
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
int main(int argc, char** argv) {


  SDL_Init(SDL_INIT_VIDEO);
  SDL_Surface* screen = SDL_SetVideoMode( 1480, 900, 32, SDL_SWSURFACE ); 
   

  mnist::MNIST_dataset<> dataset=mnist::read_dataset();
  ThreeLayerClassifier classifier(10,dataset.rows*dataset.columns);
  classifier.randomly_initialize();
  double output=0;
  std::vector<double> grad(classifier.weights.size());

  std::vector<double> momentum(classifier.weights.size());
  double log_prob=0;
  int TRAIN_COUNT=100000;
  int BATCH_SIZE=10;
  std::cout <<"TRAIN SIZE "<< dataset.training_images.size()<<"\n";

  std::default_random_engine engine;
  engine.seed(5);
  std::uniform_int_distribution<uint32_t> uint_dist10(0,dataset.training_images.size()-1);
  
  for(int i=0; i<TRAIN_COUNT; i++) {

    std::cout<<"\nTrain iteration "<<i<<"\n";

   draw_weights(screen, 28, classifier.weights);

    double total_prob=0;
    std::vector<double> total_grad(classifier.weights.size());
    
    for(int batch=0; batch<BATCH_SIZE; batch++) {
      std::cout <<"\nBatch entry "<<batch<<"\n";
      uint32_t sample=uint_dist10(engine);
      draw_digit(screen, 0, 0, dataset.training_images[sample]);
      
      std::cout<<"Image "<<sample<<"\n";
      classifier.getNNOutputAndGrad(output,grad,dataset.training_labels[sample], dataset.training_images[sample],true);
      
      std::cout <<"Correct answer " << (int)dataset.training_labels[sample] <<"\n";
      std::cout <<"Output Prob " << output<<"\n";

      log_prob+=output;
      total_prob+=exp(output);
      std::transform(total_grad.begin(), total_grad.end(), grad.begin(), total_grad.begin(),
		     std::plus<double>());
    }
    // draw_weights(gradscreen, 28, total_grad);

    std::cout <<"Total Prob " << total_prob <<"\n";
    double gradtotal = sqrt(inner_product(total_grad.begin(), total_grad.end(), total_grad.begin(), 0.0));
    std::cout <<"gradtotal "<<gradtotal <<"\n";
    
    for(uint32_t j=0; j<grad.size(); j++){
      //std::cout << total_grad[j] <<"\n";
      momentum[j]*=0.90;
      momentum[j]+=total_grad[j]/gradtotal*0.1;
      classifier.weights[j]+=momentum[j];
    }
    double momentum_total=sqrt(inner_product(momentum.begin(), momentum.end(), momentum.begin(),0.0));
    std::cout <<"momentum "<<momentum_total<<"\n\n";

       
  }
  std::cout <<"Total log prob is "<< log_prob << "\n";
  double expected_log_prob=log(0.1)*TRAIN_COUNT;
  std::cout <<"Would expect if random " << expected_log_prob << "\n";
  SDL_Quit();
  
}
