// shared memory programming with OpenMP
// MIMD architecture -> a set of processors sharing a common memory space
//                   -> each processor can access any memory location
//
// input : N = nb of nodes in the input layer
//         K = nb of layers
//   
#include "hpc.h" 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define R 3  //output layer has N - R + 1 nodes
#define THREADS 8

// each R neurons of the input produce one single neuron in the output      
typedef struct Layer Layer;
struct Layer{
    int size;
    double *neurons;
    double *weights;
    double bias;
    Layer *nextLayer;
};

typedef struct Network Network;
struct Network{
    int nb_layers;
    Layer *firstLayer;
};

// the weight matrix is a sparse matrix
void fillWeights(int rows,int columns,double *weights){

    #pragma omp parallel for collapse(2) num_threads(THREADS)
    // instead of having two for loop we can have 1 single loop
    // because the memory access is row major
    for(int i=0;i<rows;i++){ 
        for(int j=0;j<columns;j++){
            weights[i*columns+j] = 0.0; //initialization of the matrix with 0.0
        }
    }

    #pragma omp parallel for schedule(static) num_threads(THREADS)
    for(int i=0;i<rows;i++){
        int end = i + R;
        for(int j=i;j<end;j++){
            weights[i*columns+j] = (double)rand()/(double)RAND_MAX; //generate random double between 0 and 1
        }
    }
}


double *initNeurons(int size){

    double *neurons = NULL;
    neurons = malloc(size * sizeof(double));

    #pragma omp parallel for schedule(static) num_threads(THREADS)
    for(int i=0;i<size;i++){
        neurons[i] = (double)rand()/(double)RAND_MAX;
    }

    return neurons;
}

double *initNeuronsWithBias(int size,double bias){

    double *neurons = NULL;
    neurons = malloc(size * sizeof(double));

    #pragma omp parallel for schedule(static) num_threads(THREADS)
    for(int i=0;i<size;i++){ 
        neurons[i] = bias;
    }
    return neurons;
}

double *initWeight(int columns){

    double *weights = NULL;
    int rows = columns - R + 1;
    weights = malloc(rows*columns*sizeof(double *));

    fillWeights(rows,columns,weights);

    return weights;

}

Network *initialization(int init_size){

    Network *network = malloc(sizeof(Network));
    Layer *layer = malloc(sizeof(Layer));

    if(network == NULL || layer == NULL){
        exit(EXIT_FAILURE);
    }

    layer->size = init_size;
    layer->bias = (double)rand()/(double)RAND_MAX;
    layer->neurons = initNeurons(init_size);
    layer->weights = initWeight(init_size);
    layer->nextLayer = NULL;
    network->firstLayer = layer;
    network->nb_layers = 1;

    return network;

}

void addLayer(Network *network,int size){

    Layer *new_layer = malloc(sizeof(Layer));

    if(network == NULL || new_layer == NULL){
        exit(EXIT_FAILURE);
    }

    Layer *last_layer = network->firstLayer;

    int network_size = network->nb_layers;

    // can't be parallelized
    for(int i=0;i<network_size-1;i++){
        last_layer = last_layer->nextLayer;
    }

    new_layer->size = size;
    new_layer->bias = (double)rand()/(double)RAND_MAX;
    new_layer->weights = initWeight(size);
    new_layer->nextLayer = NULL;
    new_layer->neurons = malloc(size*sizeof(double));
    last_layer->nextLayer = new_layer;
    network->nb_layers +=1;
}

double sigmoid(double x){

    return 1 / (1+exp(-x));
}

double *matmul(double *weights,double *neurons,double bias,int size){

    int columns = size;
    int rows = size - R + 1;
    double *new_neurons = initNeuronsWithBias(rows,bias); // each output neurons are initialized with the bias

    #pragma omp parallel for collapse(2) schedule(static) num_threads(THREADS)
    for(int i=0;i<rows;i++){ 
        for(int j=0;j<columns;j++){ 
            // weights is a sparse matrix, when we see a 0, 
            // we skip directly to the next iteration in order to avoid a dot product 
            // which is more computational expensive
            if(weights[i*columns+j] != 0){
                #pragma omp atomic // many threads can update new_neurons[i] so we need to protect it
                new_neurons[i]+= neurons[j] * weights[i*columns+j]; // we update each output neuron 
            }
        }
    }

    #pragma omp parallel for schedule(static) num_threads(THREADS)
    for(int i=0;i<rows;i++){
        new_neurons[i] = sigmoid(new_neurons[i]);  // the value of each output neuron is the activation of the linear combination
    }

    return new_neurons;
}

void forward(Network *network){

    if(network == NULL){
        exit(EXIT_FAILURE);
    }

    Layer *current_layer = network->firstLayer;

    int network_size = network->nb_layers;

    // can't be parallelized, because layers have to be processed sequentially
    for(int i=0;i<network_size-1;i++){
        double *weights = current_layer->weights;
        double *neurons = current_layer->neurons;
        double bias = current_layer->bias;
        int size = current_layer->size;

        // compute output neurons
        double *new_neurons = matmul(weights,neurons,bias,size);

        Layer *next_layer = current_layer->nextLayer;
        next_layer->neurons = new_neurons;
        current_layer = next_layer;
    }
}

int main(int argc,char *argv[]){

    double tstart,tstop;

    srand(time(NULL));

    //these variables are used if no args are provided by the command line
    int N = 9; // default number of neurons in the 1st layer
    int K = 2; // default number of layers

    if(argc >= 3){
        N = atoi(argv[1]);
        K = atoi(argv[2]);
        while(N -(K-1)*(R-1) <= 0){
            printf("invalid N and K\n");
            printf("N : ");
            scanf("%d",&N);
            printf("K :");
            scanf("%d",&K);
        }
    }

    Network *network = initialization(N);

    // we can't parallelized because layers have to be added sequentially
    for(int i=1;i<K;i++){
        int size = N + i*(-R+1);
        addLayer(network,size);
    }

    tstart = hpc_gettime();
    forward(network);
    tstop = hpc_gettime();

    printf("execution time : %f\n",tstop - tstart);
    system("pause");

    return 0;
}