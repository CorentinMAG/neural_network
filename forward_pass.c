// input : N = nb of nodes in the input layer
//         K = nb of layers
//  
#include "hpc.h"        
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define R 3  //output layer has N - R + 1 nodes (each output neuron is linked to 3 adjacents input neurons)

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

void fillWeights(int rows,int columns,double *weights){

    for(int i=0;i<rows;i++){  // CAN BE PARALLELIZED
        for(int j=0;j<columns;j++){
            weights[i*columns+j] = 0.0; //initialization of the matrix with 0.0
        }
    }

    for(int i=0;i<rows;i++){ //CAN BE PARALLELIZED
        int end = i + R;
        for(int j=i;j<end;j++){
            weights[i*columns+j] = (double)rand()/(double)RAND_MAX; //generate random double between 0 and 1
        }
    }
}


double *initNeurons(int size){

    double *neurons = NULL;  // initialization
    neurons = malloc(size * sizeof(double));

    for(int i=0;i<size;i++){ // CAN BE PARALLELIZED
        neurons[i] = (double)rand()/(double)RAND_MAX;
    }

    return neurons;
}

double *initWeight(int columns){

    double *weights = NULL;
    int rows = columns - R + 1;
    weights = malloc(rows*columns*sizeof(double));

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

    // we retrieve the last layer added to the structure
    for(int i=0;i<network_size-1;i++){
        last_layer = last_layer->nextLayer;
    }

    new_layer->size = size;
    new_layer->bias = (double)rand()/(double)RAND_MAX;
    new_layer->weights = initWeight(size);
    new_layer->nextLayer = NULL;
    new_layer->neurons = malloc(size*sizeof(double)); // empty for now
    last_layer->nextLayer = new_layer;
    network->nb_layers +=1;
}
double sigmoid(double x){

    return 1 / (1+exp(-x));
}

double *matmul(double *weights,double *neurons,double bias,int size){

    int columns = size;
    int rows = size - R + 1;
    double *new_neurons = malloc(columns * sizeof(double));

    for(int i=0;i<rows;i++){ 
        double s= bias; // we initialize the sum with the bias
        for(int j=0;j<columns;j++){ 
            // weights can be a sparse matrix (if we have K>>1), when we see a 0, 
            // we skip directly to the next iteration in order to avoid a dot product 
            // which is more computational expensive 
            if(weights[i*columns+j] == 0) continue; 
            s+= neurons[j] * weights[i*columns+j];
        }

        new_neurons[i] = sigmoid(s);
    }

    return new_neurons;

}

void forward(Network *network){

    if(network == NULL){
        exit(EXIT_FAILURE);
    }

    Layer *current_layer = network->firstLayer;

    int network_size = network->nb_layers;

    // we call the matmul function until we reach the last layer
    for(int i=0;i<network_size-1;i++){
        double *weights = current_layer->weights;
        double *neurons = current_layer->neurons;
        double bias = current_layer->bias;
        int size = current_layer->size;

        double *new_neurons = matmul(weights,neurons,bias,size);

        Layer *next_layer = current_layer->nextLayer;
        next_layer->neurons = new_neurons; // we can now populate the neuron field of the next layer
        current_layer = next_layer;
    }
}

int main(int argc,char *argv[]){

    srand(time(NULL));

    //these variables are used if nothing is provided by the command line
    int N = 9; // default number of neurons in the 1st layer
    int K = 2; // default number of layers

    srand(time(NULL));

    if(argc == 3){
        N = atoi(argv[1]);
        K = atoi(argv[2]);
        while(N -(K-1)*(R-1) <= 0){
            printf("invalid N and/or K\n");
            printf("N : ");
            scanf("%d",&N);
            printf("K :");
            scanf("%d",&K);
        }
    }

    Network *network = initialization(N);

    // we add k-1 layers to the neural structure
    // this process has to be sequential
    for(int i=1;i<K;i++){
        int size = N + i*(-R+1);
        addLayer(network,size);
    }

    double begin = hpc_gettime();
    forward(network);
    double end = hpc_gettime();;

    printf("execution time : %f\n",end-begin);

    system("pause");

    return 0;
}