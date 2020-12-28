#include "hpc.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define R 3  //output layer has N - R + 1 nodes
#define BLKDIM 32

static double tfinal = 0.0;

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

void allocateWeights(int rows,int columns,double *weights){

    for(int i=0;i<columns*rows;i++){
        weights[i] = 0;
    }
    
    for(int i=0;i<rows;i++){ 
        int end = i + R;
        for(int j=i;j<end;j++){
            weights[i*columns+j] = (double)rand()/(double)RAND_MAX; //generate random double between 0 and 1
        }
    }
}


double *initNeurons(int size){

    double *neurons = NULL;
    neurons = (double *)malloc(size * sizeof(double));

    for(int i=0;i<size;i++){ 
        neurons[i] = (double)rand()/(double)RAND_MAX;
    }

    return neurons;
}

double *initWeight(int columns){

    double *weights = NULL;
    int rows = columns - R + 1;
    // it seems that nvcc doesn't compile if we don't cast malloc
    weights = (double *)malloc(rows*columns*sizeof(double));

    allocateWeights(rows,columns,weights);

    return weights;
}

Network *initialization(int init_size){

    Network *network = (Network *)malloc(sizeof(Network));
    Layer *layer = (Layer *)malloc(sizeof(Layer));

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

    Layer *new_layer = (Layer *)malloc(sizeof(Layer));

    if(network == NULL || new_layer == NULL){
        exit(EXIT_FAILURE);
    }

    Layer *last_layer = network->firstLayer;

    int network_size = network->nb_layers;

    for(int i=0;i<network_size-1;i++){
        last_layer = last_layer->nextLayer;
    }

    new_layer->size = size;
    new_layer->bias = (double)rand()/(double)RAND_MAX;
    new_layer->weights = initWeight(size);
    new_layer->nextLayer = NULL;
    new_layer->neurons = (double *)malloc(size*sizeof(double));
    last_layer->nextLayer = new_layer;
    network->nb_layers +=1;
}

void initRes(double *res,int rows,double bias){

    for(int i=0;i<rows;i++){
        res[i] = bias;
    }
}

void initMatRes(double *res,int cols,int rows){

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            res[i*cols+j] = 0;
        }
    }
}

double sigmoid(double x){

    return 1 / (1 + exp(-x));
}

__global__ void matmulReduction(double *weights,double *neurons,double *result,int rows,int cols){

    // only the vector need to be cached
    // because all elements of the matrix
    // are only read once
    __shared__ double local_vec[BLKDIM];

    // will be useful when we will make the
    // reduction
    __shared__ double temp[BLKDIM][BLKDIM];

    int tidx = blockIdx.x*BLKDIM + threadIdx.x;
    int tidy = blockIdx.y*BLKDIM + threadIdx.y;

    int ty = threadIdx.y;
    int tx =  threadIdx.x;

    double v = 0.0;
    
    // we have as many threads as they
    // are elements in our matrix
    if(tidx <cols && tidy < rows){

        if(ty == 0) {
            // maybe not optimal because some threads are working
            // but the majority is just waiting.
            // it changes nothing if we remove the condition but
            // it is preferable to not perform multi useless read accesses to the global memory
            local_vec[tx] = neurons[tidx];
        }

        __syncthreads();
        
        // weights can be a sparse matrix for the first layers if K>>1
        v = weights[tidy*cols+tidx] * local_vec[tx];

        // each thread overwrite the weight matrix but it's fine 
        // because only elements of vec are reused
        weights[tidy*cols+tidx] = v; 

        // reduction part
        __syncthreads();
        
        // if BLKIM is not a power of 2 it can be a problem
        int bsize = blockDim.x / 2;
        
        // temp can also be sparse if K>>1
        temp[ty][tx] = weights[tidy*cols+tidx];

        __syncthreads();

        while ( bsize > 0 ) {
            if (tidx+bsize <cols && tx < bsize ) {  // we add the condition tidx + bsize < cols so We don't need to bother o the power of 2 anymore
                    temp[ty][tx] += temp[ty][tx+bsize];
            }
            bsize = bsize / 2;
            __syncthreads();
            }
            
            if ( tx == 0) {
                result[tidy*((rows+BLKDIM-1)/BLKDIM)+blockIdx.x] = temp[ty][0];
            }
            __syncthreads();   
    }
}


double *matmul(double *weights,double *neurons,double bias,int cols,int rows){
    
    // the strategy here is to send the current neurons as well as the weight 
    // matrix to the GPU, this one perform a matmul operation and then a reduction
    // operation. Then, the result is send back to the CPU which performs the last
    // reduction to obtain finally the output neurons.

    double tstart,tstop;
    int blocky = (rows+BLKDIM-1)/BLKDIM;
    int blockx = (cols+BLKDIM-1)/BLKDIM;

    dim3 grid(blocky,blockx);
    dim3 block(BLKDIM,BLKDIM);

    const size_t size_weights = rows*cols*sizeof(double);
    const size_t size_neurons = cols*sizeof(double);
    const size_t size_result = rows*blockx*sizeof(double);
    const size_t size_new_neurons = rows*sizeof(double);

    double *new_neurons,*result;
    double *dev_weights,*dev_neurons,*dev_result;

    cudaSafeCall(cudaMalloc((void **)&dev_weights,size_weights));
    cudaSafeCall(cudaMalloc((void **)&dev_neurons,size_neurons));
    cudaSafeCall(cudaMalloc((void **)&dev_result,size_result));

    // we fill the matrix with 0
    result = (double *)malloc(size_result); initMatRes(result,blockx,rows);

    // we initialize with the bias
    new_neurons = (double *)malloc(size_new_neurons) ; initRes(new_neurons,rows,bias);

    // we send the weight matrix, the input neurons and the result matrix (full of 0) to the GPU
    cudaSafeCall(cudaMemcpy(dev_weights,weights,size_weights,cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(dev_neurons,neurons,size_neurons,cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(dev_result,result,size_result,cudaMemcpyHostToDevice));

    tstart = hpc_gettime();
    matmulReduction<<<grid,block>>>(dev_weights,dev_neurons,dev_result,rows,cols); cudaCheckError();
    cudaDeviceSynchronize();
    tstop = hpc_gettime();
    tfinal+=tstop - tstart;

    // we retrieve the result matrix
    cudaSafeCall(cudaMemcpy(result,dev_result,size_result,cudaMemcpyDeviceToHost));

    // the CPU performs the last reduction and the activation
    for(int i=0;i<rows;i++){
        for(int j=0;j<blockx;j++){
            new_neurons[i] += result[i*blockx+j];
        }
        // we could launch a new kernel to performs the activation
        new_neurons[i] = sigmoid(new_neurons[i]);
    }

    // clean up
    free(result);
    cudaFree(dev_weights);cudaFree(dev_neurons);cudaFree(dev_result);

    return new_neurons;
}

void forward(Network *network){

    if(network == NULL){
        exit(EXIT_FAILURE);
    }

    Layer *current_layer = network->firstLayer;

    int network_size = network->nb_layers;

    for(int i=0;i<network_size-1;i++){

        double *weights = current_layer->weights;
        double *neurons = current_layer->neurons;
        double bias = current_layer->bias;
        int size = current_layer->size;

        double *new_neurons = matmul(weights,neurons,bias,size,size-R+1);

        Layer *next_layer = current_layer->nextLayer;
        next_layer->neurons = new_neurons;
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
            printf("invalid N and K\n");
            printf("N : ");
            scanf("%d",&N);
            printf("K :");
            scanf("%d",&K);
        }
    }

    Network *network = initialization(N);

    // can take some times but cannot be parallelized
    // because layers have to be added sequentially
    for(int i=1;i<K;i++){
        int size = N + i*(-R+1);
        addLayer(network,size);
    }

    forward(network);

    printf("execution time : %f\n",tfinal);

    return 0;
}