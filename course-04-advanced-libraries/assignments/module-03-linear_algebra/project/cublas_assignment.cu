#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#define HA 2
#define WA 9
#define WB 2
#define HB WA 
#define WC WB   
#define HC HA  
#define index(i,j,ld) (((j)*(ld))+(i))

void printMat(float*P,int uWP,int uHP){
  int i,j;
  for(i=0;i<uHP;i++){

      printf("\n");

      for(j=0;j<uWP;j++)
          printf("%f ",P[index(i,j,uHP)]);
  }
}

__host__ float* initializeHostMemory(int height, int width, bool random, float nonRandomValue) {
  // TODO allocate host memory of type float of size height * width called hostMatrix
  float *hostMatrix = (float *)malloc(height * width * sizeof(float));

  // TODO fill hostMatrix with either random data (if random is true) else set each value to nonRandomValue
  if (random) { // if random is true, fill hostMatrix with random data
    for (int i = 0; i < height; i++) { // loop through each row
      for (int j = 0; j < width; j++) { // loop through each column
        hostMatrix[i * width + j] = (float)rand() / RAND_MAX;
      }
    }
  } else { // if random is false, fill hostMatrix with nonRandomValue for each element
    for (int i = 0; i < height; i++) { // loop through each row
      for (int j = 0; j < width; j++) { // loop through each column
        hostMatrix[i * width + j] = nonRandomValue;
      }
    }
  }

  return hostMatrix;
}

__host__ float *initializeDeviceMemoryFromHostMemory(int height, int width, float *hostMatrix) {
  // TODO allocate device memory of type float of size height * width called deviceMatrix
  float *deviceMatrix;
  cudaMalloc((void **)&deviceMatrix, height * width * sizeof(float));

  // TODO set deviceMatrix to values from hostMatrix
  cudaMemcpy(deviceMatrix, hostMatrix, height * width * sizeof(float), cudaMemcpyHostToDevice);

  return deviceMatrix;
}

__host__ float *retrieveDeviceMemory(int height, int width, float *deviceMatrix, float *hostMemory) {
  // TODO get matrix values from deviceMatrix and place results in hostMemory
  cudaMemcpy(hostMemory, deviceMatrix, height * width * sizeof(float), cudaMemcpyDeviceToHost);
  
  return hostMemory;
}

__host__ void printMatrices(float *A, float *B, float *C){
  printf("\nMatrix A:\n");
  printMat(A,WA,HA);
  printf("\n");
  printf("\nMatrix B:\n");
  printMat(B,WB,HB);
  printf("\n");
  printf("\nMatrix C:\n");
  printMat(C,WC,HC);
  printf("\n");
}

__host__ int freeMatrices(float *A, float *B, float *C, float *AA, float *BB, float *CC){
  free( A );  free( B );  free ( C );
  cudaError_t err = cudaFree(AA);
  if (err != cudaSuccess) {
    fprintf (stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }
  err = cudaFree(BB);
  if (err != cudaSuccess) {
    fprintf (stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }
  err = cudaFree(CC);
  if (err != cudaSuccess) {
    fprintf (stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int  main (int argc, char** argv) {
  cublasStatus_t status;

  // TODO initialize matrices A and B (2d arrays) of floats of size based on the HA/WA and HB/WB to be filled with random data
  float *A = initializeHostMemory(HA, WA, true, 0.0f);
  float *B = initializeHostMemory(HB, WB, true, 0.0f);

  if( A == 0 || B == 0){
    return EXIT_FAILURE;
  } else {
    // TODO create arrays of floats C filled with random value
    float *C = initializeHostMemory(HC, WC, true, 0.0f);
    // TODO create arrays of floats alpha filled with 1's
    float *alpha = (float *)malloc(sizeof(float));
    *alpha = 1.0f;
    // TODO create arrays of floats beta filled with 0's
    float *beta = (float *)malloc(sizeof(float));
    *beta = 0.0f;

    // TODO use initializeDeviceMemoryFromHostMemory to create AA from matrix A
    float *AA = initializeDeviceMemoryFromHostMemory(HA, WA, A);
    // TODO use initializeDeviceMemoryFromHostMemory to create BB from matrix B
    float *BB = initializeDeviceMemoryFromHostMemory(HB, WB, B);
    // TODO use initializeDeviceMemoryFromHostMemory to create CC from matrix C
    float *CC = initializeDeviceMemoryFromHostMemory(HC, WC, C);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // TODO perform Single-Precision Matrix to Matrix Multiplication, GEMM, on AA and BB and place results in CC
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HA, WC, WA, alpha, AA, HA, BB, HB, beta, CC, HA);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! gemm error (A)\n");
      return EXIT_FAILURE;
    }
    
    C = retrieveDeviceMemory(HC, WC, CC, C);

    printMatrices(A, B, C);

    freeMatrices(A, B, C, AA, BB, CC);
    
    cublasDestroy(handle);

    return EXIT_SUCCESS;
  }

}
