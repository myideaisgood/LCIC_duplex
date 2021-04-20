#pragma once

#define MAX_LAYERS	20
#define NUM_CTX			24
#define SYMBOL_MAX		16
#define ALPHABET_MAX	(SYMBOL_MAX+4)

#define MAX(x,y)	((x)>(y) ? (x) : (y))
#define MIN(x,y)	((x)<(y) ? (x) : (y))
#define ABS(x)		((x)>0 ? (x) : -(x))

typedef unsigned char UINT8;
typedef double WEIGHT_TYPE;

struct stNeuralNetwork {
	int n_in;
	int n_hidden;
	int n_out;
	int n_layer;
	int ctx_up;
	int ctx_left;
	WEIGHT_TYPE** Win; 	//[11][24];
	WEIGHT_TYPE** W[MAX_LAYERS]; 	//[24][24];
	WEIGHT_TYPE** Wout; 	//[24][2];
	WEIGHT_TYPE* B[MAX_LAYERS + 1];	// [24];
	WEIGHT_TYPE* Bout; 	//[2];
};
void RGB2YUV(int*** R, int*** G, int*** B, int*** Y, int*** U, int*** V, int* height, int* width);
void YUV2RGB(int*** Y, int*** U, int*** V, int*** R, int*** G, int*** B, int* height, int* width);
WEIGHT_TYPE** alloc2Dweight(int height, int width);
void free2Dweight(WEIGHT_TYPE** p);
void readWeight(char* weightfile, struct stNeuralNetwork* pNN);
void matMul(WEIGHT_TYPE* A, WEIGHT_TYPE* W, WEIGHT_TYPE* B, int nA, int nB);
void matAdd(WEIGHT_TYPE* A, WEIGHT_TYPE* B, WEIGHT_TYPE* C, int len);
void relu(WEIGHT_TYPE* A, int len);