#include <stdio.h>
#include <assert.h>
#include <windows.h>
#include <math.h>

#include "ppm_io.h"
#include "utils.h"

void RGB2YUV(int ***R, int ***G, int ***B, int ***Y, int ***U, int ***V, int *height, int* width)
{

	*Y = alloc2D(*height, *width);
	*U = alloc2D(*height, *width);
	*V = alloc2D(*height, *width);

	int r, g, b;

	for (int y = 0; y < *height; y++) {
		for (int x = 0; x < *width; x++) {

			r = (*R)[y][x];
			g = (*G)[y][x];
			b = (*B)[y][x];

			(*U)[y][x] = b - (int)round((87 * r + 169 * g) / 256.0);
			(*V)[y][x] = r - g;
			(*Y)[y][x] = g + (int)round((86 * (*V)[y][x] + 29 * (*U)[y][x]) / 256.0);
		}
	}
}

void YUV2RGB(int ***Y, int ***U, int ***V, int ***R, int ***G, int ***B, int *height, int* width)
{
	*R = alloc2D(*height, *width);
	*G = alloc2D(*height, *width);
	*B = alloc2D(*height, *width);

	int y_, u, v;

	for (int y = 0; y < *height; y++) {
		for (int x = 0; x < *width; x++) {

			y_ = (*Y)[y][x];
			u = (*U)[y][x];
			v = (*V)[y][x];

			(*G)[y][x] = y_ - (int)round((86 * v + 29 * u) / 256.0);
			(*R)[y][x] = v + (*G)[y][x];
			(*B)[y][x] = u + (int)round((87 * (*R)[y][x] + 169 * (*G)[y][x]) / 256.0);
		}
	}
}

WEIGHT_TYPE **alloc2Dweight(int height, int width) {
	WEIGHT_TYPE *arr = new WEIGHT_TYPE[height*width];
	WEIGHT_TYPE **pp = new WEIGHT_TYPE*[height];
	memset(arr, 0, height*width);

	for (int y = 0; y < height; y++)
		pp[y] = &(arr[y*width]);

	return pp;
}

void free2Dweight(WEIGHT_TYPE **p) {
	delete(p[0]);
	delete(p);
}

void readWeight(char *weightfile, struct stNeuralNetwork *pNN) {
	int i, j, k;
	FILE *fp;

	if (!(fp = fopen(weightfile, "r"))) {
		fprintf(stderr, "weightfile file open error.\n");
		exit(-1);
	}

	fscanf_s(fp, "%d", &pNN->n_in);
	fscanf_s(fp, "%d", &pNN->n_hidden);
	fscanf_s(fp, "%d", &pNN->n_out);
	fscanf_s(fp, "%d", &pNN->n_layer);
	fscanf_s(fp, "%d", &pNN->ctx_up);
	fscanf_s(fp, "%d", &pNN->ctx_left);

	assert(pNN->n_layer <= MAX_LAYERS + 2);

	pNN->Win = alloc2Dweight(pNN->n_in, pNN->n_hidden); 	//[11][24];
	pNN->Wout = alloc2Dweight(pNN->n_hidden, pNN->n_out); 	//[24][2];
	pNN->Bout = new WEIGHT_TYPE[pNN->n_out]; 	//[2];
	pNN->B[0] = new WEIGHT_TYPE[pNN->n_hidden];				// [24];

	for (i = 0; i < pNN->n_layer - 2; i++) {
		pNN->W[i] = alloc2Dweight(pNN->n_hidden, pNN->n_hidden); 	//[24][24];
		pNN->B[i + 1] = new WEIGHT_TYPE[pNN->n_hidden];				// [24];
	}

	for (i = 0; i < pNN->n_in; i++) {
		for (j = 0; j < pNN->n_hidden; j++) {
			fscanf_s(fp, "%lf", &pNN->Win[i][j]);
		}
	}

	for (k = 0; k < pNN->n_layer - 2; k++) {
		for (i = 0; i < pNN->n_hidden; i++) {
			for (j = 0; j < pNN->n_hidden; j++) {
				fscanf_s(fp, "%lf", &pNN->W[k][i][j]);
			}
		}
	}

	for (i = 0; i < pNN->n_hidden; i++) {
		for (j = 0; j < pNN->n_out; j++) {
			fscanf_s(fp, "%lf", &pNN->Wout[i][j]);
		}
	}

	for (k = 0; k < pNN->n_layer - 1; k++) {
		for (j = 0; j < pNN->n_hidden; j++) {
			fscanf_s(fp, "%lf", &pNN->B[k][j]);
		}
	}

	for (j = 0; j < pNN->n_out; j++) {
		fscanf_s(fp, "%lf", &pNN->Bout[j]);
	}

	fclose(fp);
}

void matMul(WEIGHT_TYPE *A, WEIGHT_TYPE *W, WEIGHT_TYPE *B, int nA, int nB) {
	for (int i = 0; i < nB; i++) {
		WEIGHT_TYPE sum = 0;

		for (int j = 0; j < nA; j++) {
			sum += A[j] * W[j*nB + i];
		}

		B[i] = sum;
	}
}

void matAdd(WEIGHT_TYPE *A, WEIGHT_TYPE *B, WEIGHT_TYPE *C, int len) {
	for (int i = 0; i < len; i++) {
		C[i] = A[i] + B[i];
	}
}

void relu(WEIGHT_TYPE *A, int len) {
	for (int i = 0; i < len; i++) A[i] = MAX(0, A[i]);
}
