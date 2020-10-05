#include "ppm_io.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "assert.h"
#include <windows.h>

unsigned short int endian_swap(unsigned short int x)
{
	x = (x >> 8) |
		(x << 8);

	return x;
}

unsigned short int endian_swap_int(int x)
{
	x = (x >> 8) |
		(x << 8);

	return x;
}

int **alloc2D(int height, int width) {

	signed int *arr = new signed int[height*width];
	signed int **pp = new signed int*[height];
	memset(arr, 0, height*width);

	for (int y = 0; y < height; y++)
		pp[y] = &(arr[y*width]);

	return pp;
}

void free2D(int **p) {
	delete(p[0]);
	delete(p);
}

int readPPM(char *input_file, int ***R, int ***G, int ***B, int *height, int *width, int *bitDepth) {
	FILE *fp;

	if ((fp = fopen(input_file, "rb"))) {
		char str[100];
		fscanf(fp, "%s", str);

		assert(str[0] == 'P' && str[1] == '6');

		fscanf(fp, "%d", width);
		fscanf(fp, "%d", height);
		fscanf(fp, "%d\n", bitDepth);

		assert(*bitDepth == 255);

		int size = (*width)*(*height);

		*R = alloc2D(*height, *width);
		*G = alloc2D(*height, *width);
		*B = alloc2D(*height, *width);

		unsigned char *img = new unsigned char[3 * size];

		fread(img, sizeof(unsigned char), 3 * size, fp);

		int idx;

		for (int y = 0; y < *height; y++) {
			for (int x = 0; x < *width; x++) {

				idx = (y*(*width) + x) * 3;

				(*R)[y][x] = img[idx];
				(*G)[y][x] = img[idx + 1];
				(*B)[y][x] = img[idx + 2];

			}
		}
		fclose(fp);

		return 1;
	}
	else {
		printf("readPPM: cannot open %s\n", input_file);
	}

	return -1;
}

int writePPM(char filename[], int **R, int **G, int **B, int height, int width, int bitDepth) {

	FILE *fp;

	if ((fp = fopen(filename, "wb"))) {
		fprintf(fp, "P6\n");
		fprintf(fp, "%d\n", width);
		fprintf(fp, "%d\n", height);
		fprintf(fp, "%d\n", bitDepth);

		assert(bitDepth == 255);

		unsigned char color[3];

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {

				color[0] = R[y][x];
				color[1] = G[y][x];
				color[2] = B[y][x];

				fwrite(color, 3, sizeof(unsigned char), fp);
			}
		}

		fclose(fp);

		return 1;
	}

	return -1;

}

int writePPM_gray(char filename[], int **Y, int height, int width, int bitDepth) {

	FILE *fp;

	if ((fp = fopen(filename, "wb"))) {
		fprintf(fp, "P5\n");
		fprintf(fp, "%d\n", width);
		fprintf(fp, "%d\n", height);
		fprintf(fp, "%d\n", bitDepth);

		assert(bitDepth == 255);

		unsigned char color[1];

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {

				color[0] = Y[y][x];

				fwrite(color, 1, sizeof(unsigned char), fp);
			}
		}

		fclose(fp);

		return 1;
	}

	return -1;

}