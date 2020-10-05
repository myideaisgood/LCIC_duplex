#pragma once

int readPPM(char *input_file, int ***R, int ***G, int ***B, int *height, int *width, int *bitDepth);
int writePPM(char filename[], int **R, int **G, int **B, int height, int width, int bitDepth);
int writePPM_gray(char filename[], int **Y, int height, int width, int bitDepth);
int **alloc2D(int height, int width);
void free2D(int **p);