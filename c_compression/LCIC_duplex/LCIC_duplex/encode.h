#pragma once
#include "arithmetic_codec.h"
#include "utils.h"

void initModel_y(Adaptive_Data_Model dm[]);

void initModel_uv(Adaptive_Data_Model dm[]);

int calcContext(float ctx, int channel);

void runNetwork(struct stNeuralNetwork* pNN, WEIGHT_TYPE* in, float* pred, float* ctx, float* hidden);

bool isTexture(WEIGHT_TYPE* nbr);

float runEncoder(char* infile, char* codefile_y, char* codefile_u, char* codefile_v, char* weight_smooth_y, char* weight_smooth_u, char* weight_smooth_v, char* weight_texture_y, char* weight_texture_u, char* weight_texture_v);