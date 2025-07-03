#include "../include/CNet.h"
#include<stdio.h>
#include<math.h>
#include<stdlib.h>

float mse(const struct Matrix* prediction, const struct Matrix* output){
	if(prediction->row != output->row){
		printf("Dimension Mismatch\n");
		exit(1);
	}

	float error = 0.0f;
	int n = prediction->row*prediction->col;
	for(int i=0;i<n;i++){
		float diff = prediction->data[i]-output->data[i];
		error += powf(diff,2);
	}
	error /= n;
	return error;
}

struct Matrix* d_mse(const struct Matrix *prediction, const struct Matrix *output){
	if(prediction->row != output->row){
		printf("Dimension Mismatch\n");
		exit(1);
	}

	struct Matrix* d_loss = copy_matrix(prediction);

	subt_matrix(d_loss, output);
	float scale = (float)2/(d_loss->row*d_loss->col);
	scale_matrix(d_loss, scale);

	return d_loss;
}