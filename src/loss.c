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

float cross_entropy(const struct Matrix* prediction, const struct Matrix* output){
	float loss = 0.0f;
	int n = prediction->row*prediction->col;
	for(int i=0;i<n;i++){
		loss += output->data[i]*logf(prediction->data[i] + 1e-7f);
	}
	loss *= -1.0f;

	return loss;
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

int check(const struct Matrix* prediction, const struct Matrix* output){
	int pred_i = 0, out_i = 0; 
	float max_pred = prediction->data[0];
	float max_out = output->data[0];
	int n = prediction->row * prediction->col;

	for(int i=1;i<n;i++){
		if(prediction->data[i]>max_pred){
		 	max_pred = prediction->data[i];
		 	pred_i = i;	
		}
		if(output->data[i]>max_out){
			max_out = output->data[i];
			out_i = i;
		}
	}

	if(pred_i==out_i) return 1;
	else return 0;
}

