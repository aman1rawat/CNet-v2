#include "../include/CNet.h"
#include<stdio.h>
#include<math.h>

struct Matrix* d_mse(struct Matrix *prediction, struct Matrix *output){
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