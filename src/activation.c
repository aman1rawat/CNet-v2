#include"../include/CNet.h"
#include<math.h>

struct Matrix* sigmoid(const struct Matrix* matrix){
	struct Matrix* activated_matrix = copy_matrix(matrix);

	int n = activated_matrix->row*activated_matrix->col;
	for(int i=0;i<n;i++){
		float x = matrix->data[i];
		activated_matrix->data[i] = 1.0f/(1+expf(-x)); 
	}
	return activated_matrix;
}

struct Matrix* relu(const struct Matrix* matrix){
	struct Matrix* activated_matrix = copy_matrix(matrix);

	int n = activated_matrix->row*activated_matrix->col;
	for(int i=0;i<n;i++){
		if(activated_matrix->data[i] < 0.0f) activated_matrix->data[i] = 0.0f;
	}
	return activated_matrix;
}

struct Matrix* d_sigmoid(const struct Matrix* matrix){
	struct Matrix* temp = copy_matrix(matrix);
	struct Matrix* ones = copy_matrix(matrix);
	fill_matrix(ones, 1.0f);
	subt_matrix(ones, temp);
	pointwise_product(temp, ones);
	delete_matrix(ones);
	return temp;
}

struct Martix* d_relu(const struct Matrix* matrix){
	struct Matrix* temp = copy_matrix(matrix);
	int n = matrix->row*matrix->col;
	for(int i=0;i<n;i++){
		temp->data[i] = (temp->data[i]<=0)?0.0f:1.0f;
	}

	return temp;
}