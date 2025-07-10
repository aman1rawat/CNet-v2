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

struct Matrix* linear(const struct Matrix* matrix) {
    return copy_matrix(matrix);
}

struct Matrix* softmax(const struct Matrix* matrix){
	struct Matrix* activated_matrix = copy_matrix(matrix);
	int n = activated_matrix->row * activated_matrix->col;

	float max_val = activated_matrix->data[0];
	for(int i = 1; i < n; i++){
		if(activated_matrix->data[i] > max_val)
			max_val = activated_matrix->data[i];
	}

	float sum = 0.0f;
	for(int i = 0; i < n; i++){
		activated_matrix->data[i] = expf(activated_matrix->data[i] - max_val);
		sum += activated_matrix->data[i];
	}

	for(int i = 0; i < n; i++){
		activated_matrix->data[i] /= sum;
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

struct Matrix* d_relu(const struct Matrix* matrix){
	struct Matrix* temp = copy_matrix(matrix);
	int n = matrix->row*matrix->col;
	for(int i=0;i<n;i++){
		temp->data[i] = (temp->data[i]<=0)?0.0f:1.0f;
	}

	return temp;
}


struct Matrix* d_linear(const struct Matrix* matrix) {
    struct Matrix* temp = create_matrix(matrix->row, matrix->col);
    fill_matrix(temp, 1.0f);
    return temp;
}

struct Matrix* d_softmax(const struct Matrix *prediction, const struct Matrix *output){
	struct Matrix* d_loss = copy_matrix(prediction);
	subt_matrix(d_loss, output);
	return d_loss;
}