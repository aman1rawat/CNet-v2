#ifndef CNET_H
#define CNET_H

enum Activation{sigmoid, relu};
enum Optimizer{sgd};
enum LossFunction{mse};

struct Matrix{
	int row, col;
	float *data;
};

struct Layer{
	int size;
	struct Matrix* weights;
	struct Matrix* bias;
	struct Matrix* weighted_sum;
	struct Matrix* activated_sum;
	struct Matrix* error;
	struct Layer *next_layer;
	struct Layer *prev_layer;
	Activation activation;
};

struct TrainingConfig{
	float lr;
	Optimizer optimizer;
	LossFunction lossFunction;
	int epochs;
};

struct Network{
	struct *first_layer;
	struct *last_layer;
	float loss;
	TrainingConfig config;
};

struct Matrix* create_matrix(int row, int col);
struct Matrix* copy_matrix(const struct Matrix* matrix);
void init_matrix(struct Matrix* matrix);
void fill_matrix(struct Matrix* matrix, int value);
void delete_matrix(struct Matrix* martix);
void print_martix(const struct Matrix* matrix);

void add_matrix(struct Matrix* m1, const struct Matrix* m2);
void subt_martix(struct Matrix *m1, const Matrix *m2);
void dot_product(struct Matrix *m1, const Matrix *m2);
void pointwise_product(struct Matrix *m1, const Matrix *m2);
void transpose_matrix(struct Matrix *matrix);
void scale_matrix(struct Matrix *matrix, float scale);


#endif