#ifndef CNET_H
#define CNET_H

enum Activation{SIGMOID, RELU};
enum Optimizer{SGD};
enum LossFunction{MSE};

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
	enum Activation activation;
};

struct TrainingConfig{
	float lr;
	enum Optimizer optimizer;
	enum LossFunction lossFunction;
	int epochs;
};

struct Network{
	struct Matrix*first_layer;
	struct Matrix*last_layer;
	float loss;
	struct TrainingConfig config;
};

struct Matrix* create_matrix(int row, int col);
struct Matrix* copy_matrix(const struct Matrix* matrix);
void init_matrix(struct Matrix* matrix);
void fill_matrix(struct Matrix* matrix, float value);
void delete_matrix(struct Matrix* matrix);
void print_matrix(const struct Matrix* matrix);

void add_matrix(struct Matrix* m1, const struct Matrix* m2);
void subt_matrix(struct Matrix *m1, const struct Matrix *m2);
void pointwise_product(struct Matrix *m1, const struct Matrix *m2);
void scale_matrix(struct Matrix *matrix, float scale);
struct Matrix* multiply_matrix(const struct Matrix *m1, const struct Matrix *m2);
struct Matrix* transpose_matrix(const struct Matrix *matrix);


#endif