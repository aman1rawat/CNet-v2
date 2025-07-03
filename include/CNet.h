#ifndef CNET_H
#define CNET_H

enum Activation{SIGMOID, RELU, LINEAR};
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
	struct Layer* next_layer;
	struct Layer* prev_layer;
	enum Activation activation;
};

struct TrainingConfig{
	float lr;
	enum LossFunction loss_function;
	int total_epochs;
	int total_samples;
};

struct Network{
	int input_size;
	struct Layer* first_layer;
	struct Layer* last_layer;
	float loss;
	struct Matrix* loss_derivative;
	struct TrainingConfig* config;
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

struct Network* create_network(int input_size);
void delete_network(struct Network* network);
void add_layer(struct Network *network, int size, enum Activation activation);
void delete_layer(struct Layer *layer);
void remove_layer(struct Network *network);

struct Matrix* sigmoid(const struct Matrix* matrix);
struct Matrix* relu(const struct Matrix* matrix);
struct Matrix* linear(const struct Matrix* matrix);
struct Matrix* d_sigmoid(const struct Matrix* matrix);
struct Matrix* d_relu(const struct Matrix* matrix);
struct Matrix* d_linear(const struct Matrix* matrix);

float mse(const struct Matrix* prediction, const struct Matrix* output);
struct Matrix* d_mse(const struct Matrix *prediction,const struct Matrix *output);

void forward_prop(struct Network *network, const struct Matrix* input);
void calculate_loss(struct Network *network, const struct Matrix* output);
void back_prop(struct Network *network);
void clean_network(struct Network *network);

void train_network(struct Network* network, struct Matrix** inputs, struct Matrix** outputs);

#endif