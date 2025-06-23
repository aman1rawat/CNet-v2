#ifndef CNET_H
#define CNET_H

enum Activation{sigmoid, relu};
enum Optimizer{sgd};
enum LossFunction{mse};

struct Matrix{
	int row, col;
	float **data;
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

#endif