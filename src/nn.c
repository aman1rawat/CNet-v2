#include "../include/CNet.h"
#include<stdlib.h>
#include<stdio.h>

void train_network(struct Network* network, struct DataLoader* loader){
	int epoch=1;
	int total_epochs = network->config->total_epochs;
	int total_samples = network->config->total_samples;

	while(epoch<=total_epochs){
		float loss = 0.0f;
		int correct = 0;
		printf("\nEpoch : %d\n", epoch);
		for(int i=0;i<total_samples;i++){
			load_next_sample(loader);
			// printf("Sample loaded\n");

			forward_prop(network, loader->input);
			// printf("Forward prop done\n");
			calculate_loss(network, loader->output);
			// printf("Loss calculated\n");
			loss += network->loss;
			if(i%1000 == 0){
				printf("Sample number : %d\n", i);	
				printf("Running Avg. Loss : %.4f\n", loss/(i+1));
			} 
			back_prop(network, loader->input);
			// printf("Backward prop done\n");
			
			if(network->config->loss_function == CROSS_ENTROPY){
				if(check(network->last_layer->activated_sum, loader->output)){
					correct++;
				}
			}
			
			clean_network(network);
		}

		reset_loader(loader);
		float accuracy = 100.0f*correct/(float)total_samples;
		loss /= total_samples;
		printf("------------Accuracy : %.4f------------\n\n", accuracy);
		printf("------------Average Loss : %.4f------------\n\n", loss);
		epoch++;
	}

	return;
}

void forward_prop(struct Network *network, const struct Matrix *input){
	if(network==NULL || network->first_layer==NULL) return;
	
	struct Layer *curr_layer = network->first_layer;
	while(curr_layer!=NULL){
		if(curr_layer==network->first_layer){
			curr_layer->weighted_sum = multiply_matrix(curr_layer->weights, input);
		}
		else{
			curr_layer->weighted_sum = multiply_matrix(curr_layer->weights, curr_layer->prev_layer->activated_sum);
		}

		add_matrix(curr_layer->weighted_sum, curr_layer->bias);

		switch(curr_layer->activation){
			case SIGMOID:
				curr_layer->activated_sum = sigmoid(curr_layer->weighted_sum);
				break;
			case RELU:
				curr_layer->activated_sum = relu(curr_layer->weighted_sum);
				break;
			case LINEAR:
				curr_layer->activated_sum = linear(curr_layer->weighted_sum);
				break;
			case SOFTMAX:
				curr_layer->activated_sum = softmax(curr_layer->weighted_sum);
				break;
			default:
				printf("Invalid Activation\n");
				exit(1);
				break;
		}

		curr_layer = curr_layer->next_layer;
	}
}

void calculate_loss(struct Network *network, const struct Matrix* output){
	switch(network->config->loss_function){
		case MSE:
			network->loss_derivative = d_mse(network->last_layer->activated_sum, output);
			network->loss = mse(network->last_layer->activated_sum, output);
			break;
		case CROSS_ENTROPY:
			network->loss_derivative = NULL;
			network->last_layer->error = d_softmax(network->last_layer->activated_sum, output);
			network->loss = cross_entropy(network->last_layer->activated_sum, output);
			break;
	}
	return;
}
 
void back_prop(struct Network *network, const struct Matrix* input){
	// printf("Back propagation started\n");
	struct Layer* curr_layer = network->last_layer;
	while(curr_layer!=NULL){
		switch(curr_layer->activation){
			case SIGMOID:
				// printf("sigmoid\n");
				curr_layer->error = d_sigmoid(curr_layer->weighted_sum);
				break;
			case RELU:
				// printf("relu\n");
				curr_layer->error = d_relu(curr_layer->weighted_sum);
				break;
			case LINEAR:
				// printf("linear\n");
				curr_layer->error = d_linear(curr_layer->weighted_sum);
				break;
			case SOFTMAX:
				// printf("softmax\n");
				break;
			default:
				printf("Invalid Activation\n");
				exit(1);
				break;
		}
		// printf("Switch case ended\n");

		if(curr_layer==network->last_layer && curr_layer->activation != SOFTMAX){
			// printf("if condition\n");
			pointwise_product(curr_layer->error, network->loss_derivative);
		}
		else if(curr_layer != network->last_layer){
			// printf("else case\n");
			struct Matrix* propagated_error;
			struct Matrix* transposed_weights = transpose_matrix(curr_layer->next_layer->weights);
			propagated_error = multiply_matrix(transposed_weights, curr_layer->next_layer->error);
			pointwise_product(curr_layer->error, propagated_error);
			delete_matrix(transposed_weights);
			delete_matrix(propagated_error);
		}

		// printf("Error calculated\n");

		struct Matrix *transposed_activated_sum;
		if(curr_layer==network->first_layer){
			transposed_activated_sum = transpose_matrix(input);
		}
		else{
			transposed_activated_sum = transpose_matrix(curr_layer->prev_layer->activated_sum);
		}
		
		struct Matrix *weight_gradient = multiply_matrix(curr_layer->error, transposed_activated_sum);
		scale_matrix(weight_gradient, network->config->lr);

		struct Matrix *bias_gradient = copy_matrix(curr_layer->error);
		scale_matrix(bias_gradient, network->config->lr);
		
		subt_matrix(curr_layer->weights, weight_gradient);
		subt_matrix(curr_layer->bias, bias_gradient);

		delete_matrix(transposed_activated_sum);
		delete_matrix(weight_gradient);
		delete_matrix(bias_gradient);

		curr_layer = curr_layer->prev_layer;
		// printf("Moved back to previous layer\n\n");
	}
	// printf("While loop ended\n");
	return;
}

void clean_network(struct Network* network){
	// printf("Starting cleanup\n");
	if(network==NULL){
		// printf("No network to clean\n");
		exit(1);	
	} 
	struct Layer* curr_layer = network->first_layer;
	if(curr_layer==NULL){
		// printf("No layer to clean\n");
		exit(1);
	}

	while(curr_layer!=NULL){
		if(curr_layer->weighted_sum!=NULL){
			// printf("Deleting weighted sum\n");
			delete_matrix(curr_layer->weighted_sum);
			curr_layer->weighted_sum =NULL;
		}
		if(curr_layer->activated_sum!=NULL){
			// printf("Deleting activated sum\n");
			delete_matrix(curr_layer->activated_sum);
			curr_layer->activated_sum =NULL;
		}
		if(curr_layer->error!=NULL){
			// printf("Deleting error\n");
			delete_matrix(curr_layer->error);
			curr_layer->error =NULL;
		}
		// printf("\n\n");
		curr_layer = curr_layer->next_layer;
	}
	if(network->loss_derivative != NULL){
		// printf("Deleting loss derivative\n");
		delete_matrix(network->loss_derivative);
		network->loss_derivative = NULL;
	}
	// printf("Cleanup finished\n");
	return;
}

struct Network* create_network(int input_size){
	struct Network *network = (struct Network*)malloc(sizeof(struct Network));
	network->first_layer = NULL;
	network->last_layer = NULL;
	network->input_size = input_size;
	network->config = (struct TrainingConfig*)malloc(sizeof(struct TrainingConfig));
	network->loss = 0.0f;
	network->loss_derivative = NULL;

	return network;
}

void delete_network(struct Network *network){
	if(network==NULL) return;
	if(network->config!=NULL) free(network->config);
	if(network->first_layer==NULL) return;

	if(network->first_layer==network->last_layer){
		delete_layer(network->first_layer);
		free(network);
		return;
	}

	while(network->first_layer!=network->last_layer){
		network->last_layer = network->last_layer->prev_layer;
		delete_layer(network->last_layer->next_layer);
	}
	delete_layer(network->first_layer);
	free(network);
	return;
}

void add_layer(struct Network *network, int size, enum Activation activation){
	if(network==NULL) return;

	struct Layer *layer = (struct Layer*)malloc(sizeof(struct Layer));
	layer->size = size;

	if(network->first_layer==NULL){
		layer->weights = create_matrix(size, network->input_size);	
	}
	else{
		layer->weights = create_matrix(size, network->last_layer->size);
	}

	layer->bias = create_matrix(size, 1);

	init_matrix(layer->weights);
	fill_matrix(layer->bias, 0.0f);

	layer->weighted_sum = NULL;
	layer->activated_sum = NULL;
	layer->error = NULL;
	layer->activation = activation;

	if(network->first_layer==NULL){
		layer->prev_layer = NULL;
		layer->next_layer = NULL;
		network->first_layer = network->last_layer = layer;
		return;
	}
	else{
		layer->next_layer = NULL;
		layer->prev_layer = network->last_layer;
		network->last_layer->next_layer = layer;
		network->last_layer = layer;
		return;
	}
}

void delete_layer(struct Layer* layer){
	if(layer->error != NULL) delete_matrix(layer->error);
	if(layer->activated_sum != NULL) delete_matrix(layer->activated_sum);
	if(layer->weighted_sum != NULL) delete_matrix(layer->weighted_sum);
	if(layer->bias != NULL) delete_matrix(layer->bias);
	if(layer->weights != NULL) delete_matrix(layer->weights);
	free(layer);
	return;
}

void remove_layer(struct Network* network){
	if(network==NULL || network->first_layer==NULL) return;
	if(network->last_layer==network->first_layer){
		delete_layer(network->first_layer);
		network->first_layer = network->last_layer = NULL;
		return;
	}
	else{
		network->last_layer = network->last_layer->prev_layer;
		delete_layer(network->last_layer->next_layer);
		network->last_layer->next_layer = NULL;
		return;
	}
}

