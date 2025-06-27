#include "../include/CNet.h"
#include<stdlib.h>

struct Network* create_network(int input_size){
	struct Network *network = (struct Network*)malloc(sizeof(struct Network));
	network->first_layer = NULL;
	network->last_layer = NULL;
	network->input_size = input_size;
	network->config = (struct TrainingConfig*)malloc(sizeof(struct TrainingConfig));
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

	struct Layer *layer = (struct Layer*)malloc(sieof(struct Layer));
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

void free_layer(struct Layer* layer){
	if(layer->error != NULL) free_matrix(layer->error);
	if(layer->activated_sum != NULL) free_matrix(layer->activated_sum);
	if(layer->weighted_sum != NULL) free_matrix(layer->weighted_sum);
	if(layer->bias != NULL) free_matrix(layer->bias);
	if(layer->weights != NULL) free_matrix(layer->weights);
	free(layer);
	return;
}

void remove_layer(struct Network* network){
	if(network==NULL || network->first_layer==NULL) return;
	if(network->last_layer==network->first_layer){
		free_layer(network->first_layer);
		network->first_layer = network->last_layer = NULL;
		return;
	}
	else{
		network->last_layer = network->last_layer->prev_layer;
		free_layer(network->last_layer->next_layer);
		network->last_layer->next_layer = NULL;
		return;
	}
}