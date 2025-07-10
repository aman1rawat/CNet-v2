#include"../include/CNet.h"
#include<stdlib.h>
#include<stdio.h>
#include<string.h>

#define dataset_size 10000

int main(){
	struct DataLoader* loader = init_data_loader(dataset_size);
	read_csv(loader, "example/mnist_train.csv");

	struct Network* network = create_network(784);
	add_layer(network, 256, RELU);
	add_layer(network, 128, RELU);
	add_layer(network, 64, RELU);
	add_layer(network, 10, SOFTMAX);

	network->config->total_epochs = 5;
	network->config->lr = 0.0001f;
	network->config->total_samples = dataset_size;
	network->config->loss_function = CROSS_ENTROPY;

	train_network(network, loader);

	delete_loader(loader);
	delete_network(network);

	remove("../input_data.bin");
	remove("../output_data.bin");
	remove("../main");

	return 0;

}