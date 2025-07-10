# CNet-v2
<b>
The second (and hopefully better) version of a library made purely in C for creating and training Artificial Neural Networks.<br>
I will keep updating this readme file with necessary explanations and some small snippets whenever necessary.
</b>
<br><br>
Feedbacks and suggestions [Discussions] are highly appreciated :) <br>

<hr>

<h2>HOW TO RUN</h2>
The current version of this library has been used with the MNIST dataset to create and train a Handwritten Digit Recognizer.<br>
To integrate the dataset with our library, there is an implementation file in <b>src</b> directory called <b>"mnist_loader"</b> which is basically a temporary version of a data loader similar to PyTorch's data loader.<br><br>
It converts the csv file into two binary files for input and output samples, and does sample shuffling as well.<br><br>
<b>Since the csv file itself was too big to be pushed into the repository, make sure that you download the said file and save it in "example" directory with the name "mnist_train.csv" (or with the name of your choice but then change the name in "main.c" accordingly).</b><br>
<br><b>Make sure that "Make" is installed in your system.</b><br><br>

Then simply open you terminal, make sure that you are in the root directory (./CNet-v2) and enter the command
```
	.../CNet $ make
	.../CNet-v2 $ ./main
```
<br>With the network as :<br>
```
	#define dataset_size 10000

	struct Network* network = create_network(784);
	add_layer(network, 256, RELU);
	add_layer(network, 128, RELU);
	add_layer(network, 64, RELU);
	add_layer(network, 10, SOFTMAX);

	network->config->total_epochs = 5;
	network->config->lr = 0.0001f;
	network->config->total_samples = dataset_size;
	network->config->loss_function = CROSS_ENTROPY;
```
<br>Something like this (ideally) will be logged in the terminal throughout the training:<br>
<br><img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/22d2db6b-1805-4f32-8075-3269b9e13941" />

