#include"../include/CNet.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

struct DataLoader* init_data_loader(int samples){
	srand(time(NULL));
	struct DataLoader* loader = (struct DataLoader*)malloc(sizeof(struct DataLoader));
	loader->samples = samples;

	loader->is_used = (uint8_t*)calloc((samples+7)/8, sizeof(uint8_t));

	loader->input_file = "./input_data.bin";
	loader->output_file = "./output_data.bin";

	loader->input = NULL;
	loader->output = NULL;

	return loader;
}

void delete_loader(struct DataLoader* loader){
	if(!loader) return;
	if(loader->is_used) free(loader->is_used);
	if(loader->input) delete_matrix(loader->input);
	if(loader->output) delete_matrix(loader->output);
	free(loader);
	return;
}

void read_csv(struct DataLoader* loader, const char* file){
	FILE *f_csv = fopen(file, "r");
	FILE *f_input = fopen("./input_data.bin", "wb");
	FILE *f_output = fopen("./output_data.bin", "wb");

	if(!f_csv){
		printf("Error in Loading the file\n");
		exit(1);
	}

	int line_size = 10000;
	char *line = (char*)malloc(line_size);
	float* input_buffer = (float*)malloc(785*sizeof(float));

	for(int i=0;i<loader->samples;i++){
		if(!fgets(line, line_size, f_csv)){
		  	printf("Unexpected EOF at sample %d\n", i);
		 	free(input_buffer);
			free(line);
			fclose(f_csv);
			fclose(f_input);
			fclose(f_output);
		  	exit(1);
		}

		line[strcspn(line, "\n")] = 0;
		
		char *token = strtok(line, ",");
		int input_index = 0;
		
		for(int c=0; c < 784 && token != NULL; c++){
			if(c == 0){
				int label = atoi(token);
				if(label < 0 || label >= 10){
					printf("Wrong label %d at sample %d\n", label, i);
					exit(1);
				}

				float output_data[10];
				for(int j = 0; j < 10; j++){
					output_data[j] = (j==label)? 1.0f : 0.0f;
				}
				fwrite(output_data, sizeof(float), 10, f_output);
			}
			else{
				float pixel_value = strtof(token, NULL);
				
				pixel_value = pixel_value/255.0f;
				input_buffer[input_index++] = pixel_value;
			}
			token = strtok(NULL, ",");
		}
		fwrite(input_buffer, sizeof(float), 784, f_input);
	}

	free(input_buffer);
	free(line);
	fclose(f_csv);
	fclose(f_input);
	fclose(f_output);
	
	printf("Successfully processed %d samples\n", loader->samples);
}

void reset_loader(struct DataLoader* loader){
	memset(loader->is_used, 0, (loader->samples+7)/8);
	return;
}

void load_next_sample(struct DataLoader *loader){
	int limit = loader->samples;
	int index = rand()%limit;
	
	while((loader->is_used[index/8] & (1<<(index%8))) != 0){
		index = rand()%limit;
	}

	loader->is_used[index/8] |= (1<<(index%8));

	FILE *f_input = fopen(loader->input_file, "rb");
	FILE *f_output = fopen(loader->output_file, "rb");
	if(!f_input || !f_output){
		printf("Error Reading the file\n");
		exit(1);
	}

	fseek(f_input, sizeof(float) * 784 * index, SEEK_SET);
	fseek(f_output, sizeof(float) * 10 * index, SEEK_SET);

	if(loader->input == NULL) loader->input = create_matrix(784, 1);
	if(loader->output == NULL) loader->output = create_matrix(10, 1);

	fread(loader->input->data, sizeof(float), 784, f_input);
	fread(loader->output->data, sizeof(float), 10, f_output);

	fclose(f_input);
	fclose(f_output);
	return;
}