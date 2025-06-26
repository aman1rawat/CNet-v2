#include"../include/CNet.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>

struct Matrix* create_matrix(int row, int col){
	if(row==0 || col==0){
		printf("Invalid Dimensions (%d X %d) : matrix creation not possible\n", row, col);
		return NULL;
	}
	struct Matrix* matrix = (struct Matrix*)malloc(sizeof(struct Matrix));
	if(matrix==NULL){
		printf("Allocation failed\n");
		return NULL;
	}
	matrix->row = row;
	matrix->col = col;
	matrix->data = (float*)malloc(row*col*sizeof(float));
	if(matrix->data==NULL){
		printf("Allocation failed\n");
		free(matrix);
		return NULL;
	}

	return matrix;
}

struct Matrix* copy_matrix(const struct Matrix* matrix){
	if(matrix==NULL) return NULL;

	int row = matrix->row;
	int col = matrix->col; 

	struct Matrix* new_matrix = (struct Matrix*)malloc(sizeof(struct Matrix));
	new_matrix->row = row;
	new_matrix->col = col;
	new_matrix->data = (float*)malloc(row*col*sizeof(float));

	if(new_matrix->data==NULL) return NULL;
	
	memcpy(new_matrix->data, matrix->data, row*col*sizeof(int));
	return new_matrix;
}


void init_matrix(struct Matrix* matrix){
	if(matrix==NULL) return;
	srand(time(NULL));

	float limit = sqrtf(6.0f/(matrix->row + matrix->col));

	int n = matrix->row * matrix->col;
	for(int i=0;i<n;i++){
		float num = (float)rand()/RAND_MAX;
		matrix->data[i] = limit*(2.0f*num - 1.0f);
	}
	return;
}

void fill_matrix(struct Matrix* matrix, float value){
	if(matrix==NULL) return;

	int n = matrix->row * matrix->col;
	for(int i=0;i<n;i++){
		matrix->data[i] = value;
	}
	return;
}

void delete_matrix(struct Matrix* matrix){
	if(matrix==NULL) return;
	free(matrix->data);
	free(matrix);
	return;
}

void print_matrix(const struct Matrix* matrix){
	if(matrix==NULL) return;

	int row = matrix->row;
	int col = matrix->col;
	printf(" Row: %d\n", row);
	printf(" Col: %d\n", col);

	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			float n = matrix->data[i*col+j];
			if(n<0) printf("%.2f ", n);
			else printf(" %.2f ", n);
		}
		printf("\n");
	}
	printf("\n");
	return;
}

void add_matrix(struct Matrix* m1, const struct Matrix* m2){
	if(m1==NULL || m2==NULL) return;

	if(m1->row!=m2->row || m1->col!=m2->col){
		printf("Dimension Mismatch - addition not possible\n");
		return;
	}

	int n = m1->row*m1->col;
	for(int i=0;i<n;i++){
		m1->data[i] += m2->data[i];
	}
}

void subt_matrix(struct Matrix *m1, const struct Matrix *m2){
	if(m1==NULL || m2==NULL) return;

	if(m1->row!=m2->row || m1->col!=m2->col){
		printf("Dimension Mismatch - subtraction not possible\n");
		return;
	}

	int n = m1->row*m1->col;
	for(int i=0;i<n;i++){
		m1->data[i] -= m2->data[i];
	}
}

void pointwise_product(struct Matrix *m1, const struct Matrix *m2){
	if(m1==NULL || m2==NULL) return;

	if(m1->row!=m2->row || m1->col!=m2->col){
		printf("Dimension Mismatch - pointwise multiplication not possible\n");
		return;
	}

	int n = m1->row*m1->col;
	for(int i=0;i<n;i++){
		m1->data[i] *= m2->data[i];
	}
}

void scale_matrix(struct Matrix *matrix, float scale){
	if(matrix==NULL) return;

	int n = matrix->row*matrix->col;
	for(int i=0;i<n;i++){
		matrix->data[i] *= scale;
	}
	return;
}

struct Matrix* multiply_matrix(const struct Matrix *m1, const struct Matrix *m2){
	if(m1->col!=m2->row){
		printf("Dimension Mismatch: matrix multiplication not possible\n");
		return NULL;
	}

	int row = m1->row;
	int col = m2->col;

	struct Matrix* matrix = (struct Matrix*)malloc(sizeof(struct Matrix));
    matrix->row = row;
    matrix->col = col;
    matrix->data = (float*)calloc(matrix->row * matrix->col, sizeof(float));


    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            float sum = 0.0f;
            for(int k=0;k<m1->col;k++){
                sum += m1->data[i*m1->col + k] * m2->data[k*m2->col + j];
            }
            matrix->data[i*matrix->col + j] = sum;
        }
    }

    return matrix;
}

struct Matrix* transpose_matrix(const struct Matrix *matrix){
	struct Matrix* t_matrix = (struct Matrix*)malloc(sizeof(struct Matrix));
    t_matrix->row = matrix->col;
    t_matrix->col = matrix->row;
    t_matrix->data = (float*)malloc(t_matrix->row * t_matrix->col * sizeof(float));

    for(int i=0;i<matrix->row;i++){
        for(int j=0;j<matrix->col;j++){
            t_matrix->data[j*t_matrix->col + i] = matrix->data[i*matrix->col + j];
        }
    }

    return t_matrix;
}