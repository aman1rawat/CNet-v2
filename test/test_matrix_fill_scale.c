#include"../include/CNet.h"
#include<assert.h>
#include<stddef.h>
#include<math.h>

void test(int ROW, int COL, float FILL, float SCALE){
	struct Matrix *matrix = create_matrix(ROW, COL);
	
	fill_matrix(matrix, FILL);

	for(int i=0;i<ROW*COL;i++){
		assert(fabs(matrix->data[i]-FILL)<1e-5);
	}

	scale_matrix(matrix, SCALE);

	for(int i=0;i<ROW*COL;i++){
		assert(fabs(matrix->data[i]-(FILL*SCALE))<1e-5);
	}

	delete_matrix(matrix);
}

int main(){
	test(3,3,5.0f,4.0f);
	test(1,6,2.0f,3.2f);
	test(4,7,5.6f,5.9f);
	return 0;
}