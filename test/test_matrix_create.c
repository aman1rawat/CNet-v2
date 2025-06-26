#include"../include/CNet.h"
#include<assert.h>
#include<stddef.h>

void test(int ROW, int COL){
	struct Matrix *matrix = create_matrix(ROW, COL);
	if(ROW<=0 || COL<=0){
		assert(matrix==NULL);
		return;
	}
	
	assert(matrix!=NULL);
	
	init_matrix(matrix);
	for(int i=0;i<ROW*COL;i++){
		assert(matrix->data[i]<=1.0f && matrix->data[i]>=-1.0f);
	}

	print_matrix(matrix);

	delete_matrix(matrix);
}

int main(){
	test(5,5);
	test(1,10);
	test(2,3);
	test(0,0);
	test(1,0);
	test(0,1);
	return 0;
}