#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct rec
{
	int x,y,z;
};

void PrintMatrix(int m, int n, double **A) {
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%.02f\t", A[i][j]);
		}
		printf("\n");
	}
}

//Print out a vector neatly
void PrintVector(int n, double *x) {
	for (int i = 0; i < n; i++) {
		printf("%.02f\n", x[i]);
	}
}

double **alloc_2d_init(int rows, int cols) {
	double *data = (double *)malloc(rows*cols * sizeof(double));
	double **array = (double **)malloc(rows * sizeof(double*));
	for (int i = 0; i < rows; i++)
		array[i] = &(data[cols*i]);

	return array;
}


void ludecomp(int n, double **a){
    int i,k;

    for(k = 0; k < n - 1; ++k) {
        // for the vectoriser
        for(i = k + 1; i < n; i++) {
            a[i][k] /= a[k][k];
        }

#pragma omp parallel for shared(a,n,k) private(i) schedule(static, 64)
        for(i = k + 1; i < n; i++) {
            int j;
            const double aik = a[i][k]; // some compilers will do this automatically
            for(j = k + 1; j < n; j++) {
                a[i][j] -= aik * a[k][j];
            }
        }
    }
}

int main (int argc, char *argv[]) 
{
    int i,j;
	
	int nthreads, tid;
    char f_name[50];
	int ARRAYSIZE = 16;
	
	if (argc > 1){
		printf("Receiving: %s\n", argv[1]); 
		ARRAYSIZE = atoi(argv[1]);
	}
	double **a;
	a = alloc_2d_init(ARRAYSIZE,ARRAYSIZE); 
	
    double det;
	double logsum;
    //Create filename
    sprintf(f_name,"%d.bin",ARRAYSIZE);
    printf("Reading array file %s of size %dx%d\n",f_name,ARRAYSIZE,ARRAYSIZE);
    //Open file
    FILE *datafile=fopen(f_name,"rb");
    //Read elelements
    for (i=0; i< ARRAYSIZE; i++)
        for (j=0; j< ARRAYSIZE; j++)
        {
            fread(&a[i][j],sizeof(double),1,datafile);
            //printf("a[%d][%d]=%f\n",i,j,a[i][j]);
        }
            printf("Matrix has been read.\n");

	//PrintMatrix(ARRAYSIZE,ARRAYSIZE,a);
	printf("Performing LU decomp.\n");
	ludecomp(ARRAYSIZE,a);
	//PrintMatrix(ARRAYSIZE,ARRAYSIZE,a);
	
	det = a[0][0];
	logsum = log10(fabs(a[0][0]));
	for (size_t i = 1; i < ARRAYSIZE; i++){
		det *= a[i][i];
		logsum += log10(fabs(a[i][i]));
	}
	
	printf("Determinant: %f\n",det);
	printf("log10(fabs(det)): %f\n",logsum);
}