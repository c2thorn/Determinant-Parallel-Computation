#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int ARRAYSIZE = 16;
#define INDEX(i,j) i*ARRAYSIZE+j
struct rec
{
	int x,y,z;
};

void ludecomp(int n, double *a){
    int i,k;

    for(k = 0; k < n - 1; ++k) {
        // for the vectoriser
        for(i = k + 1; i < n; i++) {
			a[INDEX(i,k)] /= a[INDEX(k,k)];
        }

#pragma omp parallel for shared(a,n,k) private(i) schedule(static, 64)
        for(i = k + 1; i < n; i++) {
            int j;
            const double aik = a[INDEX(i,k)]; // some compilers will do this automatically
            for(j = k + 1; j < n; j++) {
                a[INDEX(i,j)] -= aik * a[INDEX(k,j)];
            }
        }
    }
}

int main (int argc, char *argv[]) 
{
    int i,j;
	
	int nthreads, tid;
    char f_name[50];
	
	if (argc > 1){
		printf("Receiving: %s\n", argv[1]); 
		ARRAYSIZE = atoi(argv[1]);
	}
	
	double *a;
	a = (double*)calloc(ARRAYSIZE*ARRAYSIZE, sizeof(double));
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
		fread(&a[INDEX(i,j)],sizeof(double),1,datafile);
        }
		printf("Matrix has been read.\n");

	printf("Performing LU decomp.\n");
	ludecomp(ARRAYSIZE,a);
	
	det = a[INDEX(0,0)];
	logsum = log10(fabs(a[INDEX(0,0)]));
	for (size_t i = 1; i < ARRAYSIZE; i++){
		det *= a[INDEX(i,i)];
		logsum += log10(fabs(a[INDEX(i,i)]));
	}
	
	printf("Determinant: %f\n",det);
	printf("log10(fabs(det)): %f\n",logsum);
}