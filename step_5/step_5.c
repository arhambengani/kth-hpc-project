#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include "../libs/mmio.h"
#include <math.h>

int MAX_SIZE = 0, n = 0;

int max_col_loc(int k, double **a)
{
	int q, max_row, max_row_all_process;
	double max = 0, max_all_process = 0;

	max_row = -1;
	max_row_all_process = -1;

  #pragma omp parallel for private(q)
	for (q=k ; q<n; q++) {

		if(fabs(a[q][k]) >= max){
			max = fabs(a[q][k]);
			max_row = q;
		}
	}

	#pragma omp critical
	{
		if(max > max_all_process){
			max_all_process = max;
			max_row_all_process = max_row;
		}
	}

	return max_row_all_process;
}

void exchange_row(int r, int k, double **a, double *b)
{
	double exchange_row;
	int i;

	#pragma omp parallel for private(i,exchange_row)
	for(i = 0; i < n; i++){
		if(i >= k){
			exchange_row = a[r][i];
			a[r][i] = a[k][i];
			a[k][i] = exchange_row;
		}
	}

	exchange_row = b[r];
	b[r] = b[k];
	b[k] = exchange_row;
}

double *gauss_cyclic (double **a, double *b)
{
	double *x, l[MAX_SIZE];
	int i,j,k,r, tag=42;

	x = (double *) malloc(n * sizeof(double));

	for (k=0 ; k<n-1 ; k++) { /* Forward elimination */

		r = max_col_loc(k,a);

		if(k != r){
			exchange_row(r,k,a,b);
		}

    #pragma omp parallel for private(j,i)
		for (i = k+1; i<n; i++){
			l[i] = a[i][k] / a[k][k];
			for(j=k+1; j<n; j++){
				a[i][j]=a[i][j]-l[i]*a[k][j];
			}
			b[i]=b[i]-l[i]*b[k];
		}

	}


	for (k=n-1; k>=0; k--){ /* Backward substitution */

		double sum = 0.0;

		#pragma omp parallel for reduction(+:sum)
		for (j=k+1; j < n; j++){
			sum = sum + a[k][j] * x[j];
		}

		x[k] = 1/a[k][k] * (b[k] - sum);

	}

	return x;
}

int main(int argc, char *argv[])
{
	int ret_code;
	MM_typecode matcode;
	FILE *f, *f2;
	int M, N, nz;
	int i;
	double *I;

	if(argc < 3){
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}else{
		if ((f = fopen(argv[2], "r")) == NULL)
		exit(1);
	}

	if(mm_read_banner(f, &matcode) != 0){
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

	/*  This is how one can screen matrix types if their application */
	/*  only supports a subset of the Matrix Market data types.      */

	if(mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) ){
		printf("Sorry, this application does not support ");
		printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
		exit(1);
	}

	/* find out size of sparse matrix .... */

	if((ret_code = mm_read_mtx_array_size(f, &M, &N)) !=0)
			exit(1);

	nz = M * N;
	/* reseve memory for matrices */

	I = (double *) malloc(nz * sizeof(double));

	/* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
	/*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
	/*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	for(i=0; i<nz; i++){
		fscanf(f, "%lf\n", &I[i]);
		I[i];  /* adjust from 1-based to 0-based */
	}

	/*After reading the array initialisation of a and b starts*/
	n = N;
	MAX_SIZE = N;

	double b[N], *x;
	double **a = malloc( N * sizeof( double* ));
	if(a){
		for (i = 0; i < N; i++)
		{
			a[i] = malloc(sizeof *a[i] * M);
		}
	}
	x = (double *) malloc(N * sizeof(double));
	int count = 0;
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			a[j][i] = I[count];
			count++;
		}
		b[i] = 1;
	}

	double tbeg = omp_get_wtime();

	x = gauss_cyclic(a, b);

	double elapsedTime = omp_get_wtime() - tbeg;

	// printf("\n Time taken %f \n", elapsedTime);

	f2 = fopen(argv[4], "w+");

	mm_initialize_typecode(&matcode);
	mm_set_matrix(&matcode);
	mm_set_array(&matcode);
	mm_set_real(&matcode);

	mm_write_banner(f2, matcode);
	mm_write_mtx_array_size(f2, M, 1);

	for( i = 0; i < N; i++){
		fprintf(f2, "%20.17lf\n", x[i]);
	}


	return 0;
}
