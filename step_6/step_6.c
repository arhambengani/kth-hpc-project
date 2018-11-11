  #include <stdio.h>
  #include <mpi.h>
  #include <math.h>
  #include "../libs/mmio.h"
  #include <stdlib.h>

  int MAX_SIZE = 0, n = 0;

  int max_col_loc(int k, double **a)
  {
    int q, max_row;
    double max = 0;

    max_row = -1;

    for (q=k ; q<n; q++) {

      if(fabs(a[q][k]) >= max){
        max = fabs(a[q][k]);
        max_row = q;
      }
    }

    return max_row;
  }

  void exchange_row(int r, int k, double **a, double *b)
  {
    int p, me;

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    double exchange_row;
    int i;
    for(i = me; i < n; i += p){
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
    int p, me;

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    double *x, l[MAX_SIZE];
    int i,j,k,r, tag=42;
    int rankForBroadcast;

    MPI_Status status;

    x = (double *) malloc(n * sizeof(double));

    for (k=0 ; k<n-1 ; k++) { /* Forward elimination */

      rankForBroadcast = k % p;

      if(k % p == me){
        r = max_col_loc(k,a);
      }

      MPI_Bcast(&r,1,MPI_INT,rankForBroadcast,MPI_COMM_WORLD);

      exchange_row(r,k,a,b);

      if(k % p == me){
        for (i = k+1; i<n; i++){
          l[i] = a[i][k] / a[k][k];
        }
      }

      MPI_Bcast(l+k+1,n-(k+1),MPI_DOUBLE,rankForBroadcast,MPI_COMM_WORLD);

      for(i = me; i<n; i += p){
        if(i >= k){
          for(j=k+1;j<n;j++){
            a[j][i] = a[j][i] - l[j]*a[k][i];
          }
        }
      }
      for(j=k+1;j<n;j++){
        b[j] = b[j] - l[j]*b[k];
      }

      MPI_Barrier(MPI_COMM_WORLD);

    }


    for (k=n-1; k>=0; k--){ /* Backward substitution */

      double sum = 0.0;

      for (j=k+1; j < n; j++){
        if(j % p == me){
          sum = sum + a[k][j] * x[j];
        }
        MPI_Bcast(&sum,1,MPI_DOUBLE,j%p,MPI_COMM_WORLD);
      }

      if(k % p == me){
        x[k] = 1/a[k][k] * (b[k] - sum);
      }

      MPI_Bcast(&x[k],1,MPI_DOUBLE,k%p,MPI_COMM_WORLD);
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

    if(argc < 2){
      fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
      exit(1);
    }else{
      if ((f = fopen(argv[4], "r")) == NULL)
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

    MPI_Init(&argc, &argv);
    int p, me;

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    double tbeg = MPI_Wtime();

    x = gauss_cyclic(a, b);
    MPI_Barrier(MPI_COMM_WORLD);

    double elapsedTime = MPI_Wtime() - tbeg;

    if(me == 0){

      // printf("\n Time taken %f \n", elapsedTime);

      f2 = fopen(argv[6], "w+");

      mm_initialize_typecode(&matcode);
      mm_set_matrix(&matcode);
      mm_set_array(&matcode);
      mm_set_real(&matcode);

      mm_write_banner(f2, matcode);
      mm_write_mtx_array_size(f2, M, 1);

      for( i = 0; i < N; i++){
        fprintf(f2, "%20.17lf\n", x[i]);
      }

    }

    MPI_Finalize();

    return 0;
  }
