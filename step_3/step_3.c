#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include "../libs/mmio.h"
#include <stdlib.h>

typedef int boolean;
#define true  1
#define false 0

int MAX_SIZE = 0, n = 0, p1 = 0, p2 = 0, b1 = 0, b2 = 0, countCommRow = 0, countCommCol = 0;
MPI_Comm prime_comm1;
MPI_Comm prime_comm2;

int Co(int k)
{
  return (((k)/b2) % p2) + 1;
}

int Ro(int k)
{
  return (((k)/b1) % p1) + 1;
}

int Rop(int q)
{
  int group = (q % p1) + 1;
  return group;
}

int Cop(int q)
{
  int groupLeader = q - (Rop(q) - 1);
  int group = groupLeader/p1 + 1;
  return group;
}

boolean member(int me, int group, int rowCol){
  if(rowCol == 1){
    if((me % p1) + 1 == group){
      return true;
    }else{
      return false;
    }
  }else{
    if((me/p1) + 1 == group){
      return true;
    }else{
      return false;
    }
  }
}

int max_col_loc(int k, double **a){
  int p, me, q, max_row = -1, countRow = 0,i,j;
  double max = 0;
  double fabsCheck = 0, check = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  int rowGroup = Rop(me);
  q = (rowGroup - 1) * b1;
  countRow = q;
  for(i = q; i < n; i += b1*p1){
    for(j = 0; j < b1; j++){
      if(countRow >= k && countRow < n){
        if(a[countRow][k] < fabsCheck){
          check = a[countRow][k] * (-1);
        }else{
          check = a[countRow][k];
        }
        if(check >= max){
          max = fabs(a[countRow][k]);
          max_row = countRow;
        }
      }
      countRow += 1;
    }
    countRow += b1 * (p1-1);
  }
  return max_row;
}

int grp_leader(int group)
{
  return p1*(group-1);
}

void exchange_row_loc(int k, int r, double **a, double *b)
{
  int p, me, q, countCol = 0,i,j;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  int colGroup = Cop(me);
  q = (colGroup - 1) * b2;
  countCol = q;
  double exchange_row;

  for(i = q; i < n; i += b2*p2 ){
    for(j = 0; j < b2; j++){
      if(countCol >= k && countCol < n){
      exchange_row = a[r][countCol];
      a[r][countCol] = a[k][countCol];
      a[k][countCol] = exchange_row;
      }
      countCol++;
    }
    countCol += b2 * (p2-1);
  }

  exchange_row = b[r];
  b[r] = b[k];
  b[k] = exchange_row;
}

void copy_row_loc(int k, double *buf, double **a, double *b, int rowCol)
{
  if(rowCol == 1){
    int p, me, q, countCol = 0, count = 0,i,j;

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    int colGroup = Cop(me);
    q = (colGroup - 1) * b2;
    countCol = q;
    double exchange_row;

    for(i = q ; i < n; i += b2*p2 ){
      for(j = 0; j < b2; j++){
        if(countCol >= k && countCol < n){
        buf[count] = a[k][countCol];
        count++;
        }
        countCol++;
      }
      countCol += b2 * (p2-1);
    }
    buf[count] = b[k];
  }else{
    int p, me, q, countCol = 0, count = 0,i,j;

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    int colGroup = Cop(me);
    q = (colGroup - 1) * b2;
    countCol = q;
    double exchange_row;

    for(i = q ; i < n; i += b2*p2 ){
      for(j = 0; j < b2; j++){
        if(countCol >= k && countCol < n){
        a[k][countCol] = buf[count];
        count++;
        }
        countCol++;
      }
      countCol += b2 * (p2-1);
    }
    b[k] = buf[count];
  }

}

int compute_partner(int group, int me)
{
  int rowGroup = Rop(me);
  int colGroup = Cop(me);

  if(rowGroup < me){
    return me - (rowGroup - group);
  }else{
    return me + (group - rowGroup);
  }
}

int compute_size(int n, int k, int group, int rowCol)
{
  if(rowCol == 1){
    int p, me, q, countCol = 0, count = 0,i,j;

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    int colGroup = Cop(me);
    q = (colGroup - 1) * b2;
    countCol = q;

    for(i = q; i < n; i += b2*p2){
      for(j = 0; j < b2; j++){
        if(countCol >= k && countCol < n){
        count += 1;
        }
        countCol++;
      }
      countCol += b2 * (p2-1);
    }
    count += 1;
    return count;
  }else{
    int p, me, q, countRow = 0, count = 0,i,j;

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    int rowGroup = Rop(me);
    q = (rowGroup - 1) * b1;
    countRow = q;

    for( i = q; i < n; i += b1*p1 ){
      for(j = 0; j < b1; j++){
        if(countRow > k && countRow < n){
        count++;
        }
        countRow++;
      }
      countRow += b1 * (p1-1);
    }

    return count;
  }
}

void exchange_row_buf(int k, int r, double **a, double *b, double *buf)
{
  int p, me, q, countCol = 0, count = 0,i,j;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  int colGroup = Cop(me);
  q = (colGroup - 1) * b2;
  countCol = q;
  double exchange_row = 0;


  for( i = q; i < n; i += b2*p2){
    for( j = 0; j < b2; j++){
      if(countCol >= k && countCol < n){
        exchange_row = buf[count];
        buf[count] = a[r][countCol];
        a[r][countCol] = exchange_row;
        count++;
      }
      countCol++;
    }
    countCol += b2 * (p2-1);
  }
  exchange_row = buf[count];
  buf[count] = b[r];
  b[r] = exchange_row;
}

int rank(int q, int group, int rowCol)
{
  if(rowCol == 1){
    int count = 0, rankGroup,i;
    int rank = group - 1;
    for( i = 0; i < p2; i++){
      if(rank + (p1*count) == q){
        rankGroup = count;
        break;
      }
      count++;
    }
    return rankGroup;
  }else{
    int count = 0, rankGroup,i;
    int rank = (q/p1 * p1);
    for( i = 0; i < p1; i++){
      if(rank + i == q){
        rankGroup = count;
        break;
      }
      count++;
    }
    return rankGroup;
  }
}

double *compute_elim_fact_loc(int k,double **a, double *b, double *buf)
{
  int p, me, q, countRow = 0, count = 0,i,j;
  double elimFactor = 0, *y;
  y = (double *) malloc((n+1) * sizeof(double));

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  int rowGroup = Rop(me);
  q = (rowGroup - 1) * b1;
  countRow = q;

  for( i = q; i < n; i += b1*p1 ){
    for( j = 0; j < b1; j++){
      if(countRow > k && countRow < n){
      y[count] = a[countRow][k]/buf[0];
      count++;
      }
      countRow++;
    }
    countRow += b1 * (p1-1);
  }

  return y;
}

void compute_local_entries(int k, double **a, double *b, double *elim_buf, double *buf)
{
  int p, me, q, s, countCol = 0, count = 0, countElimBuf = 0, countRow = 0,w,d,i,j;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  int colGroup = Cop(me);
  q = (colGroup - 1) * b2;
  countCol = q;

  int rowGroup = Rop(me);
  s = (rowGroup - 1) * b1;
  countRow = s;

  for( w = s; w < n; w += b1*p1 ){
    for( d = 0; d < b1; d++){
      if(countRow > k && countRow < n){
        for( i = q; i < n; i += b2*p2 ){
          for( j = 0; j < b2; j++){
            if(countCol >= k && countCol < n){
              a[countRow][countCol] = a[countRow][countCol] - elim_buf[countElimBuf]*buf[count];
              count++;
            }
            countCol++;
          }
          countCol += b2 * (p2-1);
        }
        b[countRow] = b[countRow] - elim_buf[countElimBuf]*buf[count];
        countElimBuf += 1;
        countCol = q;
        count=0;
      }
      countRow++;
   }
   countRow += b1 * (p1-1);
 }

}

void backward_substitution(double **a, double *b, double *x)
{
  int p, me,k,j,rankToBroadcast,q,rowGroup,colGroup,ql;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  double sum = 0.0;

  for ( k=n-1; k>=0; k--) {

    double sum = 0.0;
    double sumTotal = 0.0;

    for ( j=k+1; j < n; j++){
      if(member(me,Ro(k),1)){
        if(member(me,Co(j),2)){
          sum = sum + a[k][j] * x[j];
          ql = rank(me,Rop(me),1);
        }
        MPI_Allreduce(&sum,&sumTotal,1,MPI_DOUBLE,MPI_SUM,prime_comm2);
        MPI_Barrier(prime_comm2);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (member(me,Ro(k),1) && member(me,Co(k),2)){
      x[k] = 1/a[k][k] * (b[k] - sumTotal);
    }

    rowGroup = Ro(k);
    colGroup = Co(k);
    rankToBroadcast = ((colGroup - 1) * p1) + (rowGroup - 1);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&x[k],1,MPI_DOUBLE,rankToBroadcast,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

double *gauss_double_cyclic(double **a, double *b)
{

  int world_rank, world_size, num, count;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  if(countCommRow > 0){
    MPI_Comm_free(&prime_comm2);
  }
  int color = (world_rank % p1) + 1;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &prime_comm2);
  countCommRow++;

  int row_rank, row_size;
  MPI_Comm_rank(prime_comm2, &row_rank);
  MPI_Comm_size(prime_comm2, &row_size);

  if(countCommCol > 0){
    MPI_Comm_free(&prime_comm1);
  }
  color = (world_rank/p1) + 1;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &prime_comm1);
  countCommCol++;

  MPI_Comm_rank(prime_comm1, &row_rank);
  MPI_Comm_size(prime_comm1, &row_size);

  int p, me, tag=42;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Barrier(MPI_COMM_WORLD);
  double *x, *buf, *elim_buf;
  int i,j,k,r,q, ql, size, buf_size, elim_size, psz;
  struct { double val; int pvtline; } z,y;
  MPI_Status status;
  x = (double *) malloc(n * sizeof(double));
  buf = (double *) malloc((n+1) * sizeof(double));
  elim_buf = (double *) malloc((n+1) * sizeof(double));

  for (k=0; k<n-1; k++) {
    MPI_Barrier(MPI_COMM_WORLD);

    if (member(me, Co(k),2)) {
      r = max_col_loc(k,a);
      if(r >=0){
        z.pvtline = r; z.val = fabs(a[r][k]);
      }else{
        z.pvtline = -1; z.val = -1;
      }
      MPI_Reduce(&z,&y,1,MPI_DOUBLE_INT,MPI_MAXLOC,0,prime_comm1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&y,1,MPI_DOUBLE_INT,grp_leader(Co(k)),MPI_COMM_WORLD);
    r = y.pvtline;
    MPI_Barrier(MPI_COMM_WORLD);

    if(Ro(k) == Ro(r)){
      /*pivot row and row k are in the same row group */
      if (member(me, Ro(k),1)) {
        if (r != k) exchange_row_loc(k,r,a,b);
        copy_row_loc(k,buf,a,b,1);
      }
    }
    else /* pivot row and row k are in different row groups */
    if (member(me, Ro(k),1)) {
      copy_row_loc(k,buf,a,b,1);
      q = compute_partner(Ro(r),me);
      psz = compute_size(n,k,Ro(k),1);
      MPI_Send(buf,psz,MPI_DOUBLE,q,tag,MPI_COMM_WORLD);
    }
    else if (member(me,Ro(r),1)) {
      /* executing processor contains a part of the pivot row */
      q = compute_partner(Ro(k),me);
      psz = compute_size(n,k,Ro(r),1);
      MPI_Recv(buf,psz,MPI_DOUBLE,q,tag,MPI_COMM_WORLD,&status);
      exchange_row_buf(k,r,a,b,buf);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (q=0; q<p; q++) { /* broadcast of pivot row */
      if (member(q,Ro(r),1) && member(me,Cop(q),2)) {
        ql = rank(q,Cop(q),2); buf_size = compute_size(n,k,Ro(k),1);
        MPI_Bcast(buf,buf_size,MPI_DOUBLE,ql,prime_comm1);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if ((Ro(k) != Ro(r)) && (member(me,Ro(k),1))){
      copy_row_loc(k,buf,a,b,2);
    }
    if (member(me,Co(k),2)){
      elim_buf = compute_elim_fact_loc(k,a,b,buf);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (q=0; q<p; q++) /* broadcast of elimination factors */
      if (member(q,Co(k),2) && member(me,Rop(q),1)) {
        ql = rank(q,Rop(q),1); elim_size = compute_size(n,k,Co(k),2);
        MPI_Bcast(elim_buf,elim_size,MPI_DOUBLE,ql,prime_comm2);
      }

    MPI_Barrier(MPI_COMM_WORLD);

    compute_local_entries(k,a,b,elim_buf,buf);

    MPI_Barrier(MPI_COMM_WORLD);

  }

  MPI_Barrier(MPI_COMM_WORLD);
  backward_substitution(a,b,x);

  return x;
}

int main(int argc, char *argv[])
{
  int ret_code;
  MM_typecode matcode;
  FILE *f , *f2;
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
  int count = 0,j;
  for( i = 0; i < M; i++){
    for( j = 0; j < N; j++){
      a[j][i] = I[count];
      count++;
    }
    b[i] = 1;
  }

  char *str = argv[2];
  p1 = str[0] - 48;
  p2 = str[2] - 48;
  if(n == 8){
    b1 = 1;
    b2 = 2;
  }else{
    b1 = 2;
    b2 = 2;
  }


  MPI_Init(&argc, &argv);
  int p, me;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  double tbeg = MPI_Wtime();

  x = gauss_double_cyclic(a, b);
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
