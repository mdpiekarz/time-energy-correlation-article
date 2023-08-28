#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <mkl.h>
#include <omp.h>
#include "jmb_timestamp.h"

#include "dane.h"

#define POCZ 32768
#define ROZMIAR 32768
#define KROK 1
#define WYR 64 

// program rozklada macierz A na iloczyn dwoch macierzy A = LU
// uzyte tylko c
// mkl/lapack
// autor: Beata Bylina
// data: 15.02.2017
// zmiany: 20.01.2023 BB do mierzenia energii

int main(){

printf("#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
printf("#       TESTUJEMY LU \n");
printf("#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
printf("# Podwoja precyzja \n");
int watki;

#pragma omp parallel
{
  watki= omp_get_num_threads();
}
printf("# Liczba watkow %d \n", watki);

//wyswietlanie informacji o czasie i wydajnosci (2/3*n^3/(t*10^9) Gflops)
printf("#**********************************************************\n");
printf("#n    czas_LU_MKL[s]    Performance[Gflops]  info\n");
printf("#**********************************************************\n");
   
int n  ; // n - rozmiar macierzy

double *x; //macierz dla ktorej testujemy 
double *lu;

n=ROZMIAR;

SPR(n);
SPR(x=(double*)mkl_malloc(n*n*sizeof(double),WYR));

lu=(double*)mkl_malloc(n*n*sizeof(double),WYR);


double t0,t1; //do pomiaru czasu wydarzenia
double tLU;


int info;


for(n=POCZ; n<=ROZMIAR; n = (n<1024)?(n*2):(n+KROK)){
  
  PobierzMacierzLos(n,x);
  PrzepiszMac(n,x,lu);
  
  //******************* LU z MKL********************
  jmb_show_timestamp("START LU");
  t0=omp_get_wtime();
  info=LAPACKE_mkl_dgetrfnpi(LAPACK_COL_MAJOR,n,n,n,lu,n);
  t1=omp_get_wtime();
  jmb_show_timestamp("STOP LU"); 
  tLU=t1-t0;
 // printf("W LAPACK info= %d\n", info);
  
  printf("%d  %le %lf  %d \n", n,tLU, (2.0*n*n*n)/(3.0*tLU*1e9), info);
     
}// koniec obrobki macierzy o romziarze n   
mkl_free(x);
mkl_free(lu);

return 0;
}

