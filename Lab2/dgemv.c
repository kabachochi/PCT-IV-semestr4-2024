#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <omp.h>
#include <time.h>

int m = 15000;
int n = 15000;

double wtime()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1E-9;
}

void omp_dgemv(double *a, double *b, double *c, int x, int y, int thread)
{
    #pragma omp parallel num_threads(thread)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = x / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (x - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i < ub; i++)
        {
            c[i] = 0.0;
            for (int j = 0; j < y; j++)
            {
                c[i] += a[i * y + j] * b[j];
            }
        }
        // printf ("thread %d of %d\n", threadid, nthreads);
    }
}

void dgemv(double *a, double *b, double *c, int x, int y)
{
    for (int i = 0; i < x; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < y; j++)
        {
            c[i] += a[i * y + j] * b[j];
        }
    }
}

double run_omp_dgemv(int can, int thread)
{
    double *a, *b, *c;

    a = malloc(sizeof(*a) * can * can);
    b = malloc(sizeof(*b) * can);
    c = malloc(sizeof(*a) * can);

    for (int i = 0; i < can; i++)
    {
        for (int j = 0; j < can; j++)
        {
            a[i * can + j] = i + j;
        }
    }

    for (int j = 0; j < can; j++)
    {
        b[j] = j;
    }

    double t = wtime();
    omp_dgemv(a, b, c, can, can, thread);
    t = wtime() - t;

    printf("Elapsed time (parallel): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);

    return t;
}

double run_serial_dgemv(int can)
{
    double *a, *b, *c;

    a = malloc(sizeof(*a) * can * can);
    b = malloc(sizeof(*b) * can);
    c = malloc(sizeof(*a) * can);

    for (int i = 0; i < can; i++)
    {
        for (int j = 0; j < can; j++)
        {
            a[i * can + j] = i + j;
        }
    }

    for (int j = 0; j < can; j++)
    {
        b[j] = j;
    }

    double t = wtime();
    dgemv(a, b, c, can, can);
    t = wtime() - t;

    printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);

    return t;
}

int main()
{
    for (int i = 0; i < 3; i++)
    {
        double time = 0;
        int count = m + (i * 5000);
        printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", count, count);
        printf("Memory used: %" PRIu64 " MiB\n", ((count * count + count + count) * sizeof(double)) >> 20);

        time = run_serial_dgemv(count);

        for (int j = 2; j <= 8; j = j + 2)
        {
            double s = 0;
            double time2 = 0;
            printf("Threads - %d\n", j);
            time2 = run_omp_dgemv(count, j);
            s = time / time2;
            printf("Acceleration = %f\n", s);
        }
    }
    return 0;
}