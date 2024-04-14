#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define _POSIX_S_SOURCE 1

const double PI = 3.14159265358979323846;

struct thread_data {
    double sum;
    uint8_t padding[64 - sizeof(double)];
};

double getrand()
{
    return (double)rand() / RAND_MAX;
}

double getrand2(unsigned int *seed)
{
    return (double)rand_r(seed) / RAND_MAX;
}

double func(double x)
{
    return sqrt(x * (3 - x)) / (x + 1);
}

double func2(double x, double y)
{
    return exp(x - y);
}

double runge_ser()
{
    double t = omp_get_wtime();
    const double a = 1;
    const double b = 1.2;
    const double eps = 1e-5;
    const int n0 = 10000000;
    printf("Numerical integration: [%f, %f], n = %d, eps = %f, threads = serial\n", a, b, n0, eps);

    int n = n0, k;
    double sq[2], delta = 1;

    for (k = 0; delta > eps; n *= 2, k ^= 1)
    {
        double h = (b - a) / n;
        double s = 0.0;
        for (int i = 0; i < n; i++)
        {
            s += func(a + h * (i + 0.5));
        }
        sq[k] = s * h;
        if (n > n0)
        {
            delta = fabs(sq[k] - sq[k ^ 1]) / 3.0;
        }
    }

    printf("Result PI: %.12f; Runge rule: EPS %.0e, n %d\n", sq[k] * sq[k], eps, n / 2);
    t = omp_get_wtime() - t;
    printf("Ellapsed time (sec.): %.6f\n", t);
    return t;
}

double runge_par(int threads)
{
    double t = omp_get_wtime();
    const double a = 1;
    const double b = 1.2;
    const double eps = 1e-5;
    const int n0 = 10000000;
    printf("Numerical integration: [%f, %f], n = %d, eps = %f, threads = %d\n", a, b, n0, eps, threads);

    double sq[2];
    #pragma omp parallel num_threads(threads)
    {
        int n = n0, k;
        double delta = 1;
        for (k = 0; delta > eps; n *= 2, k ^= 1)
        {
            double h = (b - a) / n;
            double s = 0.0;
            sq[k] = 0;

            #pragma omp barrier

            #pragma omp for nowait
            for (int i = 0; i < n; i++)
            {
                s += func(a + h * (i + 0.5));
            }

            #pragma omp atomic
            sq[k] += s * h;

            #pragma omp barrier
            if (n > n0)
            {
                delta = fabs(sq[k] - sq[k ^ 1]) / 3.0;
            }
            #if 0
            printf("n = %d, i = %d, sq = %.12f, delta = %.12f\n", n, k, sq[k], delta);
            #endif
        }

        #pragma omp master
        printf("Result Pi: %.12f; Runge rule: EPS %.0e, n %d\n", sq[k] * sq[k], eps, n / 2);
    }

    t = omp_get_wtime() - t;
    printf("Ellapsed time (sec.): %.6f\n", t);
    return t;
}

double monte_karlo_ser(int degree)
{
    double t = omp_get_wtime();
    int n;
    if (degree == 1)
    {
        n = 10000000;
    }
    else if (degree == 2)
    {
        n = 100000000;
    }
    printf("Numerical integration by Monte Carlo method: n = %d, threads = serial\n", n);

    int in = 0;
    double s = 0;
    
    for (int i = 0; i < n; i++)
    {
        double x = getrand() - 1;
        double y = getrand();
        if (y <= 1)
        {
            in++;
            s += func2(x, y);
        }
    }

    double v = PI * in / n;
    double res = v * s / in;
    printf("Result: %.12f, n %d\n", res, n);
    t = omp_get_wtime() - t;
    printf("Ellapsed time (sec.): %.6f\n", t);
    return t;
}

double monte_karlo_par(int threads, int degree)
{
    double t = omp_get_wtime();
    int n;
    if (degree == 1)
    {
        n = 10000000;
    }
    else if (degree == 2)
    {
        n = 100000000;
    }
    printf("Numerical integration by Monte Carlo method: n = %d, threads = %d\n", n, threads);

    int in = 0;
    double s = 0;

    #pragma omp parallel num_threads(threads)
    {
        double s_loc = 0;
        int in_loc = 0;
        unsigned int seed = omp_get_thread_num();

        #pragma omp for nowait
        for (int i = 0; i < n; i++)
        {
            double x = getrand2(&seed) - 1;
            double y = getrand2(&seed);
            if (y <= 1)
            {
                in_loc++;
                s_loc += func2(x, y);
            }
        }

        #pragma omp atomic
        s += s_loc;

        #pragma omp atomic
        in += in_loc;
    }

    double v = PI * in / n;
    double res = v * s / in;
    printf("Result: %.12f, n %d\n", res, n);
    t = omp_get_wtime() - t;
    printf("Ellapsed time (sec.): %.6f\n", t);
    return t;
}

int main(int argc, char *argv[])
{
    FILE *result1, *result2, *result3;
    
    double t_ser, t_par, res;
    if (argc == 2 && (strcmp(argv[1], "runge") == 0))
    {
        result1 = fopen("runge.dat", "w");
        t_ser = runge_ser();
        for (int i = 2; i <= 8; i += 2)
        { 
            t_par = runge_par(i);
            res = t_ser / t_par;
            fprintf(result1, "%d %f\n", i, res);
        }
        fclose(result1);
    }
    else if (argc == 2 && (strcmp(argv[1], "monte") == 0))
    {
        result2 = fopen("monte10e7.dat", "w");
        result3 = fopen("monte10e8.dat", "w");

        t_ser = monte_karlo_ser(1);
        for (int i = 2; i <= 8; i += 2)
        {
            t_par = monte_karlo_par(i, 1);
            res = t_ser / t_par;
            fprintf(result2, "%d %f\n", i, res);
        }
        fclose(result2);

        t_ser = monte_karlo_ser(2);
        for (int i = 2; i <= 8; i += 2)
        {
            t_par = monte_karlo_par(i, 2);
            res = t_ser / t_par;
            fprintf(result3, "%d %f\n", i, res);
        }
        fclose(result3);
    }

    return 0;
}