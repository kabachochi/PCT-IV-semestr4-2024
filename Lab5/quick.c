#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int threshold;

void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void partition(int *v, int *i, int *j, int low, int high)
{
    *i = low;
    *j = high;

    int pivot = v[(low + high) / 2];

    do
    {
        while (v[*i] < pivot)
        {
            (*i)++;
        }
        while (v[*j] > pivot)
        {
            (*j)--;
        }

        if (i <= j)
        {
            swap(v + *i, v + *j);
            (*i)++;
            (*j)--;
        }
    } while(*i <= *j);
}

void quick_tasks(int *v, int low, int high)
{
    int i, j;

    partition(v, &i, &j, low, high);
    
    if (high - low < threshold || (j - low < threshold || high - i < threshold))
    {
        if (low < j)
        {
            quick_tasks(v, low, j);
        }
        if (i < high)
        {
            quick_tasks(v, i, high);
        }
    }
    else
    {
        #pragma omp task
        {
            quick_tasks(v, low, j);
        }
        quick_tasks(v, i, high);
    }
}

void init(int *v, int size)
{
    for (int i = 0; i < size; i++)
    {
        v[i] = rand();
    }
}

int main()
{
    for (threshold = 1000; threshold <= 100000; threshold = threshold * 10)
    {
        for (int size = 1000000; size <= 100000000; size = size * 10)
        {
            char filename[100];
            sprintf(filename, "quick_thre_%d_size_%d.dat", threshold, size);
            FILE *fout = fopen(filename, "w");

            int *array = malloc(sizeof(int) * size);
            init(array, size);

            double serial_time = -omp_get_wtime();
            quick_tasks(array, 0, size - 2);
            serial_time = serial_time + omp_get_wtime();
            free(array);

            printf("N = %d\nSerial: %.6lf\n", size, serial_time);

            for (int threads = 2; threads <= 8; threads = threads + 2)
            {
                array = malloc(sizeof(int) * size);
                init(array, size);

                double parallel_time = -omp_get_wtime();

                #pragma omp parallel num_threads(threads)
                {
                    #pragma omp single
                    quick_tasks(array, 0, size - 1);
                }

                parallel_time = parallel_time + omp_get_wtime();
                printf("N = %d, Threshold = %d, Threads = %d\nSpeedup: %.6lf\n", size, threshold, threads, serial_time / parallel_time);
                fprintf(fout, "%d %.4lf\n", threads, serial_time / parallel_time);
                free(array);
            }
            fclose(fout);
        }
    }
    return 0;
}