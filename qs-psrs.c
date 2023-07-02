// Grupo - 5
//
// Caique Honório Cardoso - 8910222
// David Felipe Santos e Souza Dias - 11800611
// Eduardo Higa - 10262669
// Emerson Pereira Portela Filho - 11800625
// Gabriel de Avelar Las Casas Rebelo - 11800462
// Rafael Araujo Tetzner - 11801136
//
// SSC0903 - Trabalho 1
// PSRS com OpenMP


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <mpi.h>

#define SEED 1

#define MIN(a,b) (((a)<(b))?(a):(b))


/*==================================================================*/
/*==================================================================*/
/*                      Funcoes sobre o array                       */

int* gen_arr(int n) {
    int* arr = (int*) malloc(n * sizeof *arr);
    for(int i = 0; i < n; i++)
        arr[i] = rand();
    return arr;
}

void print_arr(int* arr, int start, int end) {
    for(int i = start; i < end; i++)
        printf("%d, ", arr[i]);
    printf("%d.\n", arr[end]);
}


/*==================================================================*/
/*==================================================================*/
/*                       QuickSort Sequencial                       */
void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low-1;
    for(int j = low; j < high; j++) {
        if(arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i+1], &arr[high]);
    return i+1;
}

void quicksort(int* arr, int low, int high) {
    if(low < high) {
        int pivotIdx = partition(arr, low, high);
        quicksort(arr, low, pivotIdx-1);
        quicksort(arr, pivotIdx+1, high);
    }
}

/*==================================================================*/
/*==================================================================*/
/*                       MergeSort Sequencial                       */

void merge(int arr[], int low, int mid, int high) {
    int n1 = mid - low + 1;
    int n2 = high - mid;

    int left[n1], right[n2];
    for (int i = 0; i < n1; i++)
        left[i] = arr[low + i];
    for (int j = 0; j < n2; j++)
        right[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = low;
    while (i < n1 && j < n2) {
        if (left[i] <= right[j])
            arr[k++] = left[i++];
        else
            arr[k++] = right[j++];
    }

    while (i < n1)
        arr[k++] = left[i++];
    while (j < n2)
        arr[k++] = right[j++];
}

void mergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = low + (high - low) / 2;

        mergeSort(arr, low, mid);

        mergeSort(arr, mid + 1, high);

        merge(arr, low, mid, high);
    }
}

/*==================================================================*/
/*==================================================================*/
/*                        QuickSort Paralelo                        */

struct thread_context {
    int* arr;
    int n;
    int p;
    int id;
    int low;
    int high;
};

struct thread_context get_context(int* arr, int n, int p) {
    struct thread_context ctx;
    int div = n/p;
    int rem = n%p;
    ctx.arr = arr;
    ctx.n = n;
    ctx.p = p;
    ctx.id = omp_get_thread_num();
    ctx.low = ctx.id * div + MIN(ctx.id, rem);
    ctx.high= (ctx.id+1) * div + MIN(ctx.id+1, rem) - 1;
    return ctx;
}

void phaseI(struct thread_context t, int* samples) {
    quicksort(t.arr, t.low, t.high);
    int offset = t.p * t.id;
    for(int idx = 0; idx < t.p; idx++) {
        samples[offset + idx] = t.arr[t.low + idx * t.n / (t.p * t.p)];
    }

    #pragma omp barrier
}

void phaseII(int p, int* samples, int* pivots) {
    #pragma omp single
    {
        quicksort(samples, 0, p * p - 1);
        for(int i = 1; i < p; i++)
            pivots[i-1] = samples[i*p + p/2 -1];
    }
}

void phaseIII(struct thread_context t, int* pivots, int* partitions) {
    int j = t.low;
    for (int i = 0; i < t.p - 1; i++) {
        while (j <= t.high && t.arr[j] <= pivots[i]) {
            partitions[t.id * t.p + i]++;
            j++;
        }
    }
    while (j <= t.high) {
        partitions[t.id * t.p + t.p - 1]++;
        j++;
    }
    #pragma omp barrier
}

void phaseIV(struct thread_context t, int* partitions, int* res) {
    int k = 0;
    for(int i = 0; i < t.id; i++) {
        for(int j = 0; j < t.p; j++) {
            int ptIdx = i + j * t.p;
            k += partitions[ptIdx];
        }
    }

    int start = k;

    int offset = 0;
    int offsetIdx = 0;
    for(int i = 0; i < t.p; i++) {
        int ptIdx = t.id + i * t.p;
        while(offsetIdx < ptIdx)
            offset += partitions[offsetIdx++];
        for(int j = offset; j < offset + partitions[ptIdx]; j++)
            res[k++] = t.arr[j];
        // merge(res, start, kPrev-1, k-1);
    }

    mergeSort(res, start, k-1);

    #pragma omp barrier
}

int* qs_psrs(int* arr, int n, int p) {
    int* samples = (int*) calloc(p*p, sizeof(int));
    int* pivots  = (int*) calloc(p-1, sizeof(int));
    int* partitions = (int*) calloc(p*p, sizeof(int));
    int* res = (int*) calloc(n, sizeof(int));

    #pragma omp parallel shared(arr, n, p, samples, pivots, partitions, res) num_threads(p)
    {
        struct thread_context ctx = get_context(arr, n, p);

        phaseI(ctx, samples);

        phaseII(p, samples, pivots);

        phaseIII(ctx, pivots, partitions);

        phaseIV(ctx, partitions, res);
    }

    free(samples);
    free(pivots);
    free(partitions);

    return res;
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Numero incorreto de argumentos, uso:\n\t./qs-psrs <n> <nthreads>\n");
        return EXIT_FAILURE;
    }

    srand(SEED);

    int n = atoi(argv[1]);
    int nthreads = atoi(argv[2]);
    
    int* arr = gen_arr(n);
    int* res = qs_psrs(arr, n, nthreads);
    print_arr(res, 0, n-1);

    free(arr);
    free(res);

    return EXIT_SUCCESS;
}