#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <mpi.h>

#define SEED 1

#define MIN(a,b) (((a)<(b))?(a):(b))


/*==================================================================*/
/*==================================================================*/
/*                      Funcoes sobre o array                       */

int* arr_new(int n) {
    int* arr = (int*) calloc(n, sizeof(int));
    return arr;
}

int* arr_new_debug(int* N) {
    *N = 27;
    int* arr = arr_new(*N);
    int tmp[] = {15,46,48,93,39,6,72,91,14,36,69,40,89,61,97,12,21,54,53,97,84,58,32,27,33,72,20};
    memcpy(arr, tmp, 27 * sizeof(int));
    return arr;
}

void arr_fill_rand(int* arr, int N) {
    for(int i = 0 ; i < N; i++)
        arr[i] = rand() % 10 + 1;
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

/*==================================================================*/
/*==================================================================*/
/*                              Main                                */

void master_fn(int N, int P, int T) {
    int* arr = arr_new_debug(&N); // Cria array, sempre sera o do exemplo, mudar depois TODO
    print_arr(arr, 0, N-1);

    int* samples = arr_new(T*T);
    int* pivots  = arr_new(T-1);

    #pragma omp parallel shared(arr, N, T, samples, pivots) num_threads(T)
    {
        struct thread_context t = get_context(arr, N, T);

        phaseI(t, samples); // Organiza particoes e coleta amostras

        phaseII(T, samples, pivots); // Organiza amostras e coleta pivots

        int dest = t.id + 1; // A thread com id 0 deve enviar para o processo de id 1, master nao participa das fases em mpi,
        // se alguem quiser pode mudar isso dps
        int len = t.high - t.low + 1; // tamanho da particao
        MPI_Send(pivots, T-1, MPI_INT, dest, 0, MPI_COMM_WORLD); // mando pivots
        MPI_Send(&len, 1, MPI_INT, dest, 0, MPI_COMM_WORLD); // mando tamanho
        MPI_Send(&(arr[t.low]), len, MPI_INT, dest, 0, MPI_COMM_WORLD); // mando particao
    }
}

void worker_fn(int T, int rank) {
    int src = 0; // recebe todos os dados da thread master

    int* pivots = arr_new(T-1);
    MPI_Recv(pivots, T-1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recebe pivots
    printf("Process %d received pivots: ", rank);
    print_arr(pivots, 0, T-2);

    int len;
    MPI_Recv(&len, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recebe tamanho particao
    printf("Process %d received len: %d\n", rank, len);

    int* arr = arr_new(len);
    MPI_Recv(arr, len, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recebe particao
    printf("Process %d received arr: ", rank);
    print_arr(arr, 0, len-1);


    // continuacao fase II, cada processo "sepera" sua lista em sublista baseadas no pivot
    // estou armazenando essa informacao na forma:
    // pivots: 33 69
    // arr: 12 21 36 40 54 61 69 89 97
    //
    //              33                69 
    //         12 21 | 36 40 54 61 69 | 89 97
    // pivots:   2           5            2
    // numero de elementos entre cada pivot, eventualmente 0
    int* partitions = arr_new(T);
    int j = 0;
    for(int i = 0; i < T-1; i++) {
        while(j < len && arr[j] <= pivots[i]) {
            partitions[i]++;
            j++;
        }
    }
    while(j < len) {
        partitions[T-1]++;
        j++;
    }
    printf("Process %d partitions count: ", rank);
    print_arr(partitions, 0, T-1);
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Numero incorreto de argumentos, uso:\n\tmpirun -np <P> psrs <N>\n");
        return EXIT_FAILURE;
    }

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); // Inicia MPI permitindo chamada em multiplas threads

    int P, rank, N, T;
    MPI_Comm_size(MPI_COMM_WORLD, &P); // Descobre o numero de processos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // O rank deste processo
    N = atoi(argv[1]); // O tamanho do vetor
    T = P - 1; // O numero de threads, 1 master + P - 1 workers

    if(rank == 0) {
        master_fn(N, P, T);
    } else {
        worker_fn(T, rank);
    }

    MPI_Finalize();
}