#  Grupo - 5
# 
#  Caique Honório Cardoso - 8910222
#  David Felipe Santos e Souza Dias - 11800611
#  Eduardo Higa - 10262669
#  Emerson Pereira Portela Filho - 11800625
#  Gabriel de Avelar Las Casas Rebelo - 11800462
#  Rafael Araujo Tetzner - 11801136
# 
#  SSC0903 - Trabalho 1
#  PSRS com OpenMP

CC = mpicc
FLAGS = -std=c99 -fopenmp
OUT = psrs

all: psrs

psrs: psrs.c
	$(CC) $(FLAGS) $^ -o $(OUT)


clean:
	rm -f $(OUT)