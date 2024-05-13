serial:
	gcc n_body_serial.c -fopenmp -Wall -o serial -lm

one_crit:
	gcc n_body_one_crit.c -fopenmp -Wall -o one_crit -lm

extra:
	gcc n_body_extra.c -fopenmp -Wall -o extra -lm

local:
	gcc n_body_local.c -fopenmp -Wall -o local -lm

n_locks:
	gcc n_body_n_locks.c -fopenmp -Wall -o n_locks -lm

6_atom:
	gcc n_body_6_atom.c -fopenmp -Wall -o 6_atom -lm

gdb:
	gcc n_body_local.c -fopenmp -Wall -O0 -g3 -o gdb -lm