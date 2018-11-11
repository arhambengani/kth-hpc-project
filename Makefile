CC=mpicc
CFLAGS=-lm

project:step_2/step_2.c
	$(CC) -o step_2/step_2.x step_2/step_2.c libs/mmio.c $(CFLAGS)
	$(CC) -o step_3/step_3.x step_3/step_3.c libs/mmio.c $(CFLAGS)
	icc -qopenmp step_5/step_5.c -o step_5/step_5.x libs/mmio.c $(CFLAGS)
	$(CC) -o step_6/step_6.x step_6/step_6.c libs/mmio.c $(CFLAGS)

