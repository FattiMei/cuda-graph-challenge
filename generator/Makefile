CC=nvcc
OPT=-O2

all: generator.cpp
	$(CC) -o gen generator.cpp $(OPT)

run: all
	./gen 1000 100
