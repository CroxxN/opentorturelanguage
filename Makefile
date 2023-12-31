CC=gcc
CFLAGS=-Wall -Wextra -pedantic -lOpenCL
DEBUGGER=-ggdb
OPENCL=-D CL_TARGET_OPENCL_VERSION=300

all: cltest.c vadkernel.bc
	$(CC) $(CFLAGS) $(OPENCL) cltest.c -o main

debug: cltest.c vadkernel.bc
	$(CC) cltest.c $(CFLAGS) $(DEBUGGER) $(OPENCL) -o main

run: main
	./main

.PHONY: clean
clean: main
	rm ./main
