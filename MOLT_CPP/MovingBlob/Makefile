# Makefile for building main.cpp
# and other files in subdirectory

EXE_NAME = BlobRunner

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -g -lfftw3 -lm -lstdc++ -fopenmp

# Source files
SRCS = BlobRunner.cpp

OBJS = BlobRunner.o

LDLIBS = ../Derivative.o ../FFT.o ../FD6.o ../MOLTEngine.o

# Executable
EXECUTABLE = ${EXE_NAME}.host

# Target for building the executable
# $(EXECUTABLE): $(OBJS)
# 	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(OBJS)

all: $(OBJS) ../libbase
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(OBJS) $(LDLIBS)

BlobRunner.o: BlobRunner.cpp ../Derivative.h ../FFT.h ../FD6.h ../MOLTEngine.h 
	$(CC) $(CFLAGS) -c BlobRunner.cpp

../libbase:
	$(MAKE) -C ..

# Clean target to remove object files and executable
clean:
	$(MAKE) -C .. clean
	rm -f $(EXECUTABLE) $(OBJS)

iter:
	make clean;
	clear;
	make;

run:
	./BlobRunner.host