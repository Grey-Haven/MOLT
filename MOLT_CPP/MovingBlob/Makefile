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

LDLIBS = ../Interpolate_Schemes/CubicInterpolate.o ../Interpolate_Schemes/QuadraticInterpolate.o ../Interpolate_Schemes/LinearInterpolate.o ../Derivative.o ../FFT.o ../FD6.o ../MOLTEngine.o

# Executable
EXECUTABLE = ${EXE_NAME}.host

all: $(OBJS) ../libbase
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(OBJS) $(LDLIBS)

BlobRunner.o: BlobRunner.cpp ../Derivative.h ../FFT.h ../FD6.h ../Interpolate_Schemes/Interpolate.h ../MOLTEngine.h 
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