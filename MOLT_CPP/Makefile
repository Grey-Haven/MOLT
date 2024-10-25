# Makefile for building main.cpp
# and other files in subdirectory

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -g -lfftw3 -lm -lstdc++ -fopenmp

OBJS = QuadraticInterpolate.o LinearInterpolate.o Derivative.o FD6.o FFT.o MOLTEngine.o

# Target for building the executable
all: $(OBJS)

Derivative.o: Derivative.cpp Derivative.h
	$(CC) $(CFLAGS) -c Derivative.cpp

FFT.o: FFT.cpp FFT.h Derivative.h
	$(CC) $(CFLAGS) -c FFT.cpp

FD6.o: FD6.cpp FD6.h Derivative.h
	$(CC) $(CFLAGS) -c FD6.cpp

LinearInterpolate.o: LinearInterpolate.cpp LinearInterpolate.h Interpolate.h
	$(CC) $(CFLAGS) -c LinearInterpolate.cpp

QuadraticInterpolate.o: QuadraticInterpolate.cpp QuadraticInterpolate.h Interpolate.h
	$(CC) $(CFLAGS) -c QuadraticInterpolate.cpp

MOLTEngine.o: MOLTEngine.cpp MOLTEngine.h Derivative.h FD6.h FFT.h Interpolate.h
	$(CC) $(CFLAGS) -c MOLTEngine.cpp


# Clean target to remove object files and executable
clean:
	rm -f *.o

iter:
	make clean;
	clear;
	make;

# test:
# module load googletest
# g++ -std=c++17 -o fft_test FFTTester.cpp FFT.cpp -lgtest -lgtest_main -pthread -lfftw3
# g++ -std=c++17 -o field_test FieldTester.cpp FFT.cpp -lgtest -lgtest_main -pthread -lfftw3

# g++ -std=c++17 -o fd6_test FD6Tester.cpp FD6.cpp -lgtest -lgtest_main -pthread