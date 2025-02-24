# Makefile for building main.cpp
# and other files in subdirectory

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -g -lfftw3 -lm -lstdc++ -fopenmp

SRCS = CubicInterpolate.cpp QuadraticInterpolate.cpp LinearInterpolate.cpp
OBJS = $(SRCS:.cpp=.o)

# Executable
EXECUTABLE = ${EXE_NAME}.host

# Target for building the executable
# $(EXECUTABLE): $(OBJS)
# 	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(OBJS)

all: $(OBJS)

%.o: %.cpp $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean target to remove object files and executable
clean:
	rm -f $(OBJS)

iter:
	make clean;
	clear;
	make;