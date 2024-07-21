# Makefile for building main.cpp
# and other files in subdirectory

EXE_NAME = MOLTRunner

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -g -lfftw3 -lm -lstdc++ -fopenmp

# Source files
SRC_MAIN = BlobRunner.cpp
SRC_OTHER = ../MOLTEngine.cpp

# Header files
HDR_OTHER = ../MOLTEngine.h

# Object files
OBJ_MAIN = $(SRC_MAIN:.cpp=.o)
OBJ_OTHER = $(SRC_OTHER:.cpp=.o)

# Executable
EXECUTABLE = ${EXE_NAME}.host

# Target for building the executable
$(EXECUTABLE): $(OBJ_MAIN) $(OBJ_OTHER)
	$(CC) $(CFLAGS) $^ -o $@

# Target for building the main object file
$(OBJ_MAIN): $(SRC_MAIN)
	$(CC) $(CFLAGS) -c $< -o $@

# Target for building the other object file
$(OBJ_OTHER): $(SRC_OTHER) $(HDR_OTHER)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean target to remove object files and executable
clean:
	rm -f $(EXECUTABLE) $(OBJ_MAIN) $(OBJ_OTHER)

iter:
	make clean;
	clear;
	make;

run:
	./MOLTRunner.host