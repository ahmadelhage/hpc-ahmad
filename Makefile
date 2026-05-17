.PHONY: all clean

SRC=src/

INCLUDES=-Iexternal/argparse-2.9/include -Iexternal/libnpy/include
CC=g++
MPICC=mpic++
NVCC=nvcc
# Append additional make targets to BINS for subsequent deadlines \
(e.g., cgc_cuda for deadline 2 and cgc_bonus_{PLACEHOLDER} for deadline 3).
MPI_CFLAGS  = $(shell mpicxx --showme:compile)
MPI_LDFLAGS = $(shell mpicxx --showme:link)
BINS=cgc_serial cgc_mpi cgc_cuda
# Feel free to add additional optimization flags to CFLAGS and CUFLAGS.
CFLAGS=-std=c++17 -O3 -march=native -Wall -Wextra -Wnarrowing -Wparentheses -Werror -Wno-unused-parameter -Wno-cast-function-type
CUFLAGS=-arch=sm_89

all: $(BINS) Makefile

cgc_serial: $(SRC)/serial.cpp $(SRC)/common.h
	$(CC) -o $@ $(SRC)/serial.cpp $(CFLAGS) $(INCLUDES)

# Deadline 1
cgc_mpi: $(SRC)/cgc_mpi.cpp $(SRC)/common.h
	$(MPICC) -o $@ $(SRC)/cgc_mpi.cpp $(CFLAGS) $(INCLUDES)

# Deadline 2
cgc_cuda: $(SRC)/cgc_cuda.cu $(SRC)/common.h
	$(NVCC) -o $@ $(SRC)/cgc_cuda.cu -std=c++17 -O3 $(CUFLAGS) -x=cu -ccbin=mpic++ \
		-Xcompiler "-Wall -Wextra -Wnarrowing -Wparentheses -Wno-unused-parameter -Wno-cast-function-type" \
		$(INCLUDES)
# Deadline 3
 cgc_cuda_bonus_tunes: $(SRC)/cgc_bonus_tunes.cu $(SRC)/common.h
 	$(NVCC) -o $@ $(SRC)/cgc_bonus_tunes.cu $(CFLAGS) -x=cu -ccbin=mpic++ $(CUFLAGS) $(INCLUDES)

submit:
	tar -czf submission.tar.gz src/ Makefile *.job *.pdf

clean:
	rm -f $(BINS)
