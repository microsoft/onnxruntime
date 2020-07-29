// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// This program is a simple MPI-based mock-up to assess the
// performance of reading and writing checkpoints.  It has been tested
// under WSL.
//
// It models four scenarios:
//
// - Each rank writing a checkpoint shard
// - Each rank reading its own checkpoint shard
// - Each rank reading some other rank's checkpoint shard (1 reader per shard)
// - Each rank reading all of the checkpoint shards
//
//
// To build:
// ---------
//
// $ mpiCC main.cc -O2 -Wall -o checkpoint_microbenchmark
//
//
// To write:
// ---------
//
// To generate 4 * 1 GB shard from a 4-rank MPI job:
//
// $ mpirun -n 4 ./checkpoint_microbenchmark my_test write shard 1
//
// Example shards generated from this command:
// $ ls -l my_test*
// -rwxrwxrwx 1 tiharr tiharr 1073741824 Jul 29 13:38 my_test_0
// -rwxrwxrwx 1 tiharr tiharr 1073741824 Jul 29 13:38 my_test_1
// -rwxrwxrwx 1 tiharr tiharr 1073741824 Jul 29 13:38 my_test_2
// -rwxrwxrwx 1 tiharr tiharr 1073741824 Jul 29 13:38 my_test_3                                                          
//
// Example output from this command:
// rank=3                                      Done   local_ms=11903 global_ms=11903
// rank=0                                      Done   local_ms=11753 global_ms=11903
// rank=1                                      Done   local_ms=11304 global_ms=11903
// rank=2                                      Done   local_ms=6874 global_ms=11903
//
// Timing are taken with a barrier just before IO and just after IO.
// The "local_ms" results show the time from just after the first
// barrier until the rank finishes its own IO.  The "global_ms"
// results show the time from just after the first barrier until just
// after the second barrier (until all ranks have finished their IO).
//
//
// To read:
// --------
//
// Each of 4 ranks reads its own shard (assuming stable allocation of
// ranks to hosts):
//
// $ mpirun -n 4 ./checkpoint_microbenchmark my_test read shard 1
//
// Each of 4 ranks reads some other rank's shard.  Each shard is read
// once, and no rank reads its own shard:
//
// $ mpirun -n 4 ./checkpoint_microbenchmark my_test read mix 1
//
// Each of 4 ranks reads data from _all_ of the shards.  Note that the
// total volume read is constant across the tests, and so the 1 GB here
// is split into 4 * 256 MB reads:
//
// $ mpirun -n 4 ./checkpoint_microbenchmark my_test read all 1

#include <cassert>
#include <chrono>
#include <iostream>

#include <fcntl.h>
#include <mpi.h>
#include <unistd.h>

using namespace ::std::chrono;

constexpr size_t chunk_bytes = 16ull * 1024 * 1024;
static int mpi_rank;
static int num_procs;

static uint64_t get_time_ms() {
  milliseconds ms = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
  return ms.count();
}

static ::std::string get_shard_name_from_prefix(::std::string prefix,
						int rank) {
  return prefix + "_" + std::to_string(rank);
}
							 
static void do_write_shard_test(void *buffer,
				int shard,
				size_t size_bytes,
				::std::string name) {
  std::cout << "rank=" << mpi_rank << "                          Writing shard " << shard << "\n";
  auto num_chunks = size_bytes / chunk_bytes;
  
  auto fd = open(name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd == -1) {
    std::cerr << "open() failed errno=" << errno << "\n";
    abort();
  }

  size_t total_written = 0;
  for (auto chunk = 0u; chunk < num_chunks; chunk++) {
    auto nb = write(fd, (char*)buffer+(chunk*chunk_bytes), chunk_bytes);
    if (nb != chunk_bytes) {
      std::cerr << "wrote " << nb << " not " << chunk_bytes << "\n";
      abort();
    }
    total_written += nb;
  }

  std::cout << "rank=" << mpi_rank << "                          Wrote " << total_written << " bytes\n";

  close(fd);
}

static void do_read_shard_test(void *buffer,
			       int shard,
			       size_t size_bytes,
			       ::std::string name) {
  std::cout << "rank=" << mpi_rank << "                          Reading shard " << shard << "\n";
  auto num_chunks = size_bytes / chunk_bytes;
  
  auto fd = open(name.c_str(), O_RDONLY);
  if (fd == -1) {
    std::cerr << "open() failed errno=" << errno << "\n";
    abort();
  }
  
  size_t total_read = 0;
  for (auto chunk = 0u; chunk < num_chunks; chunk++) {
    auto nb = read(fd, (char*)buffer+(chunk*chunk_bytes), chunk_bytes);
    if (nb != chunk_bytes) {
      std::cerr << "read " << nb << " not " << chunk_bytes << "\n";
      abort();
    }
    total_read += nb;
  }

  std::cout << "rank=" << mpi_rank << "                          Got " << total_read << " bytes\n";
  
  close(fd);
}

static void do_read_all_test(void *buffer,
			     size_t size_bytes,
			     ::std::string prefix) {
  for (auto i = 0; i < num_procs; i++) {
    auto shard_name = get_shard_name_from_prefix(prefix, i);
    do_read_shard_test(buffer, i, size_bytes / num_procs, shard_name);
  }
}

int main(int argc, char *argv[]) {
  auto err = MPI_Init(&argc, &argv);
  if (err) {
    std::cerr << "MPI_Init failed " << err <<"\n";
    abort();
  }

  err = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (err) {
    std::cerr << "MPI_Comm_rank failed " << err <<"\n";
    abort();
  }

  err = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  if (err) {
    std::cerr << "MPI_Comm_size failed " << err <<"\n";
    abort();
  }

  if (argc != 5) {
    if (mpi_rank == 0) {
      std::cerr << "Usage: " << argv[0] << " prefix mode pattern size_gb\n";
    }
    abort();
  }
  
  std::string prefix(argv[1]);
  std::string name = get_shard_name_from_prefix(prefix, mpi_rank);
  
  std::string mode(argv[2]);
  if (mode != "read" && mode != "write") {
    if (mpi_rank == 0) {
      std::cerr << "Expected mode 'read' or 'write'\n";
    }
    abort();
  }
  
  std::string pattern(argv[3]);
  if ((pattern != "shard" && pattern != "all" && pattern != "mix") ||
      (pattern == "all" && mode == "write") ||
      (pattern == "mix" && mode == "write")) {
    if (mpi_rank == 0) {
      std::cerr << "Expected pattern 'shard' (read/write), 'all' (read), or 'mix' (read)\n";
    }
    abort();
  }
  
  int size_gb = atoi(argv[4]);
    
  std::cout << "rank=" << mpi_rank <<
    " of " << num_procs <<
    " using name=" << name <<
    " mode=" << mode <<
    " pattern=" << pattern <<
    " size_gb=" << size_gb << std::endl;

  // This first barrier is purely cosmetic, to avoid the initialization messages getting
  // mixed with other output.
  err = MPI_Barrier(MPI_COMM_WORLD);
  if (err) {
    std::cerr << "MPI_Barrier failed at startup\n";
    abort();
  }

  std::cout << "rank=" << mpi_rank << " Allocating \n";
  size_t size_bytes = uint64_t(size_gb) * 1024 * 1024 * 1024;
  auto buffer = malloc(size_bytes);
  if (!buffer) {
    std::cerr << "Allocation of " << size_bytes << " failed\n";
    abort();
  }

  // Clear the buffer to a non-zero value, just in case there is any
  // special handling for demand-zeroing.
  std::cout << "rank=" << mpi_rank << "              Clearing\n";
  memset(buffer, 42, size_bytes);

  err = MPI_Barrier(MPI_COMM_WORLD);
  if (err) {
    std::cerr << "MPI_Barrier failed before writing\n";
    abort();
  }

  // Run the actual test
  auto start_ms = get_time_ms();
  if (mode == "write") {
    do_write_shard_test(buffer, mpi_rank, size_bytes, name);
  } else if (mode == "read") {
    if (pattern == "shard") {
      // Each rank reads its own shard
      do_read_shard_test(buffer, mpi_rank, size_bytes, name);
    } else if (pattern =="mix") {
      // Each rank reads someone else's shard (read each once)
      auto shard_to_read = (mpi_rank + num_procs/2) % num_procs;
      do_read_shard_test(buffer, shard_to_read, size_bytes, name);
    } else {
      // Each rank reads all of the shards
      assert(pattern == "all");
      do_read_all_test(buffer, size_bytes, prefix);
    }
  }

  auto local_done_ms = get_time_ms();
  err = MPI_Barrier(MPI_COMM_WORLD);
  auto global_done_ms = get_time_ms();

  if (err) {
    std::cerr << "MPI_Barrier failed after writing\n";
    abort();
  }

  std::cout << "rank=" << mpi_rank << "                                      Done  " <<
    " local_ms=" << (local_done_ms - start_ms) <<
    " global_ms=" << (global_done_ms - start_ms) <<
    "\n";
  
  MPI_Finalize();
  return 0;
}
