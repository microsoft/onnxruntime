
#pragma once
#ifndef _MEMPROFILE_
#define _MEMPROFILE_

#include <iostream>
#include <unistd.h>
#include <emscripten.h>

struct s_mallinfo {
	int arena;    /* non-mmapped space allocated from system */
	int ordblks;  /* number of free chunks */
	int smblks;   /* always 0 */
	int hblks;    /* always 0 */
	int hblkhd;   /* space in mmapped regions */
	int usmblks;  /* maximum total allocated space */
	int fsmblks;  /* always 0 */
	int uordblks; /* total allocated space */
	int fordblks; /* total free space */
	int keepcost; /* releasable (via malloc_trim) space */
};

extern "C" {
	extern s_mallinfo mallinfo();
}
/*
unsigned int getTotalMemory();
unsigned int getFreeMemory();*/
void checkMemory(const char* msg);

#endif