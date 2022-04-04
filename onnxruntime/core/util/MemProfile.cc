#include "./MemProfile.h"
/*
size_t fsmblks;    Space in freed fastbin blocks (bytes) 
size_t uordblks;   Total allocated space (bytes) 
size_t fordblks;   Total free space (bytes) 
size_t keepcost;   Top-most, releasable space (bytes) 
*/
/*
unsigned int getTotalMemory() {
	return EM_ASM_INT({return HEAP8.length;}); 
}

unsigned int getFreeMemory() {
	
	unsigned int totalMemory = getTotalMemory();
	unsigned int dynamicTop = (unsigned int)sbrk(0);
	return totalMemory - dynamicTop + i.fordblks;
}

unsigned int getMissedMemory() {
	s_mallinfo i = mallinfo();
	return i.keepcost;
}*/

void checkMemory(const char* msg){
	static int call_count=0;
	
	call_count++;

	// Smapling malloc info
	s_mallinfo i = mallinfo();
	
	//size_t fsmblks;    Space in freed fastbin blocks (bytes) 
	auto fastbin_free = i.fsmblks;
	//size_t uordblks;   Total allocated space (bytes) 
	auto TotalAlloc = i.uordblks;
	//size_t fordblks;   Total free space (bytes) 
	auto TotalFree = i.fordblks;
	// size_t keepcost;   Top-most, releasable space (bytes) 
	auto releaseable = i.keepcost;
	// Memory from JS
	auto AppTotalMem = EM_ASM_INT({return HEAP8.length;});

	printf("+++ MEM [%s] - #%d +++\n",msg,call_count);
	printf("Total App memory cost - %u, Total memory allocated via malloc %u \n",AppTotalMem,TotalAlloc);
    printf("Total free space - %u, Total memory in fastbin - %u, Total releasable memory - %u\n", TotalFree, fastbin_free, releaseable);
	printf("++++++++++++++++++++++++++++++++++\n");
}