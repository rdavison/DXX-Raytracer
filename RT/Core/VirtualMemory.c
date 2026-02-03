// TODO: I don't want to include Windows.h all the time

#ifdef _WIN32
#include <windows.h>
#else
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

// ------------------------------------------------------------------

#include "VirtualMemory.h"

#ifndef _WIN32
#ifndef MAP_ANON
#define MAP_ANON MAP_ANONYMOUS
#endif

#define VM_TABLE_SIZE 1024

typedef struct VmEntry
{
	void *address;
	size_t size;
} VmEntry;

static VmEntry g_vm_table[VM_TABLE_SIZE];

static void vm_table_insert(void *address, size_t size)
{
	if (!address)
		return;

	for (size_t i = 0; i < VM_TABLE_SIZE; ++i)
	{
		if (g_vm_table[i].address == NULL || g_vm_table[i].address == address)
		{
			g_vm_table[i].address = address;
			g_vm_table[i].size = size;
			return;
		}
	}
}

static size_t vm_table_remove(void *address)
{
	if (!address)
		return 0;

	for (size_t i = 0; i < VM_TABLE_SIZE; ++i)
	{
		if (g_vm_table[i].address == address)
		{
			size_t size = g_vm_table[i].size;
			g_vm_table[i].address = NULL;
			g_vm_table[i].size = 0;
			return size;
		}
	}

	return 0;
}
#endif

void *RT_ReserveVirtualMemory(size_t size)
{
#ifdef _WIN32
	void *result = VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_NOACCESS);
	return result;
#else
	void *result = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (result == MAP_FAILED)
		return NULL;
	vm_table_insert(result, size);
	return result;
#endif
}

bool RT_CommitVirtualMemory(void *address, size_t size)
{
#ifdef _WIN32
	void *result = VirtualAlloc(address, size, MEM_COMMIT, PAGE_READWRITE);
	return !!result;
#else
	int result = mprotect(address, size, PROT_READ | PROT_WRITE);
	return result == 0;
#endif
}

void RT_DecommitVirtualMemory(void *address, size_t size)
{
#ifdef _WIN32
	VirtualFree(address, size, MEM_DECOMMIT);
#else
	(void)madvise(address, size, MADV_DONTNEED);
	(void)mprotect(address, size, PROT_NONE);
#endif
}

void RT_ReleaseVirtualMemory(void *address)
{
#ifdef _WIN32
	VirtualFree(address, 0, MEM_RELEASE);
#else
	size_t size = vm_table_remove(address);
	if (size > 0)
	{
		(void)munmap(address, size);
	}
#endif
}
