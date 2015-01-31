/**
 * CUDA MD5 cracker
 * Copyright (C) 2015  Konrad Kusnierz <iryont@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <sstream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>

#define CONST_WORD_LIMIT 10
#define CONST_CHARSET_LIMIT 100

#define CONST_CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CONST_CHARSET_LENGTH (sizeof(CONST_CHARSET) - 1)

#define CONST_WORD_LENGTH_MIN 1
#define CONST_WORD_LENGTH_MAX 6

#define TOTAL_BLOCKS 16384UL
#define TOTAL_THREADS 512UL
#define HASHES_PER_KERNEL 128UL

#include "assert.cu"
#include "md5.cu"

/* Global variables */
uint8_t g_wordLength;

char g_word[CONST_WORD_LIMIT];
char g_charset[CONST_CHARSET_LIMIT];
char g_cracked[CONST_WORD_LIMIT];

__device__ char g_deviceCharset[CONST_CHARSET_LIMIT];
__device__ char g_deviceCracked[CONST_WORD_LIMIT];

__device__ __host__ bool next(uint8_t* length, char* word, uint32_t increment){
  uint32_t idx = 0;
  uint32_t add = 0;
  
  while(increment > 0 && idx < CONST_WORD_LIMIT){
    if(idx >= *length && increment > 0){
      increment--;
    }
    
    add = increment + word[idx];
    word[idx] = add % CONST_CHARSET_LENGTH;
    increment = add / CONST_CHARSET_LENGTH;
    idx++;
  }
  
  if(idx > *length){
    *length = idx;
  }
  
  if(idx > CONST_WORD_LENGTH_MAX){
    return false;
  }

  return true;
}

__global__ void md5Crack(uint8_t wordLength, char* charsetWord, uint32_t hash01, uint32_t hash02, uint32_t hash03, uint32_t hash04){
  uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * HASHES_PER_KERNEL;
  
  /* Shared variables */
  __shared__ char sharedCharset[CONST_CHARSET_LIMIT];
  
  /* Thread variables */
  char threadCharsetWord[CONST_WORD_LIMIT];
  char threadTextWord[CONST_WORD_LIMIT];
  uint8_t threadWordLength;
  uint32_t threadHash01, threadHash02, threadHash03, threadHash04;
  
  /* Copy everything to local memory */
  memcpy(threadCharsetWord, charsetWord, CONST_WORD_LIMIT);
  memcpy(&threadWordLength, &wordLength, sizeof(uint8_t));
  memcpy(sharedCharset, g_deviceCharset, sizeof(uint8_t) * CONST_CHARSET_LIMIT);
  
  /* Increment current word by thread index */
  next(&threadWordLength, threadCharsetWord, idx);
  
  for(uint32_t hash = 0; hash < HASHES_PER_KERNEL; hash++){
    for(uint32_t i = 0; i < threadWordLength; i++){
      threadTextWord[i] = sharedCharset[threadCharsetWord[i]];
    }
    
    md5Hash((unsigned char*)threadTextWord, threadWordLength, &threadHash01, &threadHash02, &threadHash03, &threadHash04);   

    if(threadHash01 == hash01 && threadHash02 == hash02 && threadHash03 == hash03 && threadHash04 == hash04){
      memcpy(g_deviceCracked, threadTextWord, threadWordLength);
    }
    
    if(!next(&threadWordLength, threadCharsetWord, 1)){
      break;
    }
  }
}

int main(int argc, char* argv[]){
  /* Check arguments */
  if(argc != 2 || strlen(argv[1]) != 32){
    std::cout << argv[0] << " <md5_hash>" << std::endl;
    return -1;
  }
  
  /* Amount of available devices */
  int devices;
  ERROR_CHECK(cudaGetDeviceCount(&devices));
  
  /* Sync type */
  ERROR_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
  
  /* Display amount of devices */
  std::cout << "Notice: " << devices << " device(s) found" << std::endl;
  
  /* Hash stored as u32 integers */
  uint32_t md5Hash[4];
  
  /* Parse argument */
  for(uint8_t i = 0; i < 4; i++){
    char tmp[16];
    
    strncpy(tmp, argv[1] + i * 8, 8);
    sscanf(tmp, "%x", &md5Hash[i]);   
    md5Hash[i] = (md5Hash[i] & 0xFF000000) >> 24 | (md5Hash[i] & 0x00FF0000) >> 8 | (md5Hash[i] & 0x0000FF00) << 8 | (md5Hash[i] & 0x000000FF) << 24;
  }
  
  /* Fill memory */
  memset(g_word, 0, CONST_WORD_LIMIT);
  memset(g_cracked, 0, CONST_WORD_LIMIT);
  memcpy(g_charset, CONST_CHARSET, CONST_CHARSET_LENGTH);
  
  /* Current word length = minimum word length */
  g_wordLength = CONST_WORD_LENGTH_MIN;
  
  /* Main device */
  cudaSetDevice(0);
  
  /* Time */
  cudaEvent_t clockBegin;
  cudaEvent_t clockLast;
  
  cudaEventCreate(&clockBegin);
  cudaEventCreate(&clockLast);
  cudaEventRecord(clockBegin, 0);
  
  /* Current word is different on each device */
  char** words = new char*[devices];
  
  for(int device = 0; device < devices; device++){
    cudaSetDevice(device);
    
    /* Copy to each device */
    ERROR_CHECK(cudaMemcpyToSymbol(g_deviceCharset, g_charset, sizeof(uint8_t) * CONST_CHARSET_LIMIT, 0, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpyToSymbol(g_deviceCracked, g_cracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyHostToDevice));
    
    /* Allocate on each device */
    ERROR_CHECK(cudaMalloc((void**)&words[device], sizeof(uint8_t) * CONST_WORD_LIMIT));
  }
  
  while(true){
    bool result = false;
    bool found = false;
    
    for(int device = 0; device < devices; device++){
      cudaSetDevice(device);
      
      /* Copy current data */
      ERROR_CHECK(cudaMemcpy(words[device], g_word, sizeof(uint8_t) * CONST_WORD_LIMIT, cudaMemcpyHostToDevice)); 
    
      /* Start kernel */
      md5Crack<<<TOTAL_BLOCKS, TOTAL_THREADS>>>(g_wordLength, words[device], md5Hash[0], md5Hash[1], md5Hash[2], md5Hash[3]);
      
      /* Global increment */
      result = next(&g_wordLength, g_word, TOTAL_THREADS * HASHES_PER_KERNEL * TOTAL_BLOCKS);
    }
    
    /* Display progress */
    char word[CONST_WORD_LIMIT];
    
    for(int i = 0; i < g_wordLength; i++){
      word[i] = g_charset[g_word[i]];
    }
    
    std::cout << "Notice: currently at " << std::string(word, g_wordLength) << " (" << (uint32_t)g_wordLength << ")" << std::endl;
    
    for(int device = 0; device < devices; device++){
      cudaSetDevice(device);
      
      /* Synchronize now */
      cudaDeviceSynchronize();
      
      /* Copy result */
      ERROR_CHECK(cudaMemcpyFromSymbol(g_cracked, g_deviceCracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyDeviceToHost)); 
      
      /* Check result */
      if(found = *g_cracked != 0){     
        std::cout << "Notice: cracked " << g_cracked << std::endl; 
        break;
      }
    }
    
    if(!result || found){
      if(!result && !found){
        std::cout << "Notice: found nothing (host)" << std::endl;
      }
      
      break;
    }
  }
  
  for(int device = 0; device < devices; device++){
    cudaSetDevice(device);
    
    /* Free on each device */
    cudaFree((void**)words[device]);
  }
  
  /* Free array */
  delete[] words;
  
  /* Main device */
  cudaSetDevice(0);
  
  float milliseconds = 0;
  
  cudaEventRecord(clockLast, 0);
  cudaEventSynchronize(clockLast);
  cudaEventElapsedTime(&milliseconds, clockBegin, clockLast);
  
  std::cout << "Notice: computation time " << milliseconds << " ms" << std::endl;
  
  cudaEventDestroy(clockBegin);
  cudaEventDestroy(clockLast);
}
