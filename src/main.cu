#include "cuda_runtime.h"
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include "device_launch_parameters.h"
#include <random>
#include <algorithm>
#include "person.hpp"
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define JSON_DATA_FILE "data/IFF7-4_ValinskisV_L1_dat_1.json"
#define RESULT_FILE_NAME "data/IFF7-4_ValinskisV_3_res.txt"

#define ENTRY_CNT_MAX 250
#define MAX_STRING_LEN 4096
#define CUDA_CORES 11

using namespace std;
using json = nlohmann::json;

typedef struct 
{
    int len;
    char str[MAX_STRING_LEN];
} CudaString;

// Function prototypes
int deserializeJsonFile(std::string fileName, Person arr[]);
void saveToFile(std::string fileName, Person outArr[], int outCnt);
void generate_random_array(int* array, size_t size);
__global__ void add(int* a, int* b, int* c, int* d);
__global__ void doStuff(    int *cnt, CudaString *n, int *s, double *b,
                            CudaString *nAns, int *sAns, double *bAns);
__device__ void cudaConcat(char* dest, char*source);
void convertArrsToPeopleArr(int cnt, Person people[], CudaString *nameArr, int *streetArr, double *balanceArr);
void convertPeopleArrToArrs(Person people[], int cnt, CudaString *nameArr, int *streetArr, double *balanceArr);


int main() {   
    // Delcare 
    // Host arguments
    Person people[ENTRY_CNT_MAX];
    Person outPeople[CUDA_CORES];
    CudaString *nameArr;
    int  *streetArr;
    double *balanceArr;
    // Host answers
    CudaString *nameAns;
    int *streetAns;
    double *balanceAns;
    int *thread_ids;
    // Device arguments
    int *countDev;
    CudaString *nameArrDev;
    int  *streetArrDev;
    double *balanceArrDev;
    // Device answers
    CudaString *nameAnsDev;
    int *streetAnsDev;
    double *balanceAnsDev;
    int *thread_idsDev;

    // Start
    cout << "Start L3 Cuda..." << endl;
    cout << "Reading parsing JSON file" << JSON_DATA_FILE << endl;
    int peopleCount = deserializeJsonFile(JSON_DATA_FILE, people);

    // Allocate memory on host
    // Arguments
    cout << "Allocating memory on host..." << endl;
    nameArr = (CudaString*)malloc(peopleCount * sizeof(CudaString)); 
    streetArr = (int*)malloc(peopleCount * sizeof(int));
    balanceArr = (double*)malloc(peopleCount * sizeof(double));
    // Answers
    nameAns = (CudaString*)malloc(CUDA_CORES * sizeof(CudaString));
    streetAns = (int*)malloc(CUDA_CORES * sizeof(int));
    balanceAns= (double*)malloc(CUDA_CORES * sizeof(double));


    cout << "Converting to arrays..." << endl;
    convertPeopleArrToArrs(people, peopleCount, nameArr, streetArr, balanceArr);
    // cout << "Printing.." << endl;

    for(auto i = 0; i < peopleCount; i++)
    {
        printf("Name: %s\t", nameArr[i].str);
        printf("Streetnum: %d\t", streetArr[i]);
        printf("Balance: %f\n", balanceArr[i]);
    }

    // // Alocate device memory
    // // Arguments
    cout << "Allocating memory on device..." << endl;
    cudaMalloc((void**)&countDev, sizeof(int));
    cudaMalloc((void**)&nameArrDev, peopleCount * sizeof(CudaString));
    cudaMalloc((void**)&streetArrDev, peopleCount * sizeof(int));
    cudaMalloc((void**)&balanceArrDev, peopleCount * sizeof(double));
    // Answers
    cudaMalloc((void**)&nameAnsDev, CUDA_CORES * sizeof(CudaString));
    cudaMalloc((void**)&streetAnsDev, CUDA_CORES * sizeof(int));
    cudaMalloc((void**)&balanceAnsDev, CUDA_CORES * sizeof(double));

    cout << "Copy memory form host to device..." << endl;
    cudaMemcpy(countDev, &peopleCount, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nameArrDev, nameArr, peopleCount * sizeof(CudaString), cudaMemcpyHostToDevice);
    cudaMemcpy(streetArrDev, streetArr, peopleCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(balanceArrDev, balanceArr, peopleCount * sizeof(double), cudaMemcpyHostToDevice);

    cout << "Execute kernel..." << endl;
    // Techniškai gijų skaičius turi dalytis iš 32 (wrapsize), kitu atveju bus neefektyviai išnaudojami GPU resusai
    doStuff<<<1, CUDA_CORES>>>(     countDev, nameArrDev, streetArrDev, balanceArrDev,
                                    nameAnsDev, streetAnsDev, balanceAnsDev);

    cout << "Synchronize device threads..." << endl;
    cudaDeviceSynchronize();

    cout << "Copying memmory from device to host..." << endl;
    cudaMemcpy(nameAns, nameAnsDev, CUDA_CORES * sizeof(CudaString), cudaMemcpyDeviceToHost);
    cudaMemcpy(streetAns, streetAnsDev, CUDA_CORES * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(balanceAns, balanceAnsDev, CUDA_CORES * sizeof(double), cudaMemcpyDeviceToHost);

    cout << "Freeing memmory..." << endl;
    cudaFree(nameArrDev);
    cudaFree(nameAnsDev);
    cudaFree(streetArrDev);
    cudaFree(streetAnsDev);
    cudaFree(balanceArrDev);
    cudaFree(balanceAnsDev);
    cudaFree(countDev);
    // We dont need to free host malloc, because OS will clean up after program ends


    cout << "Done computing." << endl;
    convertArrsToPeopleArr(CUDA_CORES, outPeople, nameAns, streetAns, balanceAns);
    cout << outPeople[0].InfoHeader();
    for(int i = 0; i < CUDA_CORES; i++)
    {
        cout << outPeople[i].GetStr();
    }


    cout << "Saving to file..." << endl;
    saveToFile(RESULT_FILE_NAME, outPeople, CUDA_CORES);
    return 0;
}

// Cuda doesnt have string functions like concat
__device__ void cudaConcat(char* dest, char*source)
{
    int dlen, slen;
    // Get string lenght
    for(dlen=0; dest[dlen]!='\0'; ++dlen); 

    // add chars til null terminator
    for(slen=0; source[slen]!='\0'; ++slen, ++dlen)
    {
       dest[dlen]=source[slen];
    }
}

__global__ void doStuff(    int *cnt, CudaString *n, int *s, double *b,
                            CudaString *nAns, int *sAns, double *bAns)
    {
    int thread_id = threadIdx.x;
    int blockdimx = blockDim.x;
    int work_size = *cnt / blockdimx;
    int rem = *cnt % blockdimx;
    printf("Count:%d\n", work_size);
    printf("Rem:%d\n", rem);
    
    for(int i = thread_id; i <  *cnt; i += blockdimx)
    {
        cudaConcat(nAns[thread_id].str, n[i].str);
        sAns[thread_id] += s[i];
        bAns[thread_id] += b[i];
    }
    printf("String:%s\n", n[thread_id].str);
}


void convertPeopleArrToArrs(Person people[], int cnt, CudaString *nameArr, int *streetArr, double *balanceArr)
{
    for(auto i = 0; i < cnt; i++)
    {
        nameArr[i].len = (people[i].Name.length());
        strcpy((nameArr[i].str), (people[i].Name.c_str()));
        streetArr[i] = people[i].StreetNum;
        balanceArr[i] = people[i].Balance;
    }
}



void convertArrsToPeopleArr(int cnt, Person people[], CudaString *nameArr, int *streetArr, double *balanceArr)
{
    for(int i = 0; i < cnt; i++)
    {
        string tmp(nameArr[i].str);
        people[i] = Person(tmp, streetArr[i], balanceArr[i]);
    }
}

int deserializeJsonFile(std::string fileName, Person arr[])
{
    // Read json file
    std::ifstream i(JSON_DATA_FILE);

    // Create json object
    json j;
    i >> j;

    // Deserialize json
    int count = 0;
    for (auto &x : j.items())
    {
        Person tmpPerson;
        tmpPerson.Name = x.value()["Name"].get<std::string>();
        tmpPerson.StreetNum = x.value()["StreetNum"].get<int>();
        tmpPerson.Balance = x.value()["Balance"].get<double>();
        arr[std::stoi(x.key())] = tmpPerson;
        count++;
    }

    return count;
}

// Saves people data structure to text file
void saveToFile(std::string fileName, Person outArr[], int outCnt)
{
    std::ofstream ofs(fileName);
    ofs << outArr[0].InfoHeader(); // print out table header
    for (auto i = 1; i < outCnt; i++)
    {
         ofs << outArr[i].GetStr();
    }

    ofs.close();
}