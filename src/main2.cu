#include "cuda_runtime.h"
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include "device_launch_parameters.h"
#include <random>
#include <algorithm>
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#define JSON_DATA_FILE "data/IFF7-4_ValinskisV_L1_dat_3.json"
#define RESULT_FILE_NAME "data/IFF7-4_ValinskisV_3_res.txt"

#define ENTRY_CNT_MAX 250
#define MAX_STRING_LEN 1024

using namespace std;
using json = nlohmann::json;
using namespace thrust;


struct sPerson
{
    char name[MAX_STRING_LEN];
    int streetNum;
    double balance;
};

typedef struct sPerson sPerson;


// Function prototypes
host_vector<sPerson> deserializeJsonFile(std::string fileName, int *count);
//void saveToFile(std::string fileName, Person outArr[], int outCnt);
void generate_random_array(int* array, size_t size);


// struct crumple {
//     __host__ sPerson operator ()(sPerson accumulator, sPerson item) {
//         int dlen, slen;
//         // Get string lenght
//         for(dlen=0; accumulator.name[dlen]!='\0'; ++dlen); 

//         // add chars til null terminator
//         for(slen=0; item.name[slen]!='\0'; ++slen, ++dlen)
//         {
//             accumulator.name[dlen] = item.name[slen];
//         }   
//         accumulator.streetNum = accumulator.streetNum + item.streetNum;
//         accumulator.balance = accumulator.balance + item.balance;
//         return accumulator;
//     }
// };

struct add_int
{
    __device__ int operator ()(int accumulator, int item) {
        return accumulator + item;
    }
};

int main() {   
    int peopleCount;

    cout << "Start L3 Thrust..." << endl;

    cout << "Reading parsing JSON file" << JSON_DATA_FILE << endl;
    host_vector<sPerson> people = deserializeJsonFile(JSON_DATA_FILE, &peopleCount);

    // Empty struct for initial accumulator
    sPerson temp;
    temp.name[0] = '\0';
    temp.streetNum = 0;
    temp.balance = 0.0;

    thrust::device_vector<int> deviceVec(peopleCount);
    printf("Count:%d\n", (int)deviceVec.size());


    // thrust::for_each(deviceVec.begin(), deviceVec.end(), [] (sPerson item) { cout << item.name << " \n";});
    // auto res = reduce(people.begin(), people.end(), temp, ;

    // string tmpname(res.name);
    // int tmpst = res.streetNum;
    // double tmpbal = res.balance;

    // Person resp = Person(tmpname,tmpst, tmpbal);
    // cout << resp.InfoHeader();
    // cout << resp.GetStr();

    // cout << "Saving to file..." << endl;

    // std::ofstream ofs(RESULT_FILE_NAME);
    // ofs << resp.InfoHeader(); // print out table header
    // ofs << resp.GetStr();
    // ofs.close();
    
    return 0;
}

host_vector<sPerson> deserializeJsonFile(std::string fileName, int *count)
{
    // Read json file
    std::ifstream i(JSON_DATA_FILE);
    host_vector<sPerson> tmp(ENTRY_CNT_MAX);

    // Create json object
    json j;
    i >> j;

    // Deserialize json
    *count = 0;
    for (auto &x : j.items())
    {
        sPerson tmpPerson;
        strcpy(tmpPerson.name, x.value()["Name"].get<std::string>().c_str());
        tmpPerson.streetNum = x.value()["StreetNum"].get<int>();
        tmpPerson.balance = x.value()["Balance"].get<double>();
        tmp.push_back(tmpPerson);
        (*count)++;
    }

    return tmp;
}

// Saves people data structure to text file
// void saveToFile(std::string fileName, Person outArr[], int outCnt)
// {
//     std::ofstream ofs(fileName);
//     ofs << outArr[0].InfoHeader(); // print out table header
//     for (auto i = 1; i < outCnt; i++)
//     {
//          ofs << outArr[i].GetStr();
//     }

//     ofs.close();
// }