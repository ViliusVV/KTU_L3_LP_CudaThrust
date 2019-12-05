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
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include "person.hpp"
#include<stdio.h>
#include<string.h>

#define JSON_DATA_FILE "data/IFF7-4_ValinskisV_L1_dat_3.json"
#define RESULT_FILE_NAME "data/IFF7-4_ValinskisV_3_res.txt"

#define ENTRY_CNT_MAX 128
#define MAX_STRING_LEN 512

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


struct crumple {
    __host__ __device__ sPerson operator ()(sPerson accumulator, sPerson item) {
        int dlen, slen;
        // Get string lenght
        for(dlen=0; accumulator.name[dlen]!='\0'; ++dlen); 

        // add chars til null terminator
        for(slen=0; item.name[slen]!='\0'; ++slen, ++dlen)
        {
            accumulator.name[dlen] = item.name[slen];
        }   
        accumulator.streetNum = accumulator.streetNum + item.streetNum;
        accumulator.balance = accumulator.balance + item.balance;
        return accumulator;
    }
};


int main() {   
    int peopleCount;

    cout << "Start L3 Thrust..." << endl;

    cout << "Reading parsing JSON file" << JSON_DATA_FILE << endl;
    host_vector<sPerson> people = deserializeJsonFile(JSON_DATA_FILE, &peopleCount);
    cout << "Start L3 Thrust..." << endl;
    // thrust::copy(people.begin(), people.end(), peopleHost.begin());

    thrust::host_vector<sPerson> peopleHost(peopleCount);
    thrust::copy(people.end()-peopleCount, people.end(), peopleHost.begin());

    device_vector<sPerson> peopleDevice = peopleHost;

    thrust::for_each(peopleHost.begin(), peopleHost.end(), [] (sPerson item) { cout << item.name << " \n";});
    printf("Count:%d\n", (int)peopleHost.size());

    // Empty struct for initial accumulator
    sPerson temp;
    memset(temp.name,'\0', MAX_STRING_LEN); // We need null terminators to detect end of string;
    temp.streetNum = 0;
    temp.balance = 0.0;



    // Merge everything into one person
    auto res = reduce(peopleDevice.begin(), peopleDevice.end(), temp, crumple());

    // Convert to person object
    string tmpname(res.name);
    int tmpst = res.streetNum;
    double tmpbal = res.balance;
    Person resp = Person(tmpname,tmpst, tmpbal);
    cout << resp.InfoHeader();
    cout << resp.GetStr();

    cout << "Saving to file..." << endl;

    std::ofstream ofs(RESULT_FILE_NAME);
    ofs << resp.InfoHeader(); // print out table header
    ofs << resp.GetStr();
    ofs.close();
    
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
