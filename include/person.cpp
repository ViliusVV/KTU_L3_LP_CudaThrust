#include "person.hpp"
#include <stdio.h>
#include <string.h>
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;


int Person::len = 0;
// Constructors
Person::Person(){
    Name = "";
    StreetNum = 0;
    Balance = 0.0;
    HahsValue = "";
}


Person::Person(std::string name,int streetnum, double balance)
{
  Name = name;
  StreetNum = streetnum;
  Balance = balance;
  HahsValue = "";
}


Person::Person(std::string name,int streetnum, double balance, std::string hash)
{
  Name = name;
  StreetNum = streetnum;
  Balance = balance;
  HahsValue = hash;
}

Person::Person(std::string jsonStr)
{
  Person tmp = Deserialize(jsonStr);
  Clone(tmp);
}


// Serializes this object to json
std::string Person::Serialize()
{
  json jsn;

  jsn["Name"] = Name;
  jsn["StreetNum"] = StreetNum;
  jsn["Balance"] = Balance;
  jsn["HashValue"] = HahsValue;

  return jsn.dump();
}


// Deserializes object form json 
Person Person::Deserialize(std::string jsn)
{
  json ds = json::parse(jsn);
  return Person(ds["Name"], ds["StreetNum"].get<int>(), ds["Balance"].get<double>(), ds["HashValue"]);
}


// Check if object is "empty"
bool Person::isNull()
{
    return Name == "";
}


// Clones person object by value
void Person::Clone(Person person)
{
  Name = person.Name;
  StreetNum = person.StreetNum;
  Balance = person.Balance;
  HahsValue = person.HahsValue;
}


// Table header for pretty printing
std::string Person::InfoHeader()
{
    char buff[1024];
    len = Name.length()+25;
    snprintf(buff,1024, "\n| %-*s | %15s | %15s |\n", len, "Name", "Street Number", "Balance");
    std::string strBuff(buff);
    strBuff = strBuff + std::string(strlen(buff), '-') + "\n";
    return strBuff ;
}

std::string Person::GetStr()
{
    char buff[1024];
    snprintf(buff, 1024, "| %-*s | %15d | %15f |\n", len, Name.c_str(), StreetNum, Balance);
    std::string strBuff(buff);
    return strBuff;
}


// // Output stream operator overload
// std::ostream& operator<<(std::ostream& os, const Person& person)
// {
//     char buff[512];
//     snprintf(buff,512, "| %-*s | %-13d | %-10.2f | %s |\n", person.Name.c_str(), person.StreetNum, person.Balance, person.HahsValue.c_str());
//     std::string strBuff(buff);
//     os << strBuff;
//     return os;
// }