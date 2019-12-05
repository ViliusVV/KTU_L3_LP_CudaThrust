// A2DD.h
#ifndef PERSON_H
#define PERSON_H

#include <string>

class Person
{
    public:
        static int len;
        std::string Name;
        int StreetNum;
        double Balance;
        std::string HahsValue;

        Person();
        Person(std::string name,int streetnum, double balance);
        Person(std::string name,int streetnum, double balance, std::string hash);
        Person(std::string jsonStr);
        void Clone(Person person);
        std::string Serialize();
        Person Deserialize(std::string jsn);
        bool isNull();
        int longestName();
        std::string  GetStr();

        std::string InfoHeader();
        

        // friend std::ostream& operator<<(std::ostream& os, const Person& person);
        
};



// std::ostream& operator<<(std::ostream& os, const Person& person);


#endif