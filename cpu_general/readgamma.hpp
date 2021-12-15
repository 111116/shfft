#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "sh.hpp"

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

static std::vector<TensorEntry> readGamma_ascii(int n)
{
    std::vector<TensorEntry> sparsegamma;
    std::string line;
    std::ifstream sparsefile("../gamma/sparse" + std::to_string(n));
    TensorEntry entry;
    while(getline(sparsefile, line))
    {
        std::vector<std::string> tokens = split(line.substr(1, line.length() - 3), ',');
        entry.a = std::stoi(tokens[0]);
        entry.b = std::stoi(tokens[1]);
        entry.c = std::stoi(tokens[2]);
        entry.val = std::stof(tokens[3]);
        sparsegamma.push_back(entry);
    }
    sparsefile.close();
    return sparsegamma;
}

std::vector<int> gammalist = {22,33,44,55,66,77};

static std::vector<TensorEntry> readGamma(int n)
{
    int filen = -1;
    for (int a: gammalist)
        if (a>=n) {
            filen = a;
            break;
        }
    std::vector<TensorEntry> sparsegamma;
    std::ifstream sparsefile("../gamma_bin/" + std::to_string(filen) + ".bin");
    if (!sparsefile) throw "file not found";
    struct {int a,b,c;float k;} t;
    while (sparsefile.read((char*)(&t), 16)) {
        if (t.a<n*n && t.b<n*n && t.c<n*n)
            sparsegamma.push_back((TensorEntry){(short)t.a, (short)t.b, (short)t.c, t.k});
    }
    sparsefile.close();
    return sparsegamma;
}
