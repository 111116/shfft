#pragma once

#include <vector>
#include <string>
#include <sstream>

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

static std::vector<TensorEntry> readGamma(int n)
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