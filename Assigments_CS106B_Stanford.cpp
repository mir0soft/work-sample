/*

Assignments out of CS106B Spring 2013 Section Handout 2 and 3.

Link:
https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1136/



Author: Mihran Hakobyan
Date: 05.05.2024

*/






#include <iostream>
#include <string>
#include <fstream>
#include <ctype.h>
#include "tokenscanner.h"
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <numeric>
#include "lexicon.h"


using namespace std;

// 1. ASSIGNMENT CENSORSHIP
// function censors all letters of a string "remove" in a given string "text"

void censorString(string &text, string remove){

    for (char c : remove){

        for (size_t i = 0; i < text.length(); i++){

            if (text[i] == c){
                text.erase(i,1);
            }
        }
    }

}


// 2. ASSIGNMENT READ STATS OF AN EXAM
// function calculates the minimum maximum and the mean score of a file
// whereby the file contains the results of an exam

void readStats (string fname, int & min, int & max, double & mean){

    // create ifstream object with the name file
    ifstream(file);

    file.open(fname);

    // check if successfully opened
    if (!file.is_open()){

        cerr << "ERORR: file could not be opened or does not exist." << endl;
    }

    // declaring an object named scanner of class TokenScanner,
    // and initializing the ifstream object file
    TokenScanner scanner(file);
    vector<int> grades;

    // go through all tokens and extrack the numeric ones
    while(scanner.hasMoreTokens()) {

        // declaring the next token of the file and a vector for storing grades
        string token = scanner.nextToken();
        // check if the first char of the token is a digit and put it in
        // the vector "grades" if it is
        if(isdigit(token[0])) {
            cout << token << endl;
            if(stoi(token) < 0 || stoi(token) > 100) {
                cerr << "ERROR: scores are not in the range 0 to 100: " << endl;
                max = 0;
                min = 0;
                mean = 0;
                return;
            }
            else {
                grades.emplace_back(stoi(token));
            }
        }
    }

    float count = grades.size();
    max = *max_element(grades.begin(), grades.end());
    min = *min_element(grades.begin(), grades.end());
    mean = reduce(grades.begin(), grades.end()) / count;

    file.close();

}


// 3. ASSIGNMENT - CALCULATE NUMBER OF CANNON BALLS IN PYRAMID
// function calculates the number of cannonballs stored in a pyramid
// with a height n.

int numCannonballsInStack(int height_pyramid){
    // the pyramid of height 1 has 1 cannonball stored in it
    if (height_pyramid == 1){

        return 1;
    }
    else {
        // reduce the problem to <n * n> cannonballs  and (meaning plus)
        // <height of of pyramid is lower by one>> cannonballs
        return height_pyramid * height_pyramid +
               numCannonballsInStack(height_pyramid - 1);
    }
}

// 4. ASSIGNMENT - CALCULATE MOST XZIBIT WORD
// function returns for a given Lexicon the word with the biggest
// number of substrings (words of Lexicon) contained in it

string mostXzibitWord(Lexicon& words){

    // declaring needed variables and the lexicon
    int counter_xzibit = 0;
    string xzibit_word;

    for (const string& word : words){

        int counter = 0;

        for (string sub_word : words) {
            // checking if the sub_word is a substring of word
            cout << sub_word << endl;

            if (word.find(sub_word) != string::npos){
                counter ++;
            }
        }
        if (counter > counter_xzibit) {
            counter_xzibit = counter;
            cout << counter_xzibit << endl;
            xzibit_word = word;
        }
    }
    cout << counter_xzibit << endl;
    return xzibit_word;
}

// 5. ASSIGNMENT - FIND PROTEINS
// function return all the proteins contained in an RNA
// (chemical that can encode genetic information)

vector<string> findProteins(string& rna){

    vector<string> proteins;
    string start_codon = "AUG";
    size_t start_index = 0;
    size_t end_index = -3;

    while (true) {

        // searching for the index where we encouter start codon "AUG"
        // for the first time
        start_index = rna.find(start_codon, end_index + 3);

        // making sure we leave the loop if the string rna doesnt contain
        if (start_index == string::npos) {
            break;
        }

        // check which of the stop codons is the first in the rna
        // for that protein
        // set the end_index to the last index in rna so we can compare, means
        // choose the first of three stop codons after the start codon

        size_t smallest_index = rna.length() - 1;

        for (string stop_codon : {"UAA", "UAG", "UGA"}) {
            size_t stop_codon_index = rna.find(stop_codon, start_index + 3);

            if(stop_codon_index < smallest_index){
                smallest_index = stop_codon_index;
            }
        }
        end_index = smallest_index;

        // create the proteins via calculated substring
        string protein = rna.substr(start_index, end_index - start_index + 3);
        // put the proteins at the back of a vector
        proteins.emplace_back(protein);
    }
    return proteins;
}


// 6. ASSIGNMENT - WEIGHTS AND BALANCES

// The function isMeasurable determines whether it is possible to measure out the
// desired target amount with a given set of weights,
// which is stored in the vector weights. (via Reccursion)

bool isMeasurable(int target, vector<int>& weights){
    // calculating the sum of the weights
    auto sum_rest = reduce(weights.begin(), weights.end());
    // go through all the weights in the vector and "put them on the left side
    // of balance to see if the weight is measurable", meaning balance is balanced
    for (size_t weight : weights){

        if (target + weight == sum_rest - weight || target == sum_rest) {
            return true;
        }
    }
    // creating a new weights vector by erasing the smallest weight of "weights"
    // finding pointer of minimal element
    auto min_it = min_element(weights.begin(), weights.end());
    // converting it into an integral
    size_t min_weight_index = distance(weights.begin(), min_it);
    // extracting the minimal weight of "weights"
    int min_weight = weights[min_weight_index];
    // erasing it from the vector
    weights.erase(min_it);

    // create the new target of target and the smallest weight on the balance
    // this is the new left side of the balance, hence new target
    int new_target = target + min_weight;

    // check if the weights vector is empty, if yes the desired target is not measurable
    if (weights.size() >= 1){
        // call the function recursively on the newly formed target and the remaining weights
        return isMeasurable(new_target, weights);
    }
    else {
        return false;
    }
}


// CALLING THE FUNCTIONS ABOVE TO TEST THEM OUT

int main(){

    // ASSIGNMENT 1. - run

    string text;
    cin >> text;
    censorString(text, "fog");

    cout << text << endl;

    // ASSIGNMENT 2. - run

    int min, max;
    double mean;
    readStats("exam-scores.txt", min, max, mean);

    cout << "Minimum: " << min << endl;
    cout << "MAX: " << max << endl;
    cout << "Mean: " << mean << endl;


    // ASSIGNMENT 3. - run

    cout << numCannonballsInStack(4) << endl;


    // ASSIGNMENT 4. - run

    Lexicon english("test.txt");
    vector<string> words = {"pirates", "irate", "rates", "rate", "at", "tea", "pi", "pirate", "es", "rat", "rats", "pi", "ates", "rate"};

    cout << mostXzibitWord(english) << endl;


    // ASSIGNMENT 5. - run

    string rna = "GCAUGGAUUAAUAUGAGACGACUAAUAGGAUAGUUACAACCCUUAUGUCACCGCCUUGA";
    vector<string> p =  findProteins(rna);

    for (string protein : p){
        cout << protein << endl;
    }

    // ASSIGNMENT 6. - run
    vector<int> weights = {4,2,4,1};

    if(isMeasurable(1, weights)){
        cout << "measurable" << endl;
    }
    else {
        cout << "not measurable" << endl;
    }

    return 0;
}
