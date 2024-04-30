/* One automated test for determining the complexity of a piece of text is
## the Flesch-Kincaid grade level test, which assigns a number to a piece
## of text indicating what grade level the computer things is necessary to
## understand that text. This test makes no attempt to actually understand
## the meaning of the text, and instead focuses purely on the complexity of
## the sentences and words used within that text. Specifically, the test
## counts up the total number of words and sentences within the text, along
## with the total number of syllables within each of those words. Given these
## numbers, the grade level score is then computed using the following formula:

############### Grade = C0+C1 * (number of words / num sentences) + ###############
############### C2 * (number of syllables / number of words)       ###############

## Where C0, C1, and C2 are constants chosen as follows: C0 =-15.59, C1 =0.39, C2 =11.8

## In This program we calculate the Flesch-Kincaid grad level for a given textfile.

Author: Mihran Hakobyan
Date: 30.05.2024
*/




#include <iostream>
#include <fstream>
#include <string>
#include "tokenscanner.h"
#include <ctype.h>
using namespace std;

// this function calculates the number of syllables for a given word
int counter_syllables_word(string word) {

    // declaring needed variables
    int num_syllables = 0;
    bool previous_letter_vowel = false;

    // looping through the word to count all the syllables
    for (char c : word) {
        // check if letter of the word is a vowel
        if (c == 'a' || c == 'A' || c == 'e' || c == 'E'
                || c == 'i' || c == 'I' || c == 'o' || c == 'O'
            || c == 'u' || c == 'U' || c == 'y' || c == 'Y') {

            // check if previous letter was a vowel too and go to next iteration immedeately
            if (previous_letter_vowel == true) {
                continue;
            }
            // remembering that the letter before was a vowel and count the syllable number up by 1
            previous_letter_vowel = true;
            num_syllables += 1;
        }
        // if the letter of word is not a vowel, dont count up and remember that it wasn't
        else {
            previous_letter_vowel = false;
        }

    }
    //check if an "e" is by itself at the end of the word, and count down syllable number
    // since we counted up by one in the loop above
    if (word[word.length()-1] == 'e' || word[word.length()-1] == 'E') {
        num_syllables = num_syllables - 1;
    }
    // check if syllable number is zero or smaller (edge cases like "me" and assuming
    // all words have at least one syllable) and increase to one (no words without syllables)
    if (num_syllables <= 0) {
        num_syllables = 1;
    }
    //give out the final number of syllables
    return num_syllables;
}

// this function calculates the flesch Kincaid Grade Level for a given textfile
double flesch_kincaid_grade(string filename){

    // create an ifstream object
    ifstream file;


    //open the file
    file.open(filename);

    // check if successfully opened
    if (!file.is_open()){
        return 0;
    }
    // declaring an object named scanner of class TokenScanner,
    // and initializing it ifstream object file
    TokenScanner scanner(file);

    // ignore the whitespaces in the text
    scanner.ignoreWhitespace();

    // declaring needed variables and constants
    string previous_token;
    int counter_words = 0, counter_senten = 0, counter_syllables_total = 0;
    double FKgrade;
    const double C0 = -15.59, C1 = 0.39, C2 = 11.8;

    // go through all tokens in the text to calculate the flesch kincaid grade level
    while(scanner.hasMoreTokens()) {
        // declaring the next token of the file
        string token = scanner.nextToken();

        if (token == "'") {
            // tokenizing words with an apostroph in them to one token
            // Since we already counted first part of the apostroph containing word
            // ("isn" in "isn't"), we dont need to count the whole word ("isn't) anymore
            // and can proceed with the next iteration
            // notice we call .nextToken() function so that the next loop iteration
            // is not the last part "t" of the apostroph word, but the next coming token
            scanner.nextToken();
            continue;

        }
        // remembering the previous token for the words with an apostroph in them
        previous_token = token;

        // counting the number of words in the file by checking
        // if the first character is indeed a letter or not using isalpha
        if (isalpha(token[0]) != 0) {
            counter_words += 1;
            counter_syllables_total += counter_syllables_word(token);

        }
        // counting the number of sentences in the file using punctuations
        if (token == "." || token == "!"  || token == "?") {
            counter_senten += 1;

        }
    }
    //edge case for no words and no sentences in the file
    if(counter_words == 0 || counter_senten == 0){
        counter_words = 1;
        counter_senten = 1;
    }

    cout << "\nNOTE: " << endl;
    cout << "your file contains words: " << counter_words << endl;
    cout << "................. sentences: " << counter_senten << endl;
    cout << "................. syllables: " <<  counter_syllables_total << endl;

    // calculating the Grade Level of the text using the formula
    FKgrade = C0 + C1 * (static_cast<double>(counter_words) / counter_senten) +
              C2 * (static_cast<double>(counter_syllables_total) / counter_words);

    file.close();
    return FKgrade;
}

int main(){

    string filename;
    double FKgrade;
    cout << "Please enter the file name you want to calculate "
            "the Flesch-Kincaid Grade Level for. " << endl;
    // repromting the user to enter file name correctly
    while (true) {

        cin >> filename;
        FKgrade = flesch_kincaid_grade(filename);

        if (FKgrade == 0) {
            cout << "System failed to open the file. Please try again:\n";
        }
        else {
            cout << "\nSo the Flesch-Kincaid Grade-Level of your text is: " << FKgrade << endl;
        }
    }
    return 1;
}
