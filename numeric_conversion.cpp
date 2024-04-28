#include <iostream>
#include <typeinfo>
#include <cmath>
using namespace std;

// fuction to convert int to string
string integerToString(int n){

    int first_chunk = n / 10;
    int second_chunk = n % 10;

    string second_chunk_string = string() + char(second_chunk + '0');

    // this if statement handles the case if the number is made of one digit
    if (first_chunk == 0) {
        return second_chunk_string;
    }

    // this if else statement handles the case if the number is made of two digits
    else if (first_chunk < 10) {

        // converting the digit between 1 and 9 to a string
        string first_chunk_string = string() + char(first_chunk + '0');
        // concatenating the first digit with the second digit of the number (both now of length 1)
        return first_chunk_string.append(second_chunk_string);
    }
    // this else statement handles the case if the number is made of more than two digits
    else {
        // concatenating the last digit of the number (already a string)
        // with the rest of the number by calling our function "integerToString" on
        // the remaining first part (recursively)
        return integerToString(first_chunk).append(second_chunk_string);
    }

}

// fuction to convert string to int
int stringToInteger(string str){


    int length_of_str = str.length();

    int first_digit = str[0]-'0';

    string remaining_str = str.substr(1);


    if (length_of_str == 1) { return first_digit; }

    else if (length_of_str <= 2) {

        int last_digit = str[1] -'0';

        return first_digit * 10 + last_digit;
    }

    else {

        int multiplicator = pow(10, length_of_str-1);

        return first_digit * multiplicator + stringToInteger(remaining_str);
    }

}


int main(){

    int n=7;

    string p = integerToString(170);

    cout << p << endl;


    int s = stringToInteger("43529");

    cout << s << endl;

    return 0;
}
