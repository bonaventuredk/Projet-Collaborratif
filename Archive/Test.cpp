#include <iostream>
#include <string.h>


using namespace std;


int n = getARGV("-n", 5);          // default = 5
string name = getARGV("-name", "default");

cout<< "n = " << n << endl;
cout<< "name = " << name << endl;