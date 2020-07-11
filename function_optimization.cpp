#include <iostream>
using namespace std;

float func(float x1, float x2){
	return x1*x1 + x2*x2 + 4*x1 + 5*x2 + 10;
}

float grad_x1(float v){
    return 2*v + 4;
}

float grad_x2(float v){
    return 2*v + 5;
}

int main(){
    float x1 = 10, x2 = 20;
    float lr = 0.1;
    float threshold = 0.001;
    
    float preVal, newVal;
    int maxIter = 1000, iter = 0;
    
    preVal = newVal = func(x1, x2);
    preVal+= 2;
    
    while(maxIter > iter && preVal-newVal > threshold){
        preVal = newVal;
        x1 = x1 - lr * grad_x1(x1);
        x2 = x2 - lr * grad_x1(x2);
        newVal = func(x1, x2);
        iter++;
    }
    cout << x1 << ", " << x2 << endl;
    cout << func(x1,x2) << ", " << iter;
}