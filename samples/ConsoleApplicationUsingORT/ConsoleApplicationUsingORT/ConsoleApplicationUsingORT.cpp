// ConsoleApplicationUsingORT.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include "core/framework/tensor_shape2.h"
//#include "MathFunctions.h"

int main()
{
  std::vector<int64_t> dims{3, 4, 5};
//  onnxruntime::TensorShape2 tensor_shape_2(dims);
  //onnxruntime::TensorShape2 tensor_shape_2;
//  std::string str = tensor_shape_2.ToString();
//  std::cout << "Hello World! " << str << "\n ";
  std::cout<<"hello dims size="<<dims.size()<<"\n";
//  std::cout<<"sqrt of 16:"<<mathfunctions::sqrt(16.0)<<"\n";
  std::cout<<"cube:"<<onnxruntime::cube(3)<<"\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
