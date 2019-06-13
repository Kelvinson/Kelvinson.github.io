---
title: 'class revist(Chap4 of A Tour of C++)'
date: 2019-06-11
permalink: /algos/2019/06/class-revist/
categories: 
  - 'cpp'
  - 'programming'
# collections: algos
---

**Credit** A Tour of C++, Bjarne Stroustrup

An example of complex class:
```C++
class complex {
        double re,im; //representation: two doubles;
    public:
        complex(double r,  double i): re{r}, im{i} {};
        complex(double r): re{r}, im{0} {};
        complex(): re{0}, im{0} {}

        double real() const {return re};
        double im() const {return im};
        void real(double real) {re = real};
        void im(double imag) {im = imag};
        
        complex& operator+=(complex z) {re+=z.re, im+=z.im; return *this;};
        complex& operator-=(complex z) {re-=z.re, im-=z.im; return *this;};
        complex& operator*=(complex z);
        complex& operator/=(complex z);
}

complex operator+(complex a, complex b) {return a+=b;};
complex operator-(complex a, complex b) {return a-=b;};
complex operator-(complex a) {return {-a.real(), -a.im()}};
complex operator*(complex a, complex b) {return a * b;};
complex operator*(complex a, complex b) {return a / b;}

bool operator==(complex a, complex b)
{
    return a.real()==b.real() && a.im() == b.im();
}

bool operator != (complex a, complex b) 
{
    return !(a==b);
}
```

Vector class:
```C++
class Vector{
    private:
    double * elem;
    int sz;
    public:
    //constructor, acquire resource from os
    Vector(int s):elem{new double[s]}, sz{s}
    {
        for (int i = 0;i < s; i++) 
            elem[i] = 0;
    }
    ~Vector() {delete[] elem;}

    double & operator[](int  i);
    int size() const;

    //initializer function declaration;
    Vector(std::initializer_list<double>);
    void push_back(double);
}
```
the initialization function can be as follows:
```C++
Vector::Vector(std::initializer_list<double> lst) 
      :elem{new double[lst.size()]}, sz{static_cast<int>(lst.size())}\
      {
          copy(lst.begin(), lst.end(), elem);
      }
```

std::initializer_list can be created when we use the curly brace "{}" which C++ recommmends us to use to initialize anything, it has **begin**, **end**, **size** member functions.

we can add copy and move functions to it 
```C++
class Vector{
    private:
    double * elem;
    int sz;
    public:
    //other functions as in last code excerpt
    Vector(const Vector&a); //copy constructor
    Vector& operator=(const Vector&a); //copy assignment

    Vector(Vector && a); //move constructor
    Vector& operator=(Vector &&a); //move assignment
}

//implementation details:
Vector::Vector(const Vector& a) 
      :elem{new double[a.sz]},
      sz{a.sz}
{
    for (int i =0; i<sz; ++i) \
    elem[i] = a[i];
}

Vector::Vector& operator=(const Vector& a) 
{
    double * p = new double[a.sz];
    for (int i = 0; i != a.sz; i++)\
        p[i] = a.elemp[i];
    
    delete[] elem;
    elem = p;
    sz = a.sz;
    return *this; //the object for which this function is called;
}

Vector::Vector(Vector&& a)
      :elem{a.elem}, //move from 'a'
      sz{a.sz}
{
    a.elem = nullptr; //delete original 'a'
    a.sz = 0;
}

```