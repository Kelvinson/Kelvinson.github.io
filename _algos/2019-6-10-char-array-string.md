---
title: 'char arrays vs string vs char pointer and more'
date: 2019-06-10
permalink: /algos/2019/06/chars-arrays-string/
categories: 
  - 'cpp'
  - 'programming'
# collections: algos
---

***Note*** Thanks for Geeks4Geeks, I referred to their posts [char * vs string vs char[]](https://www.geeksforgeeks.org/char-vs-stdstring-vs-char-c/) and [storage for strings in C](https://www.geeksforgeeks.org/storage-for-strings-in-c/) and [here](http://Kelvinson.github.io/files/char-vs-string.pdf) is my very messy note.

**char[]** and **string literal** 
======
In short, **char[]** and **string literal** are the same. They are interchangable. **std::string** class is an encapsulation(or interface or wrapper) of **char arrays** and saves us the effort to manage the array size and provide other convinient functions. Noticeably this is related to C++'s SBRM(Scope based resource manageemnt) aka RAII(resource acquisition  is initialization) paradigm. For more detail refer to blogs above.g



**char[]** vs <em>char*</em> 
======
However, <em>char*</em> is another thing, it is a pointer to the **char[]**, the pointer can be changed to pointer another **char[]** but it cannot change the element in the array(though it can if the char[] is dynamically allocated and  stored on stack using **malloc**), this is because When a string value is directly assigned to a pointer, in most of the compilers, it’s stored in a read-only block (generally in data segment) that is shared among functions. And we should enforce the pointed string to **const** to suppress the warning. 

![](/images/char1.png)

**Credit: https://www.geeksforgeeks.org/whats-difference-between-char-s-and-char-s-in-c/**


Example:
```C++
char hello[] = "hello";
char hello1[] = {'h', 'e', 'l', 'l', 'o', '\0'};//now hello and hello1 are the same, both with size of 6(don't forget determinator '\0')
hello1[0] = 'H'; //right, we can change one char element like other type of arrays.

char* hello2 = "hello" // or ={'h', 'e', 'l', 'l', 'o', '\0'} 
*(hello2 +1) = 'H' //wrong, "hello" is in read-only data-segment, we cannot change
//also we are recommended to write as:
const char* hello2 = "hello";

//but if create dynamically allocate the string, we can change the element;
char* hello3 =  (char*)malloc(sizeof(char) * 6)
*(hello3) = 'h';
*(hello3 + 1) = 'e';
*(hello3 + 2) = 'l';
*(hello3 + 3) = 'l';
*(hello3 + 4) = 'o';
*(hello3 + 5) = '\0';

*(hello3) = 'H'; //right, because the content hello3 points to lies on stack like other variables

```

**std::string** vs **char[]**
======
**char[]** can convert to std::string implicitly, std::string can convert to **char[]** with **.c_str()** or **.data()**
```C++
char hello[] = "hello";
string str_hello = hello; //right
auto hello1 = str_hello.c_string(); //right
char* hello2 = str_hello.data(); //right
```
