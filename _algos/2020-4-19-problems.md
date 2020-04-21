---
title: '4/19 Problems'
date: 2020-4-19
permalink: /algos/2020/04/algo/
categories: 
  - 'algorithm'
  - 'programming'
# collections: algos
---
1420. Build Array Where You Can Find The Maximum Exactly K Comparisons
find how many combs there exist so that 1)n ints 2)every int is leq m. 3)search cost is K(search cost will increase by 1 if current number is larger than maximum)

two ways to solve it:

1) dp: dp[i][j][k] is the num of ways when i int array with the maximum in the array is j and cost is k. dp[i][j][k] can be transited by a)when there is already i-1 ints and maximum is j with cost k, then we can append any number in [1, j] the cost is still k with i int and maximum j, j * dp[i-1][j][k]; b)when there is i-1 ints but the maximum is j-1 and cost is k-1 then we can only append a j, i.e. 1 * dp[i-1][j-1][k-1]. so the total transition equation is:  j * dp[i-1][j][k] + 1 * dp[i-1][j-1][k-1]. 

java code:
```java
    public int numOfArrays(int n, int m, int s) {
       int MOD = 1_000_000_007;
       long[][][] dp = new long[n+1][m+1][s+1];
       //dp[0][0][0] = 1; 
       for (int j = 1; j <= m; j++) {
           dp[1][j][1]  = 1;
       }
       for (int i = 1; i <= n; i++) {
           for (int j = 1; j <= m; j++) {
               for (int k = 1; k <= s; k++) {
                   long t = (dp[i-1][j][k] * j) % MOD;
                   for (int l = 1; l <= j-1; l++) {
                       t = (t +  dp[i-1][l][k-1]  * 1 ) % MOD;
                   }
                   dp[i][j][k] = Math.max(dp[i][j][k], t);
               }
           }
       }
       long ans = 0; 
       for (int j = 1; j <= m; j++) {
           ans = (ans + dp[n][j][s])%MOD;
       }
       return (int)ans;
    }
```
Two points to notice are: 1) initialization of dp[0][0][0] to 1 is unneccessary because we don't need it in the transition although dp[0][0][0] is definitely should be 1 according to the state definition.
2)dp[1][j][1] shoul be initialized to 1 when j in [1, m], else the answer will always be  0.
3) after transition and update, ***dp[i][j][k] =  Math.max(dp[i][j][k], t)*** but not ***dp[i][j][k] = 0***;

2)dfs + memo: follwing a top-down approach, we can find that in order to find the patterns for n ints, we need that same res for (n-1) at first, then it's a recursion problem, with memorization we can solve it as well.

dfs(i, j, k) is the ways when num of ints is i, cuMax is j, cur cost is k. its result can be recursed to the case a)dfs(i+1, j, k) when a new number of [1, j] is added; b)dfs(i+1, j, k+1) when a new number of [j+1, m] is added. Actually it's quite similar to the dp.

```java
class Solution {
   int MOD = 1_000_000_007;
    Integer[][][] dp;
    public int numOfArrays(int n, int m, int s) {
        dp = new Integer[n+1][m+1][s+1];
        return  dfs(n, m, s, 0, 0, 0);
    }
    
    private int dfs(int n, int m, int s, int i, int j, int k) {
        if (i == n) {
            if (k == s) return 1;
            return 0;
        }
        if (dp[i][j][k] != null) return dp[i][j][k];
        if (j > m || k > s) return 0;
        int ans = 0;
        ans +=  (long)j * dfs(n, m, s, i+1, j, k) % MOD;
        if (k + 1  <= s) {
            for (int jj = j + 1; jj <= m; ++jj) {
               ans = (ans + dfs(n, m, s, i+1, jj, k+1)) % MOD; 
            }
        }
        dp[i][j][k] = ans;
        return ans;
    }
}
```




1141.    Number of Ways to Paint N × 3 Grid

It's a classical dp problem(according to lee215?). Each new row is only dependent on prev row. 
For the first row, there can be the following 12 combs: 

121: 121, 131, 212, 232, 313, 323

123: 123, 132, 213, 231, 312, 321

then for each new row, the 121 case can be converted as following ways:

each 121 -> 232, 212, 313, 213, 312

each 123 -> 212, 232, 231, 312, 

so we can  find there are 5 cases where 121 can be achieved: 3 from prev 121, 2 from prev 123.
Similarily, there are 4 cases where 123 can be achieved: 2 from prev 121, 2 from prev 123.
So the transition equation is new121 = 3 * prev121 + 2 * prev123, new123 = 2 * prev121 + 2 * prev123; 

Java code:
```java
class Solution {
    public int numOfWays(int n) {
        long MOD = 1_000_000_007;
        long a121 = 6, b123 = 6; 
        for (int i = 2; i <= n; i++) {
            long t121 = (3 * a121 + 2 * b123) % MOD;
            long t123 = (2* a121 + 2 * b123) %  MOD;
            a121 = t121;
            b123 = t123;
        }
        return (int)((a121 + b123) % MOD);
    }
}
```

1416. Restore The Array

```java
class Solution {
    
    public int numberOfArrays(String s, int k) {
        long MOD = 1_000_000_007;
        if(s == null || s.length() == 0) {
            return 0;
        }
        int n = s.length();
        long[] dp = new long[n+1];
        int lenK = 0;
        int kk = k;
        while (kk != 0) {
            lenK += 1;
            kk /= 10;
        }
        char[] arr = s.toCharArray();
        dp[0] = 1;
        //dp[1] = Integer.valueOf(s.substring(0,1)) <= k ? 1 : 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= lenK && i-j+1 > 0; j++) {
                if (s.charAt(i-1-j+1) == '0') continue;
                long tmp = Long.valueOf(s.substring(i-1-j+1, i-1+1));
                dp[i] = (dp[i] +  ((tmp > 0 && tmp <= k) ? dp[i-1-j+1] : 0)) % MOD;
            }
            //System.out.println(i + " : " + dp[i]);
        }
        return (int)dp[n];
    }
}
```

Acwing 338 count digits appearing in interval[a, b]

DP solution:



Another solution not to use dp is to group the position where the current digit i can appear,say we want to find how many x appears in [1, N] where N is $abcdefg$.

To start, we want to find how many x can appear on the $4th$ positio('d'), according to value of first 3 digits $000-abc-1$: last three digits can be $000-999$, count is abc * 1000
$abc$: 
        1) x < d:  last three digits can be $000-999$, count is 1000
        2) x == d:  last three digits can be $000-efg$, count is $efg+1$
        3) x > d: no valid digits can be appended, count is 0;

```C++
#include<iostream>
#include<algorithm>
#include<vector>
using namespace std;

int get(vector<int>num, int l, int r) {
    int res = 0;
    for (int i = l; i >= r; i--) {
        res = res * 10 + num[i];
    }
    return res;
}
int power10(int x) {
    int res = 1;
    while (x--) res *= 10;
    return res;
}
int count(int n, int x) {
    if (n == 0) return 0;
    vector<int> num;
    while (n) {
        num.push_back(n % 10);
        n /= 10;
    }
    //123 -> <3,2,1> 
    n = num.size();
    int res = 0;
    for (int i = n-1-!x; i >= 0; i--) {
        if (i < n-1) {
           res += get(num, n-1, i+1) * power10(i); 
           if (!x) res -= power10(i);
        }
        if (num[i] == x) {
            res += get(num, i-1, 0) + 1;
        } else if (num[i] > x) res += power10(i); 
    }
    return res;
}
int main() 
{
    int a, b;
    while (cin>>a>>b, a || b) {
        if (a > b) swap(a, b);
        for (int i = 0; i < 10; i++) {
            cout<<count(b, i) - count(a - 1, i) << ' ';
        }
        cout<<endl;
    }
}
```