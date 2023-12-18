* LP written by GAMS Convert at 12/18/23 13:11:21
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*         3        1        0        2        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*         5        5        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        13       13        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 L  e1
 L  e2
 E  e3
COLUMNS
    x1        e1                   4
    x1        e2                   1
    x1        e3                 -12
    x2        e1                   9
    x2        e2                   1
    x2        e3                 -20
    x3        e1                   7
    x3        e2                   3
    x3        e3                 -18
    x4        e1                  10
    x4        e2                  40
    x4        e3                 -40
    x5        obj                 -1
    x5        e3                   1
RHS
    rhs       e1                6000
    rhs       e2                4000
BOUNDS
 FR bnd       x5
ENDATA
