* LP written by GAMS Convert at 12/20/23 13:12:18
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*         5        5        0        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        13       13        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        28       28        0

NAME          Convert
*
* original model was minimizing
*
ROWS
 N  obj
 E  e1
 E  e2
 E  e3
 E  e4
 E  e5
COLUMNS
    x1        e1                   1
    x1        e2                  -1
    x1        e5                  -1
    x2        e2                   1
    x2        e3                  -1
    x2        e5                  -1
    x3        e3                   1
    x3        e4                  -1
    x3        e5                  -1
    x4        e4                   1
    x4        e5                  -1
    x5        e1                   1
    x5        e5                  10
    x6        e2                   1
    x6        e5                  12
    x7        e3                   1
    x7        e5                   8
    x8        e4                   1
    x8        e5                   9
    x9        e1                  -1
    x9        e5                 -10
    x10       e2                  -1
    x10       e5                 -12
    x11       e3                  -1
    x11       e5                  -8
    x12       e4                  -1
    x12       e5                  -9
    x13       obj                  1
    x13       e5                   1
RHS
    rhs       e1                  50
BOUNDS
 UP bnd       x1                 100
 UP bnd       x2                 100
 UP bnd       x3                 100
 UP bnd       x4                 100
 FR bnd       x13
ENDATA
