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
COLUMNS
    x1        e1                   1
    x1        e2                  -1
    x1    obj    1.0
    x2        e2                   1
    x2        e3                  -1
    x2    obj    1.0
    x3        e3                   1
    x3        e4                  -1
    x3    obj    1.0
    x4        e4                   1
    x4    obj    1.0
    x5        e1                   1
    x5    obj    -10.0
    x6        e2                   1
    x6    obj    -12.0
    x7        e3                   1
    x7    obj    -8.0
    x8        e4                   1
    x8    obj    -9.0
    x9        e1                  -1
    x9    obj    10.0
    x10       e2                  -1
    x10    obj    12.0
    x11       e3                  -1
    x11    obj    8.0
    x12       e4                  -1
    x12    obj    9.0
RHS
    rhs       e1                  50
BOUNDS
 UP bnd       x1                 100
 UP bnd       x2                 100
 UP bnd       x3                 100
 UP bnd       x4                 100
 FR bnd       x13
ENDATA
