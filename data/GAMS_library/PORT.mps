* LP written by GAMS Convert at 12/18/23 13:10:13
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*         5        2        1        2        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*         7        7        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        27       27        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 G  e1
 L  e2
 L  e3
 E  e4
 E  e5
COLUMNS
    x1        e2                   2
    x1        e3                   9
    x1        e4              -0.043
    x1        e5                  -1
    x2        e2                   5
    x2        e3                   2
    x2        e4              -0.045
    x2        e5                  -1
    x3        e1                   1
    x3        e2                   2
    x3        e3                  15
    x3        e4              -0.027
    x3        e5                  -1
    x4        e1                   1
    x4        e2                   1
    x4        e3                   4
    x4        e4              -0.025
    x4        e5                  -1
    x5        e1                   1
    x5        e2                   1
    x5        e3                   3
    x5        e4              -0.022
    x5        e5                  -1
    x6        e2                -1.4
    x6        e3                  -5
    x6        e5                   1
    x7        obj                 -1
    x7        e4                   1
RHS
    rhs       e1                   4
BOUNDS
 MI bnd       x6
 UP bnd       x6                  10
 FR bnd       x7
ENDATA
