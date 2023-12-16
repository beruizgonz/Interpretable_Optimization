* LP written by GAMS Convert at 12/16/23 01:11:03
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*         6        1        3        2        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        10       10        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        25       25        0

NAME          Convert
*
* original model was minimizing
*
ROWS
 N  obj
 E  e1
 L  e2
 L  e3
 G  e4
 G  e5
 G  e6
COLUMNS
    x1        e1              -0.225
    x1        e2                   1
    x1        e4                   1
    x2        e1              -0.153
    x2        e2                   1
    x2        e5                   1
    x3        e1              -0.162
    x3        e2                   1
    x3        e6                   1
    x4        e1              -0.225
    x4        e3                   1
    x4        e4                   1
    x5        e1              -0.162
    x5        e3                   1
    x5        e5                   1
    x6        e1              -0.126
    x6        e3                   1
    x6        e6                   1
    x7        e1               -1000
    x7        e4                  -1
    x8        e1               -1000
    x8        e5                  -1
    x9        e1               -1000
    x9        e6                  -1
    x10       obj                  1
    x10       e1                   1
RHS
    rhs       e2                 350
    rhs       e3                 600
    rhs       e4                 325
    rhs       e5                 300
    rhs       e6                 275
BOUNDS
 FR bnd       x10
ENDATA
