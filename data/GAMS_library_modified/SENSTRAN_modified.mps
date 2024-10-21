* LP written by GAMS Convert at 12/20/23 12:16:22
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*         6        1        3        2        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*         7        7        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        19       19        0

NAME          Convert
*
* original model was minimizing
*
ROWS
 N  obj
 E  e1
 L  e2
 L  e3
 G  e5
 G  e6
COLUMNS
    x1        e2                   1
    x2    obj    0.153
    x2        e2                   1
    x2        e5                   1
    x3    obj    0.162
    x3        e2                   1
    x3        e6                   1
    x4    obj    0.225
    x4        e3                   1
    x4    obj    -1.0
    x5    obj    0.162
    x5        e3                   1
    x5        e5                   1
    x6    obj    0.1638
    x6        e3                   1
    x6        e6                   1
RHS
    rhs       e2                 350
    rhs       e3                 600
    rhs    obj    325
    rhs       e5                 300
    rhs       e6                 275
BOUNDS
 FR bnd       x7
ENDATA
