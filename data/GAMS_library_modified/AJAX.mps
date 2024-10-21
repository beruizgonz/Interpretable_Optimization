* LP written by GAMS Convert at 12/16/23 01:04:30
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*         8        5        0        3        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        13       13        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        37       37        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 L  e1
 L  e2
 L  e3
 E  e4
 E  e5
 E  e6
 E  e7
COLUMNS
    x1        e1        0.0188679245
    x1        e4                   1
    x1    obj    76.0
    x2        e2        0.0192307692
    x2        e4                   1
    x2    obj    75.0
    x3        e3        0.0204081633
    x3        e4                   1
    x3    obj    73.0
    x4        e1        0.0196078431
    x4        e5                   1
    x4    obj    82.0
    x5        e2        0.0204081633
    x5        e5                   1
    x5    obj    80.0
    x6        e3        0.0227272727
    x6        e5                   1
    x6    obj    78.0
    x7        e1        0.0192307692
    x7        e6                   1
    x7    obj    96.0
    x8        e2        0.0222222222
    x8        e6                   1
    x8    obj    95.0
    x9        e3        0.0212765957
    x9        e6                   1
    x9    obj    92.0
    x10       e1        0.0238095238
    x10       e7                   1
    x10    obj    72.0
    x11       e2        0.0227272727
    x11       e7                   1
    x11    obj    71.0
    x12       e3               0.025
    x12       e7                   1
    x12    obj    70.0
RHS
    rhs       e1                 672
    rhs       e2                 600
    rhs       e3                 480
    rhs       e4               30000
    rhs       e5               20000
    rhs       e6               12000
    rhs       e7                8000
    rhs    obj    5958000
BOUNDS
 FR bnd       x13
ENDATA
