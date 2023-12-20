* LP written by GAMS Convert at 12/18/23 12:02:55
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        14       14        0        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        22       22        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        58       58        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 E  e1
 E  e2
 E  e3
 E  e4
 E  e5
 E  e6
 E  e7
 E  e8
 E  e9
 E  e10
 E  e11
 E  e12
 E  e13
 E  e14
COLUMNS
    x1        e1                0.97
    x1        e14                 65
    x2        e1                0.97
    x2        e14                 65
    x3        e1                  -1
    x3        e2                   1
    x3        e14                 58
    x4        e1                  -1
    x4        e3                   1
    x4        e14                 73
    x5        e1                  -1
    x5        e4                   1
    x5        e14                 56
    x6        e1                  -1
    x6        e5                   1
    x6        e14                 71
    x7        e2                -0.6
    x7        e4                -0.4
    x7        e6                  -1
    x8        e3                -0.3
    x8        e5                -0.7
    x8        e7                  -1
    x9        e6                   1
    x9        e8                   1
    x9        e14                 40
    x10       e6                   1
    x10       e9                   1
    x10       e14                 60
    x11       e6                   1
    x11       e10                  1
    x11       e14                 70
    x12       e7                   1
    x12       e11                  1
    x12       e14                 55
    x13       e7                   1
    x13       e12                  1
    x13       e14                 50
    x14       e7                   1
    x14       e13                  1
    x14       e14                 45
    x15       e8               -0.68
    x15       e11              -0.32
    x15       e14               -265
    x16       e9               -0.45
    x16       e12              -0.55
    x16       e14               -275
    x17       e10              -0.25
    x17       e13              -0.75
    x17       e14               -310
    x18       e6                   1
    x18       e14               -120
    x19       e7                   1
    x19       e14               -150
    x20       e6                  -1
    x20       e14                120
    x21       e7                  -1
    x21       e14                150
    x22       obj                 -1
    x22       e14                  1
RHS
BOUNDS
 LO bnd       x15                 18
 UP bnd       x15                 25
 LO bnd       x16                 12
 UP bnd       x16                 15
 UP bnd       x17                  7
 UP bnd       x18                  6
 UP bnd       x19                 10
 UP bnd       x20                  6
 UP bnd       x21                 10
 FR bnd       x22
ENDATA
