* LP written by GAMS Convert at 12/16/23 01:08:53
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        13        9        0        4        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        23       23        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        75       75        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 L  e1
 L  e2
 L  e3
 L  e4
 E  e5
 E  e6
 E  e7
 E  e8
 E  e9
 E  e10
 E  e11
 E  e12
 E  e13
COLUMNS
    x1        e1                   1
    x1        e5                0.79
    x1        e9                0.21
    x1        e13              -1.73
    x2        e2                   1
    x2        e6                0.79
    x2        e10               0.21
    x2        e13               -1.8
    x3        e3                   1
    x3        e7                0.79
    x3        e11               0.21
    x3        e13               -1.6
    x4        e4                   1
    x4        e8                0.79
    x4        e12               0.21
    x4        e13               -2.2
    x5        e1                   1
    x5        e5                0.83
    x5        e9                0.17
    x5        e13              -1.82
    x6        e2                   1
    x6        e6                0.83
    x6        e10               0.17
    x6        e13               -1.9
    x7        e3                   1
    x7        e7                0.83
    x7        e11               0.17
    x7        e13               -1.7
    x8        e4                   1
    x8        e8                0.83
    x8        e12               0.17
    x8        e13              -0.95
    x9        e1                   1
    x9        e5                0.92
    x9        e9                0.08
    x9        e13              -1.05
    x10       e2                   1
    x10       e6                0.92
    x10       e10               0.08
    x10       e13               -1.1
    x11       e3                   1
    x11       e7                0.92
    x11       e11               0.08
    x11       e13              -0.95
    x12       e4                   1
    x12       e8                0.92
    x12       e12               0.08
    x12       e13              -1.33
    x13       e5                  -1
    x13       e13               0.03
    x14       e5                   1
    x14       e6                  -1
    x14       e13               0.03
    x15       e6                   1
    x15       e7                  -1
    x15       e13               0.03
    x16       e7                   1
    x16       e8                  -1
    x16       e13               0.03
    x17       e8                   1
    x17       e13              -0.02
    x18       e9                  -1
    x18       e13              0.025
    x19       e9                   1
    x19       e10                 -1
    x19       e13              0.025
    x20       e10                  1
    x20       e11                 -1
    x20       e13              0.025
    x21       e11                  1
    x21       e12                 -1
    x21       e13              0.025
    x22       e12                  1
    x22       e13               0.01
    x23       obj                 -1
    x23       e13                  1
RHS
    rhs       e1                 123
    rhs       e2                 123
    rhs       e3                 123
    rhs       e4                 123
BOUNDS
 UP bnd       x13               35.8
 UP bnd       x18               7.32
 FR bnd       x23
ENDATA
