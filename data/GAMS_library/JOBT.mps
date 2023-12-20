* LP written by GAMS Convert at 12/18/23 11:25:16
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        16       11        5        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        31       31        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        73       73        0

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
 E  e6
 E  e7
 E  e8
 E  e9
 E  e10
 G  e11
 G  e12
 G  e13
 G  e14
 G  e15
 E  e16
COLUMNS
    x1        e1                  -1
    x1        e11             -0.125
    x2        e2                  -1
    x2        e12             -0.125
    x3        e3                  -1
    x3        e13             -0.125
    x4        e4                  -1
    x4        e14             -0.125
    x5        e5                  -1
    x5        e15             -0.125
    x6        e1                   1
    x6        e2                  -1
    x6        e16                -10
    x7        e2                   1
    x7        e3                  -1
    x7        e16                -10
    x8        e3                   1
    x8        e4                  -1
    x8        e16                -10
    x9        e4                   1
    x9        e5                  -1
    x9        e16                -10
    x10       e5                   1
    x10       e16                -10
    x11       e1                  -1
    x11       e2                   1
    x11       e16                -30
    x12       e2                  -1
    x12       e3                   1
    x12       e16                -30
    x13       e3                  -1
    x13       e4                   1
    x13       e16                -30
    x14       e4                  -1
    x14       e5                   1
    x14       e16                -30
    x15       e5                  -1
    x15       e16                -30
    x16       e6                   1
    x16       e7                  -1
    x16       e11                  1
    x16       e16               -100
    x17       e7                   1
    x17       e8                  -1
    x17       e12                  1
    x17       e16               -100
    x18       e8                   1
    x18       e9                  -1
    x18       e13                  1
    x18       e16               -100
    x19       e9                   1
    x19       e10                 -1
    x19       e14                  1
    x19       e16               -100
    x20       e10                  1
    x20       e15                  1
    x20       e16               -200
    x21       e6                  -1
    x21       e11       -1.166666667
    x22       e7                  -1
    x22       e12       -1.166666667
    x23       e8                  -1
    x23       e13       -1.166666667
    x24       e9                  -1
    x24       e14       -1.166666667
    x25       e10                 -1
    x25       e15       -1.166666667
    x26       e6                   1
    x27       e7                   1
    x28       e8                   1
    x29       e9                   1
    x30       e10                  1
    x31       obj                  1
    x31       e16                  1
RHS
    rhs       e1                 -90
    rhs       e2                -200
    rhs       e3                -300
    rhs       e4                -400
    rhs       e5                -200
    rhs       e6                  20
BOUNDS
 FR bnd       x31
ENDATA
