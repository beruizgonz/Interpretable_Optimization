* LP written by GAMS Convert at 12/16/23 00:55:48
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        12        3        5        4        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        43       43        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        99       99        0

NAME          Convert
*
* original model was minimizing
*
ROWS
 N  obj
 L  e1
 L  e2
 L  e3
 L  e4
 G  e5
 G  e6
 G  e7
 G  e8
 G  e9
 E  e10
 E  e11
 E  e12
COLUMNS
    x1        e1                   1
    x1        e5                  16
    x1        e10                -18
    x2        e1                   1
    x2        e6                  15
    x2        e10                -21
    x3        e1                   1
    x3        e7                  28
    x3        e10                -18
    x4        e1                   1
    x4        e8                  23
    x4        e10                -16
    x5        e1                   1
    x5        e9                  81
    x5        e10                -10
    x6        e2                   1
    x7        e2                   1
    x7        e6                  10
    x7        e10                -15
    x8        e2                   1
    x8        e7                  14
    x8        e10                -16
    x9        e2                   1
    x9        e8                  15
    x9        e10                -14
    x10       e2                   1
    x10       e9                  57
    x10       e10                 -9
    x11       e3                   1
    x12       e3                   1
    x12       e6                   5
    x12       e10                -10
    x13       e3                   1
    x14       e3                   1
    x14       e8                   7
    x14       e10                 -9
    x15       e3                   1
    x15       e9                  29
    x15       e10                 -6
    x16       e4                   1
    x16       e5                   9
    x16       e10                -17
    x17       e4                   1
    x17       e6                  11
    x17       e10                -16
    x18       e4                   1
    x18       e7                  22
    x18       e10                -17
    x19       e4                   1
    x19       e8                  17
    x19       e10                -15
    x20       e4                   1
    x20       e9                  55
    x20       e10                -10
    x21       e5                  -1
    x21       e11                 13
    x22       e5                  -1
    x22       e11               10.4
    x23       e5                  -1
    x23       e11               9.75
    x24       e5                  -1
    x24       e11                5.2
    x25       e5                  -1
    x25       e11                2.6
    x26       e6                  -1
    x26       e11                 13
    x27       e6                  -1
    x27       e11                9.1
    x28       e7                  -1
    x28       e11                  7
    x29       e7                  -1
    x29       e11                6.3
    x30       e7                  -1
    x30       e11                4.9
    x31       e7                  -1
    x31       e11                2.1
    x32       e7                  -1
    x32       e11                0.7
    x33       e8                  -1
    x33       e11                  7
    x34       e8                  -1
    x34       e11                5.6
    x35       e8                  -1
    x35       e11                4.2
    x36       e8                  -1
    x36       e11                2.1
    x37       e8                  -1
    x37       e11                0.7
    x38       e9                  -1
    x38       e11                  1
    x39       e9                  -1
    x39       e11                0.9
    x40       e9                  -1
    x40       e11                0.1
    x41       e10                  1
    x41       e12                 -1
    x42       e11                  1
    x42       e12                 -1
    x43       obj                  1
    x43       e12                  1
RHS
    rhs       e1                  10
    rhs       e2                  19
    rhs       e3                  25
    rhs       e4                  15
    rhs       e11             7332.5
BOUNDS
 UP bnd       x21                200
 UP bnd       x22                 20
 UP bnd       x23                 30
 UP bnd       x24                 20
 UP bnd       x25                 30
 UP bnd       x26                 50
 UP bnd       x27                100
 UP bnd       x28                140
 UP bnd       x29                 20
 UP bnd       x30                 20
 UP bnd       x31                 20
 UP bnd       x32                 20
 UP bnd       x33                 10
 UP bnd       x34                 40
 UP bnd       x35                 30
 UP bnd       x36                 20
 UP bnd       x37                240
 UP bnd       x38                580
 UP bnd       x39                 20
 UP bnd       x40                 20
 FR bnd       x43
ENDATA
