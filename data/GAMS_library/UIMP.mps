* LP written by GAMS Convert at 12/20/23 13:11:06
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        21        9        0       12        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        43       43        0        0        0        0        0        0
* FX      2
*
* Nonzero counts
*     Total    const       NL
*       116      116        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 E  e1
 E  e2
 E  e3
 L  e4
 L  e5
 L  e6
 L  e7
 L  e8
 L  e9
 L  e10
 L  e11
 L  e12
 L  e13
 L  e14
 L  e15
 E  e16
 E  e17
 E  e18
 E  e19
 E  e20
 E  e21
COLUMNS
    x1        e2                  -2
    x1        e4                   4
    x1        e16                  1
    x2        e2                  -4
    x2        e5                   7
    x2        e16                  1
    x3        e2                  -1
    x3        e6                   3
    x3        e16                  1
    x4        e2                  -3
    x4        e4                   4
    x4        e17                  1
    x5        e2                  -3
    x5        e5                   6
    x5        e17                  1
    x6        e2                  -4
    x6        e4                   6
    x6        e18                  1
    x7        e2                  -2
    x7        e5                   6
    x7        e18                  1
    x8        e2                  -3
    x8        e7                   3
    x8        e16                  1
    x9        e2                  -5
    x9        e8                   6
    x9        e16                  1
    x10       e2                  -2
    x10       e9                   2
    x10       e16                  1
    x11       e2                  -4
    x11       e7                   3
    x11       e17                  1
    x12       e2                  -4
    x12       e8                   5
    x12       e17                  1
    x13       e2                  -5
    x13       e7                   5
    x13       e18                  1
    x14       e2                  -3
    x14       e8                   5
    x14       e18                  1
    x15       e2                  -3
    x15       e10                  5
    x15       e19                  1
    x16       e2                  -5
    x16       e11                  8
    x16       e19                  1
    x17       e2                  -2
    x17       e12                  4
    x17       e19                  1
    x18       e2                  -4
    x18       e10                  5
    x18       e20                  1
    x19       e2                  -4
    x19       e11                  7
    x19       e20                  1
    x20       e2                  -5
    x20       e10                  7
    x20       e21                  1
    x21       e2                  -3
    x21       e11                  7
    x21       e21                  1
    x22       e2                  -4
    x22       e13                  4
    x22       e19                  1
    x23       e2                  -6
    x23       e14                  7
    x23       e19                  1
    x24       e2                  -3
    x24       e15                  3
    x24       e19                  1
    x25       e2                  -5
    x25       e13                  4
    x25       e20                  1
    x26       e2                  -5
    x26       e14                  6
    x26       e20                  1
    x27       e2                  -6
    x27       e13                  5
    x27       e21                  1
    x28       e2                  -4
    x28       e14                  6
    x28       e21                  1
    x29       e2                  -1
    x29       e16                 -1
    x29       e19                  1
    x30       e2                  -1
    x30       e17                 -1
    x30       e20                  1
    x31       e2                  -1
    x31       e18                 -1
    x31       e21                  1
    x32       e2                  -1
    x32       e19                 -1
    x33       e2                  -1
    x33       e20                 -1
    x34       e2                  -1
    x34       e21                 -1
    x35       e3                 -10
    x35       e16                 -1
    x36       e3                 -10
    x36       e17                 -1
    x37       e3                  -9
    x37       e18                 -1
    x38       e3                 -11
    x38       e19                 -1
    x39       e3                 -11
    x39       e20                 -1
    x40       e3                 -10
    x40       e21                 -1
    x41       e1                   1
    x41       e2                   1
    x42       e1                  -1
    x42       e3                   1
    x43       obj                 -1
    x43       e1                   1
RHS
    rhs       e4                 100
    rhs       e5                 100
    rhs       e6                  40
    rhs       e7                  80
    rhs       e8                  90
    rhs       e9                  30
    rhs       e10                110
    rhs       e11                110
    rhs       e12                 50
    rhs       e13                 90
    rhs       e14                100
    rhs       e15                 40
BOUNDS
 UP bnd       x29                 20
 UP bnd       x30                 20
 FX bnd       x31                  0
 UP bnd       x32                 20
 UP bnd       x33                 20
 FX bnd       x34                  0
 LO bnd       x35                 25
 LO bnd       x36                 30
 LO bnd       x37                 30
 LO bnd       x38                 30
 LO bnd       x39                 25
 LO bnd       x40                 25
 FR bnd       x41
 FR bnd       x42
 FR bnd       x43
ENDATA
