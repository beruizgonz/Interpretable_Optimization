* LP written by GAMS Convert at 12/20/23 12:24:05
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        11        1       10        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        41       41        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       131      131        0

NAME          Convert
*
* original model was minimizing
*
ROWS
 N  obj
 E  e1
 G  e2
 G  e3
 G  e4
 G  e5
 G  e6
 G  e7
 G  e8
 G  e9
 G  e10
 G  e11
COLUMNS
    x1        e1                 -50
    x1        e2                   1
    x2        e1                 -85
    x2        e2                   1
    x2        e3                   1
    x3        e1                -115
    x3        e2                   1
    x3        e3                   1
    x3        e4                   1
    x4        e1                -143
    x4        e2                   1
    x4        e3                   1
    x4        e4                   1
    x4        e5                   1
    x5        e1               -52.5
    x5        e3                   1
    x6        e1              -89.25
    x6        e3                   1
    x6        e4                   1
    x7        e1             -120.75
    x7        e3                   1
    x7        e4                   1
    x7        e5                   1
    x8        e1             -150.15
    x8        e3                   1
    x8        e4                   1
    x8        e5                   1
    x8        e6                   1
    x9        e1                 -56
    x9        e4                   1
    x10       e1               -95.2
    x10       e4                   1
    x10       e5                   1
    x11       e1              -128.8
    x11       e4                   1
    x11       e5                   1
    x11       e6                   1
    x12       e1             -160.16
    x12       e4                   1
    x12       e5                   1
    x12       e6                   1
    x12       e7                   1
    x13       e1               -85.5
    x13       e5                   1
    x14       e1             -145.35
    x14       e5                   1
    x14       e6                   1
    x15       e1             -196.65
    x15       e5                   1
    x15       e6                   1
    x15       e7                   1
    x16       e1             -244.53
    x16       e5                   1
    x16       e6                   1
    x16       e7                   1
    x16       e8                   1
    x17       e1                 -90
    x17       e6                   1
    x18       e1                -153
    x18       e6                   1
    x18       e7                   1
    x19       e1                -207
    x19       e6                   1
    x19       e7                   1
    x19       e8                   1
    x20       e1              -257.4
    x20       e6                   1
    x20       e7                   1
    x20       e8                   1
    x20       e9                   1
    x21       e1                 -95
    x21       e7                   1
    x22       e1              -161.5
    x22       e7                   1
    x22       e8                   1
    x23       e1              -218.5
    x23       e7                   1
    x23       e8                   1
    x23       e9                   1
    x24       e1              -271.7
    x24       e7                   1
    x24       e8                   1
    x24       e9                   1
    x24       e10                  1
    x25       e1               -98.5
    x25       e8                   1
    x26       e1             -167.45
    x26       e8                   1
    x26       e9                   1
    x27       e1             -226.55
    x27       e8                   1
    x27       e9                   1
    x27       e10                  1
    x28       e1             -281.71
    x28       e8                   1
    x28       e9                   1
    x28       e10                  1
    x28       e11                  1
    x29       e1                -105
    x29       e9                   1
    x30       e1              -178.5
    x30       e9                   1
    x30       e10                  1
    x31       e1              -241.5
    x31       e9                   1
    x31       e10                  1
    x31       e11                  1
    x32       e1              -300.3
    x32       e9                   1
    x32       e10                  1
    x32       e11                  1
    x33       e1                -111
    x33       e10                  1
    x34       e1              -188.7
    x34       e10                  1
    x34       e11                  1
    x35       e1              -255.3
    x35       e10                  1
    x35       e11                  1
    x36       e1             -317.46
    x36       e10                  1
    x36       e11                  1
    x37       e1                -119
    x37       e11                  1
    x38       e1              -202.3
    x38       e11                  1
    x39       e1              -273.7
    x39       e11                  1
    x40       e1             -340.34
    x40       e11                  1
    x41       obj                  1
    x41       e1                   1
RHS
    rhs       e2                   5
    rhs       e3                   6
    rhs       e4                   7
    rhs       e5                   6
    rhs       e6                   4
    rhs       e7                   9
    rhs       e8                   8
    rhs       e9                   8
    rhs       e10                  6
    rhs       e11                  4
BOUNDS
 FR bnd       x41
ENDATA
