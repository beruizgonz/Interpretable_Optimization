* LP written by GAMS Convert at 12/16/23 23:52:51
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        41       17        0       24        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        48       48        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       184      184        0

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
 L  e16
 L  e17
 L  e18
 L  e19
 L  e20
 L  e21
 L  e22
 L  e23
 L  e24
 E  e25
 E  e26
 E  e27
 E  e28
 E  e29
 E  e30
 E  e31
 E  e32
 E  e33
 E  e34
 E  e35
 E  e36
 E  e37
 E  e38
 E  e39
 E  e40
 E  e41
COLUMNS
    x1        e1                   1
    x1        e2                   1
    x1        e3                   1
    x1        e4                   1
    x1        e5                   1
    x1        e11                0.5
    x1        e12                  1
    x1        e13               1.72
    x1        e14                0.5
    x1        e15                  1
    x1        e16                  1
    x1        e17              17.16
    x1        e18               2.34
    x1        e23               2.43
    x1        e24               1.35
    x1        e37               -150
    x1        e38                -10
    x2        e1                   1
    x2        e2                   1
    x2        e3                 0.5
    x2        e11               0.25
    x2        e12                  1
    x2        e13                4.5
    x2        e14                  1
    x2        e15                  8
    x2        e23                2.5
    x2        e24                7.5
    x3        e1                   1
    x3        e2                   1
    x3        e3                   1
    x3        e4                   1
    x3        e11               0.25
    x3        e12                  1
    x3        e13               0.75
    x3        e14               0.75
    x3        e15               0.75
    x3        e16                 16
    x3        e23                7.5
    x3        e24               0.75
    x3        e37               -200
    x3        e38                 -5
    x4        e1                   1
    x4        e2                   1
    x4        e3                   1
    x4        e4                   1
    x4        e5                0.25
    x4        e11                0.5
    x4        e12                  1
    x4        e13               5.16
    x4        e14                  5
    x4        e15                  5
    x4        e16              19.58
    x4        e17               2.42
    x4        e23              11.16
    x4        e24               4.68
    x4        e37               -375
    x4        e38                -50
    x5        e3                 0.5
    x5        e4                   1
    x5        e5                   1
    x5        e6                   1
    x5        e7                   1
    x5        e8                   1
    x5        e9                   1
    x5        e10                  1
    x5        e11               0.75
    x5        e15                  5
    x5        e16                  5
    x5        e17                  9
    x5        e18                  2
    x5        e19                1.5
    x5        e20                  2
    x5        e21                  1
    x5        e22                 26
    x5        e23                 12
    x5        e37               -525
    x5        e38                -80
    x6        e5                0.25
    x6        e6                   1
    x6        e7                   1
    x6        e8                   1
    x6        e9                   1
    x6        e10                0.5
    x6        e17                4.3
    x6        e18               5.04
    x6        e19               7.16
    x6        e20               7.97
    x6        e21               4.41
    x6        e22               1.12
    x6        e37               -140
    x6        e38                 -5
    x7        e7                0.75
    x7        e8                   1
    x7        e9                   1
    x7        e10                  1
    x7        e11               0.75
    x7        e19                 17
    x7        e20                 15
    x7        e21                 12
    x7        e22                  7
    x7        e23                  6
    x7        e37               -360
    x7        e38                -50
    x8        obj                 -1
    x8        e41                  1
    x9        e37                  1
    x9        e41                 -1
    x10       e38                  1
    x10       e41                  1
    x11       e39                  1
    x11       e41                  1
    x12       e40                  1
    x12       e41                 -1
    x13       e13                 -1
    x13       e25                 -1
    x14       e14                 -1
    x14       e26                 -1
    x15       e15                 -1
    x15       e27                 -1
    x16       e16                 -1
    x16       e28                 -1
    x17       e17                 -1
    x17       e29                 -1
    x18       e18                 -1
    x18       e30                 -1
    x19       e19                 -1
    x19       e31                 -1
    x20       e20                 -1
    x20       e32                 -1
    x21       e21                 -1
    x21       e33                 -1
    x22       e22                 -1
    x22       e34                 -1
    x23       e23                 -1
    x23       e35                 -1
    x24       e24                 -1
    x24       e36                 -1
    x25       e25                 -1
    x25       e40                 -3
    x26       e26                 -1
    x26       e40                 -3
    x27       e27                 -1
    x27       e40                 -3
    x28       e28                 -1
    x28       e40                 -3
    x29       e29                 -1
    x29       e40                 -3
    x30       e30                 -1
    x30       e40                 -3
    x31       e31                 -1
    x31       e40                 -3
    x32       e32                 -1
    x32       e40                 -3
    x33       e33                 -1
    x33       e40                 -3
    x34       e34                 -1
    x34       e40                 -3
    x35       e35                 -1
    x35       e40                 -3
    x36       e36                 -1
    x36       e40                 -3
    x37       e13                 -1
    x37       e39                 -4
    x38       e14                 -1
    x38       e39                 -4
    x39       e15                 -1
    x39       e39                 -4
    x40       e16                 -1
    x40       e39                 -4
    x41       e17                 -1
    x41       e39                 -4
    x42       e18                 -1
    x42       e39                 -4
    x43       e19                 -1
    x43       e39                 -4
    x44       e20                 -1
    x44       e39                 -4
    x45       e21                 -1
    x45       e39                 -4
    x46       e22                 -1
    x46       e39                 -4
    x47       e23                 -1
    x47       e39                 -4
    x48       e24                 -1
    x48       e39                 -4
RHS
    rhs       e1                   4
    rhs       e2                   4
    rhs       e3                   4
    rhs       e4                   4
    rhs       e5                   4
    rhs       e6                   4
    rhs       e7                   4
    rhs       e8                   4
    rhs       e9                   4
    rhs       e10                  4
    rhs       e11                  4
    rhs       e12                  4
    rhs       e25                -25
    rhs       e26                -25
    rhs       e27                -25
    rhs       e28                -25
    rhs       e29                -25
    rhs       e30                -25
    rhs       e31                -25
    rhs       e32                -25
    rhs       e33                -25
    rhs       e34                -25
    rhs       e35                -25
    rhs       e36                -25
BOUNDS
 FR bnd       x8
 FR bnd       x9
 FR bnd       x10
 FR bnd       x11
 FR bnd       x12
ENDATA
