* LP written by GAMS Convert at 12/16/23 00:59:47
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        36        7        0       29        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        49       49        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       129      129        0

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
 E  e5
 E  e6
 E  e7
 E  e8
 E  e9
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
 L  e25
 L  e26
 L  e27
 L  e28
 L  e29
 L  e30
 L  e31
 L  e32
 L  e33
 L  e34
 E  e35
 E  e36
COLUMNS
    x1        e1                   1
    x1        e5                 -16
    x1        e35                -18
    x2        e1                   1
    x2        e6                 -15
    x2        e35                -21
    x3        e1                   1
    x3        e7                 -28
    x3        e35                -18
    x4        e1                   1
    x4        e8                 -23
    x4        e35                -16
    x5        e1                   1
    x5        e9                 -81
    x5        e35                -10
    x6        e2                   1
    x6        e6                 -10
    x6        e35                -15
    x7        e2                   1
    x7        e7                 -14
    x7        e35                -16
    x8        e2                   1
    x8        e8                 -15
    x8        e35                -14
    x9        e2                   1
    x9        e9                 -57
    x9        e35                 -9
    x10       e3                   1
    x10       e6                  -5
    x10       e35                -10
    x11       e3                   1
    x11       e8                  -7
    x11       e35                 -9
    x12       e3                   1
    x12       e9                 -29
    x12       e35                 -6
    x13       e4                   1
    x13       e5                  -9
    x13       e35                -17
    x14       e4                   1
    x14       e6                 -11
    x14       e35                -16
    x15       e4                   1
    x15       e7                 -22
    x15       e35                -17
    x16       e4                   1
    x16       e8                 -17
    x16       e35                -15
    x17       e4                   1
    x17       e9                 -55
    x17       e35                -10
    x18       e5                   1
    x18       e10                 -1
    x18       e11                 -1
    x18       e12                 -1
    x18       e13                 -1
    x18       e14                 -1
    x19       e6                   1
    x19       e15                 -1
    x19       e16                 -1
    x19       e17                 -1
    x19       e18                 -1
    x19       e19                 -1
    x20       e7                   1
    x20       e20                 -1
    x20       e21                 -1
    x20       e22                 -1
    x20       e23                 -1
    x20       e24                 -1
    x21       e8                   1
    x21       e25                 -1
    x21       e26                 -1
    x21       e27                 -1
    x21       e28                 -1
    x21       e29                 -1
    x22       e9                   1
    x22       e30                 -1
    x22       e31                 -1
    x22       e32                 -1
    x22       e33                 -1
    x22       e34                 -1
    x23       e10                 -1
    x23       e36               -2.6
    x24       e11                 -1
    x24       e36              -0.65
    x25       e12                 -1
    x25       e36              -4.55
    x26       e13                 -1
    x26       e36               -2.6
    x27       e14                 -1
    x27       e36               -2.6
    x28       e15                 -1
    x28       e36               -3.9
    x29       e16                 -1
    x29       e36               -9.1
    x30       e17                 -1
    x31       e18                 -1
    x32       e19                 -1
    x33       e20                 -1
    x33       e36               -0.7
    x34       e21                 -1
    x34       e36               -1.4
    x35       e22                 -1
    x35       e36               -2.8
    x36       e23                 -1
    x36       e36               -1.4
    x37       e24                 -1
    x37       e36               -0.7
    x38       e25                 -1
    x38       e36               -1.4
    x39       e26                 -1
    x39       e36               -1.4
    x40       e27                 -1
    x40       e36               -2.1
    x41       e28                 -1
    x41       e36               -1.4
    x42       e29                 -1
    x42       e36               -0.7
    x43       e30                 -1
    x43       e36               -0.1
    x44       e31                 -1
    x44       e36               -0.8
    x45       e32                 -1
    x45       e36               -0.1
    x46       e33                 -1
    x47       e34                 -1
    x48       obj                  1
    x48       e36                  1
    x49       e35                  1
    x49       e36                 -1
RHS
    rhs       e1                  10
    rhs       e2                  19
    rhs       e3                  25
    rhs       e4                  15
    rhs       e10               -200
    rhs       e11               -220
    rhs       e12               -250
    rhs       e13               -270
    rhs       e14               -300
    rhs       e15                -50
    rhs       e16               -150
    rhs       e20               -140
    rhs       e21               -160
    rhs       e22               -180
    rhs       e23               -200
    rhs       e24               -220
    rhs       e25                -10
    rhs       e26                -50
    rhs       e27                -80
    rhs       e28               -100
    rhs       e29               -340
    rhs       e30               -580
    rhs       e31               -600
    rhs       e32               -620
BOUNDS
 FR bnd       x48
 FR bnd       x49
ENDATA
