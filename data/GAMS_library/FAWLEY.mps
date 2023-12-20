* LP written by GAMS Convert at 12/17/23 00:44:08
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        37       37        0        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        52       52        0        0        0        0        0        0
* FX      4
*
* Nonzero counts
*     Total    const       NL
*       180      180        0

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
 E  e15
 E  e16
 E  e17
 E  e18
 E  e19
 E  e20
 E  e21
 E  e22
 E  e23
 E  e24
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
COLUMNS
    x1        e1                   1
    x1        e35       -256.5850816
    x1        e36             -24.15
    x2        e2                   1
    x2        e35       -241.3769752
    x2        e36             -24.15
    x3        e3                   1
    x3        e35       -320.9963548
    x3        e36             -10.05
    x4        e1                  -1
    x4        e4               0.035
    x4        e5                0.39
    x4        e6                 0.1
    x4        e7               0.285
    x4        e8               0.165
    x4        e18       -1.165501166
    x5        e2                  -1
    x5        e4                0.03
    x5        e5                 0.3
    x5        e6               0.075
    x5        e7                0.23
    x5        e9               0.335
    x5        e18       -1.128668172
    x6        e3                  -1
    x6        e4               0.045
    x6        e5                0.43
    x6        e6               0.135
    x6        e7                0.28
    x6        e10                0.1
    x6        e18       -1.215066829
    x7        e6                  -1
    x7        e11               0.02
    x7        e12                0.9
    x7        e17               0.08
    x7        e19       -1.356852103
    x8        e5                  -1
    x8        e11               0.02
    x8        e13              0.275
    x8        e15               0.68
    x8        e17              0.025
    x8        e20       -1.128668172
    x9        e5                  -1
    x9        e11             0.0325
    x9        e14             0.3775
    x9        e15              0.555
    x9        e17              0.035
    x9        e20       -1.128668172
    x10       e7                  -1
    x10       e11               0.05
    x10       e13              0.325
    x10       e15              0.585
    x10       e17               0.04
    x10       e20       -1.086956522
    x11       e7                  -1
    x11       e11               0.06
    x11       e14               0.45
    x11       e15               0.44
    x11       e17               0.05
    x11       e20       -1.086956522
    x12       e18                  1
    x12       e34            -0.9435
    x13       e19                  1
    x13       e34             -3.774
    x14       e20                  1
    x14       e34            -4.0885
    x15       e4                  -1
    x15       e17               1.11
    x16       e11                 -1
    x16       e17               1.07
    x17       e16                  1
    x17       e35               -245
    x18       e4                  -1
    x18       e25       18.461538462
    x18       e26       132.76923077
    x18       e27       146.15384615
    x18       e30       -1.538461538
    x18       e31                 -1
    x19       e5                  -1
    x19       e28                  1
    x19       e29               14.8
    x19       e32                 -1
    x20       e7                  -1
    x20       e28                1.7
    x20       e29               21.8
    x20       e32                 -1
    x21       e8                  -1
    x21       e28                  4
    x21       e29                 48
    x21       e32                 -1
    x22       e9                  -1
    x22       e28                  5
    x22       e29                 51
    x22       e32                 -1
    x23       e10                 -1
    x23       e28                0.6
    x23       e29                 44
    x23       e32                 -1
    x24       e11                 -1
    x24       e25       131.57894737
    x24       e26       178.24561404
    x24       e27       175.43859649
    x24       e30       -1.754385965
    x24       e31                 -1
    x25       e12                 -1
    x25       e25        6.936416185
    x25       e26       118.49710983
    x25       e27       40.462427746
    x25       e30       -1.156069364
    x25       e31                 -1
    x26       e13                 -1
    x26       e25       9.5890410959
    x26       e26                130
    x26       e27       82.191780822
    x26       e30       -1.369863014
    x26       e31                 -1
    x27       e14                 -1
    x27       e25                 12
    x27       e26       132.13333333
    x27       e27       85.333333333
    x27       e30       -1.333333333
    x27       e31                 -1
    x28       e15                 -1
    x28       e28                1.5
    x28       e29                 18
    x28       e32                 -1
    x29       e16                 -1
    x29       e28                  3
    x29       e29               37.5
    x29       e32                 -1
    x30       e17                 -1
    x30       e28                3.5
    x30       e29                 44
    x30       e32                 -1
    x31       e5                -0.7
    x31       e6                -0.3
    x31       e22                 -1
    x32       e4                -0.2
    x32       e5                -0.8
    x32       e22                 -1
    x33       e5                -0.1
    x33       e6                -0.1
    x33       e15               -0.8
    x33       e22                 -1
    x34       e5                  -1
    x34       e23                 -1
    x35       e15                 -1
    x35       e23                 -1
    x36       e25                -10
    x36       e26                -99
    x36       e27                -60
    x36       e30                  1
    x36       e34              -3.12
    x37       e21                 -1
    x37       e31                  1
    x38       e24                 -1
    x38       e28                 -3
    x38       e29              -37.5
    x38       e32                  1
    x39       e25                 -1
    x40       e26                 -1
    x41       e27                  1
    x42       e28                  1
    x43       e29                  1
    x44       e21                  1
    x44       e33               -430
    x45       e22                  1
    x45       e33               -300
    x46       e23                  1
    x46       e33               -315
    x47       e24                  1
    x47       e33               -250
    x48       e33                  1
    x48       e37                 -1
    x49       e34                  1
    x49       e37                  1
    x50       e35                  1
    x50       e37                  1
    x51       e36                  1
    x51       e37                  1
    x52       obj                 -1
    x52       e37                  1
RHS
    rhs       e12               1.65
    rhs       e13              -0.58
BOUNDS
 UP bnd       x1                 110
 UP bnd       x2                 165
 UP bnd       x3                  80
 UP bnd       x12       320.34976153
 UP bnd       x13       33.386327504
 UP bnd       x14       35.612082671
 FX bnd       x44                 40
 FX bnd       x45                 20
 FX bnd       x46                 50
 FX bnd       x47                145
 FR bnd       x48
 FR bnd       x49
 FR bnd       x50
 FR bnd       x51
 FR bnd       x52
ENDATA
