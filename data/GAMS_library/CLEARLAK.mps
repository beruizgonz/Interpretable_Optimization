* LP written by GAMS Convert at 12/16/23 16:52:48
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        23       23        0        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        92       92        0        0        0        0        0        0
* FX      1
*
* Nonzero counts
*     Total    const       NL
*       157      157        0

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
COLUMNS
    x1        obj                  1
    x1        e1                   1
    x2        e2                  -1
    x2        e3                  -1
    x3        e2                   1
    x3        e4                  -1
    x3        e5                  -1
    x4        e3                   1
    x4        e6                  -1
    x4        e7                  -1
    x5        e4                   1
    x5        e8                  -1
    x5        e9                  -1
    x5        e10                 -1
    x6        e5                   1
    x6        e11                 -1
    x7        e6                   1
    x7        e12                 -1
    x8        e7                   1
    x8        e13                 -1
    x9        e8                   1
    x9        e14                 -1
    x10       e9                   1
    x10       e15                 -1
    x10       e16                 -1
    x10       e17                 -1
    x11       e10                  1
    x11       e18                 -1
    x12       e11                  1
    x12       e19                 -1
    x12       e20                 -1
    x12       e21                 -1
    x13       e12                  1
    x13       e22                 -1
    x14       e13                  1
    x14       e23                 -1
    x15       e14                  1
    x16       e15                  1
    x17       e16                  1
    x18       e17                  1
    x19       e18                  1
    x20       e19                  1
    x21       e20                  1
    x22       e21                  1
    x23       e22                  1
    x24       e23                  1
    x25       e2                   1
    x26       e3                   1
    x27       e4                   1
    x28       e5                   1
    x29       e6                   1
    x30       e7                   1
    x31       e8                   1
    x32       e9                   1
    x33       e10                  1
    x34       e11                  1
    x35       e12                  1
    x36       e13                  1
    x37       e14                  1
    x38       e15                  1
    x39       e16                  1
    x40       e17                  1
    x41       e18                  1
    x42       e19                  1
    x43       e20                  1
    x44       e21                  1
    x45       e22                  1
    x46       e23                  1
    x47       e1                 -10
    x48       e1                -7.5
    x48       e2                   1
    x49       e1                -2.5
    x49       e3                   1
    x50       e1              -5.625
    x50       e4                   1
    x51       e1              -1.875
    x51       e5                   1
    x52       e1              -1.875
    x52       e6                   1
    x53       e1              -0.625
    x53       e7                   1
    x54       e1            -1.40625
    x54       e8                   1
    x55       e1             -2.8125
    x55       e9                   1
    x56       e1            -1.40625
    x56       e10                  1
    x57       e1              -1.875
    x57       e11                  1
    x58       e1              -1.875
    x58       e12                  1
    x59       e1              -0.625
    x59       e13                  1
    x60       e1            -1.40625
    x60       e14                  1
    x61       e1           -0.703125
    x61       e15                  1
    x62       e1            -1.40625
    x62       e16                  1
    x63       e1           -0.703125
    x63       e17                  1
    x64       e1            -1.40625
    x64       e18                  1
    x65       e1            -0.46875
    x65       e19                  1
    x66       e1             -0.9375
    x66       e20                  1
    x67       e1            -0.46875
    x67       e21                  1
    x68       e1              -1.875
    x68       e22                  1
    x69       e1              -0.625
    x69       e23                  1
    x70       e1                  -5
    x71       e1               -3.75
    x71       e2                  -1
    x72       e1               -1.25
    x72       e3                  -1
    x73       e1             -2.8125
    x73       e4                  -1
    x74       e1             -0.9375
    x74       e5                  -1
    x75       e1             -0.9375
    x75       e6                  -1
    x76       e1             -0.3125
    x76       e7                  -1
    x77       e1           -0.703125
    x77       e8                  -1
    x78       e1            -1.40625
    x78       e9                  -1
    x79       e1           -0.703125
    x79       e10                 -1
    x80       e1             -0.9375
    x80       e11                 -1
    x81       e1             -0.9375
    x81       e12                 -1
    x82       e1             -0.3125
    x82       e13                 -1
    x83       e1           -0.703125
    x83       e14                 -1
    x84       e1          -0.3515625
    x84       e15                 -1
    x85       e1           -0.703125
    x85       e16                 -1
    x86       e1          -0.3515625
    x86       e17                 -1
    x87       e1           -0.703125
    x87       e18                 -1
    x88       e1           -0.234375
    x88       e19                 -1
    x89       e1            -0.46875
    x89       e20                 -1
    x90       e1           -0.234375
    x90       e21                 -1
    x91       e1             -0.9375
    x91       e22                 -1
    x92       e1             -0.3125
    x92       e23                 -1
RHS
    rhs       e2                 150
    rhs       e3                 350
    rhs       e4                 150
    rhs       e5                 350
    rhs       e6                 150
    rhs       e7                 350
    rhs       e8                 -50
    rhs       e9                 100
    rhs       e10                250
    rhs       e11                100
    rhs       e12                100
    rhs       e13                100
    rhs       e14                100
    rhs       e15                -50
    rhs       e16                100
    rhs       e17                250
    rhs       e18                100
    rhs       e19                -50
    rhs       e20                100
    rhs       e21                250
    rhs       e22                100
    rhs       e23                100
BOUNDS
 FR bnd       x1
 FX bnd       x2                 100
 UP bnd       x3                 250
 UP bnd       x4                 250
 UP bnd       x5                 250
 UP bnd       x6                 250
 UP bnd       x7                 250
 UP bnd       x8                 250
 UP bnd       x9                 250
 UP bnd       x10                250
 UP bnd       x11                250
 UP bnd       x12                250
 UP bnd       x13                250
 UP bnd       x14                250
 UP bnd       x15                250
 UP bnd       x16                250
 UP bnd       x17                250
 UP bnd       x18                250
 UP bnd       x19                250
 UP bnd       x20                250
 UP bnd       x21                250
 UP bnd       x22                250
 UP bnd       x23                250
 UP bnd       x24                250
 UP bnd       x25                200
 UP bnd       x26                200
 UP bnd       x27                200
 UP bnd       x28                200
 UP bnd       x29                200
 UP bnd       x30                200
 UP bnd       x31                200
 UP bnd       x32                200
 UP bnd       x33                200
 UP bnd       x34                200
 UP bnd       x35                200
 UP bnd       x36                200
 UP bnd       x37                200
 UP bnd       x38                200
 UP bnd       x39                200
 UP bnd       x40                200
 UP bnd       x41                200
 UP bnd       x42                200
 UP bnd       x43                200
 UP bnd       x44                200
 UP bnd       x45                200
 UP bnd       x46                200
ENDATA
