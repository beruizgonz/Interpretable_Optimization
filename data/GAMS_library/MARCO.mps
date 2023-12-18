* LP written by GAMS Convert at 12/18/23 11:39:46
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        46        9       26       11        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        61       61        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       237      237        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 G  e1
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
 G  e12
 G  e13
 G  e14
 G  e15
 G  e16
 G  e17
 G  e18
 G  e19
 G  e20
 G  e21
 G  e22
 G  e23
 G  e24
 L  e25
 L  e26
 L  e27
 L  e28
 L  e29
 E  e30
 E  e31
 E  e32
 E  e33
 E  e34
 G  e35
 G  e36
 L  e37
 L  e38
 L  e39
 L  e40
 L  e41
 L  e42
 E  e43
 E  e44
 E  e45
 E  e46
COLUMNS
    x1        e5                  -1
    x1        e8                0.31
    x1        e9                0.59
    x1        e10               0.22
    x1        e27                  1
    x1        e46              -0.08
    x2        e2               0.236
    x2        e3               0.223
    x2        e4               0.087
    x2        e5               0.111
    x2        e6               0.315
    x2        e8               0.029
    x2        e23                 -1
    x2        e25                  1
    x2        e46               -0.1
    x3        e3                  -1
    x3        e7               0.807
    x3        e8               0.129
    x3        e26                  1
    x3        e46              -0.15
    x4        e4                  -1
    x4        e8                 0.3
    x4        e9                0.59
    x4        e10               0.21
    x4        e27                  1
    x4        e46               -0.8
    x5        e46               -0.1
    x6        e16                 -1
    x6        e19               0.38
    x6        e20                0.6
    x6        e21               0.15
    x6        e27                  1
    x6        e46              -0.08
    x7        e13               0.18
    x7        e14              0.196
    x7        e15              0.073
    x7        e16              0.091
    x7        e17              0.443
    x7        e19              0.017
    x7        e24                 -1
    x7        e25                  1
    x7        e46               -0.1
    x8        e14                 -1
    x8        e18              0.836
    x8        e19              0.099
    x8        e26                  1
    x8        e46              -0.15
    x9        e15                 -1
    x9        e19               0.36
    x9        e20               0.58
    x9        e21               0.15
    x9        e27                  1
    x9        e46               -0.8
    x10       e17                 -1
    x10       e22               0.97
    x10       e46               -0.1
    x11       e30                  1
    x11       e44               -1.5
    x12       e31                  1
    x12       e35                -90
    x12       e37              -12.7
    x12       e44              -10.5
    x13       e32                  1
    x13       e36                -86
    x13       e38              -12.7
    x13       e44               -9.1
    x14       e33                  1
    x14       e39               -306
    x14       e40               -0.5
    x14       e44               -7.7
    x15       e34                  1
    x15       e41               -352
    x15       e42               -3.4
    x15       e44              -6.65
    x16       e23                  1
    x16       e28                  1
    x16       e45               -7.5
    x17       e24                  1
    x17       e29                  1
    x17       e45               -6.5
    x18       e1                   1
    x18       e45              -6.75
    x19       e12                  1
    x19       e45              -6.75
    x20       e1                  -1
    x20       e31                 -1
    x20       e35               91.8
    x20       e37              199.2
    x21       e1                  -1
    x21       e32                 -1
    x21       e36               91.8
    x21       e38              199.2
    x22       e2                  -1
    x22       e31                 -1
    x22       e35               78.5
    x22       e37               18.4
    x23       e2                  -1
    x23       e32                 -1
    x23       e36               78.5
    x23       e38               18.4
    x24       e3                  -1
    x24       e31                 -1
    x24       e35                 65
    x24       e37               6.54
    x25       e3                  -1
    x25       e32                 -1
    x25       e36                 65
    x25       e38               6.54
    x26       e3                  -1
    x26       e33                 -1
    x26       e39                272
    x26       e40              0.283
    x27       e4                  -1
    x27       e33                 -1
    x27       e39                292
    x27       e40              0.526
    x28       e5                  -1
    x28       e33                 -1
    x28       e39                295
    x28       e40               0.98
    x29       e5                  -1
    x29       e34                 -1
    x29       e41                295
    x29       e42               0.98
    x30       e6                  -1
    x30       e34                 -1
    x30       e41                343
    x30       e42                4.7
    x31       e7                  -1
    x31       e31                 -1
    x31       e35                104
    x31       e37               2.57
    x32       e7                  -1
    x32       e32                 -1
    x32       e36                104
    x32       e38               2.57
    x33       e8                  -1
    x33       e30                 -1
    x34       e9                  -1
    x34       e31                 -1
    x34       e35               93.7
    x34       e37                6.9
    x35       e9                  -1
    x35       e32                 -1
    x35       e36               93.7
    x35       e38                6.9
    x36       e10                 -1
    x36       e33                 -1
    x36       e39              294.4
    x36       e40              0.353
    x37       e10                 -1
    x37       e34                 -1
    x37       e41              294.4
    x37       e42              0.353
    x38       e11                 -1
    x38       e34                 -1
    x39       e12                 -1
    x39       e31                 -1
    x39       e35               91.8
    x39       e37              199.2
    x40       e12                 -1
    x40       e32                 -1
    x40       e36               91.8
    x40       e38              199.2
    x41       e13                 -1
    x41       e31                 -1
    x41       e35               78.5
    x41       e37               18.4
    x42       e13                 -1
    x42       e32                 -1
    x42       e36               78.5
    x42       e38               18.4
    x43       e14                 -1
    x43       e31                 -1
    x43       e35                 65
    x43       e37               6.54
    x44       e14                 -1
    x44       e32                 -1
    x44       e36                 65
    x44       e38               6.54
    x45       e14                 -1
    x45       e33                 -1
    x45       e39                272
    x45       e40               1.48
    x46       e15                 -1
    x46       e33                 -1
    x46       e39              297.6
    x46       e40               2.83
    x47       e16                 -1
    x47       e33                 -1
    x47       e39              303.3
    x47       e40               5.05
    x48       e16                 -1
    x48       e34                 -1
    x48       e41              303.3
    x48       e42               5.05
    x49       e17                 -1
    x49       e34                 -1
    x49       e41                365
    x49       e42                 11
    x50       e18                 -1
    x50       e31                 -1
    x50       e35                104
    x50       e37               2.57
    x51       e18                 -1
    x51       e32                 -1
    x51       e36                104
    x51       e38               2.57
    x52       e19                 -1
    x52       e30                 -1
    x53       e20                 -1
    x53       e31                 -1
    x53       e35               93.7
    x53       e37                6.9
    x54       e20                 -1
    x54       e32                 -1
    x54       e36               93.7
    x54       e38                6.9
    x55       e21                 -1
    x55       e33                 -1
    x55       e39              299.1
    x55       e40               1.31
    x56       e21                 -1
    x56       e34                 -1
    x56       e41              299.1
    x56       e42               1.31
    x57       e22                 -1
    x57       e34                 -1
    x57       e41                365
    x57       e42                  6
    x58       obj                 -1
    x58       e43                  1
    x59       e43                 -1
    x59       e44                  1
    x60       e43                  1
    x60       e45                  1
    x61       e43                  1
    x61       e46                  1
RHS
    rhs       e25                100
    rhs       e26                 20
    rhs       e27                 30
    rhs       e28                200
    rhs       e29                200
BOUNDS
 FR bnd       x58
 FR bnd       x59
 FR bnd       x60
 FR bnd       x61
ENDATA
