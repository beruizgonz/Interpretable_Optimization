* LP written by GAMS Convert at 12/18/23 11:46:39
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        57        1       56        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        31       31        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       135      135        0

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
 G  e25
 G  e26
 G  e27
 G  e28
 G  e29
 G  e30
 G  e31
 G  e32
 G  e33
 G  e34
 G  e35
 G  e36
 G  e37
 G  e38
 G  e39
 G  e40
 G  e41
 G  e42
 G  e43
 G  e44
 G  e45
 G  e46
 G  e47
 G  e48
 G  e49
 G  e50
 G  e51
 G  e52
 G  e53
 G  e54
 G  e55
 G  e56
 E  e57
COLUMNS
    x1        e1                   1
    x2        e2                   1
    x2        e15                  1
    x3        e3                   1
    x3        e16                  1
    x4        e17                  1
    x4        e57               1500
    x5        e4                   1
    x5        e43                  1
    x6        e5                   1
    x6        e18                  1
    x6        e29                  1
    x6        e44                  1
    x6        e57              -1000
    x7        e6                   1
    x7        e19                  1
    x7        e30                  1
    x7        e45                  1
    x8        e20                  1
    x8        e31                  1
    x8        e57               1500
    x9        e7                   1
    x9        e46                  1
    x9        e57               1000
    x10       e8                   1
    x10       e21                  1
    x10       e32                  1
    x10       e47                  1
    x10       e57               1000
    x11       e9                   1
    x11       e22                  1
    x11       e33                  1
    x11       e48                  1
    x11       e57               1500
    x12       e23                  1
    x12       e34                  1
    x12       e57               2000
    x13       e49                  1
    x13       e57               1500
    x14       e35                  1
    x14       e50                  1
    x14       e57               1500
    x15       e36                  1
    x15       e51                  1
    x15       e57               2000
    x16       e37                  1
    x16       e57               2500
    x17       e1                  -1
    x17       e10                  1
    x17       e15                 -1
    x17       e29                 -1
    x17       e43                 -1
    x17       e57              -2000
    x18       e2                  -1
    x18       e11                  1
    x18       e16                 -1
    x18       e24                  1
    x18       e30                 -1
    x18       e44                 -1
    x18       e57              -2000
    x19       e3                  -1
    x19       e17                 -1
    x19       e25                  1
    x19       e31                 -1
    x19       e45                 -1
    x19       e57               2000
    x20       e4                  -1
    x20       e12                  1
    x20       e18                 -1
    x20       e32                 -1
    x20       e46                 -1
    x20       e52                  1
    x21       e5                  -1
    x21       e13                  1
    x21       e19                 -1
    x21       e26                  1
    x21       e33                 -1
    x21       e38                  1
    x21       e47                 -1
    x21       e53                  1
    x22       e6                  -1
    x22       e20                 -1
    x22       e27                  1
    x22       e34                 -1
    x22       e39                  1
    x22       e48                 -1
    x22       e57               4000
    x23       e7                  -1
    x23       e21                 -1
    x23       e35                 -1
    x23       e49                 -1
    x23       e54                  1
    x23       e57               2000
    x24       e8                  -1
    x24       e22                 -1
    x24       e36                 -1
    x24       e40                  1
    x24       e50                 -1
    x24       e55                  1
    x24       e57               2000
    x25       e9                  -1
    x25       e23                 -1
    x25       e37                 -1
    x25       e41                  1
    x25       e51                 -1
    x25       e57               5000
    x26       e10                 -1
    x26       e14                  1
    x26       e24                 -1
    x26       e38                 -1
    x26       e52                 -1
    x26       e57             -16000
    x27       e11                 -1
    x27       e25                 -1
    x27       e28                  1
    x27       e39                 -1
    x27       e53                 -1
    x27       e57              -4000
    x28       e12                 -1
    x28       e26                 -1
    x28       e40                 -1
    x28       e54                 -1
    x28       e56                  1
    x28       e57              -2000
    x29       e13                 -1
    x29       e27                 -1
    x29       e41                 -1
    x29       e42                  1
    x29       e55                 -1
    x30       e14                 -1
    x30       e28                 -1
    x30       e42                 -1
    x30       e56                 -1
    x30       e57              -2000
    x31       obj                 -1
    x31       e57                  1
RHS
BOUNDS
 UP bnd       x1                   1
 UP bnd       x2                   1
 UP bnd       x3                   1
 UP bnd       x4                   1
 UP bnd       x5                   1
 UP bnd       x6                   1
 UP bnd       x7                   1
 UP bnd       x8                   1
 UP bnd       x9                   1
 UP bnd       x10                  1
 UP bnd       x11                  1
 UP bnd       x12                  1
 UP bnd       x13                  1
 UP bnd       x14                  1
 UP bnd       x15                  1
 UP bnd       x16                  1
 UP bnd       x17                  1
 UP bnd       x18                  1
 UP bnd       x19                  1
 UP bnd       x20                  1
 UP bnd       x21                  1
 UP bnd       x22                  1
 UP bnd       x23                  1
 UP bnd       x24                  1
 UP bnd       x25                  1
 UP bnd       x26                  1
 UP bnd       x27                  1
 UP bnd       x28                  1
 UP bnd       x29                  1
 UP bnd       x30                  1
 FR bnd       x31
ENDATA
