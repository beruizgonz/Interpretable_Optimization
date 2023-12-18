* LP written by GAMS Convert at 12/18/23 11:45:26
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        74        5       43       26        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        78       78        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       230      230        0

NAME          Convert
*
* original model was minimizing
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
 L  e41
 L  e42
 L  e43
 L  e44
 L  e45
 L  e46
 L  e47
 L  e48
 L  e49
 L  e50
 L  e51
 L  e52
 L  e53
 L  e54
 L  e55
 L  e56
 L  e57
 L  e58
 L  e59
 L  e60
 L  e61
 L  e62
 L  e63
 L  e64
 L  e65
 G  e66
 G  e67
 G  e68
 L  e69
 E  e70
 E  e71
 E  e72
 E  e73
 E  e74
COLUMNS
    x1        e6                   1
    x1        e16              -1.58
    x1        e21              -0.63
    x1        e41                  1
    x2        e7                   1
    x2        e17              -1.58
    x2        e22              -0.63
    x2        e42                  1
    x3        e8                   1
    x3        e18              -1.58
    x3        e23              -0.63
    x3        e43                  1
    x4        e9                   1
    x4        e19              -1.58
    x4        e24              -0.63
    x4        e44                  1
    x5        e10                  1
    x5        e20              -1.58
    x5        e25              -0.63
    x5        e45                  1
    x6        e11                  1
    x6        e16              -1.38
    x6        e26              -0.57
    x6        e56                  1
    x7        e12                  1
    x7        e17              -1.38
    x7        e27              -0.57
    x7        e57                  1
    x8        e13                  1
    x8        e18              -1.38
    x8        e28              -0.57
    x8        e58                  1
    x9        e14                  1
    x9        e19              -1.38
    x9        e29              -0.57
    x9        e59                  1
    x10       e15                  1
    x10       e20              -1.38
    x10       e30              -0.57
    x10       e60                  1
    x11       e1                   1
    x11       e6               -0.77
    x11       e36              -0.33
    x11       e46                  1
    x12       e2                   1
    x12       e7               -0.77
    x12       e37              -0.33
    x12       e47                  1
    x13       e3                   1
    x13       e8               -0.77
    x13       e38              -0.33
    x13       e48                  1
    x14       e4                   1
    x14       e9               -0.77
    x14       e39              -0.33
    x14       e49                  1
    x15       e5                   1
    x15       e10              -0.77
    x15       e40              -0.33
    x15       e50                  1
    x16       e1                   1
    x16       e11              -1.09
    x16       e31              -0.58
    x16       e61                  1
    x17       e2                   1
    x17       e12              -1.09
    x17       e32              -0.58
    x17       e62                  1
    x18       e3                   1
    x18       e13              -1.09
    x18       e33              -0.58
    x18       e63                  1
    x19       e4                   1
    x19       e14              -1.09
    x19       e34              -0.58
    x19       e64                  1
    x20       e5                   1
    x20       e15              -1.09
    x20       e35              -0.58
    x20       e65                  1
    x21       e1                   1
    x21       e6               -0.95
    x21       e36              -0.12
    x21       e51                  1
    x22       e2                   1
    x22       e7               -0.95
    x22       e37              -0.12
    x22       e52                  1
    x23       e3                   1
    x23       e8               -0.95
    x23       e38              -0.12
    x23       e53                  1
    x24       e4                   1
    x24       e9               -0.95
    x24       e39              -0.12
    x24       e54                  1
    x25       e5                   1
    x25       e10              -0.95
    x25       e40              -0.12
    x25       e55                  1
    x26       e1                  -1
    x26       e66                  1
    x26       e72           -12.5936
    x27       e1                  -1
    x27       e67                  1
    x27       e72            -4.3112
    x28       e1                  -1
    x28       e68                  1
    x28       e72             -11.93
    x29       e2                  -1
    x29       e66                  1
    x29       e72           -11.0228
    x30       e2                  -1
    x30       e67                  1
    x31       e2                  -1
    x31       e68                  1
    x31       e72            -11.132
    x32       e3                  -1
    x32       e66                  1
    x32       e72            -9.3596
    x33       e3                  -1
    x33       e67                  1
    x33       e72            -13.442
    x34       e3                  -1
    x34       e68                  1
    x34       e72            -8.3936
    x35       e4                  -1
    x35       e66                  1
    x35       e72           -11.0228
    x36       e4                  -1
    x36       e67                  1
    x37       e4                  -1
    x37       e68                  1
    x37       e72            -11.132
    x38       e5                  -1
    x38       e66                  1
    x38       e72             -4.034
    x39       e5                  -1
    x39       e67                  1
    x39       e72            -11.594
    x40       e5                  -1
    x40       e68                  1
    x40       e72             -8.864
    x41       e16                  1
    x41       e71              -18.7
    x42       e17                  1
    x42       e71              -18.7
    x43       e18                  1
    x43       e71              -18.7
    x44       e19                  1
    x44       e71              -18.7
    x45       e20                  1
    x45       e71              -18.7
    x46       e21                  1
    x46       e71             -52.17
    x47       e22                  1
    x47       e71             -52.17
    x48       e23                  1
    x48       e71             -52.17
    x49       e24                  1
    x49       e71             -52.17
    x50       e25                  1
    x50       e71             -52.17
    x51       e26                  1
    x51       e71                -14
    x52       e27                  1
    x52       e71                -14
    x53       e28                  1
    x53       e71                -14
    x54       e29                  1
    x54       e71                -14
    x55       e30                  1
    x55       e71                -14
    x56       e31                  1
    x56       e71                -24
    x57       e32                  1
    x57       e71                -24
    x58       e33                  1
    x58       e71                -24
    x59       e34                  1
    x59       e71                -24
    x60       e35                  1
    x60       e71                -24
    x61       e36                  1
    x61       e71               -105
    x62       e37                  1
    x62       e71               -105
    x63       e38                  1
    x63       e71               -105
    x64       e39                  1
    x64       e71               -105
    x65       e40                  1
    x65       e71               -105
    x66       e66                  1
    x66       e72            -6.0752
    x66       e73               -150
    x67       e67                  1
    x67       e72            -6.8564
    x67       e73               -150
    x68       e68                  1
    x68       e72                 -5
    x68       e73               -150
    x69       e1                  -1
    x69       e69                  1
    x69       e72            -8.6876
    x69       e74               -140
    x70       e2                  -1
    x70       e69                  1
    x70       e72            -6.8564
    x70       e74               -140
    x71       e3                  -1
    x71       e69                  1
    x71       e74               -140
    x72       e4                  -1
    x72       e69                  1
    x72       e72            -6.8564
    x72       e74               -140
    x73       e5                  -1
    x73       e69                  1
    x73       e72             -5.126
    x73       e74               -140
    x74       obj                  1
    x74       e70                  1
    x75       e70                 -1
    x75       e71                  1
    x76       e70                 -1
    x76       e72                  1
    x77       e70                 -1
    x77       e73                  1
    x78       e70                  1
    x78       e74                  1
RHS
    rhs       e41               3.25
    rhs       e42                1.4
    rhs       e43                1.1
    rhs       e46                1.5
    rhs       e47               0.85
    rhs       e51               2.07
    rhs       e52                1.5
    rhs       e53                1.3
    rhs       e59               0.98
    rhs       e60                  1
    rhs       e64               1.13
    rhs       e65               0.56
    rhs       e66            4.01093
    rhs       e67            2.18778
    rhs       e68            1.09389
    rhs       e69                  1
BOUNDS
 FR bnd       x74
 FR bnd       x75
 FR bnd       x76
 FR bnd       x77
 FR bnd       x78
ENDATA
