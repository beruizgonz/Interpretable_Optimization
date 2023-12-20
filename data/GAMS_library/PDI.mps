* LP written by GAMS Convert at 12/18/23 13:07:35
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        69       69        0        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*       134      134        0        0        0        0        0        0
* FX      4
*
* Nonzero counts
*     Total    const       NL
*       396      396        0

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
 E  e38
 E  e39
 E  e40
 E  e41
 E  e42
 E  e43
 E  e44
 E  e45
 E  e46
 E  e47
 E  e48
 E  e49
 E  e50
 E  e51
 E  e52
 E  e53
 E  e54
 E  e55
 E  e56
 E  e57
 E  e58
 E  e59
 E  e60
 E  e61
 E  e62
 E  e63
 E  e64
 E  e65
 E  e66
 E  e67
 E  e68
 E  e69
COLUMNS
    x1        e1                  -1
    x1        e17                 -1
    x1        e66                -10
    x2        e2                  -1
    x2        e18                 -1
    x2        e66                -10
    x3        e3                  -1
    x3        e19                 -1
    x3        e66                -10
    x4        e4                  -1
    x4        e20                 -1
    x4        e66                -10
    x5        e5                  -1
    x5        e17                 -1
    x5        e66                -12
    x6        e6                  -1
    x6        e18                 -1
    x6        e66                -12
    x7        e7                  -1
    x7        e19                 -1
    x7        e66                -12
    x8        e8                  -1
    x8        e20                 -1
    x8        e66                -12
    x9        e5                  -1
    x9        e21                 -1
    x9        e66                 -8
    x10       e6                  -1
    x10       e22                 -1
    x10       e66                 -8
    x11       e7                  -1
    x11       e23                 -1
    x11       e66                 -8
    x12       e8                  -1
    x12       e24                 -1
    x12       e66                 -8
    x13       e9                  -1
    x13       e21                 -1
    x13       e66                 -4
    x14       e10                 -1
    x14       e22                 -1
    x14       e66                 -4
    x15       e11                 -1
    x15       e23                 -1
    x15       e66                 -4
    x16       e12                 -1
    x16       e24                 -1
    x16       e66                 -4
    x17       e13                 -1
    x17       e21                 -1
    x17       e66                 -5
    x18       e14                 -1
    x18       e22                 -1
    x18       e66                 -5
    x19       e15                 -1
    x19       e23                 -1
    x19       e66                 -5
    x20       e16                 -1
    x20       e24                 -1
    x20       e66                 -5
    x21       e9                  -1
    x21       e25                 -1
    x21       e66                 -6
    x22       e10                 -1
    x22       e26                 -1
    x22       e66                 -6
    x23       e11                 -1
    x23       e27                 -1
    x23       e66                 -6
    x24       e12                 -1
    x24       e28                 -1
    x24       e66                 -6
    x25       e13                 -1
    x25       e25                 -1
    x25       e66                 -8
    x26       e14                 -1
    x26       e26                 -1
    x26       e66                 -8
    x27       e15                 -1
    x27       e27                 -1
    x27       e66                 -8
    x28       e16                 -1
    x28       e28                 -1
    x28       e66                 -8
    x29       e29                  1
    x29       e45                  1
    x29       e65                -70
    x29       e66                -15
    x30       e30                  1
    x30       e46                  1
    x30       e65                -70
    x30       e66                -15
    x31       e31                  1
    x31       e47                  1
    x31       e65                -77
    x31       e66                -15
    x32       e32                  1
    x32       e48                  1
    x32       e65                -77
    x32       e66                -15
    x33       e29                  1
    x33       e49                  1
    x33       e65                -68
    x33       e66                -19
    x34       e30                  1
    x34       e50                  1
    x34       e65                -68
    x34       e66                -19
    x35       e31                  1
    x35       e51                  1
    x35       e65              -74.8
    x35       e66                -19
    x36       e32                  1
    x36       e52                  1
    x36       e65              -74.8
    x36       e66                -19
    x37       e33                  1
    x37       e49                  1
    x37       e65                -68
    x37       e66                -20
    x38       e34                  1
    x38       e50                  1
    x38       e65                -68
    x38       e66                -20
    x39       e35                  1
    x39       e51                  1
    x39       e65              -74.8
    x39       e66                -20
    x40       e36                  1
    x40       e52                  1
    x40       e65              -74.8
    x40       e66                -20
    x41       e33                  1
    x41       e53                  1
    x41       e65                -65
    x41       e66                -22
    x42       e34                  1
    x42       e54                  1
    x42       e65                -65
    x42       e66                -22
    x43       e35                  1
    x43       e55                  1
    x43       e65              -71.5
    x43       e66                -22
    x44       e36                  1
    x44       e56                  1
    x44       e65              -71.5
    x44       e66                -22
    x45       e33                  1
    x45       e57                  1
    x45       e65                -72
    x45       e66                -18
    x46       e34                  1
    x46       e58                  1
    x46       e65                -72
    x46       e66                -18
    x47       e35                  1
    x47       e59                  1
    x47       e65              -79.2
    x47       e66                -18
    x48       e36                  1
    x48       e60                  1
    x48       e65              -79.2
    x48       e66                -18
    x49       e37                  1
    x49       e49                  1
    x49       e65                -68
    x49       e66                -16
    x50       e38                  1
    x50       e50                  1
    x50       e65                -68
    x50       e66                -16
    x51       e39                  1
    x51       e51                  1
    x51       e65              -74.8
    x51       e66                -16
    x52       e40                  1
    x52       e52                  1
    x52       e65              -74.8
    x52       e66                -16
    x53       e37                  1
    x53       e57                  1
    x53       e65                -72
    x53       e66                -18
    x54       e38                  1
    x54       e58                  1
    x54       e65                -72
    x54       e66                -18
    x55       e39                  1
    x55       e59                  1
    x55       e65              -79.2
    x55       e66                -18
    x56       e40                  1
    x56       e60                  1
    x56       e65              -79.2
    x56       e66                -18
    x57       e37                  1
    x57       e61                  1
    x57       e65                -71
    x57       e66                -19
    x58       e38                  1
    x58       e62                  1
    x58       e65                -71
    x58       e66                -19
    x59       e39                  1
    x59       e63                  1
    x59       e65              -78.1
    x59       e66                -19
    x60       e40                  1
    x60       e64                  1
    x60       e65              -78.1
    x60       e66                -19
    x61       e41                  1
    x61       e57                  1
    x61       e65                -72
    x61       e66                -15
    x62       e42                  1
    x62       e58                  1
    x62       e65                -72
    x62       e66                -15
    x63       e43                  1
    x63       e59                  1
    x63       e65              -79.2
    x63       e66                -15
    x64       e44                  1
    x64       e60                  1
    x64       e65              -79.2
    x64       e66                -15
    x65       e41                  1
    x65       e61                  1
    x65       e65                -71
    x65       e66                -21
    x66       e42                  1
    x66       e62                  1
    x66       e65                -71
    x66       e66                -21
    x67       e43                  1
    x67       e63                  1
    x67       e65              -78.1
    x67       e66                -21
    x68       e44                  1
    x68       e64                  1
    x68       e65              -78.1
    x68       e66                -21
    x69       e17                  1
    x69       e67                -35
    x70       e18                  1
    x70       e67                -36
    x71       e19                  1
    x71       e67                -37
    x72       e20                  1
    x72       e67                -38
    x73       e21                  1
    x73       e67                -40
    x74       e22                  1
    x74       e67                -41
    x75       e23                  1
    x75       e67                -42
    x76       e24                  1
    x76       e67                -43
    x77       e25                  1
    x77       e67                -38
    x78       e26                  1
    x78       e67                -39
    x79       e27                  1
    x79       e67                -40
    x80       e28                  1
    x80       e67                -41
    x81       e17                  1
    x81       e67                -45
    x82       e18                  1
    x82       e67                -46
    x83       e19                  1
    x83       e67                -47
    x84       e20                  1
    x84       e67                -49
    x85       e21                  1
    x85       e67                -43
    x86       e22                  1
    x86       e67                -44
    x87       e23                  1
    x87       e67                -45
    x88       e24                  1
    x88       e67                -47
    x89       e25                  1
    x90       e26                  1
    x90       e67                 -1
    x91       e27                  1
    x91       e67                 -2
    x92       e28                  1
    x92       e67                 -4
    x93       e2                  -1
    x93       e29                  1
    x93       e68                 -2
    x94       e3                  -1
    x94       e30                  1
    x94       e68                 -2
    x95       e4                  -1
    x95       e31                  1
    x95       e68                 -2
    x96       e32                  1
    x96       e68                 -2
    x97       e6                  -1
    x97       e33                  1
    x97       e68                 -2
    x98       e7                  -1
    x98       e34                  1
    x98       e68                 -2
    x99       e8                  -1
    x99       e35                  1
    x99       e68                 -2
    x100      e36                  1
    x100      e68                 -2
    x101      e10                 -1
    x101      e37                  1
    x101      e68                 -1
    x102      e11                 -1
    x102      e38                  1
    x102      e68                 -1
    x103      e12                 -1
    x103      e39                  1
    x103      e68                 -1
    x104      e40                  1
    x104      e68                 -1
    x105      e14                 -1
    x105      e41                  1
    x105      e68                 -3
    x106      e15                 -1
    x106      e42                  1
    x106      e68                 -3
    x107      e16                 -1
    x107      e43                  1
    x107      e68                 -3
    x108      e44                  1
    x108      e68                 -3
    x109      e45                 -1
    x109      e46                 -1
    x109      e47                 -1
    x109      e48                 -1
    x110      e49                 -1
    x110      e50                 -1
    x110      e51                 -1
    x110      e52                 -1
    x111      e53                 -1
    x111      e54                 -1
    x111      e55                 -1
    x111      e56                 -1
    x112      e57                 -1
    x112      e58                 -1
    x112      e59                 -1
    x112      e60                 -1
    x113      e61                 -1
    x113      e62                 -1
    x113      e63                 -1
    x113      e64                 -1
    x114      e1                   1
    x114      e29                 -1
    x115      e2                   1
    x115      e30                 -1
    x116      e3                   1
    x116      e31                 -1
    x117      e4                   1
    x117      e32                 -1
    x118      e5                   1
    x118      e33                 -1
    x119      e6                   1
    x119      e34                 -1
    x120      e7                   1
    x120      e35                 -1
    x121      e8                   1
    x121      e36                 -1
    x122      e9                   1
    x122      e37                 -1
    x123      e10                  1
    x123      e38                 -1
    x124      e11                  1
    x124      e39                 -1
    x125      e12                  1
    x125      e40                 -1
    x126      e13                  1
    x126      e41                 -1
    x127      e14                  1
    x127      e42                 -1
    x128      e15                  1
    x128      e43                 -1
    x129      e16                  1
    x129      e44                 -1
    x130      obj                 -1
    x130      e69                  1
    x131      e65                  1
    x131      e69                 -1
    x132      e66                  1
    x132      e69                  1
    x133      e67                  1
    x133      e69                  1
    x134      e68                  1
    x134      e69                  1
RHS
    rhs       e69                 10
BOUNDS
 UP bnd       x69               5000
 UP bnd       x70               5000
 UP bnd       x71               5000
 UP bnd       x72               5000
 LO bnd       x73               1200
 UP bnd       x73               3000
 LO bnd       x74               1200
 UP bnd       x74               3000
 LO bnd       x75               1200
 UP bnd       x75               3000
 LO bnd       x76               1200
 UP bnd       x76               3000
 LO bnd       x77                700
 UP bnd       x77               1500
 LO bnd       x78                700
 UP bnd       x78               1500
 LO bnd       x79                700
 UP bnd       x79               1500
 LO bnd       x80                700
 UP bnd       x80               1500
 UP bnd       x81               1000
 UP bnd       x82               1000
 UP bnd       x83               1000
 UP bnd       x84               1000
 UP bnd       x85                500
 UP bnd       x86                500
 UP bnd       x87                500
 UP bnd       x88                500
 FX bnd       x89                  0
 FX bnd       x90                  0
 FX bnd       x91                  0
 FX bnd       x92                  0
 LO bnd       x96                200
 LO bnd       x100               200
 LO bnd       x104               200
 LO bnd       x108               200
 LO bnd       x109              2000
 UP bnd       x109              2500
 UP bnd       x110              2500
 LO bnd       x111              2000
 UP bnd       x111              3000
 LO bnd       x112              1500
 UP bnd       x112              2000
 LO bnd       x113              1500
 UP bnd       x113              3000
 UP bnd       x114              3000
 UP bnd       x115              3000
 UP bnd       x116              3000
 UP bnd       x117              3000
 UP bnd       x118              2500
 UP bnd       x119              2500
 UP bnd       x120              2500
 UP bnd       x121              2500
 UP bnd       x122              4000
 UP bnd       x123              4000
 UP bnd       x124              4000
 UP bnd       x125              4000
 UP bnd       x126              2500
 UP bnd       x127              2500
 UP bnd       x128              2500
 UP bnd       x129              2500
 FR bnd       x130
 FR bnd       x131
 FR bnd       x132
 FR bnd       x133
 FR bnd       x134
ENDATA
