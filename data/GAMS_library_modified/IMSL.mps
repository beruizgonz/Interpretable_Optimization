* LP written by GAMS Convert at 12/18/23 11:15:21
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        62       62        0        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*       134      134        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       359      359        0

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
COLUMNS
    x1        e1                   1
    x1        e2        0.8333333333
    x1        e3        0.6666666667
    x1        e4                 0.5
    x1        e5        0.3333333333
    x1        e6        0.1666666667
    x2        e2        0.1666666667
    x2        e3        0.3333333333
    x2        e4                 0.5
    x2        e5        0.6666666667
    x2        e6        0.8333333333
    x2        e7                   1
    x2        e8        0.8333333333
    x2        e9        0.6666666667
    x2        e10                0.5
    x2        e11       0.3333333333
    x2        e12       0.1666666667
    x3        e8        0.1666666667
    x3        e9        0.3333333333
    x3        e10                0.5
    x3        e11       0.6666666667
    x3        e12       0.8333333333
    x3        e13                  1
    x3        e14       0.8333333333
    x3        e15       0.6666666667
    x3        e16                0.5
    x3        e17       0.3333333333
    x3        e18       0.1666666667
    x4        e14       0.1666666667
    x4        e15       0.3333333333
    x4        e16                0.5
    x4        e17       0.6666666667
    x4        e18       0.8333333333
    x4        e19       2.220446e-16
    x4        e20       0.8333333333
    x4        e21       0.6666666667
    x4        e22                0.5
    x4        e23       0.3333333333
    x4        e24       0.1666666667
    x5        e19                  1
    x5        e20       0.1666666667
    x5        e21       0.3333333333
    x5        e22                0.5
    x5        e23       0.6666666667
    x5        e24       0.8333333333
    x5        e25                  1
    x5        e26       0.8333333333
    x5        e27       0.6666666667
    x5        e28                0.5
    x5        e29       0.3333333333
    x5        e30       0.1666666667
    x6        e26       0.1666666667
    x6        e27       0.3333333333
    x6        e28                0.5
    x6        e29       0.6666666667
    x6        e30       0.8333333333
    x6        e31                  1
    x6        e32       0.8333333333
    x6        e33       0.6666666667
    x6        e34                0.5
    x6        e35       0.3333333333
    x6        e36       0.1666666667
    x7        e32       0.1666666667
    x7        e33       0.3333333333
    x7        e34                0.5
    x7        e35       0.6666666667
    x7        e36       0.8333333333
    x7        e37       2.220446e-16
    x7        e38       0.8333333333
    x7        e39       0.6666666667
    x7        e40                0.5
    x7        e41       0.3333333333
    x7        e42       0.1666666667
    x8        e37                  1
    x8        e38       0.1666666667
    x8        e39       0.3333333333
    x8        e40                0.5
    x8        e41       0.6666666667
    x8        e42       0.8333333333
    x8        e43       1.332268e-15
    x8        e44       0.8333333333
    x8        e45       0.6666666667
    x8        e46                0.5
    x8        e47       0.3333333333
    x8        e48       0.1666666667
    x9        e43                  1
    x9        e44       0.1666666667
    x9        e45       0.3333333333
    x9        e46                0.5
    x9        e47       0.6666666667
    x9        e48       0.8333333333
    x9        e49                  1
    x9        e50       0.8333333333
    x9        e51       0.6666666667
    x9        e52                0.5
    x9        e53       0.3333333333
    x9        e54       0.1666666667
    x10       e50       0.1666666667
    x10       e51       0.3333333333
    x10       e52                0.5
    x10       e53       0.6666666667
    x10       e54       0.8333333333
    x10       e55                  1
    x10       e56       0.8333333333
    x10       e57       0.6666666667
    x10       e58                0.5
    x10       e59       0.3333333333
    x10       e60       0.1666666667
    x11       e56       0.1666666667
    x11       e57       0.3333333333
    x11       e58                0.5
    x11       e59       0.6666666667
    x11       e60       0.8333333333
    x11       e61                  1
    x12       e1                  -1
    x12    obj    1.0
    x13       e2                  -1
    x13    obj    1.0
    x14       e3                  -1
    x14    obj    1.0
    x15       e4                  -1
    x15    obj    1.0
    x16       e5                  -1
    x16    obj    1.0
    x17       e6                  -1
    x17    obj    1.0
    x18       e7                  -1
    x18    obj    1.0
    x19       e8                  -1
    x19    obj    1.0
    x20       e9                  -1
    x20    obj    1.0
    x21       e10                 -1
    x21    obj    1.0
    x22       e11                 -1
    x22    obj    1.0
    x23       e12                 -1
    x23    obj    1.0
    x24       e13                 -1
    x24    obj    1.0
    x25       e14                 -1
    x25    obj    1.0
    x26       e15                 -1
    x26    obj    1.0
    x27       e16                 -1
    x27    obj    1.0
    x28       e17                 -1
    x28    obj    1.0
    x29       e18                 -1
    x29    obj    1.0
    x30       e19                 -1
    x30    obj    1.0
    x31       e20                 -1
    x31    obj    1.0
    x32       e21                 -1
    x32    obj    1.0
    x33       e22                 -1
    x33    obj    1.0
    x34       e23                 -1
    x34    obj    1.0
    x35       e24                 -1
    x35    obj    1.0
    x36       e25                 -1
    x36    obj    1.0
    x37       e26                 -1
    x37    obj    1.0
    x38       e27                 -1
    x38    obj    1.0
    x39       e28                 -1
    x39    obj    1.0
    x40       e29                 -1
    x40    obj    1.0
    x41       e30                 -1
    x41    obj    1.0
    x42       e31                 -1
    x42    obj    1.0
    x43       e32                 -1
    x43    obj    1.0
    x44       e33                 -1
    x44    obj    1.0
    x45       e34                 -1
    x45    obj    1.0
    x46       e35                 -1
    x46    obj    1.0
    x47       e36                 -1
    x47    obj    1.0
    x48       e37                 -1
    x48    obj    1.0
    x49       e38                 -1
    x49    obj    1.0
    x50       e39                 -1
    x50    obj    1.0
    x51       e40                 -1
    x51    obj    1.0
    x52       e41                 -1
    x52    obj    1.0
    x53       e42                 -1
    x53    obj    1.0
    x54       e43                 -1
    x54    obj    1.0
    x55       e44                 -1
    x55    obj    1.0
    x56       e45                 -1
    x56    obj    1.0
    x57       e46                 -1
    x57    obj    1.0
    x58       e47                 -1
    x58    obj    1.0
    x59       e48                 -1
    x59    obj    1.0
    x60       e49                 -1
    x60    obj    1.0
    x61       e50                 -1
    x61    obj    1.0
    x62       e51                 -1
    x62    obj    1.0
    x63       e52                 -1
    x63    obj    1.0
    x64       e53                 -1
    x64    obj    1.0
    x65       e54                 -1
    x65    obj    1.0
    x66       e55                 -1
    x66    obj    1.0
    x67       e56                 -1
    x67    obj    1.0
    x68       e57                 -1
    x68    obj    1.0
    x69       e58                 -1
    x69    obj    1.0
    x70       e59                 -1
    x70    obj    1.0
    x71       e60                 -1
    x71    obj    1.0
    x72       e61                 -1
    x72    obj    1.0
    x73       e1                   1
    x73    obj    1.0
    x74       e2                   1
    x74    obj    1.0
    x75       e3                   1
    x75    obj    1.0
    x76       e4                   1
    x76    obj    1.0
    x77       e5                   1
    x77    obj    1.0
    x78       e6                   1
    x78    obj    1.0
    x79       e7                   1
    x79    obj    1.0
    x80       e8                   1
    x80    obj    1.0
    x81       e9                   1
    x81    obj    1.0
    x82       e10                  1
    x82    obj    1.0
    x83       e11                  1
    x83    obj    1.0
    x84       e12                  1
    x84    obj    1.0
    x85       e13                  1
    x85    obj    1.0
    x86       e14                  1
    x86    obj    1.0
    x87       e15                  1
    x87    obj    1.0
    x88       e16                  1
    x88    obj    1.0
    x89       e17                  1
    x89    obj    1.0
    x90       e18                  1
    x90    obj    1.0
    x91       e19                  1
    x91    obj    1.0
    x92       e20                  1
    x92    obj    1.0
    x93       e21                  1
    x93    obj    1.0
    x94       e22                  1
    x94    obj    1.0
    x95       e23                  1
    x95    obj    1.0
    x96       e24                  1
    x96    obj    1.0
    x97       e25                  1
    x97    obj    1.0
    x98       e26                  1
    x98    obj    1.0
    x99       e27                  1
    x99    obj    1.0
    x100      e28                  1
    x100    obj    1.0
    x101      e29                  1
    x101    obj    1.0
    x102      e30                  1
    x102    obj    1.0
    x103      e31                  1
    x103    obj    1.0
    x104      e32                  1
    x104    obj    1.0
    x105      e33                  1
    x105    obj    1.0
    x106      e34                  1
    x106    obj    1.0
    x107      e35                  1
    x107    obj    1.0
    x108      e36                  1
    x108    obj    1.0
    x109      e37                  1
    x109    obj    1.0
    x110      e38                  1
    x110    obj    1.0
    x111      e39                  1
    x111    obj    1.0
    x112      e40                  1
    x112    obj    1.0
    x113      e41                  1
    x113    obj    1.0
    x114      e42                  1
    x114    obj    1.0
    x115      e43                  1
    x115    obj    1.0
    x116      e44                  1
    x116    obj    1.0
    x117      e45                  1
    x117    obj    1.0
    x118      e46                  1
    x118    obj    1.0
    x119      e47                  1
    x119    obj    1.0
    x120      e48                  1
    x120    obj    1.0
    x121      e49                  1
    x121    obj    1.0
    x122      e50                  1
    x122    obj    1.0
    x123      e51                  1
    x123    obj    1.0
    x124      e52                  1
    x124    obj    1.0
    x125      e53                  1
    x125    obj    1.0
    x126      e54                  1
    x126    obj    1.0
    x127      e55                  1
    x127    obj    1.0
    x128      e56                  1
    x128    obj    1.0
    x129      e57                  1
    x129    obj    1.0
    x130      e58                  1
    x130    obj    1.0
    x131      e59                  1
    x131    obj    1.0
    x132      e60                  1
    x132    obj    1.0
    x133      e61                  1
    x133    obj    1.0
RHS
    rhs       e2            0.052336
    rhs       e3            0.104528
    rhs       e4            0.156434
    rhs       e5            0.207912
    rhs       e6            0.258819
    rhs       e7            0.309017
    rhs       e8            0.358368
    rhs       e9            0.406737
    rhs       e10            0.45399
    rhs       e11                0.5
    rhs       e12           0.544639
    rhs       e13           0.587785
    rhs       e14            0.62932
    rhs       e15           0.669131
    rhs       e16           0.707107
    rhs       e17           0.743145
    rhs       e18           0.777146
    rhs       e19           0.809017
    rhs       e20           0.838671
    rhs       e21           0.866025
    rhs       e22           0.891007
    rhs       e23           0.913545
    rhs       e24            0.93358
    rhs       e25           0.951057
    rhs       e26           0.965926
    rhs       e27           0.978148
    rhs       e28           0.987688
    rhs       e29           0.994522
    rhs       e30            0.99863
    rhs       e31                  1
    rhs       e32            0.99863
    rhs       e33           0.994522
    rhs       e34           0.987688
    rhs       e35           0.978148
    rhs       e36           0.965926
    rhs       e37           0.951057
    rhs       e38            0.93358
    rhs       e39           0.913545
    rhs       e40           0.891007
    rhs       e41           0.866025
    rhs       e42           0.838671
    rhs       e43           0.809017
    rhs       e44           0.777146
    rhs       e45           0.743145
    rhs       e46           0.707107
    rhs       e47           0.669131
    rhs       e48           0.629321
    rhs       e49           0.587786
    rhs       e50            0.54464
    rhs       e51           0.500001
    rhs       e52           0.453992
    rhs       e53           0.406739
    rhs       e54           0.358371
    rhs       e55           0.309021
    rhs       e56           0.258825
    rhs       e57           0.207919
    rhs       e58           0.156444
    rhs       e59           0.104541
    rhs       e60           0.052352
    rhs       e61            2.1e-05
BOUNDS
 FR bnd       x1
 FR bnd       x2
 FR bnd       x3
 FR bnd       x4
 FR bnd       x5
 FR bnd       x6
 FR bnd       x7
 FR bnd       x8
 FR bnd       x9
 FR bnd       x10
 FR bnd       x11
 FR bnd       x134
ENDATA
