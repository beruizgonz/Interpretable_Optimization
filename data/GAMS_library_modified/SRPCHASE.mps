* LP written by GAMS Convert at 12/20/23 12:34:35
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*       157       12      145        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*       168      168        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       479      479        0

NAME          Convert
*
* original model was minimizing
*
ROWS
 N  obj
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
 G  e57
 G  e58
 G  e59
 G  e60
 G  e61
 G  e62
 G  e63
 G  e64
 G  e65
 G  e66
 G  e67
 G  e68
 G  e69
 G  e70
 G  e71
 G  e72
 G  e73
 G  e74
 G  e75
 G  e76
 G  e77
 G  e78
 G  e79
 G  e80
 G  e81
 G  e82
 G  e83
 G  e84
 G  e85
 G  e86
 G  e87
 G  e88
 G  e89
 G  e90
 G  e91
 G  e92
 G  e93
 G  e94
 G  e95
 G  e96
 G  e97
 G  e98
 G  e99
 G  e100
 G  e101
 G  e102
 G  e103
 G  e104
 G  e105
 G  e106
 G  e107
 G  e108
 G  e109
 G  e110
 G  e111
 G  e112
 G  e113
 G  e114
 G  e115
 G  e116
 G  e117
 G  e118
 G  e119
 G  e120
 G  e121
 G  e122
 G  e123
 G  e124
 G  e125
 G  e126
 G  e127
 G  e128
 G  e129
 G  e130
 G  e131
 G  e132
 G  e133
 G  e134
 G  e135
 G  e136
 G  e137
 G  e138
 G  e139
 G  e140
 G  e141
 G  e142
 G  e143
 G  e144
 G  e145
 G  e146
 G  e147
 G  e148
 G  e149
 G  e150
 G  e151
 G  e152
 G  e153
 G  e154
 G  e155
 G  e156
 G  e157
COLUMNS
    x1    obj    1.0
    x1        e2                  -1
    x2    obj    0.080974038
    x2        e3                  -1
    x3    obj    0.062981662
    x3        e4                  -1
    x4    obj    0.007258759
    x4        e5                  -1
    x5    obj    0.058226987
    x5        e6                  -1
    x6    obj    0.080088346
    x6        e7                  -1
    x7    obj    0.045833727
    x7        e8                  -1
    x8    obj    0.045539077
    x8        e9                  -1
    x9    obj    0.064089624
    x9        e10                 -1
    x10    obj    0.022962374
    x10       e11                 -1
    x11    obj    0.001120421
    x11       e12                 -1
    x12    obj    0.003365723
    x12       e13                  1
    x13    obj    0.001018274
    x13       e14                  1
    x14    obj    0.002626749
    x14       e15                  1
    x15    obj    0.002836961
    x15       e16                  1
    x16    obj    2.22302e-06
    x16       e17                  1
    x17    obj    0.000756861
    x17       e18                  1
    x18    obj    0.008159167
    x18       e19                  1
    x19    obj    3.83102e-06
    x19       e20                  1
    x20    obj    0.000619544
    x20       e21                  1
    x21    obj    0.000837212
    x21       e22                  1
    x22    obj    0.006290382
    x22       e23                  1
    x23    obj    0.001100139
    x23       e24                  1
    x24    obj    0.001726339
    x24       e25                  1
    x25    obj    0.002029397
    x25       e26                  1
    x26    obj    9.72309e-05
    x26       e27                  1
    x27    obj    0.000768678
    x27       e28                  1
    x28    obj    0.003474836
    x28       e29                  1
    x29    obj    0.001492755
    x29       e30                  1
    x30    obj    0.003659345
    x30       e31                  1
    x31    obj    0.003255202
    x31       e32                  1
    x32    obj    0.000780327
    x32       e33                  1
    x33    obj    0.002187779
    x33       e34                  1
    x34    obj    0.001258034
    x34       e35                  1
    x35    obj    0.003105806
    x35       e36                  1
    x36    obj    0.000922833
    x36       e37                  1
    x37    obj    0.007683483
    x37       e38                  1
    x38    obj    0.005543173
    x38       e39                  1
    x39    obj    0.00558771
    x39       e40                  1
    x40    obj    0.002509651
    x40       e41                  1
    x41    obj    0.004351592
    x41       e42                  1
    x42    obj    0.002168921
    x42       e43                  1
    x43    obj    0.004845292
    x43       e44                  1
    x44    obj    0.006884519
    x44       e45                  1
    x45    obj    0.003545255
    x45       e46                  1
    x46    obj    0.006790498
    x46       e47                  1
    x47    obj    0.002641523
    x47       e48                  1
    x48    obj    0.001011234
    x48       e49                  1
    x49    obj    0.003854076
    x49       e50                  1
    x50    obj    0.007714409
    x50       e51                  1
    x51    obj    0.005533847
    x51       e52                  1
    x52    obj    0.000958505
    x52       e53                  1
    x53    obj    0.007209287
    x53       e54                  1
    x54    obj    0.00231358
    x54       e55                  1
    x55    obj    0.001935024
    x55       e56                  1
    x56    obj    0.01002923
    x56       e57                  1
    x57    obj    0.00586924
    x57       e58                  1
    x58    obj    0.004409703
    x58       e59                  1
    x59    obj    0.004403358
    x59       e60                  1
    x60    obj    0.006541745
    x60       e61                  1
    x61    obj    0.005634306
    x61       e62                  1
    x62    obj    0.00020032
    x62       e63                  1
    x63    obj    0.000821619
    x63       e64                  1
    x64    obj    0.002932766
    x64       e65                  1
    x65    obj    0.007138134
    x65       e66                  1
    x66    obj    0.001846357
    x66       e67                  1
    x67    obj    0.005474219
    x67       e68                  1
    x68    obj    0.001042831
    x68       e69                  1
    x69    obj    0.003682106
    x69       e70                  1
    x70    obj    0.007168203
    x70       e71                  1
    x71    obj    8.25498e-05
    x71       e72                  1
    x72    obj    0.000819677
    x72       e73                  1
    x73    obj    0.001651754
    x73       e74                  1
    x74    obj    0.010075444
    x74       e75                  1
    x75    obj    0.0044471
    x75       e76                  1
    x76    obj    0.003208281
    x76       e77                  1
    x77    obj    0.009801905
    x77       e78                  1
    x78    obj    0.005967513
    x78       e79                  1
    x79    obj    0.0003783
    x79       e80                  1
    x80    obj    0.000496441
    x80       e81                  1
    x81    obj    0.002670645
    x81       e82                  1
    x82    obj    0.001248511
    x82       e83                  1
    x83    obj    0.005143578
    x83       e84                  1
    x84    obj    0.003351371
    x84       e85                  1
    x85    obj    0.001258577
    x85       e86                  1
    x86    obj    0.006260046
    x86       e87                  1
    x87    obj    0.001110204
    x87       e88                  1
    x88    obj    0.002770812
    x88       e89                  1
    x89    obj    0.007737124
    x89       e90                  1
    x90    obj    0.003829391
    x90       e91                  1
    x91    obj    0.002830834
    x91       e92                  1
    x92    obj    0.003059354
    x92       e93                  1
    x93    obj    0.003075135
    x93       e94                  1
    x94    obj    0.002235098
    x94       e95                  1
    x95    obj    0.000275374
    x95       e96                  1
    x96    obj    0.000701755
    x96       e97                  1
    x97    obj    0.000752317
    x97       e98                  1
    x98    obj    0.005186491
    x98       e99                  1
    x99    obj    0.001337533
    x99       e100                 1
    x100    obj    0.00432509
    x100      e101                 1
    x101    obj    0.001471558
    x101      e102                 1
    x102    obj    0.010240778
    x102      e103                 1
    x103    obj    0.000818835
    x103      e104                 1
    x104    obj    0.000935247
    x104      e105                 1
    x105    obj    0.003564502
    x105      e106                 1
    x106    obj    0.002151133
    x106      e107                 1
    x107    obj    0.002342331
    x107      e108                 1
    x108    obj    0.003251539
    x108      e109                 1
    x109    obj    0.004687603
    x109      e110                 1
    x110    obj    0.001874522
    x110      e111                 1
    x111    obj    0.004021876
    x111      e112                 1
    x112    obj    0.006123246
    x112      e113                 1
    x113    obj    0.002573324
    x113      e114                 1
    x114    obj    0.007778047
    x114      e115                 1
    x115    obj    0.00179694
    x115      e116                 1
    x116    obj    0.002839193
    x116      e117                 1
    x117    obj    0.003269555
    x117      e118                 1
    x118    obj    0.004486448
    x118      e119                 1
    x119    obj    0.007156872
    x119      e120                 1
    x120    obj    0.005350101
    x120      e121                 1
    x121    obj    0.000326837
    x121      e122                 1
    x122    obj    0.005689071
    x122      e123                 1
    x123    obj    0.004389081
    x123      e124                 1
    x124    obj    0.007849965
    x124      e125                 1
    x125    obj    0.001639688
    x125      e126                 1
    x126    obj    0.008383605
    x126      e127                 1
    x127    obj    0.004391267
    x127      e128                 1
    x128    obj    0.00116942
    x128      e129                 1
    x129    obj    0.006036493
    x129      e130                 1
    x130    obj    0.001216383
    x130      e131                 1
    x131    obj    0.004544501
    x131      e132                 1
    x132    obj    0.000151924
    x132      e133                 1
    x133    obj    0.005946289
    x133      e134                 1
    x134    obj    0.000438669
    x134      e135                 1
    x135    obj    0.005065061
    x135      e136                 1
    x136    obj    0.002475522
    x136      e137                 1
    x137    obj    0.00323869
    x137      e138                 1
    x138    obj    0.005689088
    x138      e139                 1
    x139    obj    0.003021741
    x139      e140                 1
    x140    obj    0.002424714
    x140      e141                 1
    x141    obj    0.004425194
    x141      e142                 1
    x142    obj    0.00236426
    x142      e143                 1
    x143    obj    0.002347542
    x143      e144                 1
    x144    obj    0.003718339
    x144      e145                 1
    x145    obj    0.000636058
    x145      e146                 1
    x146    obj    0.001196108
    x146      e147                 1
    x147    obj    0.000731267
    x147      e148                 1
    x148    obj    0.000601064
    x148      e149                 1
    x149    obj    0.007862276
    x149      e150                 1
    x150    obj    0.002336876
    x150      e151                 1
    x151    obj    0.001880076
    x151      e152                 1
    x152    obj    0.003001201
    x152      e153                 1
    x153    obj    0.007551515
    x153      e154                 1
    x154    obj    0.003707008
    x154      e155                 1
    x155    obj    0.000751794
    x155      e156                 1
    x156    obj    0.00424612
    x156      e157                 1
    x157      e2                   1
    x157      e3                  -1
    x157      e4                  -1
    x157      e5                  -1
    x157      e6                  -1
    x157      e7                  -1
    x157      e8                  -1
    x157      e9                  -1
    x157      e10                 -1
    x157      e11                 -1
    x157      e12                 -1
    x158      e3                   1
    x158      e19                  1
    x158      e28                  1
    x158      e41                  1
    x158      e43                  1
    x158      e49                  1
    x158      e54                  1
    x158      e78                  1
    x158      e87                  1
    x158      e90                  1
    x158      e112                 1
    x158      e121                 1
    x158      e133                 1
    x158      e140                 1
    x159      e4                   1
    x159      e29                  1
    x159      e31                  1
    x159      e51                  1
    x159      e52                  1
    x159      e61                  1
    x159      e62                  1
    x159      e72                  1
    x159      e75                  1
    x159      e76                  1
    x159      e101                 1
    x159      e111                 1
    x159      e131                 1
    x159      e144                 1
    x159      e155                 1
    x160      e5                   1
    x160      e15                  1
    x160      e21                  1
    x160      e23                  1
    x160      e25                  1
    x160      e37                  1
    x160      e63                  1
    x160      e65                  1
    x160      e84                  1
    x160      e85                  1
    x160      e152                 1
    x161      e6                   1
    x161      e47                  1
    x161      e50                  1
    x161      e56                  1
    x161      e58                  1
    x161      e64                  1
    x161      e79                  1
    x161      e92                  1
    x161      e102                 1
    x161      e113                 1
    x161      e117                 1
    x161      e129                 1
    x162      e7                   1
    x162      e14                  1
    x162      e27                  1
    x162      e46                  1
    x162      e48                  1
    x162      e66                  1
    x162      e68                  1
    x162      e70                  1
    x162      e82                  1
    x162      e96                  1
    x162      e97                  1
    x162      e105                 1
    x162      e108                 1
    x162      e114                 1
    x162      e115                 1
    x162      e119                 1
    x162      e132                 1
    x162      e134                 1
    x162      e137                 1
    x162      e138                 1
    x162      e145                 1
    x162      e147                 1
    x162      e151                 1
    x162      e157                 1
    x163      e8                   1
    x163      e17                  1
    x163      e18                  1
    x163      e26                  1
    x163      e39                  1
    x163      e40                  1
    x163      e45                  1
    x163      e57                  1
    x163      e60                  1
    x163      e67                  1
    x163      e89                  1
    x163      e99                  1
    x163      e104                 1
    x163      e110                 1
    x163      e118                 1
    x163      e122                 1
    x163      e123                 1
    x163      e126                 1
    x163      e128                 1
    x163      e148                 1
    x163      e150                 1
    x164      e9                   1
    x164      e20                  1
    x164      e24                  1
    x164      e33                  1
    x164      e59                  1
    x164      e77                  1
    x164      e86                  1
    x164      e94                  1
    x164      e95                  1
    x164      e103                 1
    x164      e107                 1
    x164      e139                 1
    x164      e141                 1
    x164      e142                 1
    x164      e143                 1
    x164      e149                 1
    x164      e156                 1
    x165      e10                  1
    x165      e35                  1
    x165      e42                  1
    x165      e55                  1
    x165      e71                  1
    x165      e74                  1
    x165      e106                 1
    x165      e109                 1
    x165      e120                 1
    x165      e127                 1
    x165      e154                 1
    x166      e11                  1
    x166      e13                  1
    x166      e16                  1
    x166      e30                  1
    x166      e32                  1
    x166      e34                  1
    x166      e38                  1
    x166      e53                  1
    x166      e69                  1
    x166      e80                  1
    x166      e83                  1
    x166      e88                  1
    x166      e91                  1
    x166      e93                  1
    x166      e100                 1
    x166      e124                 1
    x166      e125                 1
    x166      e135                 1
    x166      e136                 1
    x167      e12                  1
    x167      e22                  1
    x167      e36                  1
    x167      e44                  1
    x167      e73                  1
    x167      e81                  1
    x167      e98                  1
    x167      e116                 1
    x167      e130                 1
    x167      e146                 1
    x167      e153                 1
RHS
    rhs       e13                  1
    rhs       e14                  1
    rhs       e15                  1
    rhs       e16                  1
    rhs       e17                  1
    rhs       e18                  1
    rhs       e19                  1
    rhs       e20                  1
    rhs       e21                  1
    rhs       e22                  1
    rhs       e23                  1
    rhs       e24                  1
    rhs       e25                  1
    rhs       e26                  1
    rhs       e27                  1
    rhs       e28                  1
    rhs       e29                  1
    rhs       e30                  1
    rhs       e31                  1
    rhs       e32                  1
    rhs       e33                  1
    rhs       e34                  1
    rhs       e35                  1
    rhs       e36                  1
    rhs       e37                  1
    rhs       e38                  1
    rhs       e39                  1
    rhs       e40                  1
    rhs       e41                  1
    rhs       e42                  1
    rhs       e43                  1
    rhs       e44                  1
    rhs       e45                  1
    rhs       e46                  1
    rhs       e47                  1
    rhs       e48                  1
    rhs       e49                  1
    rhs       e50                  1
    rhs       e51                  1
    rhs       e52                  1
    rhs       e53                  1
    rhs       e54                  1
    rhs       e55                  1
    rhs       e56                  1
    rhs       e57                  1
    rhs       e58                  1
    rhs       e59                  1
    rhs       e60                  1
    rhs       e61                  1
    rhs       e62                  1
    rhs       e63                  1
    rhs       e64                  1
    rhs       e65                  1
    rhs       e66                  1
    rhs       e67                  1
    rhs       e68                  1
    rhs       e69                  1
    rhs       e70                  1
    rhs       e71                  1
    rhs       e72                  1
    rhs       e73                  1
    rhs       e74                  1
    rhs       e75                  1
    rhs       e76                  1
    rhs       e77                  1
    rhs       e78                  1
    rhs       e79                  1
    rhs       e80                  1
    rhs       e81                  1
    rhs       e82                  1
    rhs       e83                  1
    rhs       e84                  1
    rhs       e85                  1
    rhs       e86                  1
    rhs       e87                  1
    rhs       e88                  1
    rhs       e89                  1
    rhs       e90                  1
    rhs       e91                  1
    rhs       e92                  1
    rhs       e93                  1
    rhs       e94                  1
    rhs       e95                  1
    rhs       e96                  1
    rhs       e97                  1
    rhs       e98                  1
    rhs       e99                  1
    rhs       e100                 1
    rhs       e101                 1
    rhs       e102                 1
    rhs       e103                 1
    rhs       e104                 1
    rhs       e105                 1
    rhs       e106                 1
    rhs       e107                 1
    rhs       e108                 1
    rhs       e109                 1
    rhs       e110                 1
    rhs       e111                 1
    rhs       e112                 1
    rhs       e113                 1
    rhs       e114                 1
    rhs       e115                 1
    rhs       e116                 1
    rhs       e117                 1
    rhs       e118                 1
    rhs       e119                 1
    rhs       e120                 1
    rhs       e121                 1
    rhs       e122                 1
    rhs       e123                 1
    rhs       e124                 1
    rhs       e125                 1
    rhs       e126                 1
    rhs       e127                 1
    rhs       e128                 1
    rhs       e129                 1
    rhs       e130                 1
    rhs       e131                 1
    rhs       e132                 1
    rhs       e133                 1
    rhs       e134                 1
    rhs       e135                 1
    rhs       e136                 1
    rhs       e137                 1
    rhs       e138                 1
    rhs       e139                 1
    rhs       e140                 1
    rhs       e141                 1
    rhs       e142                 1
    rhs       e143                 1
    rhs       e144                 1
    rhs       e145                 1
    rhs       e146                 1
    rhs       e147                 1
    rhs       e148                 1
    rhs       e149                 1
    rhs       e150                 1
    rhs       e151                 1
    rhs       e152                 1
    rhs       e153                 1
    rhs       e154                 1
    rhs       e155                 1
    rhs       e156                 1
    rhs       e157                 1
BOUNDS
 FR bnd       x168
ENDATA
