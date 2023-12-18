* LP written by GAMS Convert at 12/18/23 11:51:24
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        21       21        0        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*       319      319        0        0        0        0        0        0
* FX    188
*
* Nonzero counts
*     Total    const       NL
*       575      575        0

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
COLUMNS
    x1        e1              -91.75
    x1        e2                  -1
    x1        e7                1.49
    x2        e1              -77.34
    x2        e2                  -1
    x2        e8                1.46
    x3        e1              -30.19
    x3        e2                  -1
    x3        e9                0.73
    x4        e1              -77.92
    x4        e2                  -1
    x4        e10                1.1
    x5        e1              -97.19
    x5        e2                  -1
    x5        e11                  1
    x6        e1              -19.42
    x6        e2                  -1
    x6        e12               0.94
    x7        e1              -67.15
    x7        e2                  -1
    x7        e13               1.09
    x8        e1              -75.73
    x8        e2                  -1
    x8        e14               1.46
    x9        e1              -53.54
    x9        e2                  -1
    x9        e15               0.73
    x10       e1               -3.46
    x10       e2                  -1
    x10       e16               1.38
    x11       e1              -50.01
    x11       e2                  -1
    x11       e18               0.51
    x12       e1               -5.51
    x12       e2                  -1
    x12       e20               1.36
    x13       e2                  -1
    x14       e2                  -1
    x15       e2                  -1
    x16       e2                  -1
    x17       e2                  -1
    x18       e2                  -1
    x19       e2                  -1
    x20       e2                  -1
    x21       e2                  -1
    x22       e2                  -1
    x23       e2                  -1
    x24       e1              -29.72
    x24       e3                  -1
    x24       e7                   1
    x25       e1              -28.05
    x25       e3                  -1
    x25       e8                0.83
    x26       e1              -94.09
    x26       e3                  -1
    x26       e9                 1.1
    x27       e1              -58.19
    x27       e3                  -1
    x27       e10                1.3
    x28       e1              -92.53
    x28       e3                  -1
    x28       e11               0.54
    x29       e1               -3.57
    x29       e3                  -1
    x29       e12               1.27
    x30       e1              -55.49
    x30       e3                  -1
    x30       e13               1.48
    x31       e1              -71.42
    x31       e3                  -1
    x31       e15               1.38
    x32       e1              -77.52
    x32       e3                  -1
    x32       e16               0.88
    x33       e1              -44.88
    x33       e3                  -1
    x33       e17               1.11
    x34       e1              -80.58
    x34       e3                  -1
    x34       e18               0.86
    x35       e1              -96.14
    x35       e3                  -1
    x35       e19               1.43
    x36       e1              -77.72
    x36       e3                  -1
    x36       e20               0.95
    x37       e3                  -1
    x38       e3                  -1
    x39       e3                  -1
    x40       e3                  -1
    x41       e3                  -1
    x42       e3                  -1
    x43       e3                  -1
    x44       e3                  -1
    x45       e3                  -1
    x46       e3                  -1
    x47       e3                  -1
    x48       e3                  -1
    x49       e3                  -1
    x50       e3                  -1
    x51       e3                  -1
    x52       e3                  -1
    x53       e3                  -1
    x54       e1              -32.41
    x54       e4                  -1
    x54       e7                1.35
    x55       e1              -28.94
    x55       e4                  -1
    x55       e8                0.65
    x56       e1                -100
    x56       e4                  -1
    x56       e9                   1
    x57       e1              -45.82
    x57       e4                  -1
    x57       e10                0.9
    x58       e1              -65.21
    x58       e4                  -1
    x58       e11                1.1
    x59       e1              -94.55
    x59       e4                  -1
    x59       e12               0.79
    x60       e1              -22.82
    x60       e4                  -1
    x60       e13               1.16
    x61       e1              -79.67
    x61       e4                  -1
    x61       e14                0.7
    x62       e1               -7.39
    x62       e4                  -1
    x62       e15               1.02
    x63       e1               -43.1
    x63       e4                  -1
    x63       e16               0.94
    x64       e1              -72.52
    x64       e4                  -1
    x64       e17                1.5
    x65       e1              -88.73
    x65       e4                  -1
    x65       e18               1.13
    x66       e1              -79.01
    x66       e4                  -1
    x66       e19               0.76
    x67       e1              -55.78
    x67       e4                  -1
    x67       e20               0.96
    x68       e4                  -1
    x69       e4                  -1
    x70       e4                  -1
    x71       e1               -1.77
    x71       e5                  -1
    x71       e7                0.94
    x72       e1               -40.3
    x72       e5                  -1
    x72       e8                1.22
    x73       e1              -44.57
    x73       e5                  -1
    x73       e9                0.85
    x74       e1              -39.42
    x74       e5                  -1
    x74       e10               1.24
    x75       e1              -16.21
    x75       e5                  -1
    x75       e11               1.04
    x76       e1              -80.34
    x76       e5                  -1
    x76       e12                0.6
    x77       e1              -70.37
    x77       e5                  -1
    x77       e13                  1
    x78       e1               -1.36
    x78       e5                  -1
    x78       e14               0.83
    x79       e1              -40.31
    x79       e5                  -1
    x79       e15               0.71
    x80       e1              -93.02
    x80       e5                  -1
    x80       e16               1.05
    x81       e1              -66.73
    x81       e5                  -1
    x81       e17               0.83
    x82       e1              -65.06
    x82       e5                  -1
    x82       e18               1.23
    x83       e1              -15.93
    x83       e5                  -1
    x83       e19               0.96
    x84       e1                -100
    x84       e5                  -1
    x84       e20                  1
    x85       e5                  -1
    x86       e5                  -1
    x87       e5                  -1
    x88       e5                  -1
    x89       e1              -43.95
    x89       e6                  -1
    x89       e7                0.51
    x90       e1              -50.85
    x90       e6                  -1
    x90       e8                1.37
    x91       e1              -84.53
    x91       e6                  -1
    x91       e9                1.06
    x92       e1              -49.91
    x92       e6                  -1
    x92       e10               0.58
    x93       e1              -87.94
    x93       e6                  -1
    x93       e11                  1
    x94       e1              -21.65
    x94       e6                  -1
    x94       e12                  1
    x95       e1                -100
    x95       e6                  -1
    x95       e13                  1
    x96       e1              -17.09
    x96       e6                  -1
    x96       e14               1.14
    x97       e1               -16.9
    x97       e6                  -1
    x97       e15               0.75
    x98       e1              -78.05
    x98       e6                  -1
    x98       e17               1.48
    x99       e1              -56.37
    x99       e6                  -1
    x99       e18               1.34
    x100      e1              -52.21
    x100      e6                  -1
    x100      e19               1.33
    x101      e1              -67.06
    x101      e6                  -1
    x101      e20               1.37
    x102      e6                  -1
    x103      e6                  -1
    x104      e6                  -1
    x105      e6                  -1
    x106      e6                  -1
    x107      e6                  -1
    x108      e6                  -1
    x109      e6                  -1
    x110      e6                  -1
    x111      e6                  -1
    x112      e6                  -1
    x113      e1               -5.85
    x113      e7                0.23
    x114      e1              -24.53
    x114      e7                  -1
    x114      e8                0.73
    x115      e1              -32.42
    x115      e7                  -1
    x115      e9                0.95
    x116      e1              -98.39
    x116      e7                  -1
    x116      e10               0.98
    x117      e1              -86.03
    x117      e7                  -1
    x117      e11               1.47
    x118      e1              -77.42
    x118      e7                  -1
    x118      e12               0.81
    x119      e1              -62.91
    x119      e7                  -1
    x119      e13                  1
    x120      e1              -10.64
    x120      e7                  -1
    x120      e14               1.06
    x121      e1              -78.08
    x121      e7                  -1
    x121      e15               0.54
    x122      e1              -39.01
    x122      e7                  -1
    x122      e16                  1
    x123      e1              -25.46
    x123      e7                  -1
    x123      e17                  1
    x124      e1              -47.83
    x124      e7                  -1
    x124      e18                  1
    x125      e1               -97.5
    x125      e7                  -1
    x125      e19               1.43
    x126      e1              -77.35
    x126      e7                  -1
    x126      e20               1.27
    x127      e7                  -1
    x128      e7                  -1
    x129      e7                  -1
    x130      e7                  -1
    x131      e7                  -1
    x132      e7                  -1
    x133      e7                  -1
    x134      e7                  -1
    x135      e7                  -1
    x136      e1               -8.07
    x136      e7                1.43
    x136      e8                  -1
    x137      e1              -41.43
    x137      e8               -0.46
    x138      e1              -60.79
    x138      e8                  -1
    x138      e9                0.93
    x139      e1              -20.04
    x139      e8                  -1
    x139      e10               1.02
    x140      e1              -75.57
    x140      e8                  -1
    x140      e11               0.74
    x141      e1              -98.21
    x141      e8                  -1
    x141      e12               0.54
    x142      e1              -51.31
    x142      e8                  -1
    x142      e13               1.01
    x143      e1               -35.9
    x143      e8                  -1
    x143      e14                  1
    x144      e1              -89.25
    x144      e8                  -1
    x144      e15                  1
    x145      e1              -57.66
    x145      e8                  -1
    x145      e16               0.72
    x146      e1              -87.33
    x146      e8                  -1
    x146      e17                  1
    x147      e1              -11.61
    x147      e8                  -1
    x147      e18               1.47
    x148      e1                -100
    x148      e8                  -1
    x148      e19                  1
    x149      e1              -75.96
    x149      e8                  -1
    x149      e21                  1
    x150      e8                  -1
    x151      e8                  -1
    x152      e8                  -1
    x153      e8                  -1
    x154      e8                  -1
    x155      e8                  -1
    x156      e8                  -1
    x157      e8                  -1
    x158      e8                  -1
    x159      e8                  -1
    x160      e8                  -1
    x161      e8                  -1
    x162      e8                  -1
    x163      e8                  -1
    x164      e8                  -1
    x165      e1               -7.16
    x165      e7                0.74
    x165      e9                  -1
    x166      e1              -84.06
    x166      e8                   1
    x166      e9                  -1
    x167      e1              -15.31
    x167      e9               -0.28
    x168      e1              -94.48
    x168      e9                  -1
    x168      e10               0.69
    x169      e1              -79.21
    x169      e9                  -1
    x169      e11                0.9
    x170      e1              -72.34
    x170      e9                  -1
    x170      e12                  1
    x171      e1              -37.99
    x171      e9                  -1
    x171      e13               0.87
    x172      e1              -99.87
    x172      e9                  -1
    x172      e14               0.66
    x173      e1              -27.02
    x173      e9                  -1
    x173      e15               1.34
    x174      e1              -58.95
    x174      e9                  -1
    x174      e16               1.06
    x175      e1              -98.37
    x175      e9                  -1
    x175      e17               0.65
    x176      e1                -100
    x176      e9                  -1
    x176      e18                  1
    x177      e1              -36.02
    x177      e9                  -1
    x177      e19                1.2
    x178      e1               -9.33
    x178      e9                  -1
    x178      e20                  1
    x179      e9                  -1
    x180      e9                  -1
    x181      e9                  -1
    x182      e9                  -1
    x183      e9                  -1
    x184      e9                  -1
    x185      e9                  -1
    x186      e9                  -1
    x187      e9                  -1
    x188      e9                  -1
    x189      e9                  -1
    x190      e9                  -1
    x191      e1              -35.66
    x191      e7                1.13
    x191      e10                 -1
    x192      e1              -64.64
    x192      e8                 0.5
    x192      e10                 -1
    x193      e1              -99.26
    x193      e9                 0.8
    x193      e10                 -1
    x194      e1              -33.79
    x194      e10               0.16
    x195      e1              -17.53
    x195      e10                 -1
    x195      e11               0.64
    x196      e1               -3.87
    x196      e10                 -1
    x196      e12               0.87
    x197      e1              -10.74
    x197      e10                 -1
    x197      e13                  1
    x198      e1                -100
    x198      e10                 -1
    x198      e14                  1
    x199      e1              -26.72
    x199      e10                 -1
    x199      e15               1.41
    x200      e1              -77.67
    x200      e10                 -1
    x200      e16                  1
    x201      e1                -100
    x201      e10                 -1
    x201      e18                  1
    x202      e1              -48.97
    x202      e10                 -1
    x202      e19                  1
    x203      e1              -63.25
    x203      e10                 -1
    x203      e20                  1
    x204      e1              -28.31
    x204      e10                 -1
    x204      e21                  1
    x205      e10                 -1
    x206      e10                 -1
    x207      e10                 -1
    x208      e10                 -1
    x209      e1              -26.36
    x209      e7                1.08
    x209      e11                 -1
    x210      e1                -100
    x210      e10                  1
    x210      e11                 -1
    x211      e1              -47.65
    x211      e11                 -1
    x211      e12               0.84
    x212      e1               -58.1
    x212      e11                 -1
    x212      e13               0.51
    x213      e1              -45.69
    x213      e11                 -1
    x213      e16               1.35
    x214      e1              -51.33
    x214      e11                 -1
    x214      e17                  1
    x215      e1              -64.08
    x215      e11                 -1
    x215      e18               1.19
    x216      e1               -6.34
    x216      e11                 -1
    x216      e20               0.71
    x217      e11                 -1
    x218      e11                 -1
    x219      e11                 -1
    x220      e12                 -1
    x221      e12                 -1
    x222      e12                 -1
    x223      e12                 -1
    x224      e12                 -1
    x225      e12                 -1
    x226      e12                 -1
    x227      e12                 -1
    x228      e12                 -1
    x229      e12                 -1
    x230      e12                 -1
    x231      e12                 -1
    x232      e12                 -1
    x233      e12                 -1
    x234      e13                 -1
    x235      e13                 -1
    x236      e13                 -1
    x237      e13                 -1
    x238      e13                 -1
    x239      e13                 -1
    x240      e13                 -1
    x241      e13                 -1
    x242      e13                 -1
    x243      e14                 -1
    x244      e14                 -1
    x245      e14                 -1
    x246      e14                 -1
    x247      e14                 -1
    x248      e14                 -1
    x249      e14                 -1
    x250      e14                 -1
    x251      e15                 -1
    x252      e15                 -1
    x253      e15                 -1
    x254      e15                 -1
    x255      e15                 -1
    x256      e15                 -1
    x257      e15                 -1
    x258      e15                 -1
    x259      e15                 -1
    x260      e15                 -1
    x261      e15                 -1
    x262      e15                 -1
    x263      e16                 -1
    x264      e16                 -1
    x265      e16                 -1
    x266      e16                 -1
    x267      e17                 -1
    x268      e17                 -1
    x269      e17                 -1
    x270      e17                 -1
    x271      e17                 -1
    x272      e17                 -1
    x273      e17                 -1
    x274      e17                 -1
    x275      e17                 -1
    x276      e17                 -1
    x277      e17                 -1
    x278      e17                 -1
    x279      e17                 -1
    x280      e17                 -1
    x281      e17                 -1
    x282      e18                 -1
    x283      e18                 -1
    x284      e18                 -1
    x285      e18                 -1
    x286      e19                 -1
    x287      e19                 -1
    x288      e19                 -1
    x289      e19                 -1
    x290      e19                 -1
    x291      e19                 -1
    x292      e19                 -1
    x293      e20                 -1
    x294      e20                 -1
    x295      e20                 -1
    x296      e20                 -1
    x297      e20                 -1
    x298      e20                 -1
    x299      e20                 -1
    x300      e20                 -1
    x301      e20                 -1
    x302      e20                 -1
    x303      e20                 -1
    x304      e20                 -1
    x305      e20                 -1
    x306      e20                 -1
    x307      e20                 -1
    x308      e20                 -1
    x309      e20                 -1
    x310      e20                 -1
    x311      e20                 -1
    x312      e20                 -1
    x313      e21                 -1
    x314      e21                 -1
    x315      e21                 -1
    x316      e21                 -1
    x317      e21                 -1
    x318      e21                 -1
    x319      obj                  1
    x319      e1                   1
RHS
    rhs       e2            -4096.15
    rhs       e3           -26320.66
    rhs       e4           -10206.05
    rhs       e5            -4781.95
    rhs       e6           -54595.19
    rhs       e12           29059.58
    rhs       e13           32301.38
    rhs       e14             550.61
    rhs       e15            1483.08
    rhs       e16            8317.77
    rhs       e17           12790.41
    rhs       e18             6913.6
    rhs       e19            2499.38
    rhs       e20            4499.99
    rhs       e21             1584.2
BOUNDS
 UP bnd       x1             1000000
 UP bnd       x2             1000000
 UP bnd       x3             1000000
 UP bnd       x4             1000000
 UP bnd       x5             1000000
 UP bnd       x6             1000000
 UP bnd       x7             1000000
 UP bnd       x8             1000000
 UP bnd       x9             1000000
 UP bnd       x10            1000000
 UP bnd       x11            1000000
 UP bnd       x12            1000000
 FX bnd       x13                  0
 FX bnd       x14                  0
 FX bnd       x15                  0
 FX bnd       x16                  0
 FX bnd       x17                  0
 FX bnd       x18                  0
 FX bnd       x19                  0
 FX bnd       x20                  0
 FX bnd       x21                  0
 FX bnd       x22                  0
 FX bnd       x23                  0
 UP bnd       x24            1000000
 UP bnd       x25            1000000
 UP bnd       x26            1000000
 UP bnd       x27            1000000
 UP bnd       x28            1000000
 UP bnd       x29            1000000
 UP bnd       x30            1000000
 UP bnd       x31            1000000
 UP bnd       x32            1000000
 UP bnd       x33            1000000
 UP bnd       x34            1000000
 UP bnd       x35            1000000
 UP bnd       x36            1000000
 FX bnd       x37                  0
 FX bnd       x38                  0
 FX bnd       x39                  0
 FX bnd       x40                  0
 FX bnd       x41                  0
 FX bnd       x42                  0
 FX bnd       x43                  0
 FX bnd       x44                  0
 FX bnd       x45                  0
 FX bnd       x46                  0
 FX bnd       x47                  0
 FX bnd       x48                  0
 FX bnd       x49                  0
 FX bnd       x50                  0
 FX bnd       x51                  0
 FX bnd       x52                  0
 FX bnd       x53                  0
 UP bnd       x54            1000000
 UP bnd       x55            1000000
 UP bnd       x56            1000000
 UP bnd       x57            1000000
 UP bnd       x58            1000000
 UP bnd       x59            1000000
 UP bnd       x60            1000000
 UP bnd       x61            1000000
 UP bnd       x62            1000000
 UP bnd       x63            1000000
 UP bnd       x64            1000000
 UP bnd       x65            1000000
 UP bnd       x66            1000000
 UP bnd       x67            1000000
 FX bnd       x68                  0
 FX bnd       x69                  0
 FX bnd       x70                  0
 UP bnd       x71            1000000
 UP bnd       x72            1000000
 UP bnd       x73            1000000
 UP bnd       x74            1000000
 UP bnd       x75            1000000
 UP bnd       x76            1000000
 UP bnd       x77            1000000
 UP bnd       x78            1000000
 UP bnd       x79            1000000
 UP bnd       x80            1000000
 UP bnd       x81            1000000
 UP bnd       x82            1000000
 UP bnd       x83            1000000
 UP bnd       x84            1000000
 FX bnd       x85                  0
 FX bnd       x86                  0
 FX bnd       x87                  0
 FX bnd       x88                  0
 UP bnd       x89            1000000
 UP bnd       x90            1000000
 UP bnd       x91            1000000
 UP bnd       x92            1000000
 UP bnd       x93            1000000
 UP bnd       x94            1000000
 UP bnd       x95            1000000
 UP bnd       x96            1000000
 UP bnd       x97            1000000
 UP bnd       x98            1000000
 UP bnd       x99            1000000
 UP bnd       x100           1000000
 UP bnd       x101           1000000
 FX bnd       x102                 0
 FX bnd       x103                 0
 FX bnd       x104                 0
 FX bnd       x105                 0
 FX bnd       x106                 0
 FX bnd       x107                 0
 FX bnd       x108                 0
 FX bnd       x109                 0
 FX bnd       x110                 0
 FX bnd       x111                 0
 FX bnd       x112                 0
 UP bnd       x113           1000000
 UP bnd       x114           1000000
 UP bnd       x115           1000000
 UP bnd       x116           1000000
 UP bnd       x117           1000000
 UP bnd       x118           1000000
 UP bnd       x119           1000000
 UP bnd       x120           1000000
 UP bnd       x121           1000000
 UP bnd       x122           1000000
 UP bnd       x123           1000000
 UP bnd       x124           1000000
 UP bnd       x125           1000000
 UP bnd       x126           1000000
 FX bnd       x127                 0
 FX bnd       x128                 0
 FX bnd       x129                 0
 FX bnd       x130                 0
 FX bnd       x131                 0
 FX bnd       x132                 0
 FX bnd       x133                 0
 FX bnd       x134                 0
 FX bnd       x135                 0
 UP bnd       x136           1000000
 UP bnd       x137           1000000
 UP bnd       x138           1000000
 UP bnd       x139           1000000
 UP bnd       x140           1000000
 UP bnd       x141           1000000
 UP bnd       x142           1000000
 UP bnd       x143           1000000
 UP bnd       x144           1000000
 UP bnd       x145           1000000
 UP bnd       x146           1000000
 UP bnd       x147           1000000
 UP bnd       x148           1000000
 UP bnd       x149           1000000
 FX bnd       x150                 0
 FX bnd       x151                 0
 FX bnd       x152                 0
 FX bnd       x153                 0
 FX bnd       x154                 0
 FX bnd       x155                 0
 FX bnd       x156                 0
 FX bnd       x157                 0
 FX bnd       x158                 0
 FX bnd       x159                 0
 FX bnd       x160                 0
 FX bnd       x161                 0
 FX bnd       x162                 0
 FX bnd       x163                 0
 FX bnd       x164                 0
 UP bnd       x165           1000000
 UP bnd       x166           1000000
 UP bnd       x167           1000000
 UP bnd       x168           1000000
 UP bnd       x169           1000000
 UP bnd       x170           1000000
 UP bnd       x171           1000000
 UP bnd       x172           1000000
 UP bnd       x173           1000000
 UP bnd       x174           1000000
 UP bnd       x175           1000000
 UP bnd       x176           1000000
 UP bnd       x177           1000000
 UP bnd       x178           1000000
 FX bnd       x179                 0
 FX bnd       x180                 0
 FX bnd       x181                 0
 FX bnd       x182                 0
 FX bnd       x183                 0
 FX bnd       x184                 0
 FX bnd       x185                 0
 FX bnd       x186                 0
 FX bnd       x187                 0
 FX bnd       x188                 0
 FX bnd       x189                 0
 FX bnd       x190                 0
 UP bnd       x191           1000000
 UP bnd       x192           1000000
 UP bnd       x193           1000000
 UP bnd       x194           1000000
 UP bnd       x195           1000000
 UP bnd       x196           1000000
 UP bnd       x197           1000000
 UP bnd       x198           1000000
 UP bnd       x199           1000000
 UP bnd       x200           1000000
 UP bnd       x201           1000000
 UP bnd       x202           1000000
 UP bnd       x203           1000000
 UP bnd       x204           1000000
 FX bnd       x205                 0
 FX bnd       x206                 0
 FX bnd       x207                 0
 FX bnd       x208                 0
 UP bnd       x209           1000000
 UP bnd       x210           1000000
 UP bnd       x211           1000000
 UP bnd       x212           1000000
 UP bnd       x213           1000000
 UP bnd       x214           1000000
 UP bnd       x215           1000000
 UP bnd       x216           1000000
 FX bnd       x217                 0
 FX bnd       x218                 0
 FX bnd       x219                 0
 FX bnd       x220                 0
 FX bnd       x221                 0
 FX bnd       x222                 0
 FX bnd       x223                 0
 FX bnd       x224                 0
 FX bnd       x225                 0
 FX bnd       x226                 0
 FX bnd       x227                 0
 FX bnd       x228                 0
 FX bnd       x229                 0
 FX bnd       x230                 0
 FX bnd       x231                 0
 FX bnd       x232                 0
 FX bnd       x233                 0
 FX bnd       x234                 0
 FX bnd       x235                 0
 FX bnd       x236                 0
 FX bnd       x237                 0
 FX bnd       x238                 0
 FX bnd       x239                 0
 FX bnd       x240                 0
 FX bnd       x241                 0
 FX bnd       x242                 0
 FX bnd       x243                 0
 FX bnd       x244                 0
 FX bnd       x245                 0
 FX bnd       x246                 0
 FX bnd       x247                 0
 FX bnd       x248                 0
 FX bnd       x249                 0
 FX bnd       x250                 0
 FX bnd       x251                 0
 FX bnd       x252                 0
 FX bnd       x253                 0
 FX bnd       x254                 0
 FX bnd       x255                 0
 FX bnd       x256                 0
 FX bnd       x257                 0
 FX bnd       x258                 0
 FX bnd       x259                 0
 FX bnd       x260                 0
 FX bnd       x261                 0
 FX bnd       x262                 0
 FX bnd       x263                 0
 FX bnd       x264                 0
 FX bnd       x265                 0
 FX bnd       x266                 0
 FX bnd       x267                 0
 FX bnd       x268                 0
 FX bnd       x269                 0
 FX bnd       x270                 0
 FX bnd       x271                 0
 FX bnd       x272                 0
 FX bnd       x273                 0
 FX bnd       x274                 0
 FX bnd       x275                 0
 FX bnd       x276                 0
 FX bnd       x277                 0
 FX bnd       x278                 0
 FX bnd       x279                 0
 FX bnd       x280                 0
 FX bnd       x281                 0
 FX bnd       x282                 0
 FX bnd       x283                 0
 FX bnd       x284                 0
 FX bnd       x285                 0
 FX bnd       x286                 0
 FX bnd       x287                 0
 FX bnd       x288                 0
 FX bnd       x289                 0
 FX bnd       x290                 0
 FX bnd       x291                 0
 FX bnd       x292                 0
 FX bnd       x293                 0
 FX bnd       x294                 0
 FX bnd       x295                 0
 FX bnd       x296                 0
 FX bnd       x297                 0
 FX bnd       x298                 0
 FX bnd       x299                 0
 FX bnd       x300                 0
 FX bnd       x301                 0
 FX bnd       x302                 0
 FX bnd       x303                 0
 FX bnd       x304                 0
 FX bnd       x305                 0
 FX bnd       x306                 0
 FX bnd       x307                 0
 FX bnd       x308                 0
 FX bnd       x309                 0
 FX bnd       x310                 0
 FX bnd       x311                 0
 FX bnd       x312                 0
 FX bnd       x313                 0
 FX bnd       x314                 0
 FX bnd       x315                 0
 FX bnd       x316                 0
 FX bnd       x317                 0
 FX bnd       x318                 0
 FR bnd       x319
ENDATA
