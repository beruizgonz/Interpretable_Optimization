* LP written by GAMS Convert at 12/18/23 11:57:02
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        20        6        4       10        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        30       30        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       167      167        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 L  e1
 L  e2
 L  e3
 L  e4
 L  e5
 L  e6
 L  e7
 L  e8
 G  e9
 G  e10
 G  e11
 G  e12
 L  e13
 L  e14
 E  e15
 E  e16
 E  e17
 E  e18
 E  e19
 E  e20
COLUMNS
    x1        e1                  12
    x1        e2                   6
    x1        e3                 0.1
    x1        e4                   1
    x1        e6                  20
    x1        e7                   7
    x1        e8                  16
    x1        e10                450
    x1        e13                  8
    x1        e14                150
    x1        e15             -658.4
    x2        e1                   8
    x2        e2                   1
    x2        e3                   1
    x2        e4                 0.1
    x2        e5                  65
    x2        e7                  21
    x2        e8                   9
    x2        e9                 530
    x2        e11                  6
    x2        e14                180
    x2        e15          -1018.932
    x3        e1                   8
    x3        e2                   1
    x3        e3                   1
    x3        e5                  60
    x3        e7                  21
    x3        e8                   9
    x3        e9                 510
    x3        e11                  8
    x3        e14                220
    x3        e15            -838.32
    x4        e2                   7
    x4        e3                   1
    x4        e5                  25
    x4        e7                  22
    x4        e8                   5
    x4        e9                 450
    x4        e10                650
    x4        e11                 31
    x4        e14                100
    x4        e15             -370.4
    x5        e1                   3
    x5        e2                   1
    x5        e3                 0.3
    x5        e4                 0.6
    x5        e6                  20
    x5        e7                   4
    x5        e8                  10
    x5        e10                 85
    x5        e13                  2
    x5        e14                120
    x5        e15            -355.56
    x6        e1                   3
    x6        e2                   3
    x6        e3                 0.3
    x6        e4                   1
    x6        e5                  30
    x6        e6                  15
    x6        e7                   2
    x6        e8                   8
    x6        e10                215
    x6        e14                 70
    x6        e15            -222.24
    x7        e2                   7
    x7        e3                   1
    x7        e4                 0.4
    x7        e5                  25
    x7        e6                  10
    x7        e7                  11
    x7        e8                  10
    x7        e9                  60
    x7        e10                130
    x7        e11                  1
    x7        e14                145
    x7        e15            -1272.7
    x8        e2                  30
    x8        e3                   1
    x8        e4                   1
    x8        e6                  45
    x8        e7                  15
    x8        e8                  62
    x8        e9                 350
    x8        e10               1775
    x8        e11                  6
    x8        e13                  1
    x8        e14                500
    x8        e15              -2235
    x9        e1                   3
    x9        e3                 0.3
    x9        e4                 0.8
    x9        e5                  15
    x9        e6                  35
    x9        e7                   3
    x9        e8                  15
    x9        e10               1940
    x9        e13                  2
    x9        e14                 65
    x10       e2                   6
    x10       e3                   1
    x10       e4                 0.1
    x10       e7                  18
    x10       e8                   6
    x10       e9                2400
    x10       e11                242
    x10       e14                120
    x11       e5                  -1
    x11       e18                -20
    x12       e6                  -1
    x12       e18                -20
    x13       e9                   1
    x13       e19               -1.3
    x14       e10                  1
    x14       e19               -1.3
    x15       e11                  1
    x15       e19                 -5
    x16       e12                  1
    x16       e19                 -5
    x17       e7                  -1
    x17       e16                -10
    x18       e8                  -1
    x18       e16                -10
    x19       e10                0.5
    x20       e9                 0.5
    x21       e13                 -1
    x21       e17                -30
    x22       e1                -100
    x22       e2                -175
    x22       e7                17.5
    x22       e8                26.5
    x22       e9               -1550
    x22       e10              -1550
    x22       e11               -155
    x22       e12               -155
    x22       e13                -40
    x22       e14                300
    x23       e7                  19
    x23       e8                  29
    x23       e9               -1050
    x23       e10              -1050
    x23       e11               -105
    x23       e12               -105
    x23       e14                250
    x23       e15               -900
    x24       e7                  14
    x24       e8                21.5
    x24       e9                -750
    x24       e10               -750
    x24       e11                -75
    x24       e12                -75
    x24       e14                180
    x24       e15               -600
    x25       e15                  1
    x25       e20                 -1
    x26       e14                  1
    x26       e16                  1
    x26       e20                  1
    x27       e14                  1
    x27       e17                  1
    x27       e20                  1
    x28       e14                  1
    x28       e18                  1
    x28       e20                  1
    x29       e14                  1
    x29       e19                  1
    x29       e20                  1
    x30       obj                 -1
    x30       e20                  1
RHS
    rhs       e3                12.5
    rhs       e4                12.5
    rhs       e5                 700
    rhs       e6                 400
    rhs       e7                 390
    rhs       e8                 530
    rhs       e14              20000
BOUNDS
 UP bnd       x8                   2
 UP bnd       x11               17.5
 UP bnd       x12               17.5
 FR bnd       x25
 FR bnd       x26
 FR bnd       x27
 FR bnd       x28
 FR bnd       x29
 FR bnd       x30
ENDATA
