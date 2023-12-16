* LP written by GAMS Convert at 12/16/23 23:54:14
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        10        1        9        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        21       21        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       180      180        0

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
 E  e10
COLUMNS
    x1        e1                44.7
    x1        e2                1411
    x1        e3                   2
    x1        e4                 365
    x1        e6                55.4
    x1        e7                33.3
    x1        e8                 441
    x1        e10                 -1
    x2        e1                  36
    x2        e2                 897
    x2        e3                 1.7
    x2        e4                  99
    x2        e5                30.9
    x2        e6                17.4
    x2        e7                 7.9
    x2        e8                 106
    x2        e10                 -1
    x3        e1                 8.4
    x3        e2                 422
    x3        e3                15.1
    x3        e4                   9
    x3        e5                  26
    x3        e6                   3
    x3        e7                23.5
    x3        e8                  11
    x3        e9                  60
    x3        e10                 -1
    x4        e1                20.6
    x4        e2                  17
    x4        e3                 0.6
    x4        e4                   6
    x4        e5                55.8
    x4        e6                 0.2
    x4        e10                 -1
    x5        e1                 7.4
    x5        e2                 448
    x5        e3                16.4
    x5        e4                  19
    x5        e5                28.1
    x5        e6                 0.8
    x5        e7                10.3
    x5        e8                   4
    x5        e10                 -1
    x6        e1                15.7
    x6        e2                 661
    x6        e3                   1
    x6        e4                  48
    x6        e6                 9.6
    x6        e7                 8.1
    x6        e8                 471
    x6        e10                 -1
    x7        e1                41.7
    x7        e5                 0.2
    x7        e7                 0.5
    x7        e8                   5
    x7        e10                 -1
    x8        e1                 2.2
    x8        e2                 333
    x8        e3                 0.2
    x8        e4                 139
    x8        e5               169.2
    x8        e6                 6.4
    x8        e7                50.8
    x8        e8                 316
    x8        e9                 525
    x8        e10                 -1
    x9        e1                 4.4
    x9        e2                 249
    x9        e3                 0.3
    x9        e4                  37
    x9        e6                18.2
    x9        e7                 3.6
    x9        e8                  79
    x9        e10                 -1
    x10       e1                 5.8
    x10       e2                 705
    x10       e3                 6.8
    x10       e4                  45
    x10       e5                 3.5
    x10       e6                   1
    x10       e7                 4.9
    x10       e8                 209
    x10       e10                 -1
    x11       e1                 2.4
    x11       e2                 138
    x11       e3                 3.7
    x11       e4                  80
    x11       e5                  69
    x11       e6                 4.3
    x11       e7                 5.8
    x11       e8                  37
    x11       e9                 862
    x11       e10                 -1
    x12       e1                 2.6
    x12       e2                 125
    x12       e3                   4
    x12       e4                  36
    x12       e5                 7.2
    x12       e6                   9
    x12       e7                 4.5
    x12       e8                  26
    x12       e9                5369
    x12       e10                 -1
    x13       e1                 5.8
    x13       e2                 166
    x13       e3                 3.8
    x13       e4                  59
    x13       e5                16.6
    x13       e6                 4.7
    x13       e7                 5.9
    x13       e8                  21
    x13       e9                1184
    x13       e10                 -1
    x14       e1                14.3
    x14       e2                 336
    x14       e3                 1.8
    x14       e4                 118
    x14       e5                 6.7
    x14       e6                29.4
    x14       e7                 7.1
    x14       e8                 198
    x14       e9                2522
    x14       e10                 -1
    x15       e1                 1.1
    x15       e2                 106
    x15       e4                 138
    x15       e5               918.4
    x15       e6                 5.7
    x15       e7                13.8
    x15       e8                  33
    x15       e9                2755
    x15       e10                 -1
    x16       e1                 9.6
    x16       e2                 138
    x16       e3                 2.7
    x16       e4                  54
    x16       e5               290.7
    x16       e6                 8.4
    x16       e7                 5.4
    x16       e8                  83
    x16       e9                1912
    x16       e10                 -1
    x17       e1                 8.5
    x17       e2                  87
    x17       e3                 1.7
    x17       e4                 173
    x17       e5                86.8
    x17       e6                 1.2
    x17       e7                 4.3
    x17       e8                  55
    x17       e9                  57
    x17       e10                 -1
    x18       e1                12.8
    x18       e2                  99
    x18       e3                 2.5
    x18       e4                 154
    x18       e5                85.7
    x18       e6                 3.9
    x18       e7                 4.3
    x18       e8                  65
    x18       e9                 257
    x18       e10                 -1
    x19       e1                17.4
    x19       e2                1055
    x19       e3                 3.7
    x19       e4                 459
    x19       e5                 5.1
    x19       e6                26.9
    x19       e7                38.2
    x19       e8                  93
    x19       e10                 -1
    x20       e1                26.9
    x20       e2                1691
    x20       e3                11.4
    x20       e4                 792
    x20       e6                38.4
    x20       e7                24.6
    x20       e8                 217
    x20       e10                 -1
    x21       obj                  1
    x21       e10                  1
RHS
    rhs       e1                   3
    rhs       e2                  70
    rhs       e3                 0.8
    rhs       e4                  12
    rhs       e5                   5
    rhs       e6                 1.8
    rhs       e7                 2.7
    rhs       e8                  18
    rhs       e9                  75
BOUNDS
 FR bnd       x21
ENDATA
