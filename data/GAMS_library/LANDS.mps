* LP written by GAMS Convert at 12/18/23 11:28:35
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        10        1        4        5        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        17       17        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        53       53        0

NAME          Convert
*
* original model was minimizing
*
ROWS
 N  obj
 E  e1
 G  e2
 L  e3
 L  e4
 L  e5
 L  e6
 L  e7
 G  e8
 G  e9
 G  e10
COLUMNS
    x1        e1                 -10
    x1        e2                   1
    x1        e3                  10
    x1        e4                  -1
    x2        e1                  -7
    x2        e2                   1
    x2        e3                   7
    x2        e5                  -1
    x3        e1                 -16
    x3        e2                   1
    x3        e3                  16
    x3        e6                  -1
    x4        e1                  -6
    x4        e2                   1
    x4        e3                   6
    x4        e7                  -1
    x5        e1                 -40
    x5        e4                   1
    x5        e8                   1
    x6        e1                 -24
    x6        e4                   1
    x6        e9                   1
    x7        e1                  -4
    x7        e4                   1
    x7        e10                  1
    x8        e1                 -45
    x8        e5                   1
    x8        e8                   1
    x9        e1                 -27
    x9        e5                   1
    x9        e9                   1
    x10       e1                -4.5
    x10       e5                   1
    x10       e10                  1
    x11       e1                 -32
    x11       e6                   1
    x11       e8                   1
    x12       e1               -19.2
    x12       e6                   1
    x12       e9                   1
    x13       e1                -3.2
    x13       e6                   1
    x13       e10                  1
    x14       e1                 -55
    x14       e7                   1
    x14       e8                   1
    x15       e1                 -33
    x15       e7                   1
    x15       e9                   1
    x16       e1                -5.5
    x16       e7                   1
    x16       e10                  1
    x17       obj                  1
    x17       e1                   1
RHS
    rhs       e2                  12
    rhs       e3                 120
    rhs       e8                   7
    rhs       e9                   3
    rhs       e10                  2
BOUNDS
 FR bnd       x17
ENDATA
