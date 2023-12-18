* LP written by GAMS Convert at 12/18/23 13:16:04
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        10        7        0        3        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        18       18        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        57       57        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 L  e1
 L  e2
 L  e3
 E  e4
 E  e5
 E  e6
 E  e7
 E  e8
 E  e9
 E  e10
COLUMNS
    x1        e1                   1
    x1        e4                   5
    x1        e7                   1
    x1        e10                -25
    x2        e2                   1
    x2        e5                   5
    x2        e8                   1
    x2        e10                -20
    x3        e3                   1
    x3        e6                   5
    x3        e9                   1
    x3        e10                -10
    x4        e1                   1
    x4        e4                   3
    x4        e7                   2
    x4        e10                -50
    x5        e2                   1
    x5        e5                   3
    x5        e8                   2
    x5        e10                -50
    x6        e3                   1
    x6        e6                   3
    x6        e9                   2
    x6        e10                -50
    x7        e1                   1
    x7        e4                   1
    x7        e7                   3
    x7        e10                -75
    x8        e2                   1
    x8        e5                   1
    x8        e8                   3
    x8        e10                -80
    x9        e3                   1
    x9        e6                   1
    x9        e9                   3
    x9        e10               -100
    x10       e4                  -1
    x10       e10                0.5
    x11       e4                   1
    x11       e5                  -1
    x11       e10                0.5
    x12       e5                   1
    x12       e6                  -1
    x12       e10                0.5
    x13       e6                   1
    x13       e10                -15
    x14       e7                  -1
    x14       e10                  2
    x15       e7                   1
    x15       e8                  -1
    x15       e10                  2
    x16       e8                   1
    x16       e9                  -1
    x16       e10                  2
    x17       e9                   1
    x17       e10                -25
    x18       obj                 -1
    x18       e10                  1
RHS
    rhs       e1                  40
    rhs       e2                  40
    rhs       e3                  40
BOUNDS
 UP bnd       x10                400
 UP bnd       x14                275
 FR bnd       x18
ENDATA
