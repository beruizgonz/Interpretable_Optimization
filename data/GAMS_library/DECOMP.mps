* LP written by GAMS Convert at 12/16/23 22:41:53
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*         9        3        4        2        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        11       11        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        30       30        0

NAME          Convert
*
* original model was minimizing
*
ROWS
 N  obj
 E  e1
 E  e2
 E  e3
 G  e4
 G  e5
 G  e6
 G  e7
 L  e8
 L  e9
COLUMNS
    x1        obj                  1
    x1        e1                   1
    x2        e3                   1
    x3        e1                  -1
    x3        e2                   1
    x4        e2                  -3
    x4        e4                   1
    x4        e8                   1
    x5        e2                  -6
    x5        e5                   1
    x5        e8                   1
    x6        e2                  -6
    x6        e3                  -2
    x6        e6                   1
    x6        e8                   1
    x7        e2                  -5
    x7        e7                   1
    x7        e8                   1
    x8        e2                  -8
    x8        e4                   1
    x8        e9                   1
    x9        e2                  -1
    x9        e3                  -2
    x9        e5                   1
    x9        e9                   1
    x10       e2                  -3
    x10       e6                   1
    x10       e9                   1
    x11       e2                  -6
    x11       e7                   1
    x11       e9                   1
RHS
    rhs       e4                   2
    rhs       e5                   7
    rhs       e6                   3
    rhs       e7                   5
    rhs       e8                   9
    rhs       e9                   8
BOUNDS
 FR bnd       x1
 FR bnd       x2
 FR bnd       x3
ENDATA
