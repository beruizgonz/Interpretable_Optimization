* LP written by GAMS Convert at 12/16/23 13:59:04
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*         4        4        0        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        10       10        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        37       37        0

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
COLUMNS
    x1        e1                  10
    x1        e2                  10
    x1        e3                  80
    x1        e4                -4.1
    x2        e1                  10
    x2        e2                  30
    x2        e3                  60
    x2        e4                -4.3
    x3        e1                  40
    x3        e2                  50
    x3        e3                  10
    x3        e4                -5.8
    x4        e1                  60
    x4        e2                  30
    x4        e3                  10
    x4        e4                  -6
    x5        e1                  30
    x5        e2                  30
    x5        e3                  40
    x5        e4                -7.6
    x6        e1                  30
    x6        e2                  40
    x6        e3                  30
    x6        e4                -7.5
    x7        e1                  30
    x7        e2                  20
    x7        e3                  50
    x7        e4                -7.3
    x8        e1                  50
    x8        e2                  40
    x8        e3                  10
    x8        e4                -6.9
    x9        e1                  20
    x9        e2                  30
    x9        e3                  50
    x9        e4                -7.3
    x10       obj                  1
    x10       e4                   1
RHS
    rhs       e1                  30
    rhs       e2                  30
    rhs       e3                  40
BOUNDS
 FR bnd       x10
ENDATA
