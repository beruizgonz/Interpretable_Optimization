* LP written by GAMS Convert at 12/18/23 11:14:09
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*         8        8        0        0        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        14       14        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*        54       54        0

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
COLUMNS
    x1        e1                   1
    x1        e2                -0.7
    x1        e3               -0.02
    x1        e4               -0.15
    x1        e5               -0.03
    x1        e6               -0.02
    x1        e7               -0.02
    x1        e8               -0.03
    x2        e1                   1
    x2        e2               -0.75
    x2        e3               -0.06
    x2        e4               -0.04
    x2        e5               -0.05
    x2        e6               -0.04
    x2        e7               -0.03
    x2        e8               -0.08
    x3        e1                   1
    x3        e2                -0.8
    x3        e3               -0.08
    x3        e4               -0.02
    x3        e5               -0.08
    x3        e6               -0.01
    x3        e8               -0.17
    x4        e1                   1
    x4        e2               -0.75
    x4        e3               -0.12
    x4        e4               -0.04
    x4        e5               -0.02
    x4        e6               -0.02
    x4        e8               -0.12
    x5        e1                   1
    x5        e2                -0.8
    x5        e3               -0.02
    x5        e4               -0.02
    x5        e5               -0.06
    x5        e6               -0.02
    x5        e7               -0.01
    x5        e8               -0.15
    x6        e1                   1
    x6        e2               -0.97
    x6        e3               -0.01
    x6        e4               -0.01
    x6        e5               -0.01
    x6        e8               -0.21
    x7        e1                   1
    x7        e3                  -1
    x7        e8               -0.38
    x8        e2                   1
    x9        e3                   1
    x10       e4                   1
    x11       e5                   1
    x12       e6                   1
    x13       e7                   1
    x14       obj                  1
    x14       e8                   1
RHS
    rhs       e1                2000
BOUNDS
 UP bnd       x1                 200
 UP bnd       x2                 750
 LO bnd       x3                 400
 UP bnd       x3                 800
 LO bnd       x4                 100
 UP bnd       x4                 700
 UP bnd       x5                1500
 LO bnd       x8                1500
 LO bnd       x9                 250
 UP bnd       x9                 300
 UP bnd       x10                 60
 UP bnd       x11                100
 UP bnd       x12                 40
 UP bnd       x13                 30
 FR bnd       x14
ENDATA
