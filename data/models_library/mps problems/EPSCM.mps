* LP written by GAMS Convert at 12/17/23 00:41:10
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        11        4        3        4        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        13       13        0        0        0        0        0        0
* FX      2
*
* Nonzero counts
*     Total    const       NL
*        43       43        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 E  e1
 E  e2
 E  e3
 L  e4
 L  e5
 L  e6
 L  e7
 G  e8
 G  e9
 G  e10
 E  e11
COLUMNS
    x1        e1                  -1
    x2        e2                  -1
    x2        e11                 -1
    x3        e3                  -1
    x4        e1                  30
    x4        e2                1.44
    x4        e3                   1
    x4        e4                   1
    x4        e8                   1
    x5        e1                  30
    x5        e2                1.44
    x5        e3                   1
    x5        e4                   1
    x5        e9                   1
    x6        e1                  75
    x6        e2                0.72
    x6        e5                   1
    x6        e9                   1
    x7        e1                  75
    x7        e2                0.72
    x7        e5                   1
    x7        e10                  1
    x8        e1                  60
    x8        e2                0.45
    x8        e6                   1
    x8        e8                   1
    x9        e1                  60
    x9        e2                0.45
    x9        e6                   1
    x9        e9                   1
    x10       e1                  60
    x10       e2                0.45
    x10       e6                   1
    x10       e10                  1
    x11       e1                  90
    x11       e3                   1
    x11       e7                   1
    x11       e8                   1
    x12       e1                  90
    x12       e3                   1
    x12       e7                   1
    x12       e10                  1
    x13       obj                 -1
    x13       e11                 -1
RHS
    rhs       e4               31000
    rhs       e5               15000
    rhs       e6               22000
    rhs       e7               10000
    rhs       e8               38400
    rhs       e9               19200
    rhs       e10               6400
BOUNDS
 FX bnd       x1                   0
 FR bnd       x2
 FX bnd       x3                   0
 FR bnd       x13
ENDATA
