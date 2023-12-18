* LP written by GAMS Convert at 12/18/23 11:26:28
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        26        1       24        1        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        29       29        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*       105      105        0

NAME          Convert
*
* original model was minimizing
*
ROWS
 N  obj
 E  e1
 L  e2
 G  e3
 G  e4
 G  e5
 G  e6
 G  e7
 G  e8
 G  e9
 G  e10
 G  e11
 G  e12
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
COLUMNS
    x1        e1                  -2
    x1        e2                   1
    x1        e3                   2
    x1        e4                   2
    x1        e5                   2
    x1        e15                  3
    x1        e16                  3
    x1        e17                  3
    x2        e1                  -2
    x2        e2                   1
    x2        e6                   2
    x2        e7                   2
    x2        e8                   2
    x2        e9                   2
    x2        e10                  2
    x2        e11                  2
    x2        e12                  2
    x2        e13                  2
    x2        e14                  2
    x2        e18                  3
    x2        e19                  3
    x2        e20                  3
    x2        e21                  3
    x2        e22                  3
    x2        e23                  3
    x2        e24                  3
    x2        e25                  3
    x2        e26                  3
    x3        e1                  -3
    x3        e2                   1
    x3        e3                   6
    x3        e4                   6
    x3        e5                   6
    x3        e15                3.4
    x3        e16                3.4
    x3        e17                3.4
    x4        e1                  -3
    x4        e2                   1
    x4        e6                   6
    x4        e7                   6
    x4        e8                   6
    x4        e9                   6
    x4        e10                  6
    x4        e11                  6
    x4        e12                  6
    x4        e13                  6
    x4        e14                  6
    x4        e18                3.4
    x4        e19                3.4
    x4        e20                3.4
    x4        e21                3.4
    x4        e22                3.4
    x4        e23                3.4
    x4        e24                3.4
    x4        e25                3.4
    x4        e26                3.4
    x5        e1                -2.1
    x5        e3                   1
    x6        e1                -2.8
    x6        e4                   1
    x7        e1                -2.1
    x7        e5                   1
    x8        e1                -0.6
    x8        e6                   1
    x9        e1                -1.5
    x9        e7                   1
    x10       e1                -0.9
    x10       e8                   1
    x11       e1                -1.2
    x11       e9                   1
    x12       e1                -1.6
    x12       e10                  1
    x13       e1                -1.2
    x13       e11                  1
    x14       e1                -1.2
    x14       e12                  1
    x15       e1                -1.2
    x15       e13                  1
    x16       e1                -0.6
    x16       e14                  1
    x17       e1                -3.6
    x17       e15                  1
    x18       e1                -4.8
    x18       e16                  1
    x19       e1                -3.6
    x19       e17                  1
    x20       e1                -0.9
    x20       e18                  1
    x21       e1               -2.25
    x21       e19                  1
    x22       e1               -1.35
    x22       e20                  1
    x23       e1                -1.8
    x23       e21                  1
    x24       e1                -2.4
    x24       e22                  1
    x25       e1                -1.8
    x25       e23                  1
    x26       e1                -1.8
    x26       e24                  1
    x27       e1                -1.8
    x27       e25                  1
    x28       e1                -0.9
    x28       e26                  1
    x29       obj                  1
    x29       e1                   1
RHS
    rhs       e2                  50
    rhs       e3                 200
    rhs       e4                 180
    rhs       e5                 160
    rhs       e6                 200
    rhs       e7                 180
    rhs       e8                 160
    rhs       e9                 200
    rhs       e10                180
    rhs       e11                160
    rhs       e12                200
    rhs       e13                180
    rhs       e14                160
    rhs       e15                180
    rhs       e16                160
    rhs       e17                140
    rhs       e18                180
    rhs       e19                160
    rhs       e20                140
    rhs       e21                180
    rhs       e22                160
    rhs       e23                140
    rhs       e24                180
    rhs       e25                160
    rhs       e26                140
BOUNDS
 FR bnd       x29
ENDATA
