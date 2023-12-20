* LP written by GAMS Convert at 12/16/23 17:00:15
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        22        2        0       20        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*         7        7        0        0        0        0        0        0
* FX      1
*
* Nonzero counts
*     Total    const       NL
*       127      127        0

NAME          Convert
*
* original model was maximizing
*
ROWS
 N  obj
 E  e1
 E  e2
 L  e3
 L  e4
 L  e5
 L  e6
 L  e7
 L  e8
 L  e9
 L  e10
 L  e11
 L  e12
 L  e13
 L  e14
 L  e15
 L  e16
 L  e17
 L  e18
 L  e19
 L  e20
 L  e21
 L  e22
COLUMNS
    x1        e2                   3
    x1        e3                  -3
    x1        e4                -2.5
    x1        e5                  -4
    x1        e6                  -6
    x1        e7                -2.3
    x1        e8                  -4
    x1        e9                  -7
    x1        e10               -4.4
    x1        e11                 -3
    x1        e12                 -5
    x1        e13                 -5
    x1        e14                 -2
    x1        e15                 -5
    x1        e16                 -4
    x1        e17                 -2
    x1        e18                 -3
    x1        e19                 -7
    x1        e20                 -4
    x1        e21                 -3
    x1        e22                 -3
    x2        e2                   5
    x2        e3                  -5
    x2        e4                -4.5
    x2        e5                  -6
    x2        e6                  -7
    x2        e7                -3.5
    x2        e8                -6.5
    x2        e9                 -10
    x2        e10               -6.4
    x2        e11                 -5
    x2        e12                 -7
    x2        e13                 -7
    x2        e14                 -4
    x2        e15                 -7
    x2        e16                 -4
    x2        e17                 -3
    x2        e18                 -6
    x2        e19                -11
    x2        e20                 -6
    x2        e21                 -4
    x2        e22                 -6
    x3        e1                 -40
    x3        e3                  40
    x3        e4                  45
    x3        e5                  55
    x3        e6                  48
    x3        e7                  28
    x3        e8                  48
    x3        e9                  80
    x3        e10                 25
    x3        e11                 45
    x3        e12                 70
    x3        e13                 45
    x3        e14                 45
    x3        e15                 65
    x3        e16                 38
    x3        e17                 20
    x3        e18                 38
    x3        e19                 68
    x3        e20                 25
    x3        e21                 45
    x3        e22                 57
    x4        e1                 -55
    x4        e3                  55
    x4        e4                  50
    x4        e5                  45
    x4        e6                  20
    x4        e7                  50
    x4        e8                  20
    x4        e9                  65
    x4        e10                 48
    x4        e11                 64
    x4        e12                 65
    x4        e13                 65
    x4        e14                 40
    x4        e15                 25
    x4        e16                 18
    x4        e17                 50
    x4        e18                 20
    x4        e19                 64
    x4        e20                 38
    x4        e21                 67
    x4        e22                 60
    x5        e1                 -30
    x5        e3                  30
    x5        e4                  40
    x5        e5                  30
    x5        e6                  60
    x5        e7                  25
    x5        e8                  65
    x5        e9                  57
    x5        e10                 30
    x5        e11                 42
    x5        e12                 48
    x5        e13                 40
    x5        e14                 44
    x5        e15                 35
    x5        e16                 64
    x5        e17                 15
    x5        e18                 60
    x5        e19                 54
    x5        e20                 20
    x5        e21                 32
    x5        e22                 40
    x6        obj                 -1
    x6        e1                   1
    x7        e1                   1
    x7        e3                  -1
    x7        e4                  -1
    x7        e5                  -1
    x7        e6                  -1
    x7        e7                  -1
    x7        e8                  -1
    x7        e9                  -1
    x7        e10                 -1
    x7        e11                 -1
    x7        e12                 -1
    x7        e13                 -1
    x7        e14                 -1
    x7        e15                 -1
    x7        e16                 -1
    x7        e17                 -1
    x7        e18                 -1
    x7        e19                 -1
    x7        e20                 -1
    x7        e21                 -1
    x7        e22                 -1
RHS
    rhs       e2                 100
BOUNDS
 LO bnd       x1              0.0001
 LO bnd       x2              0.0001
 LO bnd       x3              0.0001
 LO bnd       x4              0.0001
 LO bnd       x5              0.0001
 FR bnd       x6
 FX bnd       x7                   0
ENDATA
