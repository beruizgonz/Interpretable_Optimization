* LP written by GAMS Convert at 12/18/23 11:12:49
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*         6        1        3        2        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        12       12        0        0        0        0        0        0
* FX      5
*
* Nonzero counts
*     Total    const       NL
*        23       23        0

NAME          Convert
*
* original model was minimizing
*
ROWS
 N  obj
 E  e1
 L  e2
 L  e3
 G  e4
 G  e5
 G  e6
COLUMNS
    x1        e1              -0.225
    x1        e2                   1
    x1        e4                   1
    x2        e1              -0.153
    x2        e2                   1
    x2        e5                   1
    x3        e1              -0.162
    x3        e2                   1
    x3        e6                   1
    x4        e1              -0.225
    x4        e3                   1
    x4        e4                   1
    x5        e3                   1
    x5        e5                   1
    x6        e1              -0.126
    x6        e3                   1
    x6        e6                   1
    x7        obj                  1
    x7        e1                   1
    x8        e2                  -1
    x9        e3                  -1
    x10       e4                  -1
    x11       e5                  -1
    x12       e6                  -1
RHS
BOUNDS
 FR bnd       x7
 FX bnd       x8        349.97287028
 FX bnd       x9        599.95664663
 FX bnd       x10        324.9269431
 FX bnd       x11       300.10533238
 FX bnd       x12       274.94241832
ENDATA
