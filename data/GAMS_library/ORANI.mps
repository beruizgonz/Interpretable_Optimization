* LP written by GAMS Convert at 12/18/23 11:53:16
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*        40       39        0        1        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*        52       52        0        0        0        0        0        0
* FX     13
*
* Nonzero counts
*     Total    const       NL
*       170      170        0

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
 E  e9
 E  e10
 E  e11
 E  e12
 E  e13
 E  e14
 E  e15
 E  e16
 E  e17
 E  e18
 E  e19
 E  e20
 E  e21
 E  e22
 E  e23
 E  e24
 E  e25
 E  e26
 E  e27
 E  e28
 E  e29
 E  e30
 E  e31
 E  e32
 E  e33
 E  e34
 E  e35
 E  e36
 E  e37
 E  e38
 E  e39
 L  e40
COLUMNS
    x1        e36                  1
    x2        e1                   1
    x2        e29       -0.314814815
    x3        e2                   1
    x3        e34       -0.042857143
    x4        e3                   1
    x4        e30       -0.666666667
    x5        e4                   1
    x5        e34       -0.336134454
    x6        e39                  1
    x7        e5                  -1
    x8        e6                  -1
    x9        e5                 0.5
    x9        e29       -0.351851852
    x9        e35              -0.95
    x10       e6                0.05
    x10       e30       -0.019607843
    x10       e35              -0.05
    x11       e35                  1
    x11       e36               -0.2
    x12       e19                  1
    x12       e32                  1
    x13       e20                  1
    x13       e33                  1
    x14       e32                 -1
    x15       e33                 -1
    x16       e31                 -1
    x17       e21                  1
    x17       e31                0.5
    x18       e22                  1
    x18       e31                0.5
    x19       e34                  1
    x19       e36               0.21
    x20       e1        0.3297491039
    x20       e2        -0.670250896
    x20       e3        0.2741935484
    x20       e4        0.2741935484
    x20       e7        -0.262295082
    x20       e8        -0.795454545
    x20       e9         0.737704918
    x20       e10       0.2045454545
    x20       e11       0.0909090909
    x20       e12                0.5
    x20       e13       -0.909090909
    x20       e14               -0.5
    x20       e23       0.5737704918
    x20       e24       0.0227272727
    x20       e25                  1
    x20       e37       -0.274193548
    x21       e1        -0.039426523
    x21       e2        0.9605734767
    x21       e3        0.0161290323
    x21       e4        0.0161290323
    x21       e11       -0.090909091
    x21       e12               -0.5
    x21       e13       0.9090909091
    x21       e14                0.5
    x21       e23       -0.016393443
    x21       e24       -0.181818182
    x21       e27                  1
    x21       e37       -0.016129032
    x22       e1        0.5483870968
    x22       e2        0.5483870968
    x22       e3         0.775659824
    x22       e4        -0.224340176
    x22       e7         0.262295082
    x22       e8        0.7954545455
    x22       e9        -0.737704918
    x22       e10       -0.204545455
    x22       e15               0.25
    x22       e16       0.6666666667
    x22       e17              -0.75
    x22       e18       -0.333333333
    x22       e23       0.0163934426
    x22       e24       0.7727272727
    x22       e26                  1
    x22       e37       -0.548387097
    x23       e1        0.1612903226
    x23       e2        0.1612903226
    x23       e3        -0.065982405
    x23       e4        0.9340175953
    x23       e15              -0.25
    x23       e16       -0.666666667
    x23       e17               0.75
    x23       e18       0.3333333333
    x23       e23       -0.081967213
    x23       e24       -0.045454545
    x23       e28                  1
    x23       e37       -0.161290323
    x24       obj                  1
    x24       e37                  1
    x24       e38                 -1
    x24       e39                  1
    x24       e40                  1
    x25       e25                 -1
    x25       e26                 -1
    x25       e27                 -1
    x25       e28                 -1
    x26       e19       0.6666666667
    x26       e21       -0.333333333
    x26       e23       -0.163934426
    x27       e20                0.8
    x27       e22               -0.2
    x27       e24       -0.113636364
    x28       e5                   1
    x28       e25                 -1
    x28       e35              -0.95
    x29       e6                   1
    x29       e26                 -1
    x29       e35              -0.05
    x30       e27                 -1
    x30       e34       -0.428571429
    x31       e28                 -1
    x31       e34       -0.571428571
    x32       e7                   1
    x32       e29       0.8333333333
    x33       e8                   1
    x33       e29       0.1666666667
    x34       e9                   1
    x34       e30       0.3137254902
    x35       e10                  1
    x35       e30       0.6862745098
    x36       e27                 -1
    x37       e28                 -1
    x38       e25                 -1
    x39       e26                 -1
    x40       e19       -0.666666667
    x40       e20               -0.8
    x40       e21       0.3333333333
    x40       e22                0.2
    x40       e23       -0.327868852
    x40       e24       -0.454545455
    x40       e38                  1
    x41       e38                 -1
    x42       e11                  1
    x42       e29       -0.185185185
    x43       e12                  1
    x43       e29       -0.148148148
    x44       e13                  1
    x44       e34       -0.042857143
    x45       e14                  1
    x45       e34       -0.342857143
    x46       e15                  1
    x46       e30       -0.294117647
    x47       e16                  1
    x47       e30       -0.019607843
    x48       e17                  1
    x48       e34       -0.168067227
    x49       e18                  1
    x49       e34       -0.067226891
    x50       e1                  -1
    x50       e2                  -1
    x50       e3                  -1
    x50       e4                  -1
    x50       e39                 -1
    x51       e7                  -1
    x51       e9                  -1
    x51       e11                 -1
    x51       e13                 -1
    x51       e15                 -1
    x51       e17                 -1
    x51       e19                 -1
    x51       e21                 -1
    x52       e8                  -1
    x52       e10                 -1
    x52       e12                 -1
    x52       e14                 -1
    x52       e16                 -1
    x52       e18                 -1
    x52       e20                 -1
    x52       e22                 -1
RHS
    rhs       e40             100000
BOUNDS
 FR bnd       x1
 FR bnd       x2
 FR bnd       x3
 FR bnd       x4
 FR bnd       x5
 FR bnd       x6
 FX bnd       x7                   1
 FX bnd       x8                   1
 FR bnd       x9
 FX bnd       x10                  1
 FR bnd       x11
 FR bnd       x12
 FR bnd       x13
 FX bnd       x14                  3
 FX bnd       x15                  3
 FR bnd       x16
 FR bnd       x17
 FR bnd       x18
 FR bnd       x19
 FR bnd       x20
 FR bnd       x21
 FR bnd       x22
 FR bnd       x23
 FR bnd       x24
 FX bnd       x25                  0
 FR bnd       x26
 FR bnd       x27
 FR bnd       x28
 FR bnd       x29
 FX bnd       x30                 -2
 FX bnd       x31                 -2
 FR bnd       x32
 FR bnd       x33
 FR bnd       x34
 FR bnd       x35
 FX bnd       x36                  0
 FX bnd       x37                  0
 FX bnd       x38                  0
 FR bnd       x39
 FR bnd       x40
 FX bnd       x41                  0
 FR bnd       x42
 FR bnd       x43
 FR bnd       x44
 FR bnd       x45
 FR bnd       x46
 FR bnd       x47
 FR bnd       x48
 FR bnd       x49
 FX bnd       x50                  2
 FR bnd       x51
 FR bnd       x52
ENDATA
