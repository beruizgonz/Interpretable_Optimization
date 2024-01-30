* LP Problem in MPS Format
NAME          ExampleLP
ROWS
 N  obj
 L  c1
 E  c2
 G  c3
 L  c4
COLUMNS
    x1        c1                1
    x1        c2                1
    x1        c4                0
    x2        obj               1
    x2        c1                1
    x2        c3                1
    x2        c4                0
    x3        c2               -1
    x3        c3                2
RHS
    rhs       c1                5
    rhs       c2                4
    rhs       c3                7
    rhs       c4                8
BOUNDS
 LO bnd       x1               0
 UP bnd       x2               7
 FR bnd       x3
ENDATA
