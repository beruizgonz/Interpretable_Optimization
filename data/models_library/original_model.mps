* LP in MPS format
NAME          MyLP
ROWS
 N  obj
 E  c1
 L  c2
 G  c3
 G  c4
COLUMNS
    x1        obj               1
    x1        c1                2
    x1        c2               -1
    x1        c3               -3
    x1        c4                1
    x2        c1               -5
    x2        c2               -1
    x2        c3                2
    x3        obj               1
    x3        c4               -1
RHS
    rhs       c1                4
    rhs       c4                1
BOUNDS
 LO bnd       x1              -7
 UP bnd       x1              31
ENDATA

