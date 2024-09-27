NAME          redundant_rows
ROWS
 N  COST
 E  CON1
 E  CON2
 E  CON3
COLUMNS
    x1        COST        1
    x1        CON1        1
    x1        CON2        3
    x1        CON3        0.5
    x2        COST        1
    x2        CON1        2
    x2        CON2        5
    x2        CON3        1
    x3        COST       -2
    x3        CON1        4
    x3        CON2        8
    x3        CON3        2
    x4        COST       -3
    x4        CON1        5
    x4        CON2        4
    x4        CON3        2.5
RHS
    RHS1      CON1       10
    RHS1      CON2        2
    RHS1      CON3        5
BOUNDS
 LO BND1      x1         0
 LO BND1      x2         0
 LO BND1      x3         0
 LO BND1      x4         0
ENDATA
