NAME          singleton_inequality
ROWS
 N  COST
 L  CON1
 E  CON2
 G  CON3
 E  CON4
COLUMNS
    x1        COST       -2
    x1        CON2        4
    x1        CON3       -3
    x2        COST        4
    x2        CON2       -3
    x2        CON3        2
    x3        COST       -2
    x3        CON1       -3
    x3        CON2        8
    x3        CON4       -1
    x4        COST        2
    x4        CON2       -1
    x4        CON3       -4
RHS
    RHS1      CON1        2
    RHS1      CON2       20
    RHS1      CON3       -8
    RHS1      CON4       18
BOUNDS
 LO BND1      x1         0
 LO BND1      x2         0
 LO BND1      x3         0
 LO BND1      x4         0
ENDATA
