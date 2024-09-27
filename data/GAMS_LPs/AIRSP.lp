\ENCODING=ISO-8859-1
\Problem name: gamsmodel

Minimize
 _obj: 2.6 bh(routem1,1)#22 + 0.65 bh(routem1,2)#23 + 4.55 bh(routem1,3)#24
       + 2.6 bh(routem1,4)#25 + 2.6 bh(routem1,5)#26 + 3.9 bh(routem2,1)#27
       + 9.1 bh(routem2,2)#28 + 0.7 bh(routem3,1)#32 + 1.4 bh(routem3,2)#33
       + 2.8 bh(routem3,3)#34 + 1.4 bh(routem3,4)#35 + 0.7 bh(routem3,5)#36
       + 1.4 bh(routem4,1)#37 + 1.4 bh(routem4,2)#38 + 2.1 bh(routem4,3)#39
       + 1.4 bh(routem4,4)#40 + 0.7 bh(routem4,5)#41 + 0.1 bh(routem5,1)#42
       + 0.8 bh(routem5,2)#43 + 0.1 bh(routem5,3)#44 + oc#47 + constobj#48
Subject To
 ab(a)#0:           x(a,routem1)#0 + x(a,routem2)#1 + x(a,routem3)#2
                    + x(a,routem4)#3 + x(a,routem5)#4 <= 10
 ab(b)#1:           x(b,routem2)#5 + x(b,routem3)#6 + x(b,routem4)#7
                    + x(b,routem5)#8 <= 19
 ab(c)#2:           x(c,routem2)#9 + x(c,routem4)#10 + x(c,routem5)#11 <= 25
 ab(d)#3:           x(d,routem1)#12 + x(d,routem2)#13 + x(d,routem3)#14
                    + x(d,routem4)#15 + x(d,routem5)#16 <= 15
 cb(routem1)#4:     - 16 x(a,routem1)#0 - 9 x(d,routem1)#12 + z(routem1)#17
                     = 0
 cb(routem2)#5:     - 15 x(a,routem2)#1 - 10 x(b,routem2)#5 - 5 x(c,routem2)#9
                    - 11 x(d,routem2)#13 + z(routem2)#18  = 0
 cb(routem3)#6:     - 28 x(a,routem3)#2 - 14 x(b,routem3)#6
                    - 22 x(d,routem3)#14 + z(routem3)#19  = 0
 cb(routem4)#7:     - 23 x(a,routem4)#3 - 15 x(b,routem4)#7 - 7 x(c,routem4)#10
                    - 17 x(d,routem4)#15 + z(routem4)#20  = 0
 cb(routem5)#8:     - 81 x(a,routem5)#4 - 57 x(b,routem5)#8
                    - 29 x(c,routem5)#11 - 55 x(d,routem5)#16 + z(routem5)#21
                     = 0
 dbh(routem1,1)#9:  - z(routem1)#17 - bh(routem1,1)#22 <= -200
 dbh(routem1,2)#10: - z(routem1)#17 - bh(routem1,2)#23 <= -220
 dbh(routem1,3)#11: - z(routem1)#17 - bh(routem1,3)#24 <= -250
 dbh(routem1,4)#12: - z(routem1)#17 - bh(routem1,4)#25 <= -270
 dbh(routem1,5)#13: - z(routem1)#17 - bh(routem1,5)#26 <= -300
 dbh(routem2,1)#14: - z(routem2)#18 - bh(routem2,1)#27 <= -50
 dbh(routem2,2)#15: - z(routem2)#18 - bh(routem2,2)#28 <= -150
 dbh(routem2,3)#16: - z(routem2)#18 - bh(routem2,3)#29 <= 0
 dbh(routem2,4)#17: - z(routem2)#18 - bh(routem2,4)#30 <= 0
 dbh(routem2,5)#18: - z(routem2)#18 - bh(routem2,5)#31 <= 0
 dbh(routem3,1)#19: - z(routem3)#19 - bh(routem3,1)#32 <= -140
 dbh(routem3,2)#20: - z(routem3)#19 - bh(routem3,2)#33 <= -160
 dbh(routem3,3)#21: - z(routem3)#19 - bh(routem3,3)#34 <= -180
 dbh(routem3,4)#22: - z(routem3)#19 - bh(routem3,4)#35 <= -200
 dbh(routem3,5)#23: - z(routem3)#19 - bh(routem3,5)#36 <= -220
 dbh(routem4,1)#24: - z(routem4)#20 - bh(routem4,1)#37 <= -10
 dbh(routem4,2)#25: - z(routem4)#20 - bh(routem4,2)#38 <= -50
 dbh(routem4,3)#26: - z(routem4)#20 - bh(routem4,3)#39 <= -80
 dbh(routem4,4)#27: - z(routem4)#20 - bh(routem4,4)#40 <= -100
 dbh(routem4,5)#28: - z(routem4)#20 - bh(routem4,5)#41 <= -340
 dbh(routem5,1)#29: - z(routem5)#21 - bh(routem5,1)#42 <= -580
 dbh(routem5,2)#30: - z(routem5)#21 - bh(routem5,2)#43 <= -600
 dbh(routem5,3)#31: - z(routem5)#21 - bh(routem5,3)#44 <= -620
 dbh(routem5,4)#32: - z(routem5)#21 - bh(routem5,4)#45 <= 0
 dbh(routem5,5)#33: - z(routem5)#21 - bh(routem5,5)#46 <= 0
 ocd#34:            - 18 x(a,routem1)#0 - 21 x(a,routem2)#1 - 18 x(a,routem3)#2
                    - 16 x(a,routem4)#3 - 10 x(a,routem5)#4 - 15 x(b,routem2)#5
                    - 16 x(b,routem3)#6 - 14 x(b,routem4)#7 - 9 x(b,routem5)#8
                    - 10 x(c,routem2)#9 - 9 x(c,routem4)#10 - 6 x(c,routem5)#11
                    - 17 x(d,routem1)#12 - 16 x(d,routem2)#13
                    - 17 x(d,routem3)#14 - 15 x(d,routem4)#15
                    - 10 x(d,routem5)#16 + oc#47  = 0
Bounds
      oc#47 Free
      constobj#48 = 0
End
