\ENCODING=ISO-8859-1
\Problem name: gamsmodel

Minimize
 _obj: oc#40 + bc#41 + constobj#42
Subject To
 ab(a)#0:       x(a,routem1)#0 + x(a,routem2)#1 + x(a,routem3)#2
                + x(a,routem4)#3 + x(a,routem5)#4 <= 10
 ab(b)#1:       x(b,routem1)#5 + x(b,routem2)#6 + x(b,routem3)#7
                + x(b,routem4)#8 + x(b,routem5)#9 <= 19
 ab(c)#2:       x(c,routem1)#10 + x(c,routem2)#11 + x(c,routem3)#12
                + x(c,routem4)#13 + x(c,routem5)#14 <= 25
 ab(d)#3:       x(d,routem1)#15 + x(d,routem2)#16 + x(d,routem3)#17
                + x(d,routem4)#18 + x(d,routem5)#19 <= 15
 db(routem1)#4: 16 x(a,routem1)#0 + 9 x(d,routem1)#15 - y(routem1,1)#20
                - y(routem1,2)#21 - y(routem1,3)#22 - y(routem1,4)#23
                - y(routem1,5)#24 >= 0
 db(routem2)#5: 15 x(a,routem2)#1 + 10 x(b,routem2)#6 + 5 x(c,routem2)#11
                + 11 x(d,routem2)#16 - y(routem2,1)#25 - y(routem2,2)#26 >= 0
 db(routem3)#6: 28 x(a,routem3)#2 + 14 x(b,routem3)#7 + 22 x(d,routem3)#17
                - y(routem3,1)#27 - y(routem3,2)#28 - y(routem3,3)#29
                - y(routem3,4)#30 - y(routem3,5)#31 >= 0
 db(routem4)#7: 23 x(a,routem4)#3 + 15 x(b,routem4)#8 + 7 x(c,routem4)#13
                + 17 x(d,routem4)#18 - y(routem4,1)#32 - y(routem4,2)#33
                - y(routem4,3)#34 - y(routem4,4)#35 - y(routem4,5)#36 >= 0
 db(routem5)#8: 81 x(a,routem5)#4 + 57 x(b,routem5)#9 + 29 x(c,routem5)#14
                + 55 x(d,routem5)#19 - y(routem5,1)#37 - y(routem5,2)#38
                - y(routem5,3)#39 >= 0
 ocd#9:         - 18 x(a,routem1)#0 - 21 x(a,routem2)#1 - 18 x(a,routem3)#2
                - 16 x(a,routem4)#3 - 10 x(a,routem5)#4 - 15 x(b,routem2)#6
                - 16 x(b,routem3)#7 - 14 x(b,routem4)#8 - 9 x(b,routem5)#9
                - 10 x(c,routem2)#11 - 9 x(c,routem4)#13 - 6 x(c,routem5)#14
                - 17 x(d,routem1)#15 - 16 x(d,routem2)#16 - 17 x(d,routem3)#17
                - 15 x(d,routem4)#18 - 10 x(d,routem5)#19 + oc#40  = 0
 bcd1#10:       13 y(routem1,1)#20 + 10.4 y(routem1,2)#21
                + 9.75 y(routem1,3)#22 + 5.2 y(routem1,4)#23
                + 2.6 y(routem1,5)#24 + 13 y(routem2,1)#25
                + 9.1 y(routem2,2)#26 + 7 y(routem3,1)#27 + 6.3 y(routem3,2)#28
                + 4.9 y(routem3,3)#29 + 2.1 y(routem3,4)#30
                + 0.7 y(routem3,5)#31 + 7 y(routem4,1)#32 + 5.6 y(routem4,2)#33
                + 4.2 y(routem4,3)#34 + 2.1 y(routem4,4)#35
                + 0.7 y(routem4,5)#36 + y(routem5,1)#37 + 0.9 y(routem5,2)#38
                + 0.1 y(routem5,3)#39 + bc#41  = 7332.5
Bounds
 0 <= y(routem1,1)#20 <= 200
 0 <= y(routem1,2)#21 <= 20
 0 <= y(routem1,3)#22 <= 30
 0 <= y(routem1,4)#23 <= 20
 0 <= y(routem1,5)#24 <= 30
 0 <= y(routem2,1)#25 <= 50
 0 <= y(routem2,2)#26 <= 100
 0 <= y(routem3,1)#27 <= 140
 0 <= y(routem3,2)#28 <= 20
 0 <= y(routem3,3)#29 <= 20
 0 <= y(routem3,4)#30 <= 20
 0 <= y(routem3,5)#31 <= 20
 0 <= y(routem4,1)#32 <= 10
 0 <= y(routem4,2)#33 <= 40
 0 <= y(routem4,3)#34 <= 30
 0 <= y(routem4,4)#35 <= 20
 0 <= y(routem4,5)#36 <= 240
 0 <= y(routem5,1)#37 <= 580
 0 <= y(routem5,2)#38 <= 20
 0 <= y(routem5,3)#39 <= 20
      constobj#42 = 0
End
