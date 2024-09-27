\ENCODING=ISO-8859-1
\Problem name: gamsmodel

Maximize
 _obj: 934 cons(one)#38 + 934 cons(two)#39 + 934 cons(three)#40 - rationr#75
       + revenue#96 - cropcost#97 - labcost#98 - vetcost#99 + constobj#100
Subject To
 landb(good)#0:        xcrop(cropm05,good)#2 + xcrop(cropm10,good)#4
                       + xcrop(cropm15,good)#6 + xcrop(cropm16,good)#7
                       + xcrop(cropm17,good)#9 + xcrop(cropm19,good)#11
                       + xcrop(cropm25,good)#13 + xcrop(cropm33,good)#19
                       + xcrop(cropm36,good)#21 + lswitch(good)#27 <= 8.775
 landb(medium)#1:      xcrop(cropm05,medium)#3 + xcrop(cropm10,medium)#5
                       + xcrop(cropm16,medium)#8 + xcrop(cropm17,medium)#10
                       + xcrop(cropm19,medium)#12 + xcrop(cropm25,medium)#14
                       + xcrop(cropm33,medium)#20 + xcrop(cropm36,medium)#22
                       + 1.407 xliver(recm1)#23 + 0.611 xliver(recm2)#24
                       + 0.631 xliver(recm3)#25 - lswitch(good)#27
                       + lswitch(medium)#28 <= 11.64
 landb(pasture)#2:     0.209 xliver(recm1)#23 + 2.03 xliver(recm2)#24
                       + 0.9 xliver(recm3)#25 - lswitch(medium)#28 <= 21.92
 lbal#3:               - xliver(recm1)#23 - xliver(recm2)#24 - xliver(recm3)#25
                       + xlive#26  = 0
 rliv#4:               - 5.141 xliver(recm1)#23 - 21.646 xliver(recm2)#24
                       - 49.845 xliver(recm3)#25 + rationr#75  = 0
 mbalc(cottonmh)#5:    0.848 xcrop(cropm02,good)#0
                       + 0.569 xcrop(cropm02,medium)#1
                       + 0.269 xcrop(cropm29,good)#15
                       + 0.149 xcrop(cropm29,medium)#16
                       + 0.403 xcrop(cropm30,good)#17
                       + 0.133 xcrop(cropm30,medium)#18 - sales(cottonmh)#41
                       >= 0
 mbalc(banana)#6:      0.221 xcrop(cropm05,good)#2
                       + 0.174 xcrop(cropm05,medium)#3 - sales(banana)#42 >= 0
 mbalc(sugarmcane)#7:  0.045 xcrop(cropm10,good)#4
                       + 0.03 xcrop(cropm10,medium)#5 - sales(sugarmcane)#43
                       >= 0
 mbalc(beansmarr)#8:   0.285 xcrop(cropm29,good)#15
                       + 0.221 xcrop(cropm29,medium)#16
                       + 0.274 xcrop(cropm33,good)#19
                       + 0.26 xcrop(cropm33,medium)#20
                       + 0.288 xcrop(cropm36,good)#21
                       + 0.287 xcrop(cropm36,medium)#22 - 0.225 cons(one)#38
                       - 0.152 cons(two)#39 - 0.15 cons(three)#40
                       - sales(beansmarr)#44 >= 0
 mbalc(beansmcor)#9:   0.251 xcrop(cropm25,good)#13
                       + 0.211 xcrop(cropm25,medium)#14
                       + 0.115 xcrop(cropm30,good)#17
                       + 0.352 xcrop(cropm30,medium)#18 - sales(beansmcor)#45
                       >= 0
 mbalc(oranges)#10:    0.092 xcrop(cropm15,good)#6 - sales(oranges)#46 >= 0
 mbalc(manioc)#11:     4.456 xcrop(cropm16,good)#7
                       + 3.964 xcrop(cropm16,medium)#8
                       + 3.408 xcrop(cropm36,good)#21
                       + 1.031 xcrop(cropm36,medium)#22 - 0.965 cons(one)#38
                       - 2.64 cons(two)#39 - 0.935 cons(three)#40
                       - sales(manioc)#47 >= 0
 mbalc(corn)#12:       0.725 xcrop(cropm17,good)#9
                       + 0.563 xcrop(cropm17,medium)#10
                       + 0.373 xcrop(cropm25,good)#13
                       + 0.264 xcrop(cropm25,medium)#14
                       + 0.536 xcrop(cropm29,good)#15
                       + 0.544 xcrop(cropm29,medium)#16
                       + 0.361 xcrop(cropm30,good)#17
                       + 0.212 xcrop(cropm30,medium)#18
                       + 0.594 xcrop(cropm33,good)#19
                       + 0.442 xcrop(cropm33,medium)#20
                       + 0.503 xcrop(cropm36,good)#21
                       + 0.328 xcrop(cropm36,medium)#22 - 0.235 cons(one)#38
                       - 0.232 cons(two)#39 - 0.581 cons(three)#40
                       - sales(corn)#48 >= 0
 mbalc(sisal)#13:      2.244 xcrop(cropm19,good)#11
                       + 1.666 xcrop(cropm19,medium)#12 - sales(sisal)#49 >= 0
 dprod(cottonmh)#14:   - 0.848 xcrop(cropm02,good)#0
                       - 0.569 xcrop(cropm02,medium)#1
                       - 0.269 xcrop(cropm29,good)#15
                       - 0.149 xcrop(cropm29,medium)#16
                       - 0.403 xcrop(cropm30,good)#17
                       - 0.133 xcrop(cropm30,medium)#18 + xprod(cottonmh)#29
                        = 0
 dprod(banana)#15:     - 0.221 xcrop(cropm05,good)#2
                       - 0.174 xcrop(cropm05,medium)#3 + xprod(banana)#30  = 0
 dprod(sugarmcane)#16: - 0.045 xcrop(cropm10,good)#4
                       - 0.03 xcrop(cropm10,medium)#5 + xprod(sugarmcane)#31
                        = 0
 dprod(beansmarr)#17:  - 0.285 xcrop(cropm29,good)#15
                       - 0.221 xcrop(cropm29,medium)#16
                       - 0.274 xcrop(cropm33,good)#19
                       - 0.26 xcrop(cropm33,medium)#20
                       - 0.288 xcrop(cropm36,good)#21
                       - 0.287 xcrop(cropm36,medium)#22 + xprod(beansmarr)#32
                        = 0
 dprod(beansmcor)#18:  - 0.251 xcrop(cropm25,good)#13
                       - 0.211 xcrop(cropm25,medium)#14
                       - 0.115 xcrop(cropm30,good)#17
                       - 0.352 xcrop(cropm30,medium)#18 + xprod(beansmcor)#33
                        = 0
 dprod(oranges)#19:    - 0.092 xcrop(cropm15,good)#6 + xprod(oranges)#34  = 0
 dprod(manioc)#20:     - 4.456 xcrop(cropm16,good)#7
                       - 3.964 xcrop(cropm16,medium)#8
                       - 3.408 xcrop(cropm36,good)#21
                       - 1.031 xcrop(cropm36,medium)#22 + xprod(manioc)#35  = 0
 dprod(corn)#21:       - 0.725 xcrop(cropm17,good)#9
                       - 0.563 xcrop(cropm17,medium)#10
                       - 0.373 xcrop(cropm25,good)#13
                       - 0.264 xcrop(cropm25,medium)#14
                       - 0.536 xcrop(cropm29,good)#15
                       - 0.544 xcrop(cropm29,medium)#16
                       - 0.361 xcrop(cropm30,good)#17
                       - 0.212 xcrop(cropm30,medium)#18
                       - 0.594 xcrop(cropm33,good)#19
                       - 0.442 xcrop(cropm33,medium)#20
                       - 0.503 xcrop(cropm36,good)#21
                       - 0.328 xcrop(cropm36,medium)#22 + xprod(corn)#36  = 0
 dprod(sisal)#22:      - 2.244 xcrop(cropm19,good)#11
                       - 1.666 xcrop(cropm19,medium)#12 + xprod(sisal)#37  = 0
 cond#23:              cons(one)#38 + cons(two)#39 + cons(three)#40  = 1
 labc(jan)#24:         4.79 xcrop(cropm02,good)#0
                       + 4.79 xcrop(cropm02,medium)#1
                       + 8.24 xcrop(cropm05,good)#2
                       + 8.24 xcrop(cropm05,medium)#3
                       + 5.78 xcrop(cropm10,good)#4
                       + 5.78 xcrop(cropm10,medium)#5 + 2 xcrop(cropm15,good)#6
                       + 5.71 xcrop(cropm16,good)#7
                       + 5.71 xcrop(cropm16,medium)#8
                       + 9.19 xcrop(cropm17,good)#9
                       + 9.19 xcrop(cropm17,medium)#10
                       + 1.22 xcrop(cropm19,good)#11
                       + 1.22 xcrop(cropm19,medium)#12
                       + 12.46 xcrop(cropm25,good)#13
                       + 12.46 xcrop(cropm25,medium)#14
                       + 5.54 xcrop(cropm29,good)#15
                       + 5.54 xcrop(cropm29,medium)#16
                       + 9.18 xcrop(cropm30,good)#17
                       + 9.18 xcrop(cropm30,medium)#18
                       + 6.87 xcrop(cropm33,good)#19
                       + 6.87 xcrop(cropm33,medium)#20
                       + 7.44 xcrop(cropm36,good)#21
                       + 7.44 xcrop(cropm36,medium)#22 + 4.261 xliver(recm1)#23
                       + 1.873 xliver(recm2)#24 + 1.933 xliver(recm3)#25
                       - flab(jan)#50 - tlab(jan)#62 - 25 plab#74 <= 0
 labc(feb)#25:         10.89 xcrop(cropm02,good)#0
                       + 10.89 xcrop(cropm02,medium)#1
                       + 0.89 xcrop(cropm05,good)#2
                       + 0.89 xcrop(cropm05,medium)#3
                       + 3.37 xcrop(cropm10,good)#4
                       + 3.37 xcrop(cropm10,medium)#5
                       + 2.36 xcrop(cropm15,good)#6 + 7.6 xcrop(cropm16,good)#7
                       + 7.6 xcrop(cropm16,medium)#8
                       + 14.42 xcrop(cropm17,good)#9
                       + 14.42 xcrop(cropm17,medium)#10
                       + 1.11 xcrop(cropm19,good)#11
                       + 1.11 xcrop(cropm19,medium)#12
                       + 11.35 xcrop(cropm25,good)#13
                       + 11.35 xcrop(cropm25,medium)#14
                       + 10.3 xcrop(cropm29,good)#15
                       + 10.3 xcrop(cropm29,medium)#16
                       + 3.94 xcrop(cropm30,good)#17
                       + 3.94 xcrop(cropm30,medium)#18
                       + 10.19 xcrop(cropm33,good)#19
                       + 10.19 xcrop(cropm33,medium)#20
                       + 10.21 xcrop(cropm36,good)#21
                       + 10.21 xcrop(cropm36,medium)#22
                       + 2.854 xliver(recm1)#23 + 1.262 xliver(recm2)#24
                       + 1.302 xliver(recm3)#25 - flab(feb)#51 - tlab(feb)#63
                       - 25 plab#74 <= 0
 labc(mar)#26:         18.7 xcrop(cropm02,good)#0
                       + 18.7 xcrop(cropm02,medium)#1
                       + 5.43 xcrop(cropm05,good)#2
                       + 5.43 xcrop(cropm05,medium)#3
                       + 4.03 xcrop(cropm10,good)#4
                       + 4.03 xcrop(cropm10,medium)#5
                       + 4.13 xcrop(cropm15,good)#6
                       + 7.28 xcrop(cropm16,good)#7
                       + 7.28 xcrop(cropm16,medium)#8
                       + 10.59 xcrop(cropm17,good)#9
                       + 10.59 xcrop(cropm17,medium)#10
                       + 0.56 xcrop(cropm19,good)#11
                       + 0.56 xcrop(cropm19,medium)#12
                       + 24.38 xcrop(cropm25,good)#13
                       + 24.38 xcrop(cropm25,medium)#14
                       + 9.24 xcrop(cropm29,good)#15
                       + 9.24 xcrop(cropm29,medium)#16
                       + 16.15 xcrop(cropm30,good)#17
                       + 16.15 xcrop(cropm30,medium)#18
                       + 10.61 xcrop(cropm33,good)#19
                       + 10.61 xcrop(cropm33,medium)#20
                       + 9.63 xcrop(cropm36,good)#21
                       + 9.63 xcrop(cropm36,medium)#22 + 0.04 xliver(recm1)#23
                       + 0.04 xliver(recm2)#24 + 0.04 xliver(recm3)#25
                       - flab(mar)#52 - tlab(mar)#64 - 25 plab#74 <= 0
 labc(apr)#27:         9.83 xcrop(cropm02,good)#0
                       + 9.83 xcrop(cropm02,medium)#1
                       + 3.31 xcrop(cropm05,good)#2
                       + 3.31 xcrop(cropm05,medium)#3
                       + 7.09 xcrop(cropm10,good)#4
                       + 7.09 xcrop(cropm10,medium)#5
                       + 2.13 xcrop(cropm15,good)#6
                       + 8.15 xcrop(cropm16,good)#7
                       + 8.15 xcrop(cropm16,medium)#8
                       + 12.01 xcrop(cropm17,good)#9
                       + 12.01 xcrop(cropm17,medium)#10
                       + 0.78 xcrop(cropm19,good)#11
                       + 0.78 xcrop(cropm19,medium)#12
                       + 22.03 xcrop(cropm25,good)#13
                       + 22.03 xcrop(cropm25,medium)#14
                       + 8.97 xcrop(cropm29,good)#15
                       + 8.97 xcrop(cropm29,medium)#16
                       + 23.69 xcrop(cropm30,good)#17
                       + 23.69 xcrop(cropm30,medium)#18
                       + 9.14 xcrop(cropm33,good)#19
                       + 9.14 xcrop(cropm33,medium)#20
                       + 16.18 xcrop(cropm36,good)#21
                       + 16.18 xcrop(cropm36,medium)#22 + 0.04 xliver(recm1)#23
                       + 0.04 xliver(recm2)#24 + 0.04 xliver(recm3)#25
                       - flab(apr)#53 - tlab(apr)#65 - 25 plab#74 <= 0
 labc(may)#28:         8.03 xcrop(cropm02,good)#0
                       + 8.03 xcrop(cropm02,medium)#1
                       + 11.13 xcrop(cropm05,good)#2
                       + 11.13 xcrop(cropm05,medium)#3
                       + 4.64 xcrop(cropm10,good)#4
                       + 4.64 xcrop(cropm10,medium)#5
                       + 7.18 xcrop(cropm15,good)#6
                       + 8.48 xcrop(cropm16,good)#7
                       + 8.48 xcrop(cropm16,medium)#8
                       + 3.47 xcrop(cropm17,good)#9
                       + 3.47 xcrop(cropm17,medium)#10
                       + 1.19 xcrop(cropm19,good)#11
                       + 1.19 xcrop(cropm19,medium)#12
                       + 16.91 xcrop(cropm25,good)#13
                       + 16.91 xcrop(cropm25,medium)#14
                       + 15.82 xcrop(cropm29,good)#15
                       + 15.82 xcrop(cropm29,medium)#16
                       + 28.12 xcrop(cropm30,good)#17
                       + 28.12 xcrop(cropm30,medium)#18
                       + 18.38 xcrop(cropm33,good)#19
                       + 18.38 xcrop(cropm33,medium)#20
                       + 20.11 xcrop(cropm36,good)#21
                       + 20.11 xcrop(cropm36,medium)#22 + 0.04 xliver(recm1)#23
                       + 0.04 xliver(recm2)#24 + 0.04 xliver(recm3)#25
                       - flab(may)#54 - tlab(may)#66 - 25 plab#74 <= 0
 labc(jun)#29:         20.95 xcrop(cropm02,good)#0
                       + 20.95 xcrop(cropm02,medium)#1
                       + 10.33 xcrop(cropm05,good)#2
                       + 10.33 xcrop(cropm05,medium)#3
                       + 5.91 xcrop(cropm10,good)#4
                       + 5.91 xcrop(cropm10,medium)#5
                       + 5.88 xcrop(cropm15,good)#6
                       + 10.71 xcrop(cropm16,good)#7
                       + 10.71 xcrop(cropm16,medium)#8
                       + 10.75 xcrop(cropm17,good)#9
                       + 10.75 xcrop(cropm17,medium)#10
                       + 0.44 xcrop(cropm19,good)#11
                       + 0.44 xcrop(cropm19,medium)#12
                       + 9.47 xcrop(cropm25,good)#13
                       + 9.47 xcrop(cropm25,medium)#14
                       + 13 xcrop(cropm29,good)#15
                       + 13 xcrop(cropm29,medium)#16
                       + 20.58 xcrop(cropm30,good)#17
                       + 20.58 xcrop(cropm30,medium)#18
                       + 11.5 xcrop(cropm33,good)#19
                       + 11.5 xcrop(cropm33,medium)#20
                       + 16.86 xcrop(cropm36,good)#21
                       + 16.86 xcrop(cropm36,medium)#22 + 0.04 xliver(recm1)#23
                       + 0.04 xliver(recm2)#24 + 0.04 xliver(recm3)#25
                       - flab(jun)#55 - tlab(jun)#67 - 25 plab#74 <= 0
 labc(jul)#30:         15.67 xcrop(cropm02,good)#0
                       + 15.67 xcrop(cropm02,medium)#1
                       + 3.27 xcrop(cropm05,good)#2
                       + 3.27 xcrop(cropm05,medium)#3
                       + 7.67 xcrop(cropm10,good)#4
                       + 7.67 xcrop(cropm10,medium)#5
                       + 3.21 xcrop(cropm15,good)#6
                       + 9.34 xcrop(cropm16,good)#7
                       + 9.34 xcrop(cropm16,medium)#8
                       + 5.88 xcrop(cropm17,good)#9
                       + 5.88 xcrop(cropm17,medium)#10
                       + 1.89 xcrop(cropm19,good)#11
                       + 1.89 xcrop(cropm19,medium)#12
                       + 10.97 xcrop(cropm25,good)#13
                       + 10.97 xcrop(cropm25,medium)#14
                       + 14.74 xcrop(cropm29,good)#15
                       + 14.74 xcrop(cropm29,medium)#16
                       + 17.73 xcrop(cropm30,good)#17
                       + 17.73 xcrop(cropm30,medium)#18
                       + 9.22 xcrop(cropm33,good)#19
                       + 9.22 xcrop(cropm33,medium)#20
                       + 14.86 xcrop(cropm36,good)#21
                       + 14.86 xcrop(cropm36,medium)#22 + 0.04 xliver(recm1)#23
                       + 0.04 xliver(recm2)#24 + 0.04 xliver(recm3)#25
                       - flab(jul)#56 - tlab(jul)#68 - 25 plab#74 <= 0
 labc(aug)#31:         15.48 xcrop(cropm02,good)#0
                       + 15.48 xcrop(cropm02,medium)#1
                       + 2.65 xcrop(cropm05,good)#2
                       + 2.65 xcrop(cropm05,medium)#3
                       + 9.23 xcrop(cropm10,good)#4
                       + 9.23 xcrop(cropm10,medium)#5
                       + 9.2 xcrop(cropm15,good)#6
                       + 12.76 xcrop(cropm16,good)#7
                       + 12.76 xcrop(cropm16,medium)#8
                       + 7.3 xcrop(cropm17,good)#9
                       + 7.3 xcrop(cropm17,medium)#10
                       + 3.04 xcrop(cropm19,good)#11
                       + 3.04 xcrop(cropm19,medium)#12
                       + 9.7 xcrop(cropm25,good)#13
                       + 9.7 xcrop(cropm25,medium)#14
                       + 10.64 xcrop(cropm29,good)#15
                       + 10.64 xcrop(cropm29,medium)#16
                       + 15.06 xcrop(cropm30,good)#17
                       + 15.06 xcrop(cropm30,medium)#18
                       + 13.36 xcrop(cropm33,good)#19
                       + 13.36 xcrop(cropm33,medium)#20
                       + 14 xcrop(cropm36,good)#21
                       + 14 xcrop(cropm36,medium)#22 + 0.04 xliver(recm1)#23
                       + 0.04 xliver(recm2)#24 + 0.04 xliver(recm3)#25
                       - flab(aug)#57 - tlab(aug)#69 - 25 plab#74 <= 0
 labc(sep)#32:         11.35 xcrop(cropm02,good)#0
                       + 11.35 xcrop(cropm02,medium)#1
                       + 5.47 xcrop(cropm05,good)#2
                       + 5.47 xcrop(cropm05,medium)#3
                       + 5.93 xcrop(cropm10,good)#4
                       + 5.93 xcrop(cropm10,medium)#5
                       + 6.89 xcrop(cropm15,good)#6
                       + 8.93 xcrop(cropm16,good)#7
                       + 8.93 xcrop(cropm16,medium)#8
                       + 4.14 xcrop(cropm17,good)#9
                       + 4.14 xcrop(cropm17,medium)#10
                       + 2.93 xcrop(cropm19,good)#11
                       + 2.93 xcrop(cropm19,medium)#12
                       + 4.93 xcrop(cropm25,good)#13
                       + 4.93 xcrop(cropm25,medium)#14
                       + 5.64 xcrop(cropm29,good)#15
                       + 5.64 xcrop(cropm29,medium)#16
                       + 4.58 xcrop(cropm30,good)#17
                       + 4.58 xcrop(cropm30,medium)#18
                       + 3.85 xcrop(cropm33,good)#19
                       + 3.85 xcrop(cropm33,medium)#20
                       + 8.04 xcrop(cropm36,good)#21
                       + 8.04 xcrop(cropm36,medium)#22 + 0.04 xliver(recm1)#23
                       + 0.04 xliver(recm2)#24 + 0.04 xliver(recm3)#25
                       - flab(sep)#58 - tlab(sep)#70 - 25 plab#74 <= 0
 labc(oct)#33:         8.38 xcrop(cropm02,good)#0
                       + 8.38 xcrop(cropm02,medium)#1
                       + 3.67 xcrop(cropm05,good)#2
                       + 3.67 xcrop(cropm05,medium)#3
                       + 9.05 xcrop(cropm10,good)#4
                       + 9.05 xcrop(cropm10,medium)#5
                       + 3.3 xcrop(cropm15,good)#6
                       + 10.22 xcrop(cropm16,good)#7
                       + 10.22 xcrop(cropm16,medium)#8
                       + 4.25 xcrop(cropm17,good)#9
                       + 4.25 xcrop(cropm17,medium)#10
                       + 4.99 xcrop(cropm19,good)#11
                       + 4.99 xcrop(cropm19,medium)#12
                       + 6.41 xcrop(cropm25,good)#13
                       + 6.41 xcrop(cropm25,medium)#14
                       + 4.9 xcrop(cropm29,good)#15
                       + 4.9 xcrop(cropm29,medium)#16
                       + 9.76 xcrop(cropm30,good)#17
                       + 9.76 xcrop(cropm30,medium)#18
                       + 3.87 xcrop(cropm33,good)#19
                       + 3.87 xcrop(cropm33,medium)#20
                       + 6.69 xcrop(cropm36,good)#21
                       + 6.69 xcrop(cropm36,medium)#22 + 0.04 xliver(recm1)#23
                       + 0.04 xliver(recm2)#24 + 0.04 xliver(recm3)#25
                       - flab(oct)#59 - tlab(oct)#71 - 25 plab#74 <= 0
 labc(nov)#34:         8.23 xcrop(cropm02,good)#0
                       + 8.23 xcrop(cropm02,medium)#1
                       + 3.18 xcrop(cropm05,good)#2
                       + 3.18 xcrop(cropm05,medium)#3
                       + 7.59 xcrop(cropm10,good)#4
                       + 7.59 xcrop(cropm10,medium)#5
                       + 9.7 xcrop(cropm15,good)#6 + 6.18 xcrop(cropm16,good)#7
                       + 6.18 xcrop(cropm16,medium)#8
                       + 3.49 xcrop(cropm17,good)#9
                       + 3.49 xcrop(cropm17,medium)#10
                       + 7.73 xcrop(cropm19,good)#11
                       + 7.73 xcrop(cropm19,medium)#12
                       + 0.53 xcrop(cropm25,good)#13
                       + 0.53 xcrop(cropm25,medium)#14
                       + 4.73 xcrop(cropm29,good)#15
                       + 4.73 xcrop(cropm29,medium)#16
                       + 7.39 xcrop(cropm30,good)#17
                       + 7.39 xcrop(cropm30,medium)#18
                       + 2.65 xcrop(cropm33,good)#19
                       + 2.65 xcrop(cropm33,medium)#20
                       + 5.49 xcrop(cropm36,good)#21
                       + 5.49 xcrop(cropm36,medium)#22 + 7.075 xliver(recm1)#23
                       + 3.095 xliver(recm2)#24 + 3.195 xliver(recm3)#25
                       - flab(nov)#60 - tlab(nov)#72 - 25 plab#74 <= 0
 labc(dec)#35:         4.54 xcrop(cropm02,good)#0
                       + 4.54 xcrop(cropm02,medium)#1
                       + 1.44 xcrop(cropm05,good)#2
                       + 1.44 xcrop(cropm05,medium)#3
                       + 6.37 xcrop(cropm10,good)#4
                       + 6.37 xcrop(cropm10,medium)#5
                       + 8.87 xcrop(cropm15,good)#6
                       + 8.34 xcrop(cropm16,good)#7
                       + 8.34 xcrop(cropm16,medium)#8
                       + 2.11 xcrop(cropm17,good)#9
                       + 2.11 xcrop(cropm17,medium)#10
                       + 9.44 xcrop(cropm19,good)#11
                       + 9.44 xcrop(cropm19,medium)#12
                       + 0.89 xcrop(cropm25,good)#13
                       + 0.89 xcrop(cropm25,medium)#14
                       + 4.92 xcrop(cropm29,good)#15
                       + 4.92 xcrop(cropm29,medium)#16
                       + 2.67 xcrop(cropm30,good)#17
                       + 2.67 xcrop(cropm30,medium)#18
                       + 1.09 xcrop(cropm33,good)#19
                       + 1.09 xcrop(cropm33,medium)#20
                       + 5.26 xcrop(cropm36,good)#21
                       + 5.26 xcrop(cropm36,medium)#22 + 7.075 xliver(recm1)#23
                       + 3.095 xliver(recm2)#24 + 3.195 xliver(recm3)#25
                       - flab(dec)#61 - tlab(dec)#73 - 25 plab#74 <= 0
 ddev(1960)#36:        - 602.060322854716 sales(banana)#42
                       - 4703.49984422058 sales(sugarmcane)#43
                       - 215.898400752587 sales(beansmcor)#45
                       - 62.2127194392644 sales(manioc)#47
                       - 34.3822170900693 sales(corn)#48 - pdev(1960)#76
                       + ndev(1960)#86  = 0
 ddev(1961)#37:        350.573491928632 sales(banana)#42
                       - 487.070308443245 sales(sugarmcane)#43
                       - 124.1768579492 sales(beansmcor)#45
                       - 1.32274451803741 sales(manioc)#47
                       + 13.6836027713626 sales(corn)#48 - pdev(1961)#77
                       + ndev(1961)#87  = 0
 ddev(1962)#38:        893.266779949023 sales(banana)#42
                       - 505.244573683664 sales(sugarmcane)#43
                       + 1065.85136406397 sales(beansmcor)#45
                       + 120.56652305318 sales(manioc)#47
                       + 85.5658198614319 sales(corn)#48 - pdev(1962)#78
                       + ndev(1962)#88  = 0
 ddev(1963)#39:        556.60577740017 sales(banana)#42
                       + 5528.61148613563 sales(sugarmcane)#43
                       + 31.0442144873003 sales(beansmcor)#45
                       + 9.17175744325124 sales(manioc)#47
                       - 7.53464203233256 sales(corn)#48 - pdev(1963)#79
                       + ndev(1963)#89  = 0
 ddev(1964)#40:        241.716227697536 sales(banana)#42
                       + 6164.71076955032 sales(sugarmcane)#43
                       - 394.637817497648 sales(beansmcor)#45
                       - 44.7218828371166 sales(manioc)#47
                       - 11.4318706697459 sales(corn)#48 - pdev(1964)#80
                       + ndev(1964)#90  = 0
 ddev(1965)#41:        - 161.852166525064 sales(banana)#42
                       + 2566.20625194724 sales(sugarmcane)#43
                       - 39.5108184383818 sales(beansmcor)#45
                       - 19.1415343064755 sales(manioc)#47
                       + 1.99191685912241 sales(corn)#48 - pdev(1965)#81
                       + ndev(1965)#91  = 0
 ddev(1966)#42:        - 854.82158028887 sales(banana)#42
                       - 7756.77640461107 sales(sugarmcane)#43
                       - 77.140169332079 sales(beansmcor)#45
                       - 21.6558420680342 sales(manioc)#47
                       - 15.7621247113164 sales(corn)#48 - pdev(1966)#82
                       + ndev(1966)#92  = 0
 ddev(1967)#43:        - 422.047578589635 sales(banana)#42
                       - 414.373247481566 sales(sugarmcane)#43
                       - 281.749764816557 sales(beansmcor)#45
                       + 16.0587743553469 sales(manioc)#47
                       - 10.5658198614319 sales(corn)#48 - pdev(1967)#83
                       + ndev(1967)#93  = 0
 ddev(1968)#44:        - 165.569243840272 sales(banana)#42
                       - 450.721777962404 sales(sugarmcane)#43
                       - 279.397930385701 sales(beansmcor)#45
                       + 0.535656870940797 sales(manioc)#47
                       - 22.6905311778291 sales(corn)#48 - pdev(1968)#84
                       + ndev(1968)#94  = 0
 ddev(1969)#45:        164.188615123194 sales(banana)#42
                       + 58.1576487693447 sales(sugarmcane)#43
                       + 315.616180620884 sales(beansmcor)#45
                       + 2.72201144620925 sales(manioc)#47
                       + 1.12586605080831 sales(corn)#48 - pdev(1969)#85
                       + ndev(1969)#95  = 0
 arev#46:              - 211 xlive#26 - 1900 sales(cottonmh)#41
                       - 4000 sales(banana)#42 - 35000 sales(sugarmcane)#43
                       - 2500 sales(beansmarr)#44 - 1500 sales(beansmcor)#45
                       - 10000 sales(oranges)#46 - 170 sales(manioc)#47
                       - 150 sales(corn)#48 - 1000 sales(sisal)#49 + revenue#96
                        = 0
 acrop#47:             - 122 xcrop(cropm02,good)#0
                       - 122 xcrop(cropm02,medium)#1 - 45 xcrop(cropm05,good)#2
                       - 45 xcrop(cropm05,medium)#3 - 36 xcrop(cropm10,good)#4
                       - 36 xcrop(cropm10,medium)#5 - 291 xcrop(cropm15,good)#6
                       - 20 xcrop(cropm16,good)#7 - 20 xcrop(cropm16,medium)#8
                       - 97 xcrop(cropm17,good)#9 - 97 xcrop(cropm17,medium)#10
                       - 45 xcrop(cropm29,good)#15
                       - 45 xcrop(cropm29,medium)#16
                       - 27 xcrop(cropm30,good)#17
                       - 27 xcrop(cropm30,medium)#18
                       - 53 xcrop(cropm33,good)#19
                       - 53 xcrop(cropm33,medium)#20
                       - 105 xcrop(cropm36,good)#21
                       - 105 xcrop(cropm36,medium)#22 + cropcost#97  = 0
 alab#48:              - 3 flab(jan)#50 - 3 flab(feb)#51 - 3 flab(mar)#52
                       - 3 flab(apr)#53 - 3 flab(may)#54 - 3 flab(jun)#55
                       - 3 flab(jul)#56 - 3 flab(aug)#57 - 3 flab(sep)#58
                       - 3 flab(oct)#59 - 3 flab(nov)#60 - 3 flab(dec)#61
                       - 10 tlab(jan)#62 - 10 tlab(feb)#63 - 10 tlab(mar)#64
                       - 10 tlab(apr)#65 - 10 tlab(may)#66 - 10 tlab(jun)#67
                       - 10 tlab(jul)#68 - 10 tlab(aug)#69 - 10 tlab(sep)#70
                       - 10 tlab(oct)#71 - 10 tlab(nov)#72 - 10 tlab(dec)#73
                       - 2054 plab#74 + labcost#98  = 0
 awcc#49:              10 tlab(jan)#62 + 10 tlab(feb)#63 + 10 tlab(mar)#64
                       + 10 tlab(apr)#65 + 10 tlab(may)#66 + 10 tlab(jun)#67
                       + 10 tlab(jul)#68 + 10 tlab(aug)#69 + 10 tlab(sep)#70
                       + 10 tlab(oct)#71 + 10 tlab(nov)#72 + 10 tlab(dec)#73
                       + 2054 plab#74 + rationr#75 + cropcost#97 + vetcost#99
                       <= 7123.2
 avet#50:              - xlive#26 + vetcost#99  = 0
Bounds
 0 <= xcrop(cropm19,good)#11 <= 0.036
 0 <= xcrop(cropm19,medium)#12 <= 0.298
      xlive#26 Free
      xprod(cottonmh)#29 Free
      xprod(banana)#30 Free
      xprod(sugarmcane)#31 Free
      xprod(beansmarr)#32 Free
      xprod(beansmcor)#33 Free
      xprod(oranges)#34 Free
      xprod(manioc)#35 Free
      xprod(corn)#36 Free
      xprod(sisal)#37 Free
 0 <= flab(jan)#50 <= 70.5
 0 <= flab(feb)#51 <= 70.5
 0 <= flab(mar)#52 <= 70.5
 0 <= flab(apr)#53 <= 70.5
 0 <= flab(may)#54 <= 70.5
 0 <= flab(jun)#55 <= 70.5
 0 <= flab(jul)#56 <= 70.5
 0 <= flab(aug)#57 <= 70.5
 0 <= flab(sep)#58 <= 70.5
 0 <= flab(oct)#59 <= 70.5
 0 <= flab(nov)#60 <= 70.5
 0 <= flab(dec)#61 <= 70.5
      rationr#75 Free
      revenue#96 Free
      cropcost#97 Free
      labcost#98 Free
      vetcost#99 Free
      constobj#100 = 0
End
