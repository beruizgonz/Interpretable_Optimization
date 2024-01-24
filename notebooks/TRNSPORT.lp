\ENCODING=ISO-8859-1
\Problem name: gamsmodel

Minimize
 _obj: 0.225 x(seattle,newmyork)#0 + 0.153 x(seattle,chicago)#1
       + 0.162 x(seattle,topeka)#2 + 0.225 x(sanmdiego,newmyork)#3
       + 0.162 x(sanmdiego,chicago)#4 + 0.126 x(sanmdiego,topeka)#5
       + constobj#6
Subject To
 supply(seattle)#0:   x(seattle,newmyork)#0 + x(seattle,chicago)#1
                      + x(seattle,topeka)#2 <= 350
 supply(sanmdiego)#1: x(sanmdiego,newmyork)#3 + x(sanmdiego,chicago)#4
                      + x(sanmdiego,topeka)#5 <= 600
 demand(newmyork)#2:  x(seattle,newmyork)#0 + x(sanmdiego,newmyork)#3 >= 325
 demand(chicago)#3:   x(seattle,chicago)#1 + x(sanmdiego,chicago)#4 >= 300
 demand(topeka)#4:    x(seattle,topeka)#2 + x(sanmdiego,topeka)#5 >= 275
Bounds
      constobj#6 = 0
End
