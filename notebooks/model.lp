\ Model Convert
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x7
Subject To
 e1: - 0.225 x1 - 0.153 x2 - 0.162 x3 - 0.225 x4 - 0.162 x5 - 0.126 x6 + x7
   = 0
 e2: x1 + x2 + x3 <= 350
 e3: x4 + x5 + x6 <= 600
 e4: x1 + x4 >= 325
 e5: x2 + x5 >= 300
 e6: x3 + x6 >= 275
Bounds
 x7 free
End
