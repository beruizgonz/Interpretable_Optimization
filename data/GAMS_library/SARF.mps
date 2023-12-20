* LP written by GAMS Convert at 12/18/23 13:41:06
*
* Equation counts
*     Total        E        G        L        N        X        C        B
*       532      149        0      383        0        0        0        0
*
* Variable counts
*                  x        b        i      s1s      s2s       sc       si
*     Total     cont   binary  integer     sos1     sos2    scont     sint
*       542      542        0        0        0        0        0        0
* FX      0
*
* Nonzero counts
*     Total    const       NL
*      3949     3949        0

NAME          Convert
*
* original model was maximizing
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
 E  e40
 E  e41
 E  e42
 E  e43
 E  e44
 E  e45
 E  e46
 E  e47
 E  e48
 E  e49
 E  e50
 E  e51
 E  e52
 E  e53
 E  e54
 E  e55
 E  e56
 E  e57
 E  e58
 E  e59
 E  e60
 E  e61
 E  e62
 E  e63
 E  e64
 E  e65
 E  e66
 E  e67
 E  e68
 E  e69
 E  e70
 E  e71
 E  e72
 E  e73
 E  e74
 E  e75
 E  e76
 E  e77
 E  e78
 E  e79
 E  e80
 E  e81
 E  e82
 E  e83
 E  e84
 E  e85
 E  e86
 E  e87
 E  e88
 E  e89
 E  e90
 E  e91
 E  e92
 E  e93
 E  e94
 E  e95
 E  e96
 E  e97
 E  e98
 E  e99
 E  e100
 E  e101
 E  e102
 E  e103
 E  e104
 E  e105
 E  e106
 E  e107
 E  e108
 E  e109
 E  e110
 E  e111
 E  e112
 E  e113
 E  e114
 E  e115
 E  e116
 E  e117
 E  e118
 E  e119
 E  e120
 E  e121
 E  e122
 E  e123
 E  e124
 E  e125
 E  e126
 E  e127
 E  e128
 E  e129
 E  e130
 E  e131
 E  e132
 E  e133
 E  e134
 E  e135
 E  e136
 E  e137
 E  e138
 E  e139
 E  e140
 E  e141
 E  e142
 L  e143
 L  e144
 L  e145
 L  e146
 L  e147
 L  e148
 L  e149
 L  e150
 L  e151
 L  e152
 L  e153
 L  e154
 L  e155
 L  e156
 L  e157
 L  e158
 L  e159
 L  e160
 L  e161
 L  e162
 L  e163
 L  e164
 L  e165
 L  e166
 L  e167
 L  e168
 L  e169
 L  e170
 L  e171
 L  e172
 L  e173
 L  e174
 L  e175
 L  e176
 L  e177
 L  e178
 L  e179
 L  e180
 L  e181
 L  e182
 L  e183
 L  e184
 L  e185
 L  e186
 L  e187
 L  e188
 L  e189
 L  e190
 L  e191
 L  e192
 L  e193
 L  e194
 L  e195
 L  e196
 L  e197
 L  e198
 L  e199
 L  e200
 L  e201
 L  e202
 L  e203
 L  e204
 L  e205
 L  e206
 L  e207
 L  e208
 L  e209
 L  e210
 L  e211
 L  e212
 L  e213
 L  e214
 L  e215
 L  e216
 L  e217
 L  e218
 L  e219
 L  e220
 L  e221
 L  e222
 L  e223
 L  e224
 L  e225
 L  e226
 L  e227
 L  e228
 L  e229
 L  e230
 L  e231
 L  e232
 L  e233
 L  e234
 L  e235
 L  e236
 L  e237
 L  e238
 L  e239
 L  e240
 L  e241
 L  e242
 L  e243
 L  e244
 L  e245
 L  e246
 L  e247
 L  e248
 L  e249
 L  e250
 L  e251
 L  e252
 L  e253
 L  e254
 L  e255
 L  e256
 L  e257
 L  e258
 L  e259
 L  e260
 L  e261
 L  e262
 L  e263
 L  e264
 L  e265
 L  e266
 L  e267
 L  e268
 L  e269
 L  e270
 L  e271
 L  e272
 L  e273
 L  e274
 L  e275
 L  e276
 L  e277
 L  e278
 L  e279
 L  e280
 L  e281
 L  e282
 L  e283
 L  e284
 L  e285
 L  e286
 L  e287
 L  e288
 L  e289
 L  e290
 L  e291
 L  e292
 L  e293
 L  e294
 L  e295
 L  e296
 L  e297
 L  e298
 L  e299
 L  e300
 L  e301
 L  e302
 L  e303
 L  e304
 L  e305
 L  e306
 L  e307
 L  e308
 L  e309
 L  e310
 L  e311
 L  e312
 L  e313
 L  e314
 L  e315
 L  e316
 L  e317
 L  e318
 L  e319
 L  e320
 L  e321
 L  e322
 L  e323
 L  e324
 L  e325
 L  e326
 L  e327
 L  e328
 L  e329
 L  e330
 L  e331
 L  e332
 L  e333
 L  e334
 L  e335
 L  e336
 L  e337
 L  e338
 L  e339
 L  e340
 L  e341
 L  e342
 L  e343
 L  e344
 L  e345
 L  e346
 L  e347
 L  e348
 L  e349
 L  e350
 L  e351
 L  e352
 L  e353
 L  e354
 L  e355
 L  e356
 L  e357
 L  e358
 L  e359
 L  e360
 L  e361
 L  e362
 L  e363
 L  e364
 L  e365
 L  e366
 L  e367
 L  e368
 L  e369
 L  e370
 L  e371
 L  e372
 L  e373
 L  e374
 L  e375
 L  e376
 L  e377
 L  e378
 L  e379
 L  e380
 L  e381
 L  e382
 L  e383
 L  e384
 L  e385
 L  e386
 L  e387
 L  e388
 L  e389
 L  e390
 L  e391
 L  e392
 L  e393
 L  e394
 L  e395
 L  e396
 L  e397
 L  e398
 L  e399
 L  e400
 L  e401
 L  e402
 L  e403
 L  e404
 L  e405
 L  e406
 L  e407
 L  e408
 L  e409
 L  e410
 L  e411
 L  e412
 L  e413
 L  e414
 L  e415
 L  e416
 L  e417
 L  e418
 L  e419
 L  e420
 L  e421
 L  e422
 L  e423
 L  e424
 L  e425
 L  e426
 L  e427
 L  e428
 L  e429
 L  e430
 L  e431
 L  e432
 L  e433
 L  e434
 L  e435
 L  e436
 L  e437
 L  e438
 L  e439
 L  e440
 L  e441
 L  e442
 L  e443
 L  e444
 L  e445
 L  e446
 L  e447
 L  e448
 L  e449
 L  e450
 L  e451
 L  e452
 L  e453
 L  e454
 L  e455
 L  e456
 L  e457
 L  e458
 L  e459
 L  e460
 L  e461
 L  e462
 L  e463
 L  e464
 L  e465
 L  e466
 L  e467
 L  e468
 L  e469
 L  e470
 L  e471
 L  e472
 L  e473
 L  e474
 L  e475
 L  e476
 L  e477
 L  e478
 L  e479
 L  e480
 L  e481
 L  e482
 L  e483
 L  e484
 L  e485
 L  e486
 L  e487
 L  e488
 L  e489
 L  e490
 L  e491
 L  e492
 L  e493
 L  e494
 L  e495
 L  e496
 L  e497
 L  e498
 L  e499
 L  e500
 L  e501
 L  e502
 L  e503
 L  e504
 L  e505
 L  e506
 L  e507
 L  e508
 L  e509
 L  e510
 L  e511
 L  e512
 L  e513
 L  e514
 L  e515
 L  e516
 L  e517
 L  e518
 L  e519
 L  e520
 L  e521
 L  e522
 L  e523
 L  e524
 L  e525
 E  e526
 E  e527
 E  e528
 E  e529
 E  e530
 E  e531
 E  e532
COLUMNS
    x1        e13                  1
    x1        e29                  1
    x1        e39                  1
    x1        e56                  1
    x1        e63                  1
    x1        e74                  1
    x1        e104                 1
    x1        e111                 1
    x1        e123             1.155
    x1        e136                 1
    x1        e143                 1
    x1        e149                 1
    x1        e150                 1
    x1        e151                 1
    x1        e152                 1
    x1        e153                 1
    x1        e154                 1
    x1        e155                 1
    x1        e156                 1
    x1        e157                 1
    x1        e158                 1
    x1        e159                 1
    x1        e161                 1
    x1        e162                 1
    x1        e163                 1
    x1        e164                 1
    x1        e165                 1
    x1        e166                 1
    x1        e167                 1
    x1        e168                 1
    x1        e169                 1
    x1        e170                 1
    x1        e171                 1
    x1        e172                 1
    x1        e178                 2
    x1        e183                 2
    x1        e527            -6.855
    x2        e14                  1
    x2        e30                  1
    x2        e40                  1
    x2        e57                  1
    x2        e64                  1
    x2        e75                  1
    x2        e105                 1
    x2        e112                 1
    x2        e124             1.155
    x2        e136                 1
    x2        e143                 1
    x2        e149                 1
    x2        e150                 1
    x2        e151                 1
    x2        e152                 1
    x2        e153                 1
    x2        e154                 1
    x2        e155                 1
    x2        e156                 1
    x2        e157                 1
    x2        e158                 1
    x2        e159                 1
    x2        e160                 1
    x2        e162                 1
    x2        e163                 1
    x2        e164                 1
    x2        e165                 1
    x2        e166                 1
    x2        e167                 1
    x2        e168                 1
    x2        e169                 1
    x2        e170                 1
    x2        e171                 1
    x2        e172                 1
    x2        e179                 2
    x2        e184                 2
    x2        e527            -6.855
    x3        e15                  1
    x3        e31                  1
    x3        e40                  1
    x3        e57                  1
    x3        e65                  1
    x3        e75                  1
    x3        e106                 1
    x3        e113                 1
    x3        e125             1.155
    x3        e136                 1
    x3        e143                 1
    x3        e149                 1
    x3        e150                 1
    x3        e151                 1
    x3        e152                 1
    x3        e153                 1
    x3        e154                 1
    x3        e155                 1
    x3        e156                 1
    x3        e157                 1
    x3        e158                 1
    x3        e159                 1
    x3        e160                 1
    x3        e161                 1
    x3        e163                 1
    x3        e164                 1
    x3        e165                 1
    x3        e166                 1
    x3        e167                 1
    x3        e168                 1
    x3        e169                 1
    x3        e170                 1
    x3        e171                 1
    x3        e172                 1
    x3        e179                 2
    x3        e185                 2
    x3        e527            -6.855
    x4        e16                  1
    x4        e32                  1
    x4        e41                  1
    x4        e58                  1
    x4        e65                  1
    x4        e75                  1
    x4        e106                 1
    x4        e113                 1
    x4        e125             1.155
    x4        e136                 1
    x4        e143                 1
    x4        e149                 1
    x4        e150                 1
    x4        e151                 1
    x4        e152                 1
    x4        e153                 1
    x4        e154                 1
    x4        e155                 1
    x4        e156                 1
    x4        e157                 1
    x4        e158                 1
    x4        e159                 1
    x4        e160                 1
    x4        e161                 1
    x4        e164                 1
    x4        e165                 1
    x4        e166                 1
    x4        e167                 1
    x4        e168                 1
    x4        e169                 1
    x4        e170                 1
    x4        e171                 1
    x4        e172                 1
    x4        e179                 2
    x4        e185                 2
    x4        e527            -6.855
    x5        e17                  1
    x5        e33                  1
    x5        e41                  1
    x5        e58                  1
    x5        e65                  1
    x5        e75                  1
    x5        e106                 1
    x5        e113                 1
    x5        e125             1.155
    x5        e136                 1
    x5        e143                 1
    x5        e149                 1
    x5        e150                 1
    x5        e151                 1
    x5        e152                 1
    x5        e153                 1
    x5        e154                 1
    x5        e155                 1
    x5        e156                 1
    x5        e157                 1
    x5        e158                 1
    x5        e159                 1
    x5        e160                 1
    x5        e161                 1
    x5        e165                 1
    x5        e166                 1
    x5        e167                 1
    x5        e168                 1
    x5        e169                 1
    x5        e170                 1
    x5        e171                 1
    x5        e172                 1
    x5        e179                 2
    x5        e185                 2
    x5        e527            -6.855
    x6        e17                  1
    x6        e32                  1
    x6        e39                  1
    x6        e56                  1
    x6        e63                  1
    x6        e74                  1
    x6        e104                 1
    x6        e111                 1
    x6        e123             1.155
    x6        e136                 1
    x6        e143                 1
    x6        e149                 1
    x6        e150                 1
    x6        e151                 1
    x6        e152                 1
    x6        e153                 1
    x6        e154                 1
    x6        e155                 1
    x6        e156                 1
    x6        e157                 1
    x6        e158                 1
    x6        e159                 1
    x6        e165                 1
    x6        e166                 1
    x6        e167                 1
    x6        e168                 1
    x6        e169                 1
    x6        e170                 1
    x6        e171                 1
    x6        e172                 1
    x6        e178                 2
    x6        e183                 2
    x6        e527            -6.855
    x7        e17                  1
    x7        e33                  1
    x7        e39                  1
    x7        e56                  1
    x7        e63                  1
    x7        e74                  1
    x7        e104                 1
    x7        e111                 1
    x7        e123             1.155
    x7        e136                 1
    x7        e143                 1
    x7        e149                 1
    x7        e150                 1
    x7        e151                 1
    x7        e152                 1
    x7        e153                 1
    x7        e154                 1
    x7        e155                 1
    x7        e156                 1
    x7        e157                 1
    x7        e158                 1
    x7        e159                 1
    x7        e165                 1
    x7        e166                 1
    x7        e167                 1
    x7        e168                 1
    x7        e169                 1
    x7        e170                 1
    x7        e171                 1
    x7        e172                 1
    x7        e178                 2
    x7        e183                 2
    x7        e527            -6.855
    x8        e18                  1
    x8        e23                  1
    x8        e35                  1
    x8        e43                  1
    x8        e59                  1
    x8        e73                  1
    x8        e84                  1
    x8        e114                 1
    x8        e128              1.34
    x8        e137                 1
    x8        e144                 1
    x8        e149                 1
    x8        e150                 1
    x8        e151                 1
    x8        e152                 1
    x8        e153                 1
    x8        e154                 1
    x8        e155                 1
    x8        e156                 1
    x8        e157                 1
    x8        e158                 1
    x8        e159                 1
    x8        e160                 1
    x8        e161                 1
    x8        e162                 1
    x8        e163                 1
    x8        e164                 1
    x8        e171                 1
    x8        e172                 1
    x8        e177                20
    x8        e188                 1
    x8        e527              -4.5
    x9        e18                  1
    x9        e24                  1
    x9        e36                  1
    x9        e44                  1
    x9        e59                  1
    x9        e74                  1
    x9        e84                  1
    x9        e114                 1
    x9        e128              1.34
    x9        e137                 1
    x9        e144                 1
    x9        e149                 1
    x9        e150                 1
    x9        e151                 1
    x9        e152                 1
    x9        e153                 1
    x9        e154                 1
    x9        e155                 1
    x9        e156                 1
    x9        e157                 1
    x9        e158                 1
    x9        e159                 1
    x9        e160                 1
    x9        e161                 1
    x9        e162                 1
    x9        e163                 1
    x9        e164                 1
    x9        e171                 1
    x9        e172                 1
    x9        e178                20
    x9        e188                 1
    x9        e527              -4.5
    x10       e18                  1
    x10       e25                  1
    x10       e37                  1
    x10       e45                  1
    x10       e59                  1
    x10       e74                  1
    x10       e84                  1
    x10       e114                 1
    x10       e128              1.34
    x10       e137                 1
    x10       e144                 1
    x10       e149                 1
    x10       e150                 1
    x10       e151                 1
    x10       e152                 1
    x10       e153                 1
    x10       e154                 1
    x10       e155                 1
    x10       e156                 1
    x10       e157                 1
    x10       e158                 1
    x10       e159                 1
    x10       e160                 1
    x10       e161                 1
    x10       e162                 1
    x10       e163                 1
    x10       e164                 1
    x10       e171                 1
    x10       e172                 1
    x10       e178                20
    x10       e188                 1
    x10       e527              -4.5
    x11       e7                   1
    x11       e23                  1
    x11       e35                  1
    x11       e43                  1
    x11       e60                  1
    x11       e73                  1
    x11       e85                  1
    x11       e115                 1
    x11       e129              1.34
    x11       e137                 1
    x11       e144                 1
    x11       e149                 1
    x11       e150                 1
    x11       e151                 1
    x11       e152                 1
    x11       e153                 1
    x11       e154                 1
    x11       e155                 1
    x11       e156                 1
    x11       e157                 1
    x11       e158                 1
    x11       e159                 1
    x11       e160                 1
    x11       e161                 1
    x11       e162                 1
    x11       e163                 1
    x11       e164                 1
    x11       e165                 1
    x11       e177                20
    x11       e189                 1
    x11       e527              -4.5
    x12       e8                   1
    x12       e23                  1
    x12       e35                  1
    x12       e43                  1
    x12       e60                  1
    x12       e73                  1
    x12       e85                  1
    x12       e115                 1
    x12       e129              1.34
    x12       e137                 1
    x12       e144                 1
    x12       e150                 1
    x12       e151                 1
    x12       e152                 1
    x12       e153                 1
    x12       e154                 1
    x12       e155                 1
    x12       e156                 1
    x12       e157                 1
    x12       e158                 1
    x12       e159                 1
    x12       e160                 1
    x12       e161                 1
    x12       e162                 1
    x12       e163                 1
    x12       e164                 1
    x12       e165                 1
    x12       e177                20
    x12       e189                 1
    x12       e527              -4.5
    x13       e11                  1
    x13       e25                  1
    x13       e37                  1
    x13       e45                  1
    x13       e60                  1
    x13       e74                  1
    x13       e85                  1
    x13       e115                 1
    x13       e129              1.34
    x13       e137                 1
    x13       e144                 1
    x13       e153                 1
    x13       e154                 1
    x13       e155                 1
    x13       e156                 1
    x13       e157                 1
    x13       e158                 1
    x13       e159                 1
    x13       e160                 1
    x13       e161                 1
    x13       e162                 1
    x13       e163                 1
    x13       e164                 1
    x13       e165                 1
    x13       e178                20
    x13       e189                 1
    x13       e527              -4.5
    x14       e12                  1
    x14       e25                  1
    x14       e37                  1
    x14       e45                  1
    x14       e59                  1
    x14       e74                  1
    x14       e84                  1
    x14       e114                 1
    x14       e128              1.34
    x14       e137                 1
    x14       e144                 1
    x14       e154                 1
    x14       e155                 1
    x14       e156                 1
    x14       e157                 1
    x14       e158                 1
    x14       e159                 1
    x14       e160                 1
    x14       e161                 1
    x14       e162                 1
    x14       e163                 1
    x14       e164                 1
    x14       e178                20
    x14       e188                 1
    x14       e527              -4.5
    x15       e27                  1
    x15       e52                  1
    x15       e71                  1
    x15       e78                  1
    x15       e88                  1
    x15       e116                 1
    x15       e131                 9
    x15       e138                 1
    x15       e145                 1
    x15       e160                 1
    x15       e161                 1
    x15       e162                 1
    x15       e163                 1
    x15       e164                 1
    x15       e165                 1
    x15       e166                 1
    x15       e167                 1
    x15       e188                 8
    x15       e527             -5.75
    x16       e27                  1
    x16       e53                  1
    x16       e72                  1
    x16       e79                  1
    x16       e89                  1
    x16       e117                 1
    x16       e132                 9
    x16       e138                 1
    x16       e145                 1
    x16       e160                 1
    x16       e161                 1
    x16       e162                 1
    x16       e163                 1
    x16       e164                 1
    x16       e165                 1
    x16       e166                 1
    x16       e167                 1
    x16       e168                 1
    x16       e189                 8
    x16       e527             -5.75
    x17       e28                  1
    x17       e52                  1
    x17       e71                  1
    x17       e78                  1
    x17       e88                  1
    x17       e116                 1
    x17       e131                 9
    x17       e138                 1
    x17       e145                 1
    x17       e161                 1
    x17       e162                 1
    x17       e163                 1
    x17       e164                 1
    x17       e165                 1
    x17       e166                 1
    x17       e167                 1
    x17       e188                 8
    x17       e527             -5.75
    x18       e28                  1
    x18       e53                  1
    x18       e72                  1
    x18       e79                  1
    x18       e89                  1
    x18       e117                 1
    x18       e132                 9
    x18       e138                 1
    x18       e145                 1
    x18       e161                 1
    x18       e162                 1
    x18       e163                 1
    x18       e164                 1
    x18       e165                 1
    x18       e166                 1
    x18       e167                 1
    x18       e168                 1
    x18       e189                 8
    x18       e527             -5.75
    x19       e28                  1
    x19       e54                  1
    x19       e72                  1
    x19       e78                  1
    x19       e89                  1
    x19       e117                 1
    x19       e132                 9
    x19       e138                 1
    x19       e145                 1
    x19       e161                 1
    x19       e162                 1
    x19       e163                 1
    x19       e164                 1
    x19       e165                 1
    x19       e166                 1
    x19       e167                 1
    x19       e168                 1
    x19       e188                 8
    x19       e527             -5.75
    x20       e27                  1
    x20       e53                  1
    x20       e72                  1
    x20       e78                  1
    x20       e89                  1
    x20       e117                 1
    x20       e132                 9
    x20       e138                 1
    x20       e145                 1
    x20       e160                 1
    x20       e161                 1
    x20       e162                 1
    x20       e163                 1
    x20       e164                 1
    x20       e165                 1
    x20       e166                 1
    x20       e167                 1
    x20       e168                 1
    x20       e188                 8
    x20       e527             -5.75
    x21       e28                  1
    x21       e52                  1
    x21       e71                  1
    x21       e78                  1
    x21       e88                  1
    x21       e116                 1
    x21       e132                 9
    x21       e138                 1
    x21       e145                 1
    x21       e161                 1
    x21       e162                 1
    x21       e163                 1
    x21       e164                 1
    x21       e165                 1
    x21       e166                 1
    x21       e167                 1
    x21       e168                 1
    x21       e188                 8
    x21       e527             -5.75
    x22       e18                  1
    x22       e25                  1
    x22       e37                  1
    x22       e45                  1
    x22       e51                  1
    x22       e52                  1
    x22       e54                  1
    x22       e56                  1
    x22       e61                  1
    x22       e74                  1
    x22       e76                  1
    x22       e77                  1
    x22       e78                  1
    x22       e80                  1
    x22       e94                  1
    x22       e95                  1
    x22       e96                  1
    x22       e97                  1
    x22       e98                  1
    x22       e99                  1
    x22       e100                 1
    x22       e101                 1
    x22       e102                 1
    x22       e103                 1
    x22       e105                 1
    x22       e107                 1
    x22       e108                 1
    x22       e109                 1
    x22       e110                 1
    x22       e124               1.2
    x22       e126               1.2
    x22       e127               1.2
    x22       e129               1.2
    x22       e131               1.2
    x22       e139                 1
    x22       e146                 1
    x22       e149                 1
    x22       e150                 1
    x22       e151                 1
    x22       e152                 1
    x22       e153                 1
    x22       e154                 1
    x22       e155                 1
    x22       e156                 1
    x22       e157                 1
    x22       e158                 1
    x22       e159                 1
    x22       e160                 1
    x22       e161                 1
    x22       e162                 1
    x22       e163                 1
    x22       e164                 1
    x22       e165                 1
    x22       e166                 1
    x22       e167                 1
    x22       e171                 1
    x22       e172                 1
    x22       e527             -3.13
    x23       e19                  1
    x23       e25                  1
    x23       e37                  1
    x23       e45                  1
    x23       e51                  1
    x23       e52                  1
    x23       e54                  1
    x23       e56                  1
    x23       e61                  1
    x23       e74                  1
    x23       e76                  1
    x23       e77                  1
    x23       e78                  1
    x23       e80                  1
    x23       e94                  1
    x23       e95                  1
    x23       e96                  1
    x23       e97                  1
    x23       e98                  1
    x23       e99                  1
    x23       e100                 1
    x23       e101                 1
    x23       e102                 1
    x23       e103                 1
    x23       e105                 1
    x23       e107                 1
    x23       e108                 1
    x23       e109                 1
    x23       e110                 1
    x23       e124               1.2
    x23       e126               1.2
    x23       e127               1.2
    x23       e129               1.2
    x23       e131               1.2
    x23       e139                 1
    x23       e146                 1
    x23       e149                 1
    x23       e150                 1
    x23       e151                 1
    x23       e152                 1
    x23       e153                 1
    x23       e154                 1
    x23       e155                 1
    x23       e156                 1
    x23       e157                 1
    x23       e158                 1
    x23       e159                 1
    x23       e160                 1
    x23       e161                 1
    x23       e162                 1
    x23       e163                 1
    x23       e164                 1
    x23       e165                 1
    x23       e166                 1
    x23       e167                 1
    x23       e172                 1
    x23       e527             -3.13
    x24       e7                   1
    x24       e25                  1
    x24       e37                  1
    x24       e45                  1
    x24       e51                  1
    x24       e52                  1
    x24       e54                  1
    x24       e56                  1
    x24       e61                  1
    x24       e74                  1
    x24       e76                  1
    x24       e77                  1
    x24       e78                  1
    x24       e80                  1
    x24       e94                  1
    x24       e95                  1
    x24       e96                  1
    x24       e97                  1
    x24       e98                  1
    x24       e99                  1
    x24       e100                 1
    x24       e101                 1
    x24       e102                 1
    x24       e103                 1
    x24       e105                 1
    x24       e107                 1
    x24       e108                 1
    x24       e109                 1
    x24       e110                 1
    x24       e124               1.2
    x24       e126               1.2
    x24       e127               1.2
    x24       e129               1.2
    x24       e131               1.2
    x24       e139                 1
    x24       e146                 1
    x24       e149                 1
    x24       e150                 1
    x24       e151                 1
    x24       e152                 1
    x24       e153                 1
    x24       e154                 1
    x24       e155                 1
    x24       e156                 1
    x24       e157                 1
    x24       e158                 1
    x24       e159                 1
    x24       e160                 1
    x24       e161                 1
    x24       e162                 1
    x24       e163                 1
    x24       e164                 1
    x24       e165                 1
    x24       e166                 1
    x24       e167                 1
    x24       e527             -3.13
    x25       e8                   1
    x25       e26                  1
    x25       e38                  1
    x25       e45                  1
    x25       e51                  1
    x25       e52                  1
    x25       e54                  1
    x25       e56                  1
    x25       e61                  1
    x25       e74                  1
    x25       e76                  1
    x25       e77                  1
    x25       e78                  1
    x25       e80                  1
    x25       e94                  1
    x25       e95                  1
    x25       e96                  1
    x25       e97                  1
    x25       e98                  1
    x25       e99                  1
    x25       e100                 1
    x25       e101                 1
    x25       e102                 1
    x25       e103                 1
    x25       e105                 1
    x25       e107                 1
    x25       e108                 1
    x25       e109                 1
    x25       e110                 1
    x25       e124               1.2
    x25       e126               1.2
    x25       e127               1.2
    x25       e129               1.2
    x25       e131               1.2
    x25       e139                 1
    x25       e146                 1
    x25       e150                 1
    x25       e151                 1
    x25       e152                 1
    x25       e153                 1
    x25       e154                 1
    x25       e155                 1
    x25       e156                 1
    x25       e157                 1
    x25       e158                 1
    x25       e159                 1
    x25       e160                 1
    x25       e161                 1
    x25       e162                 1
    x25       e163                 1
    x25       e164                 1
    x25       e165                 1
    x25       e166                 1
    x25       e167                 1
    x25       e527             -3.13
    x26       e9                   1
    x26       e26                  1
    x26       e38                  1
    x26       e46                  1
    x26       e51                  1
    x26       e52                  1
    x26       e54                  1
    x26       e56                  1
    x26       e62                  1
    x26       e75                  1
    x26       e76                  1
    x26       e77                  1
    x26       e78                  1
    x26       e80                  1
    x26       e94                  1
    x26       e95                  1
    x26       e96                  1
    x26       e97                  1
    x26       e98                  1
    x26       e99                  1
    x26       e100                 1
    x26       e101                 1
    x26       e102                 1
    x26       e103                 1
    x26       e105                 1
    x26       e107                 1
    x26       e108                 1
    x26       e109                 1
    x26       e110                 1
    x26       e124               1.2
    x26       e126               1.2
    x26       e127               1.2
    x26       e129               1.2
    x26       e131               1.2
    x26       e139                 1
    x26       e146                 1
    x26       e151                 1
    x26       e152                 1
    x26       e153                 1
    x26       e154                 1
    x26       e155                 1
    x26       e156                 1
    x26       e157                 1
    x26       e158                 1
    x26       e159                 1
    x26       e160                 1
    x26       e161                 1
    x26       e162                 1
    x26       e163                 1
    x26       e164                 1
    x26       e165                 1
    x26       e166                 1
    x26       e167                 1
    x26       e527             -3.13
    x27       e10                  1
    x27       e25                  1
    x27       e37                  1
    x27       e46                  1
    x27       e51                  1
    x27       e52                  1
    x27       e54                  1
    x27       e56                  1
    x27       e62                  1
    x27       e75                  1
    x27       e76                  1
    x27       e77                  1
    x27       e78                  1
    x27       e80                  1
    x27       e94                  1
    x27       e95                  1
    x27       e96                  1
    x27       e97                  1
    x27       e98                  1
    x27       e99                  1
    x27       e100                 1
    x27       e101                 1
    x27       e102                 1
    x27       e103                 1
    x27       e105                 1
    x27       e107                 1
    x27       e108                 1
    x27       e109                 1
    x27       e110                 1
    x27       e124               1.2
    x27       e126               1.2
    x27       e127               1.2
    x27       e129               1.2
    x27       e131               1.2
    x27       e139                 1
    x27       e146                 1
    x27       e152                 1
    x27       e153                 1
    x27       e154                 1
    x27       e155                 1
    x27       e156                 1
    x27       e157                 1
    x27       e158                 1
    x27       e159                 1
    x27       e160                 1
    x27       e161                 1
    x27       e162                 1
    x27       e163                 1
    x27       e164                 1
    x27       e165                 1
    x27       e166                 1
    x27       e167                 1
    x27       e527             -3.13
    x28       e11                  1
    x28       e26                  1
    x28       e38                  1
    x28       e46                  1
    x28       e51                  1
    x28       e52                  1
    x28       e54                  1
    x28       e56                  1
    x28       e62                  1
    x28       e75                  1
    x28       e76                  1
    x28       e77                  1
    x28       e78                  1
    x28       e80                  1
    x28       e94                  1
    x28       e95                  1
    x28       e96                  1
    x28       e97                  1
    x28       e98                  1
    x28       e99                  1
    x28       e100                 1
    x28       e101                 1
    x28       e102                 1
    x28       e103                 1
    x28       e105                 1
    x28       e107                 1
    x28       e108                 1
    x28       e109                 1
    x28       e110                 1
    x28       e124               1.2
    x28       e126               1.2
    x28       e127               1.2
    x28       e129               1.2
    x28       e131               1.2
    x28       e139                 1
    x28       e146                 1
    x28       e153                 1
    x28       e154                 1
    x28       e155                 1
    x28       e156                 1
    x28       e157                 1
    x28       e158                 1
    x28       e159                 1
    x28       e160                 1
    x28       e161                 1
    x28       e162                 1
    x28       e163                 1
    x28       e164                 1
    x28       e165                 1
    x28       e166                 1
    x28       e167                 1
    x28       e527             -3.13
    x29       e12                  1
    x29       e25                  1
    x29       e37                  1
    x29       e46                  1
    x29       e51                  1
    x29       e52                  1
    x29       e54                  1
    x29       e56                  1
    x29       e62                  1
    x29       e75                  1
    x29       e76                  1
    x29       e77                  1
    x29       e78                  1
    x29       e80                  1
    x29       e94                  1
    x29       e95                  1
    x29       e96                  1
    x29       e97                  1
    x29       e98                  1
    x29       e99                  1
    x29       e100                 1
    x29       e101                 1
    x29       e102                 1
    x29       e103                 1
    x29       e105                 1
    x29       e107                 1
    x29       e108                 1
    x29       e109                 1
    x29       e110                 1
    x29       e124               1.2
    x29       e126               1.2
    x29       e127               1.2
    x29       e129               1.2
    x29       e131               1.2
    x29       e139                 1
    x29       e146                 1
    x29       e154                 1
    x29       e155                 1
    x29       e156                 1
    x29       e157                 1
    x29       e158                 1
    x29       e159                 1
    x29       e160                 1
    x29       e161                 1
    x29       e162                 1
    x29       e163                 1
    x29       e164                 1
    x29       e165                 1
    x29       e166                 1
    x29       e167                 1
    x29       e527             -3.13
    x30       e19                  1
    x30       e20                  1
    x30       e34                  1
    x30       e38                  1
    x30       e42                  1
    x30       e47                  1
    x30       e48                  1
    x30       e49                  1
    x30       e52                  1
    x30       e53                  1
    x30       e54                  1
    x30       e66                  1
    x30       e74                  1
    x30       e81                  1
    x30       e83                  1
    x30       e90                0.5
    x30       e91                0.5
    x30       e132             7.525
    x30       e133             7.525
    x30       e140                 1
    x30       e147                 1
    x30       e149                 1
    x30       e150                 1
    x30       e151                 1
    x30       e152                 1
    x30       e153                 1
    x30       e154                 1
    x30       e155                 1
    x30       e156                 1
    x30       e157                 1
    x30       e158                 1
    x30       e159                 1
    x30       e160                 1
    x30       e161                 1
    x30       e162                 1
    x30       e163                 1
    x30       e164                 1
    x30       e165                 1
    x30       e166                 1
    x30       e167                 1
    x30       e168                 1
    x30       e169                 1
    x30       e172                 1
    x30       e178                26
    x30       e180                 6
    x30       e192              16.5
    x30       e193              16.5
    x30       e527                -8
    x31       e7                   1
    x31       e21                  1
    x31       e35                  1
    x31       e38                  1
    x31       e43                  1
    x31       e47                  1
    x31       e48                  1
    x31       e49                  1
    x31       e52                  1
    x31       e53                  1
    x31       e54                  1
    x31       e66                  1
    x31       e74                  1
    x31       e81                  1
    x31       e83                  1
    x31       e90                0.5
    x31       e91                0.5
    x31       e132             7.525
    x31       e133             7.525
    x31       e140                 1
    x31       e147                 1
    x31       e149                 1
    x31       e150                 1
    x31       e151                 1
    x31       e152                 1
    x31       e153                 1
    x31       e154                 1
    x31       e155                 1
    x31       e156                 1
    x31       e157                 1
    x31       e158                 1
    x31       e159                 1
    x31       e160                 1
    x31       e161                 1
    x31       e162                 1
    x31       e163                 1
    x31       e164                 1
    x31       e165                 1
    x31       e166                 1
    x31       e167                 1
    x31       e168                 1
    x31       e169                 1
    x31       e178                26
    x31       e180                 6
    x31       e192              16.5
    x31       e193              16.5
    x31       e527                -8
    x32       e8                   1
    x32       e22                  1
    x32       e35                  1
    x32       e38                  1
    x32       e42                  1
    x32       e47                  1
    x32       e48                  1
    x32       e49                  1
    x32       e52                  1
    x32       e53                  1
    x32       e54                  1
    x32       e66                  1
    x32       e74                  1
    x32       e81                  1
    x32       e83                  1
    x32       e90                0.5
    x32       e91                0.5
    x32       e132             7.525
    x32       e133             7.525
    x32       e140                 1
    x32       e147                 1
    x32       e150                 1
    x32       e151                 1
    x32       e152                 1
    x32       e153                 1
    x32       e154                 1
    x32       e155                 1
    x32       e156                 1
    x32       e157                 1
    x32       e158                 1
    x32       e159                 1
    x32       e160                 1
    x32       e161                 1
    x32       e162                 1
    x32       e163                 1
    x32       e164                 1
    x32       e165                 1
    x32       e166                 1
    x32       e167                 1
    x32       e168                 1
    x32       e169                 1
    x32       e178                26
    x32       e180                 6
    x32       e192              16.5
    x32       e193              16.5
    x32       e527                -8
    x33       e9                   1
    x33       e23                  1
    x33       e35                  1
    x33       e38                  1
    x33       e43                  1
    x33       e48                  1
    x33       e49                  1
    x33       e50                  1
    x33       e51                  1
    x33       e53                  1
    x33       e54                  1
    x33       e67                  1
    x33       e75                  1
    x33       e82                  1
    x33       e84                  1
    x33       e91                0.5
    x33       e92                0.5
    x33       e133             7.525
    x33       e134             7.525
    x33       e140                 1
    x33       e147                 1
    x33       e151                 1
    x33       e152                 1
    x33       e153                 1
    x33       e154                 1
    x33       e155                 1
    x33       e156                 1
    x33       e157                 1
    x33       e158                 1
    x33       e159                 1
    x33       e160                 1
    x33       e161                 1
    x33       e162                 1
    x33       e163                 1
    x33       e164                 1
    x33       e165                 1
    x33       e166                 1
    x33       e167                 1
    x33       e168                 1
    x33       e169                 1
    x33       e170                 1
    x33       e179                26
    x33       e181                 6
    x33       e193              16.5
    x33       e194              16.5
    x33       e527                -8
    x34       e9                   1
    x34       e23                  1
    x34       e36                  1
    x34       e38                  1
    x34       e44                  1
    x34       e48                  1
    x34       e49                  1
    x34       e50                  1
    x34       e51                  1
    x34       e53                  1
    x34       e54                  1
    x34       e67                  1
    x34       e75                  1
    x34       e82                  1
    x34       e84                  1
    x34       e91                0.5
    x34       e92                0.5
    x34       e133             7.525
    x34       e134             7.525
    x34       e140                 1
    x34       e147                 1
    x34       e151                 1
    x34       e152                 1
    x34       e153                 1
    x34       e154                 1
    x34       e155                 1
    x34       e156                 1
    x34       e157                 1
    x34       e158                 1
    x34       e159                 1
    x34       e160                 1
    x34       e161                 1
    x34       e162                 1
    x34       e163                 1
    x34       e164                 1
    x34       e165                 1
    x34       e166                 1
    x34       e167                 1
    x34       e168                 1
    x34       e169                 1
    x34       e170                 1
    x34       e179                26
    x34       e181                 6
    x34       e193              16.5
    x34       e194              16.5
    x34       e527                -8
    x35       e9                   1
    x35       e23                  1
    x35       e36                  1
    x35       e38                  1
    x35       e44                  1
    x35       e49                  1
    x35       e50                  1
    x35       e51                  1
    x35       e52                  1
    x35       e53                  1
    x35       e54                  1
    x35       e68                  1
    x35       e75                  1
    x35       e83                  1
    x35       e85                  1
    x35       e92                0.5
    x35       e93                0.5
    x35       e134             7.525
    x35       e135             7.525
    x35       e140                 1
    x35       e147                 1
    x35       e151                 1
    x35       e152                 1
    x35       e153                 1
    x35       e154                 1
    x35       e155                 1
    x35       e156                 1
    x35       e157                 1
    x35       e158                 1
    x35       e159                 1
    x35       e160                 1
    x35       e161                 1
    x35       e162                 1
    x35       e163                 1
    x35       e164                 1
    x35       e165                 1
    x35       e166                 1
    x35       e167                 1
    x35       e168                 1
    x35       e169                 1
    x35       e170                 1
    x35       e171                 1
    x35       e180                26
    x35       e182                 6
    x35       e194              16.5
    x35       e195              16.5
    x35       e527                -8
    x36       e9                   1
    x36       e23                  1
    x36       e48                  1
    x36       e49                  1
    x36       e50                  1
    x36       e51                  1
    x36       e52                  1
    x36       e53                  1
    x36       e69                  1
    x36       e86                  1
    x36       e87                  1
    x36       e118                 1
    x36       e128               3.5
    x36       e141                 1
    x36       e148                 1
    x36       e151                 1
    x36       e152                 1
    x36       e153                 1
    x36       e154                 1
    x36       e155                 1
    x36       e156                 1
    x36       e157                 1
    x36       e158                 1
    x36       e159                 1
    x36       e160                 1
    x36       e161                 1
    x36       e162                 1
    x36       e163                 1
    x36       e164                 1
    x36       e182               1.3
    x36       e183             11.05
    x36       e184             11.05
    x36       e185               1.3
    x36       e186               1.3
    x36       e187               1.3
    x36       e188              11.7
    x36       e527            -4.425
    x37       e10                  1
    x37       e24                  1
    x37       e49                  1
    x37       e50                  1
    x37       e51                  1
    x37       e52                  1
    x37       e53                  1
    x37       e54                  1
    x37       e70                  1
    x37       e86                  1
    x37       e87                  1
    x37       e119                 1
    x37       e129               3.5
    x37       e141                 1
    x37       e148                 1
    x37       e152                 1
    x37       e153                 1
    x37       e154                 1
    x37       e155                 1
    x37       e156                 1
    x37       e157                 1
    x37       e158                 1
    x37       e159                 1
    x37       e160                 1
    x37       e161                 1
    x37       e162                 1
    x37       e163                 1
    x37       e164                 1
    x37       e165                 1
    x37       e183             11.05
    x37       e184             11.05
    x37       e185               1.3
    x37       e186               1.3
    x37       e187               1.3
    x37       e188               1.3
    x37       e189              11.7
    x37       e527            -4.425
    x38       e11                  1
    x38       e25                  1
    x38       e48                  1
    x38       e49                  1
    x38       e50                  1
    x38       e51                  1
    x38       e52                  1
    x38       e53                  1
    x38       e69                  1
    x38       e86                  1
    x38       e87                  1
    x38       e118               0.5
    x38       e119               0.5
    x38       e128              1.75
    x38       e129              1.75
    x38       e141                 1
    x38       e148                 1
    x38       e153                 1
    x38       e154                 1
    x38       e155                 1
    x38       e156                 1
    x38       e157                 1
    x38       e158                 1
    x38       e159                 1
    x38       e160                 1
    x38       e161                 1
    x38       e162                 1
    x38       e163                 1
    x38       e164                 1
    x38       e165                 1
    x38       e182               1.3
    x38       e183             11.05
    x38       e184             11.05
    x38       e185               1.3
    x38       e186               1.3
    x38       e187               1.3
    x38       e188              5.85
    x38       e189              5.85
    x38       e527            -4.425
    x39       e12                  1
    x39       e26                  1
    x39       e48                  1
    x39       e49                  1
    x39       e50                  1
    x39       e51                  1
    x39       e52                  1
    x39       e53                  1
    x39       e69                  1
    x39       e86                  1
    x39       e87                  1
    x39       e118               0.5
    x39       e119               0.5
    x39       e128              1.75
    x39       e129              1.75
    x39       e141                 1
    x39       e148                 1
    x39       e154                 1
    x39       e155                 1
    x39       e156                 1
    x39       e157                 1
    x39       e158                 1
    x39       e159                 1
    x39       e160                 1
    x39       e161                 1
    x39       e162                 1
    x39       e163                 1
    x39       e164                 1
    x39       e165                 1
    x39       e182               1.3
    x39       e183             11.05
    x39       e184             11.05
    x39       e185               1.3
    x39       e186               1.3
    x39       e187               1.3
    x39       e188              5.85
    x39       e189              5.85
    x39       e527            -4.425
    x40       e12                  1
    x40       e26                  1
    x40       e48                  1
    x40       e49                  1
    x40       e50                  1
    x40       e51                  1
    x40       e54                  1
    x40       e56                  1
    x40       e70                  1
    x40       e86                  1
    x40       e87                  1
    x40       e119              0.34
    x40       e120              0.33
    x40       e121              0.33
    x40       e129              1.19
    x40       e130             1.155
    x40       e131             1.155
    x40       e141                 1
    x40       e148                 1
    x40       e154                 1
    x40       e155                 1
    x40       e156                 1
    x40       e157                 1
    x40       e158                 1
    x40       e159                 1
    x40       e160                 1
    x40       e161                 1
    x40       e162                 1
    x40       e163                 1
    x40       e164                 1
    x40       e165                 1
    x40       e166                 1
    x40       e167                 1
    x40       e182               1.3
    x40       e183             11.05
    x40       e184             11.05
    x40       e185               1.3
    x40       e188               1.3
    x40       e189               3.9
    x40       e190               5.2
    x40       e191               3.9
    x40       e527            -4.425
    x41       e9                   1
    x41       e23                  1
    x41       e49                  1
    x41       e50                  1
    x41       e51                  1
    x41       e54                  1
    x41       e56                  1
    x41       e57                  1
    x41       e70                  1
    x41       e86                  1
    x41       e87                  1
    x41       e119              0.25
    x41       e120              0.25
    x41       e121              0.25
    x41       e122              0.25
    x41       e129             0.875
    x41       e130             0.875
    x41       e131             0.875
    x41       e132             0.875
    x41       e141                 1
    x41       e148                 1
    x41       e151                 1
    x41       e152                 1
    x41       e153                 1
    x41       e154                 1
    x41       e155                 1
    x41       e156                 1
    x41       e157                 1
    x41       e158                 1
    x41       e159                 1
    x41       e160                 1
    x41       e161                 1
    x41       e162                 1
    x41       e163                 1
    x41       e164                 1
    x41       e165                 1
    x41       e166                 1
    x41       e167                 1
    x41       e168                 1
    x41       e183             11.05
    x41       e184             11.05
    x41       e185               1.3
    x41       e188               1.3
    x41       e189              2.93
    x41       e190              4.23
    x41       e191              4.23
    x41       e192              2.93
    x41       e527            -4.425
    x42       e11                  1
    x42       e26                  1
    x42       e48                  1
    x42       e49                  1
    x42       e50                  1
    x42       e51                  1
    x42       e52                  1
    x42       e53                  1
    x42       e69                  1
    x42       e86                  1
    x42       e87                  1
    x42       e118                 1
    x42       e128               3.5
    x42       e141                 1
    x42       e148                 1
    x42       e153                 1
    x42       e154                 1
    x42       e155                 1
    x42       e156                 1
    x42       e157                 1
    x42       e158                 1
    x42       e159                 1
    x42       e160                 1
    x42       e161                 1
    x42       e162                 1
    x42       e163                 1
    x42       e164                 1
    x42       e182               1.3
    x42       e183             11.05
    x42       e184             11.05
    x42       e185               1.3
    x42       e186               1.3
    x42       e187               1.3
    x42       e188              11.7
    x42       e527            -4.425
    x43       e1                -3.5
    x43       e136                -1
    x43       e142          -0.00644
    x44       e1               -3.43
    x44       e136                -1
    x44       e142         -0.005796
    x45       e1               -3.29
    x45       e136                -1
    x45       e142         -0.005152
    x46       e1              -2.625
    x46       e136                -1
    x46       e142          -0.00322
    x47       e2                  -2
    x47       e137                -1
    x47       e142          -0.01583
    x48       e2                -1.9
    x48       e137                -1
    x48       e142        -0.0131389
    x49       e2               -1.58
    x49       e137                -1
    x49       e142        -0.0087065
    x50       e2               -1.06
    x50       e137                -1
    x50       e142        -0.0044324
    x51       e3                 -60
    x51       e138                -1
    x51       e142          -0.01111
    x52       e3               -58.8
    x52       e138                -1
    x52       e142         -0.009999
    x53       e3               -56.4
    x53       e138                -1
    x53       e142         -0.008888
    x54       e3                 -45
    x54       e138                -1
    x54       e142         -0.005555
    x55       e4                 -12
    x55       e139                -1
    x55       e142          -0.02255
    x56       e4               -11.4
    x56       e139                -1
    x56       e142        -0.0187165
    x57       e4               -9.48
    x57       e139                -1
    x57       e142        -0.0124025
    x58       e4               -6.36
    x58       e139                -1
    x58       e142         -0.006314
    x59       e5                 -35
    x59       e140                -1
    x59       e142          -0.01683
    x60       e5               -32.9
    x60       e140                -1
    x60       e142        -0.0102663
    x61       e5              -31.15
    x61       e140                -1
    x61       e142        -0.0063954
    x62       e5              -29.05
    x62       e140                -1
    x62       e142        -0.0038709
    x63       e6                -3.5
    x63       e141                -1
    x63       e142          -0.01437
    x64       e6               -3.36
    x64       e141                -1
    x64       e142        -0.0083346
    x65       e6              -2.485
    x65       e141                -1
    x65       e142        -0.0035925
    x66       e142                 1
    x66       e528            -0.267
    x67       e7                  -1
    x67       e173                 2
    x67       e220                 2
    x67       e454                 2
    x67       e529            -0.516
    x68       e7                  -1
    x68       e173                 3
    x68       e233                 3
    x68       e454                 3
    x68       e529            -0.693
    x69       e7                  -1
    x69       e173                 3
    x69       e233                 3
    x69       e478                 3
    x69       e529            -0.372
    x70       e7                  -1
    x70       e173                45
    x70       e246                45
    x70       e502                45
    x70       e529             -0.81
    x71       e8                  -1
    x71       e174                 2
    x71       e221                 2
    x71       e455                 2
    x71       e529            -0.516
    x72       e8                  -1
    x72       e174                 3
    x72       e234                 3
    x72       e455                 3
    x72       e529            -0.693
    x73       e8                  -1
    x73       e174                 3
    x73       e234                 3
    x73       e479                 3
    x73       e529            -0.372
    x74       e8                  -1
    x74       e174                45
    x74       e247                45
    x74       e503                45
    x74       e529             -0.81
    x75       e9                  -1
    x75       e175                 2
    x75       e222                 2
    x75       e456                 2
    x75       e529            -0.516
    x76       e9                  -1
    x76       e175                 3
    x76       e235                 3
    x76       e456                 3
    x76       e529            -0.693
    x77       e9                  -1
    x77       e175                 3
    x77       e235                 3
    x77       e480                 3
    x77       e529            -0.372
    x78       e9                  -1
    x78       e175                45
    x78       e248                45
    x78       e504                45
    x78       e529             -0.81
    x79       e10                 -1
    x79       e176                 2
    x79       e223                 2
    x79       e457                 2
    x79       e529            -0.516
    x80       e10                 -1
    x80       e176                 3
    x80       e236                 3
    x80       e457                 3
    x80       e529            -0.693
    x81       e10                 -1
    x81       e176                 3
    x81       e236                 3
    x81       e481                 3
    x81       e529            -0.372
    x82       e10                 -1
    x82       e176                45
    x82       e249                45
    x82       e505                45
    x82       e529             -0.81
    x83       e11                 -1
    x83       e177                 2
    x83       e224                 2
    x83       e458                 2
    x83       e529            -0.516
    x84       e11                 -1
    x84       e177                 3
    x84       e237                 3
    x84       e458                 3
    x84       e529            -0.693
    x85       e11                 -1
    x85       e177                 3
    x85       e237                 3
    x85       e482                 3
    x85       e529            -0.372
    x86       e11                 -1
    x86       e177                45
    x86       e250                45
    x86       e506                45
    x86       e529             -0.81
    x87       e12                 -1
    x87       e178                 2
    x87       e225                 2
    x87       e459                 2
    x87       e529            -0.516
    x88       e12                 -1
    x88       e178                 3
    x88       e238                 3
    x88       e459                 3
    x88       e529            -0.693
    x89       e12                 -1
    x89       e178                 3
    x89       e238                 3
    x89       e483                 3
    x89       e529            -0.372
    x90       e12                 -1
    x90       e178                45
    x90       e251                45
    x90       e507                45
    x90       e529             -0.81
    x91       e13                 -1
    x91       e185                 2
    x91       e226                 2
    x91       e466                 2
    x91       e529            -0.516
    x92       e13                 -1
    x92       e185                 3
    x92       e239                 3
    x92       e466                 3
    x92       e529            -0.693
    x93       e13                 -1
    x93       e185                 3
    x93       e239                 3
    x93       e490                 3
    x93       e529            -0.372
    x94       e13                 -1
    x94       e185                45
    x94       e252                45
    x94       e514                45
    x94       e529             -0.81
    x95       e14                 -1
    x95       e186                 2
    x95       e227                 2
    x95       e467                 2
    x95       e529            -0.516
    x96       e14                 -1
    x96       e186                 3
    x96       e240                 3
    x96       e467                 3
    x96       e529            -0.693
    x97       e14                 -1
    x97       e186                 3
    x97       e240                 3
    x97       e491                 3
    x97       e529            -0.372
    x98       e14                 -1
    x98       e186                45
    x98       e253                45
    x98       e515                45
    x98       e529             -0.81
    x99       e15                 -1
    x99       e187                 2
    x99       e228                 2
    x99       e468                 2
    x99       e529            -0.516
    x100      e15                 -1
    x100      e187                 3
    x100      e241                 3
    x100      e468                 3
    x100      e529            -0.693
    x101      e15                 -1
    x101      e187                 3
    x101      e241                 3
    x101      e492                 3
    x101      e529            -0.372
    x102      e15                 -1
    x102      e187                45
    x102      e254                45
    x102      e516                45
    x102      e529             -0.81
    x103      e16                 -1
    x103      e188                 2
    x103      e229                 2
    x103      e469                 2
    x103      e529            -0.516
    x104      e16                 -1
    x104      e188                 3
    x104      e242                 3
    x104      e469                 3
    x104      e529            -0.693
    x105      e16                 -1
    x105      e188                 3
    x105      e242                 3
    x105      e493                 3
    x105      e529            -0.372
    x106      e16                 -1
    x106      e188                45
    x106      e255                45
    x106      e517                45
    x106      e529             -0.81
    x107      e17                 -1
    x107      e189                 2
    x107      e230                 2
    x107      e470                 2
    x107      e529            -0.516
    x108      e17                 -1
    x108      e189                 3
    x108      e243                 3
    x108      e470                 3
    x108      e529            -0.693
    x109      e17                 -1
    x109      e189                 3
    x109      e243                 3
    x109      e494                 3
    x109      e529            -0.372
    x110      e17                 -1
    x110      e189                45
    x110      e256                45
    x110      e518                45
    x110      e529             -0.81
    x111      e18                 -1
    x111      e195                 2
    x111      e231                 2
    x111      e476                 2
    x111      e529            -0.516
    x112      e18                 -1
    x112      e195                 3
    x112      e244                 3
    x112      e476                 3
    x112      e529            -0.693
    x113      e18                 -1
    x113      e195                 3
    x113      e244                 3
    x113      e500                 3
    x113      e529            -0.372
    x114      e18                 -1
    x114      e195                45
    x114      e257                45
    x114      e524                45
    x114      e529             -0.81
    x115      e19                 -1
    x115      e196                 2
    x115      e232                 2
    x115      e477                 2
    x115      e529            -0.516
    x116      e19                 -1
    x116      e196                 3
    x116      e245                 3
    x116      e477                 3
    x116      e529            -0.693
    x117      e19                 -1
    x117      e196                 3
    x117      e245                 3
    x117      e501                 3
    x117      e529            -0.372
    x118      e19                 -1
    x118      e196                45
    x118      e258                45
    x118      e525                45
    x118      e529             -0.81
    x119      e20                 -1
    x119      e173               0.6
    x119      e259               0.6
    x119      e454               0.6
    x119      e529           -0.1644
    x120      e20                 -1
    x120      e173                 1
    x120      e273                 1
    x120      e478                 1
    x120      e529            -0.118
    x121      e20                 -1
    x121      e173                 6
    x121      e289                 6
    x121      e502                 6
    x121      e529            -0.108
    x122      e21                 -1
    x122      e174               0.6
    x122      e260               0.6
    x122      e455               0.6
    x122      e529           -0.1644
    x123      e21                 -1
    x123      e174                 1
    x123      e274                 1
    x123      e479                 1
    x123      e529            -0.118
    x124      e21                 -1
    x124      e174                 6
    x124      e290                 6
    x124      e503                 6
    x124      e529            -0.108
    x125      e22                 -1
    x125      e175               0.6
    x125      e261               0.6
    x125      e456               0.6
    x125      e529           -0.1644
    x126      e22                 -1
    x126      e175                 1
    x126      e275                 1
    x126      e480                 1
    x126      e529            -0.118
    x127      e22                 -1
    x127      e175                 6
    x127      e291                 6
    x127      e504                 6
    x127      e529            -0.108
    x128      e23                 -1
    x128      e176               0.6
    x128      e262               0.6
    x128      e457               0.6
    x128      e529           -0.1644
    x129      e23                 -1
    x129      e176                 1
    x129      e276                 1
    x129      e481                 1
    x129      e529            -0.118
    x130      e23                 -1
    x130      e176                 6
    x130      e292                 6
    x130      e505                 6
    x130      e529            -0.108
    x131      e24                 -1
    x131      e177               0.6
    x131      e263               0.6
    x131      e458               0.6
    x131      e529           -0.1644
    x132      e24                 -1
    x132      e177                 1
    x132      e277                 1
    x132      e482                 1
    x132      e529            -0.118
    x133      e24                 -1
    x133      e177                 6
    x133      e293                 6
    x133      e506                 6
    x133      e529            -0.108
    x134      e25                 -1
    x134      e178               0.6
    x134      e264               0.6
    x134      e459               0.6
    x134      e529           -0.1644
    x135      e25                 -1
    x135      e178                 1
    x135      e278                 1
    x135      e483                 1
    x135      e529            -0.118
    x136      e25                 -1
    x136      e178                 6
    x136      e294                 6
    x136      e507                 6
    x136      e529            -0.108
    x137      e26                 -1
    x137      e179               0.6
    x137      e265               0.6
    x137      e460               0.6
    x137      e529           -0.1644
    x138      e26                 -1
    x138      e179                 1
    x138      e279                 1
    x138      e484                 1
    x138      e529            -0.118
    x139      e26                 -1
    x139      e179                 6
    x139      e295                 6
    x139      e508                 6
    x139      e529            -0.108
    x140      e27                 -1
    x140      e184               0.6
    x140      e266               0.6
    x140      e465               0.6
    x140      e529           -0.1644
    x141      e27                 -1
    x141      e184                 1
    x141      e280                 1
    x141      e489                 1
    x141      e529            -0.118
    x142      e27                 -1
    x142      e184                 6
    x142      e296                 6
    x142      e513                 6
    x142      e529            -0.108
    x143      e28                 -1
    x143      e185               0.6
    x143      e267               0.6
    x143      e466               0.6
    x143      e529           -0.1644
    x144      e28                 -1
    x144      e185                 1
    x144      e281                 1
    x144      e490                 1
    x144      e529            -0.118
    x145      e28                 -1
    x145      e185                 6
    x145      e297                 6
    x145      e514                 6
    x145      e529            -0.108
    x146      e29                 -1
    x146      e186               0.6
    x146      e268               0.6
    x146      e467               0.6
    x146      e529           -0.1644
    x147      e29                 -1
    x147      e186                 1
    x147      e282                 1
    x147      e491                 1
    x147      e529            -0.118
    x148      e29                 -1
    x148      e186                 6
    x148      e298                 6
    x148      e515                 6
    x148      e529            -0.108
    x149      e30                 -1
    x149      e187               0.6
    x149      e269               0.6
    x149      e468               0.6
    x149      e529           -0.1644
    x150      e30                 -1
    x150      e187                 1
    x150      e283                 1
    x150      e492                 1
    x150      e529            -0.118
    x151      e30                 -1
    x151      e187                 6
    x151      e299                 6
    x151      e516                 6
    x151      e529            -0.108
    x152      e31                 -1
    x152      e188               0.6
    x152      e270               0.6
    x152      e469               0.6
    x152      e529           -0.1644
    x153      e31                 -1
    x153      e188                 1
    x153      e284                 1
    x153      e493                 1
    x153      e529            -0.118
    x154      e31                 -1
    x154      e188                 6
    x154      e300                 6
    x154      e517                 6
    x154      e529            -0.108
    x155      e32                 -1
    x155      e189               0.6
    x155      e271               0.6
    x155      e470               0.6
    x155      e529           -0.1644
    x156      e32                 -1
    x156      e189                 1
    x156      e285                 1
    x156      e494                 1
    x156      e529            -0.118
    x157      e32                 -1
    x157      e189                 6
    x157      e301                 6
    x157      e518                 6
    x157      e529            -0.108
    x158      e33                 -1
    x158      e190               0.6
    x158      e272               0.6
    x158      e471               0.6
    x158      e529           -0.1644
    x159      e33                 -1
    x159      e190                 1
    x159      e286                 1
    x159      e495                 1
    x159      e529            -0.118
    x160      e33                 -1
    x160      e190                 6
    x160      e302                 6
    x160      e519                 6
    x160      e529            -0.108
    x161      e34                 -1
    x161      e175                50
    x162      e34                 -1
    x162      e175               1.6
    x162      e275               1.6
    x162      e480               1.6
    x162      e529           -0.1888
    x163      e34                 -1
    x163      e175                 2
    x163      e291                 2
    x163      e504                 2
    x163      e529            -0.036
    x164      e35                 -1
    x164      e176                50
    x165      e35                 -1
    x165      e176               1.6
    x165      e276               1.6
    x165      e481               1.6
    x165      e529           -0.1888
    x166      e35                 -1
    x166      e176                 2
    x166      e292                 2
    x166      e505                 2
    x166      e529            -0.036
    x167      e36                 -1
    x167      e177                50
    x168      e36                 -1
    x168      e177               1.6
    x168      e277               1.6
    x168      e482               1.6
    x168      e529           -0.1888
    x169      e36                 -1
    x169      e177                 2
    x169      e293                 2
    x169      e506                 2
    x169      e529            -0.036
    x170      e37                 -1
    x170      e178                50
    x171      e37                 -1
    x171      e178               1.6
    x171      e278               1.6
    x171      e483               1.6
    x171      e529           -0.1888
    x172      e37                 -1
    x172      e178                 2
    x172      e294                 2
    x172      e507                 2
    x172      e529            -0.036
    x173      e38                 -1
    x173      e179                50
    x174      e38                 -1
    x174      e179               1.6
    x174      e279               1.6
    x174      e484               1.6
    x174      e529           -0.1888
    x175      e38                 -1
    x175      e179                 2
    x175      e295                 2
    x175      e508                 2
    x175      e529            -0.036
    x176      e39                 -1
    x176      e190                50
    x177      e39                 -1
    x177      e190               1.6
    x177      e286               1.6
    x177      e495               1.6
    x177      e529           -0.1888
    x178      e39                 -1
    x178      e190                 2
    x178      e302                 2
    x178      e519                 2
    x178      e529            -0.036
    x179      e40                 -1
    x179      e191                50
    x180      e40                 -1
    x180      e191               1.6
    x180      e287               1.6
    x180      e496               1.6
    x180      e529           -0.1888
    x181      e40                 -1
    x181      e191                 2
    x181      e303                 2
    x181      e520                 2
    x181      e529            -0.036
    x182      e41                 -1
    x182      e192                50
    x183      e41                 -1
    x183      e192               1.6
    x183      e288               1.6
    x183      e497               1.6
    x183      e529           -0.1888
    x184      e41                 -1
    x184      e192                 2
    x184      e304                 2
    x184      e521                 2
    x184      e529            -0.036
    x185      e42                 -1
    x185      e175               0.5
    x185      e305               0.5
    x185      e456               0.5
    x185      e529           -0.1155
    x186      e42                 -1
    x186      e175               0.5
    x186      e305               0.5
    x186      e480               0.5
    x186      e529            -0.062
    x187      e42                 -1
    x187      e175                 8
    x187      e322                 8
    x187      e504                 8
    x187      e529            -0.144
    x188      e42                 -1
    x188      e175                 8
    x188      e437                 8
    x189      e43                 -1
    x189      e176               0.5
    x189      e306               0.5
    x189      e457               0.5
    x189      e529           -0.1155
    x190      e43                 -1
    x190      e176               0.5
    x190      e306               0.5
    x190      e481               0.5
    x190      e529            -0.062
    x191      e43                 -1
    x191      e176                 8
    x191      e323                 8
    x191      e505                 8
    x191      e529            -0.144
    x192      e43                 -1
    x192      e176                 8
    x192      e438                 8
    x193      e44                 -1
    x193      e177               0.5
    x193      e307               0.5
    x193      e458               0.5
    x193      e529           -0.1155
    x194      e44                 -1
    x194      e177               0.5
    x194      e307               0.5
    x194      e482               0.5
    x194      e529            -0.062
    x195      e44                 -1
    x195      e177                 8
    x195      e324                 8
    x195      e506                 8
    x195      e529            -0.144
    x196      e44                 -1
    x196      e177                 8
    x196      e439                 8
    x197      e45                 -1
    x197      e178               0.5
    x197      e308               0.5
    x197      e459               0.5
    x197      e529           -0.1155
    x198      e45                 -1
    x198      e178               0.5
    x198      e308               0.5
    x198      e483               0.5
    x198      e529            -0.062
    x199      e45                 -1
    x199      e178                 8
    x199      e325                 8
    x199      e507                 8
    x199      e529            -0.144
    x200      e45                 -1
    x200      e178                 8
    x200      e440                 8
    x201      e46                 -1
    x201      e179               0.5
    x201      e309               0.5
    x201      e460               0.5
    x201      e529           -0.1155
    x202      e46                 -1
    x202      e179               0.5
    x202      e309               0.5
    x202      e484               0.5
    x202      e529            -0.062
    x203      e46                 -1
    x203      e179                 8
    x203      e326                 8
    x203      e508                 8
    x203      e529            -0.144
    x204      e46                 -1
    x204      e179                 8
    x204      e441                 8
    x205      e47                 -1
    x205      e181               0.5
    x205      e310               0.5
    x205      e462               0.5
    x205      e529           -0.1155
    x206      e47                 -1
    x206      e181               0.5
    x206      e310               0.5
    x206      e486               0.5
    x206      e529            -0.062
    x207      e47                 -1
    x207      e181                 8
    x207      e327                 8
    x207      e510                 8
    x207      e529            -0.144
    x208      e47                 -1
    x208      e181                 8
    x208      e442                 8
    x209      e48                 -1
    x209      e182               0.5
    x209      e311               0.5
    x209      e463               0.5
    x209      e529           -0.1155
    x210      e48                 -1
    x210      e182               0.5
    x210      e311               0.5
    x210      e487               0.5
    x210      e529            -0.062
    x211      e48                 -1
    x211      e182                 8
    x211      e328                 8
    x211      e511                 8
    x211      e529            -0.144
    x212      e48                 -1
    x212      e182                 8
    x212      e443                 8
    x213      e49                 -1
    x213      e183               0.5
    x213      e312               0.5
    x213      e464               0.5
    x213      e529           -0.1155
    x214      e49                 -1
    x214      e183               0.5
    x214      e312               0.5
    x214      e488               0.5
    x214      e529            -0.062
    x215      e49                 -1
    x215      e183                 8
    x215      e329                 8
    x215      e512                 8
    x215      e529            -0.144
    x216      e49                 -1
    x216      e183                 8
    x216      e444                 8
    x217      e50                 -1
    x217      e184               0.5
    x217      e313               0.5
    x217      e465               0.5
    x217      e529           -0.1155
    x218      e50                 -1
    x218      e184               0.5
    x218      e313               0.5
    x218      e489               0.5
    x218      e529            -0.062
    x219      e50                 -1
    x219      e184                 8
    x219      e330                 8
    x219      e513                 8
    x219      e529            -0.144
    x220      e50                 -1
    x220      e184                 8
    x220      e445                 8
    x221      e51                 -1
    x221      e185               0.5
    x221      e314               0.5
    x221      e466               0.5
    x221      e529           -0.1155
    x222      e51                 -1
    x222      e185               0.5
    x222      e314               0.5
    x222      e490               0.5
    x222      e529            -0.062
    x223      e51                 -1
    x223      e185                 8
    x223      e331                 8
    x223      e514                 8
    x223      e529            -0.144
    x224      e51                 -1
    x224      e185                 8
    x224      e446                 8
    x225      e52                 -1
    x225      e186               0.5
    x225      e315               0.5
    x225      e467               0.5
    x225      e529           -0.1155
    x226      e52                 -1
    x226      e186               0.5
    x226      e315               0.5
    x226      e491               0.5
    x226      e529            -0.062
    x227      e52                 -1
    x227      e186                 8
    x227      e332                 8
    x227      e515                 8
    x227      e529            -0.144
    x228      e52                 -1
    x228      e186                 8
    x228      e447                 8
    x229      e53                 -1
    x229      e187               0.5
    x229      e316               0.5
    x229      e468               0.5
    x229      e529           -0.1155
    x230      e53                 -1
    x230      e187               0.5
    x230      e316               0.5
    x230      e492               0.5
    x230      e529            -0.062
    x231      e53                 -1
    x231      e187                 8
    x231      e333                 8
    x231      e516                 8
    x231      e529            -0.144
    x232      e53                 -1
    x232      e187                 8
    x232      e448                 8
    x233      e54                 -1
    x233      e188               0.5
    x233      e317               0.5
    x233      e469               0.5
    x233      e529           -0.1155
    x234      e54                 -1
    x234      e188               0.5
    x234      e317               0.5
    x234      e493               0.5
    x234      e529            -0.062
    x235      e54                 -1
    x235      e188                 8
    x235      e334                 8
    x235      e517                 8
    x235      e529            -0.144
    x236      e54                 -1
    x236      e188                 8
    x236      e449                 8
    x237      e55                 -1
    x237      e189               0.5
    x237      e318               0.5
    x237      e470               0.5
    x237      e529           -0.1155
    x238      e55                 -1
    x238      e189               0.5
    x238      e318               0.5
    x238      e494               0.5
    x238      e529            -0.062
    x239      e55                 -1
    x239      e189                 8
    x239      e335                 8
    x239      e518                 8
    x239      e529            -0.144
    x240      e55                 -1
    x240      e189                 8
    x240      e450                 8
    x241      e56                 -1
    x241      e190               0.5
    x241      e319               0.5
    x241      e471               0.5
    x241      e529           -0.1155
    x242      e56                 -1
    x242      e190               0.5
    x242      e319               0.5
    x242      e495               0.5
    x242      e529            -0.062
    x243      e56                 -1
    x243      e190                 8
    x243      e336                 8
    x243      e519                 8
    x243      e529            -0.144
    x244      e56                 -1
    x244      e190                 8
    x244      e451                 8
    x245      e57                 -1
    x245      e191               0.5
    x245      e320               0.5
    x245      e472               0.5
    x245      e529           -0.1155
    x246      e57                 -1
    x246      e191               0.5
    x246      e320               0.5
    x246      e496               0.5
    x246      e529            -0.062
    x247      e57                 -1
    x247      e191                 8
    x247      e337                 8
    x247      e520                 8
    x247      e529            -0.144
    x248      e57                 -1
    x248      e191                 8
    x248      e452                 8
    x249      e58                 -1
    x249      e192               0.5
    x249      e321               0.5
    x249      e473               0.5
    x249      e529           -0.1155
    x250      e58                 -1
    x250      e192               0.5
    x250      e321               0.5
    x250      e497               0.5
    x250      e529            -0.062
    x251      e58                 -1
    x251      e192                 8
    x251      e338                 8
    x251      e521                 8
    x251      e529            -0.144
    x252      e58                 -1
    x252      e192                 8
    x252      e453                 8
    x253      e59                 -1
    x253      e179               0.5
    x253      e197               0.5
    x253      e460               0.5
    x253      e529            -0.146
    x254      e59                 -1
    x254      e179               0.5
    x254      e197               0.5
    x254      e484               0.5
    x254      e529           -0.0925
    x255      e59                 -1
    x255      e179                 8
    x255      e339                 8
    x255      e508                 8
    x255      e529            -0.144
    x256      e60                 -1
    x256      e180               0.5
    x256      e198               0.5
    x256      e461               0.5
    x256      e529            -0.146
    x257      e60                 -1
    x257      e180               0.5
    x257      e198               0.5
    x257      e485               0.5
    x257      e529           -0.0925
    x258      e60                 -1
    x258      e180                 8
    x258      e340                 8
    x258      e509                 8
    x258      e529            -0.144
    x259      e61                 -1
    x259      e181               0.5
    x259      e199               0.5
    x259      e462               0.5
    x259      e529            -0.146
    x260      e61                 -1
    x260      e181               0.5
    x260      e199               0.5
    x260      e486               0.5
    x260      e529           -0.0925
    x261      e61                 -1
    x261      e181                 8
    x261      e341                 8
    x261      e510                 8
    x261      e529            -0.144
    x262      e62                 -1
    x262      e182               0.5
    x262      e200               0.5
    x262      e463               0.5
    x262      e529            -0.146
    x263      e62                 -1
    x263      e182               0.5
    x263      e200               0.5
    x263      e487               0.5
    x263      e529           -0.0925
    x264      e62                 -1
    x264      e182                 8
    x264      e342                 8
    x264      e511                 8
    x264      e529            -0.144
    x265      e63                 -1
    x265      e191               0.5
    x265      e201               0.5
    x265      e472               0.5
    x265      e529            -0.146
    x266      e63                 -1
    x266      e191               0.5
    x266      e201               0.5
    x266      e496               0.5
    x266      e529           -0.0925
    x267      e63                 -1
    x267      e191                 8
    x267      e343                 8
    x267      e520                 8
    x267      e529            -0.144
    x268      e64                 -1
    x268      e192               0.5
    x268      e202               0.5
    x268      e473               0.5
    x268      e529            -0.146
    x269      e64                 -1
    x269      e192               0.5
    x269      e202               0.5
    x269      e497               0.5
    x269      e529           -0.0925
    x270      e64                 -1
    x270      e192                 8
    x270      e344                 8
    x270      e521                 8
    x270      e529            -0.144
    x271      e65                 -1
    x271      e193               0.5
    x271      e203               0.5
    x271      e474               0.5
    x271      e529            -0.146
    x272      e65                 -1
    x272      e193               0.5
    x272      e203               0.5
    x272      e498               0.5
    x272      e529           -0.0925
    x273      e65                 -1
    x273      e193                 8
    x273      e345                 8
    x273      e522                 8
    x273      e529            -0.144
    x274      e66                 -1
    x274      e176               1.5
    x274      e346               1.5
    x274      e457               1.5
    x274      e529            -0.399
    x275      e66                 -1
    x275      e176               1.5
    x275      e346               1.5
    x275      e481               1.5
    x275      e529           -0.2385
    x276      e66                 -1
    x276      e176                20
    x276      e353                20
    x276      e505                20
    x276      e529             -0.36
    x277      e67                 -1
    x277      e177               1.5
    x277      e347               1.5
    x277      e458               1.5
    x277      e529            -0.399
    x278      e67                 -1
    x278      e177               1.5
    x278      e347               1.5
    x278      e482               1.5
    x278      e529           -0.2385
    x279      e67                 -1
    x279      e177                20
    x279      e354                20
    x279      e506                20
    x279      e529             -0.36
    x280      e68                 -1
    x280      e178               1.5
    x280      e348               1.5
    x280      e459               1.5
    x280      e529            -0.399
    x281      e68                 -1
    x281      e178               1.5
    x281      e348               1.5
    x281      e483               1.5
    x281      e529           -0.2385
    x282      e68                 -1
    x282      e178                20
    x282      e355                20
    x282      e507                20
    x282      e529             -0.36
    x283      e69                 -1
    x283      e180               1.5
    x283      e349               1.5
    x283      e461               1.5
    x283      e529            -0.399
    x284      e69                 -1
    x284      e180               1.5
    x284      e349               1.5
    x284      e485               1.5
    x284      e529           -0.2385
    x285      e69                 -1
    x285      e180                20
    x285      e356                20
    x285      e509                20
    x285      e529             -0.36
    x286      e70                 -1
    x286      e181               1.5
    x286      e350               1.5
    x286      e462               1.5
    x286      e529            -0.399
    x287      e70                 -1
    x287      e181               1.5
    x287      e350               1.5
    x287      e486               1.5
    x287      e529           -0.2385
    x288      e70                 -1
    x288      e181                20
    x288      e357                20
    x288      e510                20
    x288      e529             -0.36
    x289      e71                 -1
    x289      e185               1.5
    x289      e351               1.5
    x289      e466               1.5
    x289      e529            -0.399
    x290      e71                 -1
    x290      e185               1.5
    x290      e351               1.5
    x290      e490               1.5
    x290      e529           -0.2385
    x291      e71                 -1
    x291      e185                20
    x291      e358                20
    x291      e514                20
    x291      e529             -0.36
    x292      e72                 -1
    x292      e186               1.5
    x292      e352               1.5
    x292      e467               1.5
    x292      e529            -0.399
    x293      e72                 -1
    x293      e186               1.5
    x293      e352               1.5
    x293      e491               1.5
    x293      e529           -0.2385
    x294      e72                 -1
    x294      e186                20
    x294      e359                20
    x294      e515                20
    x294      e529             -0.36
    x295      e73                 -1
    x295      e177                 8
    x296      e73                 -1
    x296      e177                 4
    x296      e360                 4
    x296      e458                 4
    x296      e529              -0.9
    x297      e73                 -1
    x297      e177                 4
    x297      e360                 4
    x297      e482                 4
    x297      e529            -0.472
    x298      e73                 -1
    x298      e177                 8
    x298      e368                 8
    x298      e506                 8
    x298      e529            -0.144
    x299      e74                 -1
    x299      e178                 8
    x300      e74                 -1
    x300      e178                 4
    x300      e361                 4
    x300      e459                 4
    x300      e529              -0.9
    x301      e74                 -1
    x301      e178                 4
    x301      e361                 4
    x301      e483                 4
    x301      e529            -0.472
    x302      e74                 -1
    x302      e178                 8
    x302      e369                 8
    x302      e507                 8
    x302      e529            -0.144
    x303      e75                 -1
    x303      e179                 8
    x304      e75                 -1
    x304      e179                 4
    x304      e362                 4
    x304      e460                 4
    x304      e529              -0.9
    x305      e75                 -1
    x305      e179                 4
    x305      e362                 4
    x305      e484                 4
    x305      e529            -0.472
    x306      e75                 -1
    x306      e179                 8
    x306      e370                 8
    x306      e508                 8
    x306      e529            -0.144
    x307      e76                 -1
    x307      e185                 8
    x308      e76                 -1
    x308      e185                 4
    x308      e363                 4
    x308      e466                 4
    x308      e529              -0.9
    x309      e76                 -1
    x309      e185                 4
    x309      e363                 4
    x309      e490                 4
    x309      e529            -0.472
    x310      e76                 -1
    x310      e185                 8
    x310      e371                 8
    x310      e514                 8
    x310      e529            -0.144
    x311      e77                 -1
    x311      e186                 8
    x312      e77                 -1
    x312      e186                 4
    x312      e364                 4
    x312      e467                 4
    x312      e529              -0.9
    x313      e77                 -1
    x313      e186                 4
    x313      e364                 4
    x313      e491                 4
    x313      e529            -0.472
    x314      e77                 -1
    x314      e186                 8
    x314      e372                 8
    x314      e515                 8
    x314      e529            -0.144
    x315      e78                 -1
    x315      e188                 8
    x316      e78                 -1
    x316      e188                 4
    x316      e365                 4
    x316      e469                 4
    x316      e529              -0.9
    x317      e78                 -1
    x317      e188                 4
    x317      e365                 4
    x317      e493                 4
    x317      e529            -0.472
    x318      e78                 -1
    x318      e188                 8
    x318      e373                 8
    x318      e517                 8
    x318      e529            -0.144
    x319      e79                 -1
    x319      e189                 8
    x320      e79                 -1
    x320      e189                 4
    x320      e366                 4
    x320      e470                 4
    x320      e529              -0.9
    x321      e79                 -1
    x321      e189                 4
    x321      e366                 4
    x321      e494                 4
    x321      e529            -0.472
    x322      e79                 -1
    x322      e189                 8
    x322      e374                 8
    x322      e518                 8
    x322      e529            -0.144
    x323      e80                 -1
    x323      e190                 8
    x324      e80                 -1
    x324      e190                 4
    x324      e367                 4
    x324      e471                 4
    x324      e529              -0.9
    x325      e80                 -1
    x325      e190                 4
    x325      e367                 4
    x325      e495                 4
    x325      e529            -0.472
    x326      e80                 -1
    x326      e190                 8
    x326      e375                 8
    x326      e519                 8
    x326      e529            -0.144
    x327      e81                 -1
    x327      e178                50
    x328      e81                 -1
    x328      e178               0.5
    x328      e376               0.5
    x328      e459               0.5
    x328      e529            -0.125
    x329      e81                 -1
    x329      e178                 1
    x329      e376                 1
    x329      e483                 1
    x329      e529            -0.143
    x330      e81                 -1
    x330      e178                20
    x330      e385                20
    x330      e507                20
    x330      e529             -0.36
    x331      e82                 -1
    x331      e179                50
    x332      e82                 -1
    x332      e179               0.5
    x332      e377               0.5
    x332      e460               0.5
    x332      e529            -0.125
    x333      e82                 -1
    x333      e179                 1
    x333      e377                 1
    x333      e484                 1
    x333      e529            -0.143
    x334      e82                 -1
    x334      e179                20
    x334      e386                20
    x334      e508                20
    x334      e529             -0.36
    x335      e83                 -1
    x335      e180                50
    x336      e83                 -1
    x336      e180               0.5
    x336      e378               0.5
    x336      e461               0.5
    x336      e529            -0.125
    x337      e83                 -1
    x337      e180                 1
    x337      e378                 1
    x337      e485                 1
    x337      e529            -0.143
    x338      e83                 -1
    x338      e180                20
    x338      e387                20
    x338      e509                20
    x338      e529             -0.36
    x339      e84                 -1
    x339      e181                50
    x340      e84                 -1
    x340      e181               0.5
    x340      e379               0.5
    x340      e462               0.5
    x340      e529            -0.125
    x341      e84                 -1
    x341      e181                 1
    x341      e379                 1
    x341      e486                 1
    x341      e529            -0.143
    x342      e84                 -1
    x342      e181                20
    x342      e388                20
    x342      e510                20
    x342      e529             -0.36
    x343      e85                 -1
    x343      e182                50
    x344      e85                 -1
    x344      e182               0.5
    x344      e380               0.5
    x344      e463               0.5
    x344      e529            -0.125
    x345      e85                 -1
    x345      e182                 1
    x345      e380                 1
    x345      e487                 1
    x345      e529            -0.143
    x346      e85                 -1
    x346      e182                20
    x346      e389                20
    x346      e511                20
    x346      e529             -0.36
    x347      e86                 -1
    x347      e183                50
    x348      e86                 -1
    x348      e183               0.5
    x348      e381               0.5
    x348      e464               0.5
    x348      e529            -0.125
    x349      e86                 -1
    x349      e183                 1
    x349      e381                 1
    x349      e488                 1
    x349      e529            -0.143
    x350      e86                 -1
    x350      e183                20
    x350      e390                20
    x350      e512                20
    x350      e529             -0.36
    x351      e87                 -1
    x351      e184                50
    x352      e87                 -1
    x352      e184               0.5
    x352      e382               0.5
    x352      e465               0.5
    x352      e529            -0.125
    x353      e87                 -1
    x353      e184                 1
    x353      e382                 1
    x353      e489                 1
    x353      e529            -0.143
    x354      e87                 -1
    x354      e184                20
    x354      e391                20
    x354      e513                20
    x354      e529             -0.36
    x355      e88                 -1
    x355      e187                50
    x356      e88                 -1
    x356      e187               0.5
    x356      e383               0.5
    x356      e468               0.5
    x356      e529            -0.125
    x357      e88                 -1
    x357      e187                 1
    x357      e383                 1
    x357      e492                 1
    x357      e529            -0.143
    x358      e88                 -1
    x358      e187                20
    x358      e392                20
    x358      e516                20
    x358      e529             -0.36
    x359      e89                 -1
    x359      e188                50
    x360      e89                 -1
    x360      e188               0.5
    x360      e384               0.5
    x360      e469               0.5
    x360      e529            -0.125
    x361      e89                 -1
    x361      e188                 1
    x361      e384                 1
    x361      e493                 1
    x361      e529            -0.143
    x362      e89                 -1
    x362      e188                20
    x362      e393                20
    x362      e517                20
    x362      e529             -0.36
    x363      e90                 -1
    x363      e192               2.5
    x363      e204               2.5
    x363      e473               2.5
    x363      e529           -0.6125
    x364      e90                 -1
    x364      e192               2.5
    x364      e204               2.5
    x364      e497               2.5
    x364      e529            -0.345
    x365      e90                 -1
    x365      e192               200
    x366      e91                 -1
    x366      e193               2.5
    x366      e205               2.5
    x366      e474               2.5
    x366      e529           -0.6125
    x367      e91                 -1
    x367      e193               2.5
    x367      e205               2.5
    x367      e498               2.5
    x367      e529            -0.345
    x368      e91                 -1
    x368      e193               200
    x369      e92                 -1
    x369      e194               2.5
    x369      e206               2.5
    x369      e475               2.5
    x369      e529           -0.6125
    x370      e92                 -1
    x370      e194               2.5
    x370      e206               2.5
    x370      e499               2.5
    x370      e529            -0.345
    x371      e92                 -1
    x371      e194               200
    x372      e93                 -1
    x372      e195               2.5
    x372      e207               2.5
    x372      e476               2.5
    x372      e529           -0.6125
    x373      e93                 -1
    x373      e195               2.5
    x373      e207               2.5
    x373      e500               2.5
    x373      e529            -0.345
    x374      e93                 -1
    x374      e195               200
    x375      e94                 -1
    x375      e184                 1
    x375      e394                 1
    x375      e465                 1
    x375      e529            -0.235
    x376      e94                 -1
    x376      e184                 1
    x376      e394                 1
    x376      e489                 1
    x376      e529            -0.128
    x377      e95                 -1
    x377      e186                 1
    x377      e395                 1
    x377      e467                 1
    x377      e529            -0.235
    x378      e95                 -1
    x378      e186                 1
    x378      e395                 1
    x378      e491                 1
    x378      e529            -0.128
    x379      e96                 -1
    x379      e187                 1
    x379      e396                 1
    x379      e468                 1
    x379      e529            -0.235
    x380      e96                 -1
    x380      e187                 1
    x380      e396                 1
    x380      e492                 1
    x380      e529            -0.128
    x381      e97                 -1
    x381      e189                 1
    x381      e397                 1
    x381      e470                 1
    x381      e529            -0.235
    x382      e97                 -1
    x382      e189                 1
    x382      e397                 1
    x382      e494                 1
    x382      e529            -0.128
    x383      e98                 -1
    x383      e191                 1
    x383      e398                 1
    x383      e472                 1
    x383      e529            -0.235
    x384      e98                 -1
    x384      e191                 1
    x384      e398                 1
    x384      e496                 1
    x384      e529            -0.128
    x385      e99                 -1
    x385      e184               0.5
    x385      e406               0.5
    x385      e465               0.5
    x385      e529            -0.116
    x386      e99                 -1
    x386      e184               0.5
    x386      e406               0.5
    x386      e489               0.5
    x386      e529           -0.0625
    x387      e100                -1
    x387      e186               0.5
    x387      e407               0.5
    x387      e467               0.5
    x387      e529            -0.116
    x388      e100                -1
    x388      e186               0.5
    x388      e407               0.5
    x388      e491               0.5
    x388      e529           -0.0625
    x389      e101                -1
    x389      e187               0.5
    x389      e408               0.5
    x389      e468               0.5
    x389      e529            -0.116
    x390      e101                -1
    x390      e187               0.5
    x390      e408               0.5
    x390      e492               0.5
    x390      e529           -0.0625
    x391      e102                -1
    x391      e189               0.5
    x391      e409               0.5
    x391      e470               0.5
    x391      e529            -0.116
    x392      e102                -1
    x392      e189               0.5
    x392      e409               0.5
    x392      e494               0.5
    x392      e529           -0.0625
    x393      e103                -1
    x393      e191               0.5
    x393      e410               0.5
    x393      e472               0.5
    x393      e529            -0.116
    x394      e103                -1
    x394      e191               0.5
    x394      e410               0.5
    x394      e496               0.5
    x394      e529           -0.0625
    x395      e104                -1
    x395      e183               0.8
    x395      e399               0.8
    x395      e464               0.8
    x395      e529            -0.232
    x396      e104                -1
    x396      e183                 2
    x396      e399                 2
    x396      e488                 2
    x396      e529            -0.366
    x397      e105                -1
    x397      e184               0.8
    x397      e400               0.8
    x397      e465               0.8
    x397      e529            -0.232
    x398      e105                -1
    x398      e184                 2
    x398      e400                 2
    x398      e489                 2
    x398      e529            -0.366
    x399      e106                -1
    x399      e185               0.8
    x399      e401               0.8
    x399      e466               0.8
    x399      e529            -0.232
    x400      e106                -1
    x400      e185                 2
    x400      e401                 2
    x400      e490                 2
    x400      e529            -0.366
    x401      e107                -1
    x401      e186               0.8
    x401      e402               0.8
    x401      e467               0.8
    x401      e529            -0.232
    x402      e107                -1
    x402      e186                 2
    x402      e402                 2
    x402      e491                 2
    x402      e529            -0.366
    x403      e108                -1
    x403      e187               0.8
    x403      e403               0.8
    x403      e468               0.8
    x403      e529            -0.232
    x404      e108                -1
    x404      e187                 2
    x404      e403                 2
    x404      e492                 2
    x404      e529            -0.366
    x405      e109                -1
    x405      e189               0.8
    x405      e404               0.8
    x405      e470               0.8
    x405      e529            -0.232
    x406      e109                -1
    x406      e189                 2
    x406      e404                 2
    x406      e494                 2
    x406      e529            -0.366
    x407      e110                -1
    x407      e191               0.8
    x407      e405               0.8
    x407      e472               0.8
    x407      e529            -0.232
    x408      e110                -1
    x408      e191                 2
    x408      e405                 2
    x408      e496                 2
    x408      e529            -0.366
    x409      e111                -1
    x409      e183                 1
    x409      e208                 1
    x409      e529             -0.77
    x410      e112                -1
    x410      e184                 1
    x410      e209                 1
    x410      e529             -0.77
    x411      e113                -1
    x411      e185                 1
    x411      e210                 1
    x411      e529             -0.77
    x412      e114                -1
    x412      e188                 1
    x412      e211                 1
    x412      e529             -0.77
    x413      e115                -1
    x413      e189                 1
    x413      e212                 1
    x413      e529             -0.77
    x414      e116                -1
    x414      e191              1.25
    x414      e213              1.25
    x414      e529           -1.6375
    x415      e117                -1
    x415      e192              1.25
    x415      e214              1.25
    x415      e529           -1.6375
    x416      e42                  1
    x417      e43                  1
    x418      e44                  1
    x419      e45                  1
    x420      e46                  1
    x421      e47                  1
    x422      e48                  1
    x423      e49                  1
    x424      e50                  1
    x425      e51                  1
    x426      e52                  1
    x427      e53                  1
    x428      e118                -1
    x428      e188               504
    x429      e6                 0.7
    x429      e54                  1
    x429      e118                -1
    x429      e188                 1
    x429      e215                 1
    x429      e529             -0.77
    x430      e119                -1
    x430      e189               504
    x431      e6                 0.7
    x431      e55                  1
    x431      e119                -1
    x431      e189                 1
    x431      e216                 1
    x431      e529             -0.77
    x432      e120                -1
    x432      e190               504
    x433      e6                 0.7
    x433      e56                  1
    x433      e120                -1
    x433      e190                 1
    x433      e217                 1
    x433      e529             -0.77
    x434      e121                -1
    x434      e191               504
    x435      e6                 0.7
    x435      e57                  1
    x435      e121                -1
    x435      e191                 1
    x435      e218                 1
    x435      e529             -0.77
    x436      e122                -1
    x436      e192               504
    x437      e6                 0.7
    x437      e58                  1
    x437      e122                -1
    x437      e192                 1
    x437      e219                 1
    x437      e529             -0.77
    x438      e123                -1
    x438      e183              0.25
    x438      e411              0.25
    x438      e464              0.25
    x438      e529            -0.062
    x439      e123                -1
    x439      e183              0.25
    x439      e411              0.25
    x439      e488              0.25
    x439      e529          -0.03525
    x440      e123                -1
    x440      e183                 4
    x440      e424                 4
    x440      e512                 4
    x440      e529            -0.072
    x441      e124                -1
    x441      e184              0.25
    x441      e412              0.25
    x441      e465              0.25
    x441      e529            -0.062
    x442      e124                -1
    x442      e184              0.25
    x442      e412              0.25
    x442      e489              0.25
    x442      e529          -0.03525
    x443      e124                -1
    x443      e184                 4
    x443      e425                 4
    x443      e513                 4
    x443      e529            -0.072
    x444      e125                -1
    x444      e185              0.25
    x444      e413              0.25
    x444      e466              0.25
    x444      e529            -0.062
    x445      e125                -1
    x445      e185              0.25
    x445      e413              0.25
    x445      e490              0.25
    x445      e529          -0.03525
    x446      e125                -1
    x446      e185                 4
    x446      e426                 4
    x446      e514                 4
    x446      e529            -0.072
    x447      e126                -1
    x447      e186              0.25
    x447      e414              0.25
    x447      e467              0.25
    x447      e529            -0.062
    x448      e126                -1
    x448      e186              0.25
    x448      e414              0.25
    x448      e491              0.25
    x448      e529          -0.03525
    x449      e126                -1
    x449      e186                 4
    x449      e427                 4
    x449      e515                 4
    x449      e529            -0.072
    x450      e127                -1
    x450      e187              0.25
    x450      e415              0.25
    x450      e468              0.25
    x450      e529            -0.062
    x451      e127                -1
    x451      e187              0.25
    x451      e415              0.25
    x451      e492              0.25
    x451      e529          -0.03525
    x452      e127                -1
    x452      e187                 4
    x452      e428                 4
    x452      e516                 4
    x452      e529            -0.072
    x453      e128                -1
    x453      e188              0.25
    x453      e416              0.25
    x453      e469              0.25
    x453      e529            -0.062
    x454      e128                -1
    x454      e188              0.25
    x454      e416              0.25
    x454      e493              0.25
    x454      e529          -0.03525
    x455      e128                -1
    x455      e188                 4
    x455      e429                 4
    x455      e517                 4
    x455      e529            -0.072
    x456      e129                -1
    x456      e189              0.25
    x456      e417              0.25
    x456      e470              0.25
    x456      e529            -0.062
    x457      e129                -1
    x457      e189              0.25
    x457      e417              0.25
    x457      e494              0.25
    x457      e529          -0.03525
    x458      e129                -1
    x458      e189                 4
    x458      e430                 4
    x458      e518                 4
    x458      e529            -0.072
    x459      e130                -1
    x459      e190              0.25
    x459      e418              0.25
    x459      e471              0.25
    x459      e529            -0.062
    x460      e130                -1
    x460      e190              0.25
    x460      e418              0.25
    x460      e495              0.25
    x460      e529          -0.03525
    x461      e130                -1
    x461      e190                 4
    x461      e431                 4
    x461      e519                 4
    x461      e529            -0.072
    x462      e131                -1
    x462      e191              0.25
    x462      e419              0.25
    x462      e472              0.25
    x462      e529            -0.062
    x463      e131                -1
    x463      e191              0.25
    x463      e419              0.25
    x463      e496              0.25
    x463      e529          -0.03525
    x464      e131                -1
    x464      e191                 4
    x464      e432                 4
    x464      e520                 4
    x464      e529            -0.072
    x465      e132                -1
    x465      e192              0.25
    x465      e420              0.25
    x465      e473              0.25
    x465      e529            -0.062
    x466      e132                -1
    x466      e192              0.25
    x466      e420              0.25
    x466      e497              0.25
    x466      e529          -0.03525
    x467      e132                -1
    x467      e192                 4
    x467      e433                 4
    x467      e521                 4
    x467      e529            -0.072
    x468      e133                -1
    x468      e193              0.25
    x468      e421              0.25
    x468      e474              0.25
    x468      e529            -0.062
    x469      e133                -1
    x469      e193              0.25
    x469      e421              0.25
    x469      e498              0.25
    x469      e529          -0.03525
    x470      e133                -1
    x470      e193                 4
    x470      e434                 4
    x470      e522                 4
    x470      e529            -0.072
    x471      e134                -1
    x471      e194              0.25
    x471      e422              0.25
    x471      e475              0.25
    x471      e529            -0.062
    x472      e134                -1
    x472      e194              0.25
    x472      e422              0.25
    x472      e499              0.25
    x472      e529          -0.03525
    x473      e134                -1
    x473      e194                 4
    x473      e435                 4
    x473      e523                 4
    x473      e529            -0.072
    x474      e135                -1
    x474      e195              0.25
    x474      e423              0.25
    x474      e476              0.25
    x474      e529            -0.062
    x475      e135                -1
    x475      e195              0.25
    x475      e423              0.25
    x475      e500              0.25
    x475      e529          -0.03525
    x476      e135                -1
    x476      e195                 4
    x476      e436                 4
    x476      e524                 4
    x476      e529            -0.072
    x477      e1                   1
    x477      e526               -10
    x478      e2                   1
    x478      e526            -23.96
    x479      e3                   1
    x479      e526             -1.08
    x480      e4                   1
    x480      e526                -5
    x481      e5                   1
    x481      e526              -1.8
    x482      e6                   1
    x482      e526            -16.47
    x483      e197              -140
    x483      e198              -140
    x483      e199              -140
    x483      e200              -140
    x483      e201              -140
    x483      e202              -140
    x483      e203              -140
    x483      e530       -38.1523805
    x484      e204              -140
    x484      e205              -140
    x484      e206              -140
    x484      e207              -140
    x484      e530       -3.69872833
    x485      e454              -140
    x485      e455              -140
    x485      e456              -140
    x485      e457              -140
    x485      e458              -140
    x485      e459              -140
    x485      e460              -140
    x485      e461              -140
    x485      e462              -140
    x485      e463              -140
    x485      e464              -140
    x485      e465              -140
    x485      e466              -140
    x485      e467              -140
    x485      e468              -140
    x485      e469              -140
    x485      e470              -140
    x485      e471              -140
    x485      e472              -140
    x485      e473              -140
    x485      e474              -140
    x485      e475              -140
    x485      e476              -140
    x485      e477              -140
    x485      e530      -224.6271135
    x486      e478              -140
    x486      e479              -140
    x486      e480              -140
    x486      e481              -140
    x486      e482              -140
    x486      e483              -140
    x486      e484              -140
    x486      e485              -140
    x486      e486              -140
    x486      e487              -140
    x486      e488              -140
    x486      e489              -140
    x486      e490              -140
    x486      e491              -140
    x486      e492              -140
    x486      e493              -140
    x486      e494              -140
    x486      e495              -140
    x486      e496              -140
    x486      e497              -140
    x486      e498              -140
    x486      e499              -140
    x486      e500              -140
    x486      e501              -140
    x486      e530      -121.2986413
    x487      e502              -100
    x487      e503              -100
    x487      e504              -100
    x487      e505              -100
    x487      e506              -100
    x487      e507              -100
    x487      e508              -100
    x487      e509              -100
    x487      e510              -100
    x487      e511              -100
    x487      e512              -100
    x487      e513              -100
    x487      e514              -100
    x487      e515              -100
    x487      e516              -100
    x487      e517              -100
    x487      e518              -100
    x487      e519              -100
    x487      e520              -100
    x487      e521              -100
    x487      e522              -100
    x487      e523              -100
    x487      e524              -100
    x487      e525              -100
    x487      e530      -19.07619025
    x488      e208              -140
    x488      e209              -140
    x488      e210              -140
    x488      e211              -140
    x488      e212              -140
    x488      e530      -286.1428538
    x489      e213              -140
    x489      e214              -140
    x489      e530      -184.9364165
    x490      e215               -80
    x490      e216               -80
    x490      e217               -80
    x490      e218               -80
    x490      e219               -80
    x490      e530      -594.1113282
    x491      e220              -140
    x491      e221              -140
    x491      e222              -140
    x491      e223              -140
    x491      e224              -140
    x491      e225              -140
    x491      e226              -140
    x491      e227              -140
    x491      e228              -140
    x491      e229              -140
    x491      e230              -140
    x491      e231              -140
    x491      e232              -140
    x491      e530      -31.44779589
    x492      e233              -140
    x492      e234              -140
    x492      e235              -140
    x492      e236              -140
    x492      e237              -140
    x492      e238              -140
    x492      e239              -140
    x492      e240              -140
    x492      e241              -140
    x492      e242              -140
    x492      e243              -140
    x492      e244              -140
    x492      e245              -140
    x492      e530      -12.32909443
    x493      e246              -100
    x493      e247              -100
    x493      e248              -100
    x493      e249              -100
    x493      e250              -100
    x493      e251              -100
    x493      e252              -100
    x493      e253              -100
    x493      e254              -100
    x493      e255              -100
    x493      e256              -100
    x493      e257              -100
    x493      e258              -100
    x493      e530      -1.081045618
    x494      e259              -140
    x494      e260              -140
    x494      e261              -140
    x494      e262              -140
    x494      e263              -140
    x494      e264              -140
    x494      e265              -140
    x494      e266              -140
    x494      e267              -140
    x494      e268              -140
    x494      e269              -140
    x494      e270              -140
    x494      e271              -140
    x494      e272              -140
    x494      e530       -44.9254227
    x495      e273              -140
    x495      e274              -140
    x495      e275              -140
    x495      e276              -140
    x495      e277              -140
    x495      e278              -140
    x495      e279              -140
    x495      e280              -140
    x495      e281              -140
    x495      e282              -140
    x495      e283              -140
    x495      e284              -140
    x495      e285              -140
    x495      e286              -140
    x495      e287              -140
    x495      e288              -140
    x495      e530      -5.198474122
    x496      e289              -100
    x496      e290              -100
    x496      e291              -100
    x496      e292              -100
    x496      e293              -100
    x496      e294              -100
    x496      e295              -100
    x496      e296              -100
    x496      e297              -100
    x496      e298              -100
    x496      e299              -100
    x496      e300              -100
    x496      e301              -100
    x496      e302              -100
    x496      e303              -100
    x496      e304              -100
    x496      e530      -1.081045618
    x497      e305              -140
    x497      e306              -140
    x497      e307              -140
    x497      e308              -140
    x497      e309              -140
    x497      e310              -140
    x497      e311              -140
    x497      e312              -140
    x497      e313              -140
    x497      e314              -140
    x497      e315              -140
    x497      e316              -140
    x497      e317              -140
    x497      e318              -140
    x497      e319              -140
    x497      e320              -140
    x497      e321              -140
    x497      e530      -30.85488508
    x498      e322              -100
    x498      e323              -100
    x498      e324              -100
    x498      e325              -100
    x498      e326              -100
    x498      e327              -100
    x498      e328              -100
    x498      e329              -100
    x498      e330              -100
    x498      e331              -100
    x498      e332              -100
    x498      e333              -100
    x498      e334              -100
    x498      e335              -100
    x498      e336              -100
    x498      e337              -100
    x498      e338              -100
    x498      e530      -3.963833931
    x499      e339              -100
    x499      e340              -100
    x499      e341              -100
    x499      e342              -100
    x499      e343              -100
    x499      e344              -100
    x499      e345              -100
    x499      e530       -4.68453101
    x500      e346              -140
    x500      e347              -140
    x500      e348              -140
    x500      e349              -140
    x500      e350              -140
    x500      e351              -140
    x500      e352              -140
    x500      e530       -38.1523805
    x501      e353              -100
    x501      e354              -100
    x501      e355              -100
    x501      e356              -100
    x501      e357              -100
    x501      e358              -100
    x501      e359              -100
    x501      e530      -7.206970784
    x502      e360              -140
    x502      e361              -140
    x502      e362              -140
    x502      e363              -140
    x502      e364              -140
    x502      e365              -140
    x502      e366              -140
    x502      e367              -140
    x502      e530      -9.729410559
    x503      e368              -100
    x503      e369              -100
    x503      e370              -100
    x503      e371              -100
    x503      e372              -100
    x503      e373              -100
    x503      e374              -100
    x503      e375              -100
    x503      e530      -1.081045618
    x504      e376              -140
    x504      e377              -140
    x504      e378              -140
    x504      e379              -140
    x504      e380              -140
    x504      e381              -140
    x504      e382              -140
    x504      e383              -140
    x504      e384              -140
    x504      e530      -28.07838919
    x505      e385              -100
    x505      e386              -100
    x505      e387              -100
    x505      e388              -100
    x505      e389              -100
    x505      e390              -100
    x505      e391              -100
    x505      e392              -100
    x505      e393              -100
    x505      e530       -4.68453101
    x506      e394              -140
    x506      e395              -140
    x506      e396              -140
    x506      e397              -140
    x506      e398              -140
    x506      e530      -23.42265505
    x507      e399              -140
    x507      e400              -140
    x507      e401              -140
    x507      e402              -140
    x507      e403              -140
    x507      e404              -140
    x507      e405              -140
    x507      e530       -77.1372127
    x508      e406              -140
    x508      e407              -140
    x508      e408              -140
    x508      e409              -140
    x508      e410              -140
    x508      e530      -13.77450227
    x509      e411              -140
    x509      e412              -140
    x509      e413              -140
    x509      e414              -140
    x509      e415              -140
    x509      e416              -140
    x509      e417              -140
    x509      e418              -140
    x509      e419              -140
    x509      e420              -140
    x509      e421              -140
    x509      e422              -140
    x509      e423              -140
    x509      e530      -40.43288043
    x510      e424              -100
    x510      e425              -100
    x510      e426              -100
    x510      e427              -100
    x510      e428              -100
    x510      e429              -100
    x510      e430              -100
    x510      e431              -100
    x510      e432              -100
    x510      e433              -100
    x510      e434              -100
    x510      e435              -100
    x510      e436              -100
    x510      e530      -5.615677837
    x511      e437               -84
    x511      e438               -84
    x511      e439               -84
    x511      e440               -84
    x511      e441               -84
    x511      e442               -84
    x511      e443               -84
    x511      e444               -84
    x511      e445               -84
    x511      e446               -84
    x511      e447               -84
    x511      e448               -84
    x511      e449               -84
    x511      e450               -84
    x511      e451               -84
    x511      e452               -84
    x511      e453               -84
    x511      e530      -5.832156863
    x512      e173                -6
    x512      e531            -0.135
    x513      e174                -6
    x513      e531            -0.135
    x514      e175                -6
    x514      e531            -0.135
    x515      e176                -6
    x515      e531            -0.135
    x516      e177                -6
    x516      e531            -0.135
    x517      e178                -6
    x517      e531            -0.135
    x518      e179                -6
    x518      e531            -0.135
    x519      e180                -6
    x519      e531            -0.135
    x520      e181                -6
    x520      e531            -0.135
    x521      e182                -6
    x521      e531            -0.135
    x522      e183                -6
    x522      e531            -0.135
    x523      e184                -6
    x523      e531            -0.135
    x524      e185                -6
    x524      e531            -0.135
    x525      e186                -6
    x525      e531            -0.135
    x526      e187                -6
    x526      e531            -0.135
    x527      e188                -6
    x527      e531            -0.135
    x528      e189                -6
    x528      e531            -0.135
    x529      e190                -6
    x529      e531            -0.135
    x530      e191                -6
    x530      e531            -0.135
    x531      e192                -6
    x531      e531            -0.135
    x532      e193                -6
    x532      e531            -0.135
    x533      e194                -6
    x533      e531            -0.135
    x534      e195                -6
    x534      e531            -0.135
    x535      e196                -6
    x535      e531            -0.135
    x536      e526                 1
    x536      e532                -1
    x537      e527                 1
    x537      e532                 1
    x538      e528                 1
    x538      e532                 1
    x539      e529                 1
    x539      e532                 1
    x540      e531                 1
    x540      e532                 1
    x541      e530                 1
    x541      e532                 1
    x542      obj                 -1
    x542      e532                 1
RHS
    rhs       e143              1600
    rhs       e144              1600
    rhs       e145               240
    rhs       e146               400
    rhs       e147              1600
    rhs       e148              1600
    rhs       e149              1600
    rhs       e150              1600
    rhs       e151              1600
    rhs       e152              1600
    rhs       e153              1600
    rhs       e154              1600
    rhs       e155              1600
    rhs       e156              1600
    rhs       e157              1600
    rhs       e158              1600
    rhs       e159              1600
    rhs       e160              1600
    rhs       e161              1600
    rhs       e162              1600
    rhs       e163              1600
    rhs       e164              1600
    rhs       e165              1600
    rhs       e166              1600
    rhs       e167              1600
    rhs       e168              1600
    rhs       e169              1600
    rhs       e170              1600
    rhs       e171              1600
    rhs       e172              1600
BOUNDS
 MI bnd       x66
 UP bnd       x66              21.73
 LO bnd       x477               875
 FR bnd       x542
ENDATA
