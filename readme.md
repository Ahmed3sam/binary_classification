### dataset:
the dataset has 3700 rows and 19 columns (18 features & 1 class label)
first thing we will see if there is duplicates rows

### check duplicates rows:
there are many duplicates. We have now only 490 rows out of 3700
and there is the problems we have to get answers:
    1 - nulls
    2 - is there duplicate columns 
    3 - is there bias columns
    4 - the columns has 2 values in a row
    5 - convert the characters to numbers
    
so we need to make more analysis on the data

### more analysis:
We saw that:

    1 - variable2 & variable3 & variable8 have 2 values in a row
    2 - variable1 has 2 characters.
    3 - variable4 has 3 characters.
    4 - variable5 has 3 charchters.
    note that variable4 and variable5 are the same where
    u=g, y=p, l=gg
    
    5 - variable6 has 14 characters.
    6 - variable7 has 9 characters
    7 - variable9 has 2 characters.
    8 - variable10 has 2 characters.
    note the f values in variable10 equal 0 in variable11
    
    9 - variable12 has 2 characters
    10 - variable13 has 3 characters
    11 - variable17 = variable14 * 10000
    12 - variable18 has 2 characters
    13 - variable19 equal classLabel
    
    14 - we have all the alphabets :) and (aa,bb,cc,dd,ff,gg)
    15 - Null problem of course

### solutions:

    1 - we will drop duplicates columns (variable5 - variable10 - variable17)
    2 - we will drop bias columns (variable19)
    
    3 - we will take the mean of the 2 values in variable2 & variable8
    4 - the values in variable3 is very small so we will add them
    
    5 - we will change the characters to ascii (and douple character will multiply by 2)
    6 - we will fill nulls with 0 because the dataset became small after dropping the duplicates
    7 - we will apply that in the validation set either

### machine learning:

using svm model 

### results:

confusuin matrix: 
[[83 24]
 [ 5 88]]

avg/ total
precision: 0.87
recall:    0.85
f1-score:  0.85

