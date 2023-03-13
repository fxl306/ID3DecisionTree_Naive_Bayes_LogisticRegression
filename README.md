This application implements the ID3 decision tree, naïve Bayes and the Logistic Regression algorithm.

## ID3 Decision Tree

This implemented ID3 decision Tree algorithm handles nominal and continues attributes, and use both information gain and gain ratio as the split criterion. It takes four options:

1. The path to the data. If this is “/a/b/someproblem” then it will load “/a/b/someproblem.names” and “/a/b/someproblem.data”. 

2. A nonnegative integer that sets the maximum depth of the tree (the number of tests on any path from root to leaf). If this value is zero, you should grow the full tree. If this value is positive, grow the tree to the given value. Note that if this value is too large, it will have the same effect as when the option is zero. 

3. A flag called --no-cv which disables cross validation and runs on the full dataset. If the flag is not present, run cross validation by default. 
4.  A flag called --gain-ratio which uses gain ratio as the split criterion. If the flag is not present, use information gain by default. 

For example, running dtree.py 440data/volcano 5 --gain-ratio will train on the volcano dataset in the path 440data/ with a tree depth limit of 5, will use cross validation, and will use gain ratio as the split criterion. 

When the code is run, it should first construct 5 folds using stratified cross validation (implement the function *cv_split* in *utils.py*) if the no-cv flag is not provided. To ensure repeatability, set the random seed for the PRNG to 12345. Then it should produce decision trees on each fold (or the sample according to the option) and report the following. 

Output format:

>  When the code is run on any problem, it will produce output in the following format: 
>
> > Accuracy: 0.xyz 
> >
> > Size: xyz 
> >
> > Maximum Depth: xyz 
> >
> > First Feature: <name> 
>
> “Accuracy” is the (average) fraction of examples in the test sets (if cross validating) or training set (if not) that were correctly classified by the learned decision tree. It is implement this in the function *accuracy* in *utils.py* . “Size” is the size of the tree in number of nodes, and maximum depth is the length of the longest sequence of tests from root to leaf. The “First Feature” is the name of the first feature that was used to partition the data, or “None” if the tree was empty. 

## Naïve  Bayes

In this implemented naïve bayes algorithm, the range of the feature will be partitioned into *k* bins (value set through an option). To do this, divide the range of the feature into *k* equal-length disjoint intervals. The *x*th bin is the *x*th interval, so replace the original feature with a discrete feature that takes value *x* if the original feature’s value maps to bin *x*. The *m*-estimates is used to smooth the probability estimates. The logs is used whenever possible to avoid multiplying too many probabilities together. The program takes four options: 

1. The path to the data (see the first assignment). 
2.  The –no-cv option (see the first assignment). 
3. A positive integer (at least 2), which is the number of bins for any continuous feature. 
4. A nonnegative integer *m* for the *m*-estimate. If this value is negative, use Laplace smoothing. Note that *m*=0 is maximum likelihood estimation. The value of *p* in the *m*-estimate should be fixed to 1/*v* for a variable with *v* values. 

When the code is run, it will first construct 5 folds using stratified cross validation if this option is provided. To ensure repeatability, the random seed is set for the PRNG to 12345. Then it should produce naïve Bayes models on each fold (or the sample according to the option).

## Logistic Regression

In this implemented logistic regression algorithm, during learning, it will minimize the negative conditional log likelihood plus a constant (*λ*) times a penalty term, half of the 2-norm of the weights squared. The standard gradient descent is used for the minimization. Nominal attributes are encoded as 1-of-N vectors. This program takes three options: 

1. the first two options above.
2. a nonnegative real number that sets the value of the constant *λ*. 
3. The same notes about 5 fold stratified CV from above, etc. apply in this case. 

Output format:

>When either algorithm is run on any problem, it must produce output in exactly the following format: 
>
>> Accuracy: 0.xyz 0.abc 
>>
>> Precision: 0.xyz 0.abc 
>>
>> Recall: 0.xyz 0.abc 
>>
>> Area under ROC: 0.xyz 
>
>For all metrics expect Area under ROC, “0.xyz” is the average value of each quantity over five folds. “0.abc” is the standard deviation. For Area under ROC, use the “pooling” method. Here, after running the classifier on each fold, store all the test examples' classes and confidence values in an array. After all folds are done, use this global array to calculate the area. To calculate the area under ROC, first calculate the TP and FP rates at each confidence level, using the numeric value output by the classifier as the confidence. Each pair of adjacent points is joined by a line, so the area under the curve is the sum over the areas of all trapezoids bounded by the FP rates of the adjacent points, the TP rates of the adjacent points, and the line joining the TP rates. 

## Contents

* 440data

The data files for testing this application.

- notebooks

  The Jupyter notebooks with the testing results

- src 

  source code



