# OpenFisca-Reformator

Algorithmic optimization of the tax and benefit system leading to a reform

### Why a reformator?

The tax and benefit legislation is complex. The full implementation is more than 100,000 lines of code.
This tool helps to make reform that returns a complete and simplified tax and benefit system.

It can be applied on small parts of the legislation, or on the whole legislation.

### How it works:

1. Define your concepts (e.g., 'nb_enfants', 'age', 'parent isolé') and budget (e.g.: cost less than 2 millions euros)
2. The machine learning algorithm helps you adjust the parameters of your reform to approximate the current legislation and fit your budget
3. From the biggest discrepencies with the current legislation, you can improve your concepts (or the legislation)
4. Repeat until you reach a legislation that matches your own goals. The algorithm takes care of minimizing your budget and maximizing the similarity to current legislation.

### Limitations:

The results can only be as good as the input data, that should contain at least 10,000 individuals.

### Current Results:

We got a tax reform for "aides sociales" and taxes for people with no salary that is:

* a few lines long 
* similar to the existing in average at more than 95%

### Demo

https://github.com/openfisca/reformator/blob/master/notebooks/visual_reformator.ipynb
