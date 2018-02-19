# -*- coding: utf-8 -*-
"""
@project: buckminsterfullerene
@descr:   A library for creating various features from a given dataframe
@author:  eyazici
@author:  Emre YAZICI
@email:   yaziciemre@gmail.com 
@website: www.yaziciemre.com
@licence: BSD 3-Clause License
@version: 001
@changes: * Initial version
"""

## Imports
import pandas as pd # Data related Operations
import numpy as np # Number related Operations
from sklearn.metrics.cluster import normalized_mutual_info_score # Mutual info score
# =============================================================================

#: Variations for single columns
SINGLE_VARIATIONS_NUMERIC = ["log", "log2", \
                     "sqrt", "pow2.0", "pow0.3", "pow0.1", "pow1.7", \
                     "sign", "abs", "sigmoid", "tanh", "arctan", \
                     "gaussian", "valueOverMean", "valueMinusMean", \
                     "valueMinusMeanOverStd", "valueMinMax", \
                     "null", "positive", ">(x)", "<(x)", ">q25", ">q75"]

SINGLE_VARIATIONS_CATEGORIC = ["=mode", "mean", 'max', 'min', 'std']

MULTI_VARIATIONS = ["a+b", "a*b", "a/b", "a>b", \
                    "max(a, b)", "min(a, b)", "a==b",
                    "a(x)+b", "a+b(x)", "a^(x)+b", "a+b^(x)", \
                    "a*b>0"]

#: Created temp feature name
COLUMN_TEMP = "bmftemp"
#: Create temp feature name prefix
COLUMN_PREFIX = "bmf_"

# method: gini
# Calculates and returns gini
# @actual: The actual values
# @pred: The predicted values
# @return: The gini value
# @completed
def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses

     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)

# method: gini_normalized
# Calculates and returns gini normalized
# @a: The actual values
# @p: The predicted values
# @return: The gini value
# @completed
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

# method: sigmoid
# Sigmoid function
# @x: The input values
# @return: The output values
# @completed
def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

# method: gaussian
# Gaussian function
# @x: The input values
# @return: The output values
# @completed
def gaussian(x):
    return np.exp(-x*x)

# class: buckminsterfullerene
# The main class for the module
# @completed
class buckminsterfullerene(object):
    
    # method: __init__
    # Initializes the class
    # @dataFrame, dataframe: The input dataframe
    # @target, str: The input dataframe target name
    # @completed
    def __init__( self, dataFrame, target ):
        #: Set the dataframe to be processed by shuffling it
        self.dataFrame = dataFrame.reindex(np.random.permutation(dataFrame.index))
        #: Set the name of the target column
        self.target = target
        #: Set the feature index
        self.featureIndex = 0
        #: Create features
        self.features = {}
        #: Iteration
        self.iteration = 0
        #: Create a dictionary of the types of columns
        self.types = {}
        #: Loop to find datatype
        for c in dataFrame:
            self.types[ c ] = self.__determineType__( c )
        
    # __determineType__
    # Determines the type of the column
    # @column, str: The input column
    # @return, str: The type of the column
    # @completed
    def __determineType__( self, column ):
        #: Get first "one hundred" values
        series = self.dataFrame[ column ].head(1000).values
        #: Loop for each
        for v in series[0:10]:
            #: Return according to type
            if type(v) is str:
                return "str"
            if type(v) is float or type(v) is np.float64:
                return "float"
            if type(v) is int or type(v) is np.int64:
                if len(np.unique(series)) == 2:
                    return "bool"
                elif len(np.unique(series)) == 1000:
                    return "unique"
                else:
                    return "int"

    # method: createMultiFeature
    # Creates multi features
    # @column1/column2, str: The input column names
    # @variation, str: The variation
    # @param, str: The parameter
    # @return, column: The derived column
    # @completed
    def createMultiFeature( self, column1, column2, variation, param = 0 ):
        #: Switch according to criteria
        if variation == "a*b>0":
            return (self.dataFrame[ column1 ] * self.dataFrame[ column2 ]) > 0
        if variation == "a+b":
            return self.dataFrame[ column1 ] + self.dataFrame[ column2 ]
        if variation == "a*b":
            return self.dataFrame[ column1 ] * self.dataFrame[ column2 ]
        if variation == "a/b":
            return self.dataFrame[ column1 ] / self.dataFrame[ column2 ]
        if variation == "a>b":
            return self.dataFrame[ column1 ] > self.dataFrame[ column2 ]
        if variation == "max(a, b)":
            return np.maximum(self.dataFrame[ column1 ], self.dataFrame[ column2 ])
        if variation == "min(a, b)":
            return np.minimum(self.dataFrame[ column1 ], self.dataFrame[ column2 ])
        if variation == "a==b":
            return self.dataFrame[ column1 ] == self.dataFrame[ column2 ]
        if variation == "a(x)+b":
            return self.dataFrame[ column1 ]*param + self.dataFrame[ column2 ]
        if variation == "a+b(x)":
            return self.dataFrame[ column1 ] + self.dataFrame[ column2 ]*param
        if variation == "a^(x)+b":
            return np.power(self.dataFrame[ column1 ], param) + self.dataFrame[ column2 ]
        if variation == "a+b^(x)":
            return self.dataFrame[ column1 ] + np.power(self.dataFrame[ column2 ], param)
        #: Raise error
        raise Exception('Invalid variation:', variation)
        
    # method: createSingleFeature
    # Creates multi features
    # @column, str: The input column name
    # @variation, str: The variation
    # @param, float: The parameter
    # @return, column: The derived column
    # @completed        
    def createSingleFeature( self, column, variation, param = 0 ):
        #: Switch according to criteria
        if variation == "max":
            valmap = df.groupby([ column ])[ self.target ].max().to_dict()
            return self.dataFrame[ column ].apply(lambda value: valmap[ value ])
        if variation == "min":
            valmap = df.groupby([ column ])[ self.target ].min().to_dict()
            return self.dataFrame[ column ].apply(lambda value: valmap[ value ])
        if variation == "std":
            valmap = df.groupby([ column ])[ self.target ].std().to_dict()
            return self.dataFrame[ column ].apply(lambda value: valmap[ value ])
        if variation == "mean":
            valmap = df.groupby([ column ])[ self.target ].mean().to_dict()
            return self.dataFrame[ column ].apply(lambda value: valmap[ value ])
        if variation == "=mode":
            return self.dataFrame[ column ] == self.dataFrame[ column ].mode()[0]
        if variation == ">q75":
            return self.dataFrame[ column ] < self.dataFrame[ column ].quantile(.75)
        if variation == ">q25":
            return self.dataFrame[ column ] < self.dataFrame[ column ].quantile(.25)
        if variation == "<(x)":
            return self.dataFrame[ column ] < param
        if variation == ">(x)":
            return self.dataFrame[ column ] > param
        if variation == "positive":
            return self.dataFrame[ column ] > 0
        if variation == "null":
            return np.isnan( self.dataFrame[ column ] )
        if variation == "log":
            return np.log( self.dataFrame[ column ] )
        if variation == "sqrt":
            return np.sqrt( self.dataFrame[ column ] )
        if variation == "pow2.0":
            return np.power( self.dataFrame[ column ], 2.0 )
        if variation == "pow0.3":
            return np.power( self.dataFrame[ column ], 0.3 )
        if variation == "pow0.1":
            return np.power( self.dataFrame[ column ], 0.1 )
        if variation == "pow1.7":
            return np.power( self.dataFrame[ column ], 1.7 )
        if variation == "abs":
            return np.abs( self.dataFrame[ column ] )
        if variation == "sign":
            return np.sign( self.dataFrame[ column ] )
        if variation == "log2":
            return np.log2( self.dataFrame[ column ] )
        if variation == "sigmoid":
            return sigmoid( self.dataFrame[ column ] )
        if variation == "valueOverMean":
            return self.dataFrame[ column ] / self.dataFrame[ column ].mean()
        if variation == "valueMinusMean":
            return self.dataFrame[ column ] - self.dataFrame[ column ].mean()
        if variation == "tanh":
            return np.tanh(self.dataFrame[ column ])
        if variation == "arctan":
            return np.arctan(self.dataFrame[ column ])
        if variation == "gaussian":
            return gaussian( self.dataFrame[ column ] )
        if variation == "valueMinusMeanOverStd":
            return (self.dataFrame[ column ] - self.dataFrame[ column ].mean()) / self.dataFrame[ column ].std()        
        if variation == "valueMinMax":
            return (self.dataFrame[ column ] - self.dataFrame[ column ].min()) / (self.dataFrame[ column ].max() - self.dataFrame[ column ].min())
        #: Raise error
        raise Exception('Invalid variation:', variation)
    
    # method: __singleValid__
    # Validates that the newly created features gain is more important
    # @base, float: The base importance
    # @new, float: The new importances
    # @minAddition, float: The minimum additional amount to be gained
    # @minRatio, float: The minimum ratio amount to be gained
    # @threshold, float: The minimum limit
    # @return, bool: Valid or not
    # @completed
    def __singleValid__( self, base, new, minAddition, minRatio, threshold = 0.0 ):
        return new > base and \
            new > base + minAddition and \
            new > base * minRatio and \
            new > threshold

    # method: __multiValid__
    # Validates that the newly created features gain is more important
    # @base1, float: The base 1 importance
    # @base2, float: The base 2 importance
    # @new, float: The new importances
    # @minAddition, float: The minimum additional amount to be gained
    # @minRatio, float: The minimum ratio amount to be gained
    # @threshold, float: The minimum limit
    # @return, bool: Valid or not
    # @completed
    def __multiValid__( self, base1, base2, new, minAddition, minRatio, threshold = 0.0 ):
        return new > base1 and \
            new > base2 and \
            new > base1 + minAddition and \
            new > base2 + minAddition and \
            new > base1 * minRatio and \
            new > base2 * minRatio and \
            new > threshold

    # __information__
    # Returns the mutual information 
    # @column, str: The column to be checked
    # @criteria, string: The type of criteria: (correlation, gini)
    # @return, float: The mutual information
    # @completed
    def __information__( self, column, criteria ):
        #: Switch according to criteria
        if criteria == "correlation":
            return abs( self.dataFrame[ self.target ].corr( self.dataFrame[ column ] ) )
        if criteria == "gini":
            return gini_normalized( self.dataFrame[ self.target ], self.dataFrame[ column ] )
        if criteria == "mutual":
            return normalized_mutual_info_score( self.dataFrame[ self.target ], self.dataFrame[ column ] )        
        #: Raise error
        raise Exception('Invalid criteria:', criteria)
    
    # method: single
    # Creates new features for single columns
    # @criteria, string: The type of criteria: (correlation, gini)
    # @excludeColumns, array: Which columns to be excluded
    # @excludeVariations, array: Which variations to be excluded
    # @minAddition, float: The minimum additional amount to be gained
    # @minRatio, float: The minimum ratio amount to be gained
    # @threshold, float: The minimum limit
    # @completed
    def single( self, \
               criteria = 'correlation', \
               excludeColumns = [], \
               excludeVariations = [], \
               minAddition = 0.05, \
               minRatio = 1.10, \
               threshold = 0.10):
        #: P values
        ps = [0.01, 0.1, 0.5, 1, 1.5, 3, 5, 10, 20, 100]
        #: For each column
        for c in self.dataFrame.columns:
            #: Skip the target column
            if c != self.target and c not in excludeColumns:
                #: Find the correlation
                cs = self.__information__( c, criteria )
                #: For each variation
                for v in self.__getSingleVariations__( c ):
                    #: Check if the variation is not blocked
                    if v not in excludeVariations:
                        #: Check if the variation has parameter?
                        for p in ps if "(x)" in v else [0]:
                            #: Apply the variation
                            self.dataFrame[ COLUMN_TEMP ] = self.createSingleFeature( c, v, p )
                            #: Measure the correlation
                            ct = self.__information__( COLUMN_TEMP, criteria )
                            #: If a better correlation
                            if self.__singleValid__( cs, ct, minAddition, minRatio, threshold):
                                #: Add new feature
                                self.__addNewFeature__( ('single', v, c, cs, ct, p) )
        #: Delete the temp if exists
        if COLUMN_TEMP in self.dataFrame:
            #: Delete
            del self.dataFrame[ COLUMN_TEMP ]
        #: Increment the iteration
        self.iteration += 1
        
    # __getSingleVariations__
    # Returns the variations for the given column
    # @column, str: The name of the column
    # @return, array: The possible variations for the column
    # @completed
    def __getSingleVariations__( self, column ):
        if self.types[column] in ["int", "float"]:
            return SINGLE_VARIATIONS_NUMERIC
        if self.types[column] in ["str", "bool"]:
            return SINGLE_VARIATIONS_CATEGORIC
        #: Raise error
        raise Exception('Invalid type:', column, self.types[column])
        
    # __addNewFeature__
    # Adds a new feature to the list
    # @tupleVar: The tuple variable
    # @completed
    def __addNewFeature__( self, tupleVar ):
        #: Print
        print (tupleVar)
        #: Increment the feature inex
        self.featureIndex += 1
        #: Assign it to a new column
        self.dataFrame[ COLUMN_PREFIX + chr(65+self.iteration) + "_" + str(self.featureIndex) ] = self.dataFrame[ COLUMN_TEMP ]
        #: Create the feature definition
        self.features[ COLUMN_PREFIX + chr(65+self.iteration) + "_" + str(self.featureIndex) ] = tupleVar
        #: As type
        self.dataFrame[ COLUMN_PREFIX + chr(65+self.iteration) + "_" + str(self.featureIndex) ] = self.dataFrame[ COLUMN_PREFIX + chr(65+self.iteration) + "_" + str(self.featureIndex) ].astype(float)
        
    # method: multi
    # Creates new features for multi columns
    # @criteria, string: The type of criteria: (correlation, gini)
    # @excludeColumns, array: Which columns to be excluded
    # @excludeVariations, array: Which variations to be excluded
    # @minAddition, float: The minimum additional amount to be gained
    # @minRatio, float: The minimum ratio amount to be gained
    # @threshold, float: The minimum limit
    # @completed
    def multi( self, \
               criteria = 'correlation', \
               excludeColumns = [], \
               excludeVariations = [], \
               minAddition = 0.10, \
               minRatio = 1.15, \
               threshold = 0.20 ):
        #: P values
        ps = [0.01, 0.1, 0.5, 1, 1.5, 3, 5, 10, 20, 100]
        #: For each column
        for c1 in self.dataFrame.columns:
            #: For each column
            for c2 in self.dataFrame.columns:
                #: Skip the target column
                if c1 != self.target and c1 not in excludeColumns and \
                    c2 != self.target and c2 not in excludeColumns and \
                    c1 > c2:
                    #: Find the correlation
                    c1s = self.__information__( c1, criteria )
                    c2s = self.__information__( c2, criteria )
                    #: For each variation
                    for v in MULTI_VARIATIONS:
                        #: Check if the variation is not blocked
                        if v not in excludeVariations:
                            #: Check if the variation has parameter?
                            for p in ps if "(x)" in v else [0]:
                                #: Apply the variation
                                self.dataFrame[ COLUMN_TEMP ] = self.createMultiFeature( c1, c2, v )
                                #: Measure the correlation
                                ct = self.__information__( COLUMN_TEMP, criteria )
                                #: If a better correlation
                                if self.__multiValid__( c1s, c2s, ct, minAddition, minRatio, threshold):
                                    #: Add new feature
                                    self.__addNewFeature__( ('multi', v, c1, c2, c1s, c2s, ct, p) )
        #: Delete the temp if exists
        if COLUMN_TEMP in self.dataFrame:
            #: Delete
            del self.dataFrame[ COLUMN_TEMP ]
        #: Increment the iteration
        self.iteration += 1
