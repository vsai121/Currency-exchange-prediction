import numpy as np
import tensorflow as tf

import csv

INPUT_SIZE = 3
NUM_STEPS = 2

def read_csv_file(filename):
    name = filename

    """initializing the titles and rows list"""
    fields = []
    rows = []

    """reading csv file"""
    with open(name, 'r') as csvfile:
        """creating a csv reader object"""
        csvreader = csv.reader(csvfile)

        """extracting field names through first row"""
        fields = csvreader.next()

        """extracting each data row one by one"""
        for row in csvreader:
            rows.append(row)

        """get total number of rows"""
        print("Total no. of rows: %d"%(csvreader.line_num))

    """printing the field names"""
    print('Field names are:' + ', '.join(field for field in fields))

    """printing first 5 rows"""
    print('\nFirst 5 rows are:\n')
    for row in rows[:5]:

        for col in row:
            print("%10s"%col),
        print('\n')


    return rows


def read_prices():
    data = read_csv_file('USD_INR.csv')
    prices = []

    """reversing data (to predict future prices)"""
    for row in (data[::-1]):

        prices.append(row[1])

    return prices


def split_data(input):

    """
    Splits a sequence into windows
    """

    seq = [np.array(input[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input) // INPUT_SIZE)]

    """Split into groups of `num_steps"""
    X = np.array([seq[i: i + NUM_STEPS] for i in range(len(seq) - NUM_STEPS)])
    y = np.array([seq[i + NUM_STEPS] for i in range(len(seq) - NUM_STEPS)])


    return X , y


def train_test_split(X , y):

    """
    Splitting data into training and test data"
    """

    X_train = X[0:3000]
    X_test= X[3000:]

    y_train = y[0:3000]
    y_test = y[3000:]

    """
    Validating shapes

    print("X_train" , X_train.shape)
    print("Y_train" , y_train.shape)
    print("X_test" , X_test.shape)
    print("Y_test" , y_test.shape)

    """

    return X_train , y_train , X_test , y_test


prices = read_prices()
X , y = split_data(prices)

X_train , y_train , X_test , y_test = train_test_split(X,y)
