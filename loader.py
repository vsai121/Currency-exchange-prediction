import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
import csv

INPUT_SIZE = 1
NUM_STEPS = 30
TRAIN_TEST_RATIO = 0.1

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

        prices.append(float(row[1]))

    return prices


def normalize_prices(prices):

    prices = np.asarray(prices)
    temp = prices.reshape(len(prices),1)

    scaler = StandardScaler()
    scaler = scaler.fit(temp)

    prices = scaler.transform(temp)

    return prices


def normalize_seq(seq):

    #print(seq)
    seq = [seq[0] / seq[0][0] - 1.0] + [curr / seq[i-1][-1] - 1.0 for i, curr in enumerate(seq[1:])]
    return seq

def split_data(input):

    """
    Splits a sequence into windows
    """

    seq = [np.array(input[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input) // INPUT_SIZE)]

    #Normalizing seq
    seq = normalize_seq(seq)

    """Split into groups of num_steps"""
    X = np.array([seq[i: i + NUM_STEPS] for i in range(len(seq) - NUM_STEPS)])
    y = np.array([seq[i + NUM_STEPS] for i in range(len(seq) - NUM_STEPS)])

    return X , y


def train_test_split(X , y):

    """
    Splitting data into training and test data"
    """

    train_size = int(len(X) * (1.0 - TRAIN_TEST_RATIO))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]


    return X_train , y_train , X_test , y_test


def process():
    prices = read_prices()
    #print("Prices" , prices)
    X , y = split_data(prices)

    #print("X" , "Y")
    #print(X[1])
    #print(y[1])

    X_train , y_train , X_test , y_test = train_test_split(X,y)
    #print(X.shape)
    #print(y.shape)

    return X_train , y_train , X_test , y_test
