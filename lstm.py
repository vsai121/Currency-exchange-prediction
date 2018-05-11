import numpy as np
import tensorflow as tf

import csv


def read_csv_file(filename):
    name = filename

    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(name, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = csvreader.next()

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

        # get total number of rows
        print("Total no. of rows: %d"%(csvreader.line_num))

    # printing the field names
    print('Field names are:' + ', '.join(field for field in fields))

    #  printing first 5 rows
    print('\nFirst 5 rows are:\n')
    for row in rows[:5]:

        for col in row:
            print("%10s"%col),
        print('\n')


    return rows


data = read_csv_file('USD_INR.csv')
print(data)            
