# data metrics
import csv


with open('subset.csv', newline='') as csvfile:
    list = []
    entry = {}
    ml_data = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in ml_data:
        entry = {
            "CUSTOMER": row[0],
            "DISTRIBUTOR": row[1],
            "CONTRACT_NUMBER": row[2],
            "TRANSACTION_TYPE": row[3],
            "RAW_NAME1": row[4],
            "RAW_NAME2": row[5],
            "RAW_NAME3": row[6],
            "RAW_ADDRESS1": row[7],
            "RAW_ADDRESS2": row[8],
            "RAW_CITY": row[9],
            "RAW_STATE": row[10],
            "RAW_ZIPCODE": row[11],
            "TARGET_LOC_ID": row[12]
        }

        list.append(entry)

    for entry in list:
        print(entry, "\n")
