# data metrics
import csv


def split(data):
    '''Split the list of dicts into sublists'''
    customer, distrib, contract, transaction, name1, name2, name3, addr1, addr2, city, state, zipcode, loc_id = ([] for i in range(13))
    for entry in data:
        customer.append(entry["CUSTOMER"])
        distrib.append(entry["DISTRIBUTOR"])
        contract.append(entry["CONTRACT_NUMBER"])
        transaction.append(entry["TRANSACTION_TYPE"])
        name1.append(entry["RAW_NAME1"])
        name2.append(entry["RAW_NAME2"])
        name3.append(entry["RAW_NAME3"])
        addr1.append(entry["RAW_ADDRESS1"])
        addr2.append(entry["RAW_ADDRESS2"])
        city.append(entry["RAW_CITY"])
        state.append(entry["RAW_STATE"])
        zipcode.append(entry["RAW_ZIPCODE"])
        loc_id.append(entry["TARGET_LOC_ID"])


def print_data():
    '''Want to print out Field Uniques Duplicates and Total'''


if __name__ == "__main__":
    '''Turn CSV into a list of dictionaries'''

    with open('subset.csv', newline='') as csvfile:
        data_list = []
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

            data_list.append(entry)
    split(data_list)

    #for entry in data_list:
        #print(entry, "\n")

    
