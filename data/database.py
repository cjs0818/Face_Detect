# -*- coding: utf-8 -*-

import pprint
import csv
import os

# Mongo DB
from pymongo import MongoClient


# ----------------------------------------------------
# database from CSV file
# ----------------------------------------------------
class Database():
    #def __init__(self):

    def get_datatbase(self, filename):
        #filename = 'RMI_researchers.csv'

        with open(filename, 'r', encoding='UTF-8-sig') as f:
            csv_data = csv.reader(f, delimiter=',')
            print("-------------")
            dict = {}
            row_cnt = 0
            for row in csv_data:
                row_cnt = row_cnt + 1
                if row_cnt == 1:
                    key = row
                else:
                    for i in range(0, len(row), 1):
                        if i == 0:
                            # print(dict_name)
                            dict_info = {}
                        else:
                            dict_info.update({key[i]: row[i]})
                            # print(dict_info)
                    dict.update({row[0]: dict_info})
                    # print("dict_name = ", dict_name)

        # json_data = json.dumps(dict, indent=4, ensure_ascii=False)
        # print(json_data)

        return dict

    def search(self, db, key, value):
        # ----------------------------
        # -- Search from csv file
        kor_name = []
        for name in db.keys():
            info = db[name]
            if info[key] == value:
                kor_name = name

        return kor_name

# ----------------------------------------------------
# Mongo DB
#    folder: BASE_DIR/data/db
# ----------------------------------------------------
class MongoDB():
    def __init__(self, db_name="DB_reception", coll_name="RMI_researchers"):
        self.db_client = MongoClient('localhost', 27017)
        #self.db = self.db_client["DB_Episode"]
        #self.coll = self.db.coll_Test
        self.db = self.db_client[db_name]
        self.coll = self.db[coll_name]

    def insert(self, post):
        post_id = self.mdb_collection.insert_one(post).inserted_id
        coll_list = self.mdb.collection_names()
        print(coll_list)

    def search(self, key, value):
        # ------------------------
        # Search from MongoDB
        kor_name = []
        name_dict = { key: value}
        result = self.coll.find(name_dict)
        try:
            kor_name = result[0]["name"]
        except Exception as e:
            pass

        return kor_name
# ----------------------------------------------------
# Main
# ----------------------------------------------------
if __name__ == '__main__':

    # ------------------------
    # Load Database
    PARENT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
    #DB_DIR = os.path.join(PARENT_DIR, "Receptionbot_Danbee/receptionbot")
    filename = PARENT_DIR + "/RMI_researchers.csv"

    C_db = Database()
    db = C_db.get_datatbase(filename)

    #print(db)
    pprint.pprint(db)


    # --------------------------------
    # Mongo DB
    db_name = "DB_reception"        # define a Database
    coll_name = "RMI_researchers"   # define a Collection
    mgdb = MongoDB(db_name, coll_name)

    db_name = "DB_reception"        # define a Database
    coll_name = "Event"             # define a Collection
    mgdb_event = MongoDB(db_name, coll_name)

    # Test the 'find' of Mongo DB
    eng_name = "jschoi"
    name_dict = { "english_name": eng_name }
    result = mgdb.coll.find(name_dict)
    pprint.pprint(result[0])
    print(result[0]["name"])
