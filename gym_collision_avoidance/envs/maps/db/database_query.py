from tinydb import TinyDB, Query
import time

if __name__ == "__main__":
    db = TinyDB("map_db.json")
    db2 = TinyDB("map_selection_db.json")

    db2.truncate()

    table1 = db2.table("level1")
    Map = Query()
    result = db.search(
        (Map.size <= 10) & (Map.rooms == 2) & (Map.area <= 50) & (Map.area >= 30)
    )
    print(len(result))
    print(result[200])
    table1.insert_multiple(result)

    table2 = db2.table("level2")

    result2 = db.search(
        (Map.size < 20)
        & (Map.rooms <= 5)
        & (Map.rooms > 2)
        & (Map.area <= 200)
        & (Map.area >= 150)
    )
    print(len(result2))
    print(result2[200])
    table2.insert_multiple(result2)

    table3 = db2.table("level3")

    result3 = db.search(
        (Map.size < 20)
        & (Map.rooms <= 9)
        & (Map.rooms > 5)
        & (Map.area <= 400)
        & (Map.area >= 300)
    )
    print(len(result3))
    print(result3[200])
    table3.insert_multiple(result3)
