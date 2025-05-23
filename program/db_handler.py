import sqlite3
from pathlib import Path
import csv 

class DBHandler:
    def __init__(self, db_path="ais_data.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
       # Create tables if they do not exist
       
       # vessels: id, mmsi, ship_type   
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS vessels (
            id INTEGER PRIMARY KEY,
            mmsi INTEGER UNIQUE,
            ship_type TEXT
        )
        ''')

        # raw_ais_data: id, vessel_id (foreign key), timestamp, lat, lon, speed, heading
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_ais_data (
            id INTEGER PRIMARY KEY,
            vessel_id INTEGER,
            timestamp TEXT,
            latitude REAL,
            longitude REAL,
            navigational_status TEXT,
            rot REAL,
            sog REAL,
            cog REAL,
            heading REAL,
            behavior_label TEXT,

            FOREIGN KEY (vessel_id) REFERENCES vessels(id)
        )
        ''')

        

    def insert_raw_data(self, data):
        """
        data: List of tuples (
            mmsi, ship_type, timestamp, lat, lon, nav_status, rot,
            sog, cog, heading, behavior_label
        )
        """
        insert_rows = []

        for row in data:
            mmsi, ship_type, timestamp, lat, lon, nav_status, rot, sog, cog, heading, behavior_label = row

            # Get or insert vessel
            self.cursor.execute('SELECT id FROM vessels WHERE mmsi = ?', (mmsi,))
            result = self.cursor.fetchone()

            if result:
                vessel_id = result[0]
            else:
                self.cursor.execute(
                    'INSERT INTO vessels (mmsi, ship_type) VALUES (?, ?)',
                    (mmsi, ship_type)
                )
                vessel_id = self.cursor.lastrowid
                print(f"➕ New vessel added: MMSI {mmsi}, Ship Type '{ship_type}'")

            insert_rows.append((
                vessel_id, timestamp, lat, lon, nav_status, rot,
                sog, cog, heading, behavior_label
            ))

        self.cursor.executemany('''
            INSERT INTO raw_ais_data (
                vessel_id, timestamp, latitude, longitude,
                navigational_status, rot, sog, cog, heading,
                behavior_label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', insert_rows)

        self.conn.commit()



    def load_csv_and_insert_raw_data(db, csv_path):
        data = []

        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    mmsi = int(row['MMSI'])
                    ship_type = row['Ship type'].strip()
                    timestamp = row['Timestamp']
                    lat = float(row['Latitude'])
                    lon = float(row['Longitude'])
                    nav_status = row['Navigational status'].strip()
                    rot = float(row['ROT']) if row['ROT'] else 0.0
                    sog = float(row['SOG']) if row['SOG'] else 0.0
                    cog = float(row['COG']) if row['COG'] else 0.0
                    heading = float(row['Heading']) if row['Heading'] else 0.0
                    behavior_label = row['Behavior Label'].strip()

                    data.append((
                        mmsi, ship_type, timestamp, lat, lon, nav_status, rot,
                        sog, cog, heading, behavior_label
                    ))
                except (ValueError, KeyError) as e:
                    print(f"⚠️ Skipping row due to error: {e}")

        db.insert_raw_data(data)


    def close(self):
        self.conn.close()
