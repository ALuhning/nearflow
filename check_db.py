#!/usr/bin/env python3
import sqlite3

def check_schema():
    try:
        # Connect to the database
        conn = sqlite3.connect('langflow.db')
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute("PRAGMA table_info(user);")
        columns = cursor.fetchall()
        
        print("=== User Table Schema ===")
        for column in columns:
            print(f"{column[1]} ({column[2]})")
        
        # Query for the superuser with correct column names
        cursor.execute("SELECT * FROM user WHERE username = 'newadmin.near';")
        result = cursor.fetchone()
        
        if result:
            print(f"\n=== Superuser Data ===")
            column_names = [desc[1] for desc in columns]
            for i, value in enumerate(result):
                print(f"{column_names[i]}: {value}")
        else:
            print("\n❌ Superuser not found in database!")
            
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_schema()
