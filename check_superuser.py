#!/usr/bin/env python3
import sqlite3
import sys

def check_superuser():
    try:
        # Connect to the database
        conn = sqlite3.connect('langflow.db')
        cursor = conn.cursor()
        
        # Query for the superuser
        cursor.execute("SELECT username, is_active, is_superuser, created_at, last_login_at FROM user WHERE username = 'newadmin.near';")
        result = cursor.fetchone()
        
        if result:
            username, is_active, is_superuser, created_at, last_login_at = result
            print(f"=== Superuser Check ===")
            print(f"Username: {username}")
            print(f"Is Active: {is_active}")
            print(f"Is Superuser: {is_superuser}")
            print(f"Created At: {created_at}")
            print(f"Last Login At: {last_login_at}")
            
            if is_active and is_superuser:
                print("✅ Superuser is properly configured!")
            else:
                print("❌ Superuser configuration issue!")
                
        else:
            print("❌ Superuser not found in database!")
            
        conn.close()
        
    except Exception as e:
        print(f"Error checking superuser: {e}")

if __name__ == "__main__":
    check_superuser()
