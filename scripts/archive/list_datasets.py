from gridstatusio import GridStatusClient
from dotenv import load_dotenv
import os

load_dotenv()

def list_ercot_datasets():
    api_key = os.getenv("GRIDSTATUS_API_KEY")
    if not api_key:
        print("Error: GRIDSTATUS_API_KEY not found.")
        return

    client = GridStatusClient(api_key=api_key)
    
    # List datasets filtering for ERCOT
    # filtering is done client side if filter_term isn't supported or behaves differently
    try:
        datasets = client.list_datasets()
        print(f"Found {len(datasets)} datasets for 'ercot'.")
        
        for d in datasets:
            # We are looking for Day-Ahead Settlement Point Prices
            name = str(d.get('name', '')).lower()
            d_id = str(d.get('id', '')).lower()
            if 'spp' in d_id and ('day' in name or 'da' in d_id):
                print(f"MATCH -> ID: {d['id']} | Name: {d['name']}")
                
    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    list_ercot_datasets()
