import requests
import pandas as pd
import io
import zipfile
import os
from datetime import datetime

OUTPUT_FILE = "data/ercot_da_spp_combined.csv"
HIST_JSON_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=13060"
DOWNLOAD_URL_TEMPLATE = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={}"

TARGET_YEARS = [2024, 2025, 2026] 
TARGET_NODE = "HB_NORTH"

def process_dataframe(df_year, year, filename, sheet_name):
    """
    Process a single dataframe (from one sheet or CSV), filter for HB_NORTH, 
    and return the cleaned dataframe. Returns None if empty or invalid.
    """
    try:
        # Normalize columns
        df_year.columns = [str(c).strip() for c in df_year.columns]
        
        # Identify columns
        spp_col = next((c for c in df_year.columns if "Price" in c), "Settlement Point Price")
        node_col = next((c for c in df_year.columns if "Point" in c and "Price" not in c), "Settlement Point")
        date_col = next((c for c in df_year.columns if "Date" in c), "Delivery Date")
        hour_col = next((c for c in df_year.columns if "Hour" in c), "Hour Ending")
        
        if node_col not in df_year.columns:
            return None

        filtered = df_year[df_year[node_col] == TARGET_NODE].copy()
        
        if len(filtered) == 0:
            return None

        # Construct timestamp
        filtered[date_col] = pd.to_datetime(filtered[date_col])
        
        def parse_he(date, he_str):
            try:
                # Handle 24, 1, etc
                he = int(float(he_str))
            except:
                # Handle "01:00" strings
                he = int(str(he_str).split(":")[0])
            
            delta = pd.Timedelta(hours=he-1)
            return date + delta

        filtered['interval_start_utc'] = filtered.apply(lambda row: parse_he(row[date_col], row[hour_col]), axis=1)
        
        # Rename for consistency
        filtered = filtered.rename(columns={spp_col: 'spp'})
        filtered['location'] = TARGET_NODE
        
        # Select only needed
        final_df = filtered[['interval_start_utc', 'spp', 'location']]
        return final_df

    except Exception as e:
        print(f"Error parsing sheet {sheet_name} in {filename}: {e}")
        return None

def fetch_historical_data():
    all_dfs = []
    
    # 1. Get list of documents
    print(f"Fetching document list from {HIST_JSON_URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    resp = requests.get(HIST_JSON_URL, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    
    docs = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
    print(f"Found {len(docs)} documents.")
    
    # 2. Filter for target years
    target_docs = []
    for entry in docs:
        doc = entry.get("Document", {})
        friendly_name = doc.get("FriendlyName", "")
        # Expect naming confirmation: DAMLZHBSPP_2024
        for year in TARGET_YEARS:
            if f"_{year}" in friendly_name and "DAMLZHBSPP" in friendly_name:
                target_docs.append((year, doc))
                print(f"Found match: {friendly_name} (DocID: {doc.get('DocID')})")
                
    # 3. Download and Parse
    for year, doc in target_docs:
        doc_id = doc.get("DocID")
        url = DOWNLOAD_URL_TEMPLATE.format(doc_id)
        print(f"Downloading {year} data from {url}...")
        
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for filename in z.namelist():
                    if filename.endswith(".xlsx") or filename.endswith(".csv"):
                        print(f"  Parsing {filename}...")
                        with z.open(filename) as f:
                            if filename.endswith(".xlsx"):
                                # Read all sheets
                                xls = pd.ExcelFile(f)
                                sheet_names = xls.sheet_names
                                print(f"    Sheets found: {len(sheet_names)}")
                                for sheet in sheet_names:
                                    df_sheet = pd.read_excel(xls, sheet_name=sheet)
                                    res = process_dataframe(df_sheet, year, filename, sheet)
                                    if res is not None:
                                        all_dfs.append(res)
                            else:
                                df_csv = pd.read_csv(f)
                                res = process_dataframe(df_csv, year, filename, "csv")
                                if res is not None:
                                    all_dfs.append(res)
                        
        except Exception as e:
            print(f"Error processing {year}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Combine and Save
    if all_dfs:
        full_df = pd.concat(all_dfs)
        full_df = full_df.sort_values(by="interval_start_utc")
        # Remove duplicates if overlaps
        full_df = full_df.drop_duplicates(subset=['interval_start_utc'])
        
        print(f"Total rows collected: {len(full_df)}")
        print(full_df.head())
        print(full_df.tail())
        
        full_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved combined data to {OUTPUT_FILE}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    fetch_historical_data()
