
import requests
import pandas as pd
import io
import zipfile
import os
from datetime import datetime

OUTPUT_FILE = "data/raw/ercot_da_spp_5y_all_hubs.parquet"
DATA_DIR = "data/raw"

# Doc Type 12331 is "Day-Ahead Market (DAM) Settlement Point Prices"
# But sometimes historicals are in different report types.
# fetch_historical_ercot.py used 13060 (Historical RTM? No, likely DAM).
# Let's use the one from that script: 13060 "Historical DAM Locational Marginal Prices (LMP)"?
# Wait, we want SPP (Settlement Point Prices).
# 12331 is often the daily report.
# 13060 is "Day-Ahead Market (DAM) Settlement Point Prices (SPP) - Historical"

HIST_JSON_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=13060"
DOWNLOAD_URL_TEMPLATE = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={}"

TARGET_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026] 

TARGET_HUBS = [
    "HB_NORTH",
    "HB_SOUTH",
    "HB_WEST",
    "HB_HOUSTON",
    "HB_PAN",
    "HB_BUSAVG",
    "HB_HUBAVG"
]

def process_dataframe(df_year, year, filename, sheet_name):
    try:
        # Normalize columns
        df_year.columns = [str(c).strip() for c in df_year.columns]
        
        # Identify columns
        spp_col = next((c for c in df_year.columns if "Price" in c), "Settlement Point Price")
        node_col = next((c for c in df_year.columns if "Point" in c and "Price" not in c), "Settlement Point")
        date_col = next((c for c in df_year.columns if "Date" in c), "Delivery Date")
        hour_col = next((c for c in df_year.columns if "Hour" in c), "Hour Ending")
        
        if node_col not in df_year.columns:
            print(f"Skipping {sheet_name}: '{node_col}' column not found in {df_year.columns.tolist()}")
            return None

        # Clean ID values
        df_year[node_col] = df_year[node_col].astype(str).str.strip()

        # Filter for target hubs
        filtered = df_year[df_year[node_col].isin(TARGET_HUBS)].copy()
        
        if len(filtered) == 0:
            print(f"Skipping {sheet_name}: No target hubs found. (Unique: {df_year[node_col].unique()[:5]})")
            return None

        # Construct timestamp
        filtered[date_col] = pd.to_datetime(filtered[date_col])
        
        def parse_he(date, he_str):
            try:
                he = int(float(he_str))
            except:
                he = int(str(he_str).split(":")[0])
            
            # HE 24 is 00:00 next day, but pandas Timedelta logic:
            # interval_start = date + (he-1) hours
            delta = pd.Timedelta(hours=he-1)
            return date + delta

        filtered['interval_start_utc'] = filtered.apply(lambda row: parse_he(row[date_col], row[hour_col]), axis=1)
        
        # Rename for consistency
        filtered = filtered.rename(columns={spp_col: 'spp', node_col: 'location'})
        
        # Select only needed
        final_df = filtered[['interval_start_utc', 'spp', 'location']]
        return final_df

    except Exception as e:
        print(f"Error parsing sheet {sheet_name} in {filename}: {e}")
        return None

def fetch_historical_archive():
    all_dfs = []
    
    print(f"Fetching document list from {HIST_JSON_URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    
    try:
        resp = requests.get(HIST_JSON_URL, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        docs = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
        print(f"Found {len(docs)} documents.")
        
        # Filter for target years
        target_docs = []
        for entry in docs:
            doc = entry.get("Document", {})
            friendly_name = doc.get("FriendlyName", "")
            # Expect naming: dam_spp_2020.zip or similar. Usually contains "spp" and year.
            # Filename often: "dam_spp_yyyy.zip" or "DAMLZHBSPP_yyyy.zip"
            
            for year in TARGET_YEARS:
                if str(year) in friendly_name and "spp" in friendly_name.lower():
                    # Deduplicate if multiple matches for same year?
                    # Usually ERCOT puts one per year.
                    target_docs.append((year, doc))
                    print(f"Found match: {friendly_name} (DocID: {doc.get('DocID')})")
                    
        # Sort to ensure order
        target_docs.sort(key=lambda x: x[0])
        
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
                                    xls = pd.ExcelFile(f)
                                    sheet_names = xls.sheet_names
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
                
        if all_dfs:
            full_df = pd.concat(all_dfs)
            # Ensure UTC conversion if not already (it is naive datetime + timedelta above)
            # Actually ERCOT data is usually local time (CST), need to localize.
            # For simplicity in this feasibility check, we'll treat as local-ish or simplistic UTC.
            # But gridstatus standardized it.
            # Standard: ERCOT is CST/CDT. 
            # Let's assume input 'interval_start_utc' is essentially the local time for now, 
            # or standardize later. The forecast model uses lags so relative time matters most.
            # But let's try to set to UTC if we can. 
            # Usually column is 'Interval Ending' in local time. 
            # Let's just keep as datetime for now.
            
            full_df = full_df.sort_values(by=['location', 'interval_start_utc'])
            full_df = full_df.drop_duplicates(subset=['location', 'interval_start_utc'])
            
            print(f"Total rows collected: {len(full_df)}")
            print("Breakdown by hub:")
            print(full_df['location'].value_counts())
            
            os.makedirs(DATA_DIR, exist_ok=True)
            full_df.to_parquet(OUTPUT_FILE, index=False)
            print(f"Saved combined data to {OUTPUT_FILE}")
        else:
            print("No data collected.")

    except Exception as e:
        print(f"Global error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_historical_archive()
