
import requests
import pandas as pd
import io
import zipfile

HIST_JSON_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=13060"
DOWNLOAD_URL_TEMPLATE = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={}"

def debug_2025():
    print("Fetching document list...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    resp = requests.get(HIST_JSON_URL, headers=headers)
    data = resp.json()
    docs = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
    
    doc_2025 = None
    for entry in docs:
        doc = entry.get("Document", {})
        friendly_name = doc.get("FriendlyName", "")
        if "2025" in friendly_name and "spp" in friendly_name.lower():
            doc_2025 = doc
            print(f"Found 2025 doc: {friendly_name} (ID: {doc.get('DocID')})")
            break
            
    if not doc_2025:
        print("2025 doc not found.")
        return

    url = DOWNLOAD_URL_TEMPLATE.format(doc_2025.get("DocID"))
    print(f"Downloading from {url}...")
    r = requests.get(url, headers=headers)
    
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        print("\nFiles in ZIP:")
        for f in z.namelist():
            print(f" - {f}")
            if f.endswith(".xlsx"):
                with z.open(f) as xls_file:
                    df = pd.read_excel(xls_file) # Read first sheet by default
                    print("\nFirst Sheet Columns:")
                    print(df.columns.tolist())
                    print("\nFirst 5 rows:")
                    print(df.head())
                    print("\nFirst 5 Hour Ending:")
                    print(df['Hour Ending'].head())
                    print("\nUnique Settlement Points (first 10):")
                    print(df['Settlement Point'].unique()[:10])
                    
                    target = "HB_NORTH"
                    if target in df['Settlement Point'].values:
                        print(f"\n{target} FOUND in values.")
                    else:
                        print(f"\n{target} NOT FOUND. Did you mean one of the above?")

if __name__ == "__main__":
    debug_2025()
