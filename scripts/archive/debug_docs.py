import requests
import json

HIST_JSON_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=13060"

def list_all_docs():
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    resp = requests.get(HIST_JSON_URL, headers=headers)
    data = resp.json()
    docs = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
    
    print(f"Total documents: {len(docs)}")
    for entry in docs:
        doc = entry.get("Document", {})
        print(f"FriendlyName: {doc.get('FriendlyName')} | PublishDate: {doc.get('PublishDate')} | ContentSize: {doc.get('ContentSize')}")

if __name__ == "__main__":
    list_all_docs()
