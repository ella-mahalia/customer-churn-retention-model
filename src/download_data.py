
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = DATA_DIR / "Telco-Customer-Churn.csv"

print("Downloading IBM Telco Customer Churn dataset...")
urllib.request.urlretrieve(URL, OUT)

sha256 = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(f"Saved to: {OUT}")
print(f"Size: {OUT.stat().st_size:,} bytes")
print(f"SHA-256: {sha256}")
