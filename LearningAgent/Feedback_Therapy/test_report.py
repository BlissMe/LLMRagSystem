# test_report.py
import requests, json

def main():
    url = "http://127.0.0.1:8000/report"
    payload = {"user_id": 41}
    try:
        r = requests.post(url, json=payload, timeout=30)
        print("Status:", r.status_code)
        try:
            data = r.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except ValueError:
            print("Response text:", r.text)
    except Exception as e:
        print("Request failed:", e)

if __name__ == "__main__":
    main()
