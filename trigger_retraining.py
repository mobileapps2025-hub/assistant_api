import requests
import sys

def trigger_retraining():
    url = "https://mclassistant-faaagzbyf7atebc9.westeurope-01.azurewebsites.net/admin/train"
    # url = "http://localhost:8000/admin/train"
    print(f"Triggering retraining at: {url}")
    
    try:
        # The endpoint expects a POST request.
        # Based on the router definition, it doesn't strictly require a body, 
        # but sometimes empty JSON {} is safer for some frameworks if they expect a body.
        # In this case, the router function takes no arguments, so a simple POST should work.
        response = requests.post(url)
        
        if response.status_code == 200:
            print("Retraining triggered successfully!")
            print("Response:", response.json())
        else:
            print(f"Failed to trigger retraining. Status code: {response.status_code}")
            try:
                print("Error details:", response.text)
            except:
                pass
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    trigger_retraining()
