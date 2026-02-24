import requests
import sys

url = "https://multilobate-miss-overzealous.ngrok-free.dev/generate"

if len(sys.argv) < 2:
    print("Usage: python client.py \"your prompt here\"")
    sys.exit(1)

prompt = sys.argv[1]

response = requests.post(
    url,
    json={"prompt": prompt}
)

if response.status_code == 200:
    with open("output.png", "wb") as f:
        f.write(response.content)
    print("Image saved as output.png")
    import subprocess
    subprocess.run(["start", "output.png"], shell=True)
else:
    print("Error:", response.status_code, response.text)
