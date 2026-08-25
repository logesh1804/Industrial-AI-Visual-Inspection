import socket
import urllib.request

local_ips = socket.gethostbyname_ex(socket.gethostname())[2]
print("My Local IPs:", local_ips)

for ip in local_ips:
    if ip.startswith("127."):
        continue
    prefix = ".".join(ip.split(".")[:3])
    print(f"Scanning {prefix}.1 to {prefix}.254 on port 8080...")
    for last in range(1, 120):
        url = f"http://{prefix}.{last}:8080/video"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=0.08) as s:
                print(f"[FOUND ACTIVE IP CAMERA] {url}")
        except Exception:
            pass
print("Scan done.")
