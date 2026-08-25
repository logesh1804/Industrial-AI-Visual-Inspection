import socket
import urllib.request

hostname = socket.gethostname()
local_ips = socket.gethostbyname_ex(hostname)[2]
print(f"Local IP addresses: {local_ips}")

for ip in local_ips:
    if ip.startswith("127."):
        continue
    prefix = ".".join(ip.split(".")[:3])
    print(f"Scanning subnet: {prefix}.1 to {prefix}.254 on port 8080 & 4747...")
    # Fast scan of common IP camera IPs on this subnet
    for last in range(1, 100):
        target_ip = f"{prefix}.{last}"
        for port in [8080, 4747]:
            url = f"http://{target_ip}:{port}/video"
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=0.15) as stream:
                    print(f"[FOUND ACTIVE IP CAMERA] {url}")
            except Exception:
                pass
print("Subnet scan finished.")
