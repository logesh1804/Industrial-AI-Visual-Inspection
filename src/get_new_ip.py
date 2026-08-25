import socket

hostname = socket.gethostname()
ips = socket.gethostbyname_ex(hostname)[2]
print(f"Current Laptop Local IPs on new Wi-Fi: {ips}")
