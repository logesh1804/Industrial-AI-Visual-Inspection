import socket
import subprocess

hostname = socket.gethostname()
print("Hostname:", hostname)
ips = socket.gethostbyname_ex(hostname)[2]
print("IPs:", ips)
