import paramiko
import os

# Read local file
with open(r'c:\Users\lpnhu\Downloads\uq_capstone\src\model_utils.py', 'r') as f:
    content = f.read()

# Connect and upload
hostname = 'amarel.rutgers.edu'
username = 'hpl14'
password = input("Enter password: ")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, username=username, password=password)

sftp = ssh.open_sftp()
remote_path = '/scratch/hpl14/uq_capstone/src/model_utils.py'

# Write file
with sftp.file(remote_path, 'w') as remote_file:
    remote_file.write(content)

print(f"✓ Uploaded model_utils.py to {remote_path}")
sftp.close()
ssh.close()
