import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os
import paramiko


ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname="bi.nsc.liu.se",,password="castro1997...")
def handler(title, instructions, fields):
    if len(fields) > 1:
        raise SSHException("Expecting one field only.")
    return [password]

transport = paramiko.Transport('bi.nsc.liu.se')
transport.connect(username='sm_valfa')
transport.auth_interactive(username, handler)

stdin,stdout,stderr=transport.exec_command("ls")
print(stdout.readlines())
