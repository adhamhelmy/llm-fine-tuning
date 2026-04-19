# University Data Science Server — Setup Guide

## Server Specs

| Component | Details |
|-----------|---------|
| **Hostname** | R940-N1 |
| **User** | usern1 |
| **GPU** | NVIDIA Tesla V100 PCIe 32GB |
| **VRAM** | 32 GB HBM2 |
| **CUDA Cores** | 5,120 |
| **Tensor Cores** | 640 (Volta generation) |
| **Compute Capability** | 7.0 |
| **Memory Bandwidth** | 900 GB/s |
| **FP32 Performance** | ~14 TFLOPS |
| **FP16 Performance** | ~28 TFLOPS |
| **Server IP (internal)** | 10.102.0.71 |

### Unsloth Fine-Tuning Capacity

| Method | Max Model Size | Notes |
|--------|---------------|-------|
| QLoRA (4-bit) | ~32B | ~26 GB, fits comfortably |
| QLoRA (4-bit) | 40B | ~30 GB, tight |
| LoRA (16-bit) | 14B | ~33 GB, fits |
| LoRA (16-bit) | 11B | ~29 GB, fits with headroom |

Recommended: **Qwen2.5-32B or Llama-3.3-32B with QLoRA**. Use `use_gradient_checkpointing="unsloth"` and batch size 1–2.

---

## Connecting to the Server Remotely 

The server is on the university's internal wired network and not publicly accessible. So, a reverse tunnel via ngrok is used to bridge the two.

### Architecture
```
PC ←→ ngrok cloud ←→ Server (university network)
```

### Step 1 — Generate SSH key on the server (one time only)
```bash
ssh-keygen -t ed25519 -C "tunnel" -N "" -f ~/.ssh/id_ed25519
```

### Step 2 — Install ngrok on the server (one time only)
```bash
curl -Lo ngrok.tgz https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xf ngrok.tgz
./ngrok --version
```

### Step 3 — Create a free ngrok account (one time only)
1. Go to [ngrok.com](https://ngrok.com) and sign up
2. Get your authtoken from [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
3. Add it to ngrok on the server:
```bash
./ngrok config add-authtoken YOUR_TOKEN_HERE
```

### Step 4 — Start the tunnel (every session)
Run this on the server each time you want to connect remotely:
```bash
./ngrok tcp 22
```

ngrok will print something like:
```
Forwarding  tcp://7.tcp.eu.ngrok.io:18796 -> localhost:22
```

Note the hostname and port — they change each session unless you have a paid ngrok plan.

### Step 5 — Configure SSH on your Mac (one time only)
Edit `~/.ssh/config` on your Mac:
```
Host uni-server
    HostName 7.tcp.eu.ngrok.io
    User usern1
    Port 18796
```

Update `HostName` and `Port` each session with the new values from ngrok output.

### Step 6 — Connect from Mac terminal
```bash
ssh uni-server
```

---

## VS Code Remote SSH Setup (Mac)

### Install the extension (one time only)
In VS Code: `Cmd+Shift+X` → search **Remote - SSH** (by Microsoft) → Install

### Connect
1. `Cmd+Shift+P` → **Remote-SSH: Connect to Host**
2. Select **uni-server**
3. Wait for VS Code server to install on the remote (first time only, ~1 min)
4. **File → Open Folder** → navigate to your project

### Install remote extensions (one time only)
Once connected, install these **on the remote** via the Extensions panel:
- **Python** (by Microsoft)
- **Jupyter** (by Microsoft)

These must be installed on the remote separately from your local Mac extensions.

---

## Project Setup on the Server

Since the project is on GitHub, clone it directly onto the server:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

Push changes back as normal:
```bash
git add .
git commit -m "your message"
git push
```

---

## Every Session Checklist

1. SSH into the server from the lab (or any network-connected machine)
2. Run `./ngrok tcp 22` and note the new hostname/port
3. Update `~/.ssh/config` on your Mac with the new hostname/port
4. Connect via `ssh uni-server` or VS Code Remote-SSH
5. Keep the ngrok terminal session alive — closing it drops the tunnel