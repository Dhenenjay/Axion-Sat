# 💾 Disk Space Analysis & Recommendations

**Current Status:** 74.81 GB free  
**Target Needed:** 110 GB for BigEarthNet  
**Gap:** Need to free **35-40 GB more**

---

## 🎯 Quick Wins (Safe & Easy to Delete)

### 1. **PIP Cache** - 12.2 GB ⭐ HIGH PRIORITY
**Location:** `C:\Users\Dhenenjay\AppData\Local\pip\cache`  
**Safe to delete?** ✅ YES - Will re-download packages when needed  
**Command:**
```powershell
Remove-Item -Path "$env:LOCALAPPDATA\pip\cache" -Recurse -Force
```

### 2. **Temp Files** - 6.8 GB ⭐ HIGH PRIORITY
**Location:** `C:\Users\Dhenenjay\AppData\Local\Temp`  
**Safe to delete?** ✅ YES - Temporary files  
**Command:**
```powershell
Remove-Item -Path "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
```

### 3. **Windows Temp** - 0.31 GB
**Location:** `C:\Windows\Temp`  
**Safe to delete?** ✅ YES (with admin rights)  
**Command:**
```powershell
# Run PowerShell as Administrator
Remove-Item -Path "C:\Windows\Temp\*" -Recurse -Force -ErrorAction SilentlyContinue
```

**TOTAL FROM QUICK WINS: ~19.3 GB** ✅

---

## 🔧 Node Modules (Can Regenerate with npm install)

### Large node_modules Directories - ~9.5 GB total

| Project | Size | Safe to Delete? |
|---------|------|-----------------|
| axionorbital | 1.35 GB + 0.30 GB | ⚠️ Only if not actively using |
| suna/frontend | 1.11 GB | ⚠️ Only if not actively using |
| vreo-us | 0.85 GB + 0.76 GB | ⚠️ Only if not actively using |
| Various MCP projects | ~4 GB | ⚠️ Only if not actively using |

**How to delete and regenerate:**
```powershell
# Delete (example for one project)
Remove-Item -Path "C:\Users\Dhenenjay\axionorbital\node_modules" -Recurse -Force

# Regenerate when needed
cd C:\Users\Dhenenjay\axionorbital
npm install
```

**POTENTIAL: 5-10 GB** (if you delete unused project node_modules)

---

## 🐍 Python Virtual Environments (Can Recreate)

### Large Python venvs - ~9 GB total

| Environment | Size | Safe to Delete? |
|-------------|------|-----------------|
| Axion-Sat\.venv | 7.16 GB | ⚠️ CURRENT PROJECT - Keep or recreate |
| PycharmProjects venvs | 1.16 GB | ✅ If not using those projects |
| suna/backend/.venv | 0.56 GB | ⚠️ Only if not actively using |
| phoenix-ide/server/.venv | 0.12 GB | ⚠️ Only if not actively using |

**Option 1: Delete unused project venvs**
```powershell
# Example: Delete old PyCharm project venvs
Remove-Item -Path "C:\Users\Dhenenjay\PycharmProjects\UAV_OP\venv" -Recurse -Force
Remove-Item -Path "C:\Users\Dhenenjay\PycharmProjects\Inertialogic\venv" -Recurse -Force
```

**Option 2: Recreate current Axion-Sat venv (if needed)**
```powershell
# Delete current venv
Remove-Item -Path ".venv" -Recurse -Force

# Recreate after BigEarthNet download
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**POTENTIAL: 1-7 GB** (depending on which you delete)

---

## 📂 Old/Duplicate Projects

### Similar Projects (May be duplicates or old versions)

| Project | Size | Notes |
|---------|------|-------|
| Axion-Sat | 13.27 GB | **CURRENT - KEEP** |
| axionorbital | 3.96 GB | Similar name - check if duplicate |
| Axion-MCP | 1.24 GB | Related project? |
| axion-orbital-glow | 0.53 GB | Related project? |
| axion-planetary-mcp | 1.07 GB | Related project? |

**Action:** Review if these are duplicates or old versions you don't need

**POTENTIAL: 5-7 GB** (if duplicates exist)

---

## 📥 Downloads Folder - 3.42 GB

**Location:** `C:\Users\Dhenenjay\Downloads`

**Action:** Manually review and delete:
- Old installers
- Duplicate downloads
- Archived files you've already extracted

**POTENTIAL: 2-3 GB**

---

## 🎮 VSCode Extensions Cache

**Location:** `.vscode\extensions`

Some extensions have node_modules:
- `prisma.prisma-6.17.0` - 0.14 GB
- Other extensions - ~0.38 GB

**Action:** Consider uninstalling unused VSCode extensions

**POTENTIAL: 0.5 GB**

---

## 📊 Recommended Action Plan

### **Phase 1: Safe & Easy (Get ~19 GB immediately)** ⭐
```powershell
# 1. Clear pip cache (12.2 GB)
Remove-Item -Path "$env:LOCALAPPDATA\pip\cache" -Recurse -Force

# 2. Clear temp files (6.8 GB)
Remove-Item -Path "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
```

### **Phase 2: Cleanup Old Projects (Get ~5-10 GB)**
```powershell
# Delete old PyCharm project venvs (1.16 GB)
Remove-Item -Path "C:\Users\Dhenenjay\PycharmProjects\UAV_OP\venv" -Recurse -Force
Remove-Item -Path "C:\Users\Dhenenjay\PycharmProjects\Inertialogic\venv" -Recurse -Force

# Delete unused node_modules in old projects
# Example: If you're not using suna backend
Remove-Item -Path "C:\Users\Dhenenjay\suna\backend\.venv" -Recurse -Force
```

### **Phase 3: Clean Downloads (Get ~2-3 GB)**
```powershell
# Open Downloads folder and manually review
explorer C:\Users\Dhenenjay\Downloads
```

### **Phase 4: If Still Need More Space (Get ~7 GB)**
```powershell
# Option A: Delete and recreate current Axion-Sat venv (7.16 GB)
# You'll need to recreate it after downloading BigEarthNet
cd C:\Users\Dhenenjay\Axion-Sat
Remove-Item -Path ".venv" -Recurse -Force

# Option B: Delete unused node_modules from old projects
# Review and delete node_modules from projects you're not using
```

---

## ✅ Safe Commands to Run Now

**Run these to get ~19 GB immediately:**
```powershell
# 1. Clear pip cache
Write-Host "Clearing pip cache..." -ForegroundColor Yellow
Remove-Item -Path "$env:LOCALAPPDATA\pip\cache" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "✅ Cleared pip cache" -ForegroundColor Green

# 2. Clear temp files
Write-Host "Clearing temp files..." -ForegroundColor Yellow
Remove-Item -Path "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "✅ Cleared temp files" -ForegroundColor Green

# 3. Check new space
Write-Host "`nChecking disk space..." -ForegroundColor Cyan
Get-PSDrive C | Select-Object @{Name="FreeGB";Expression={[math]::Round($_.Free/1GB,2)}}
```

---

## 🚨 What NOT to Delete

- ❌ **Axion-Sat** folder (except .venv if desperate)
- ❌ **Documents, Pictures, Music** (personal data)
- ❌ **AppData\Roaming** (application settings)
- ❌ **Active project node_modules** (unless you can regenerate)
- ❌ **System files** (Windows, Program Files)

---

## 📈 Expected Results

| Action | Space Freed | Cumulative Total |
|--------|-------------|------------------|
| Starting point | - | 74.81 GB free |
| Clear pip cache | +12.2 GB | ~87 GB free |
| Clear temp files | +6.8 GB | ~94 GB free |
| Delete old venvs | +1-2 GB | ~96 GB free |
| Clean Downloads | +2-3 GB | ~99 GB free |
| Delete unused node_modules | +5-10 GB | **~109 GB free** ✅ |

**Alternative if needed:**
- Delete current Axion-Sat .venv (+7 GB) → **~101 GB free**
- Delete more node_modules (+5 GB) → **~106 GB free**
- Review duplicate projects (+5 GB) → **~111 GB free** ✅

---

## 🎯 Next Steps

1. **Run Phase 1 commands** (get 19 GB immediately)
2. **Check disk space** - see if you need more
3. **Review Downloads folder** (get 2-3 GB more)
4. **Delete old project dependencies** if still needed
5. **Download BigEarthNet**
6. **Recreate .venv** if you deleted it

Good luck! 🚀
