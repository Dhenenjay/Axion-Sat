# Google Colab Setup Instructions

## Overview
Run Stage 1 precompute on Google Colab (15-18 hours instead of 51 hours!)

---

## Step 1: Upload Data to Google Drive (One-time)

### Option A: Manual Upload (Easiest)
1. Open Google Drive in your browser
2. Create folder: `Axion-Sat-Data/tiles`
3. Upload your `C:\Users\Dhenenjay\Axion-Sat\data\tiles\benv2_catalog` folder there
4. Wait for upload to complete (~200K files)

### Option B: Using Google Drive Desktop App (Recommended)
1. Install [Google Drive for Desktop](https://www.google.com/drive/download/)
2. Sign in and sync your Google Drive
3. Copy `C:\Users\Dhenenjay\Axion-Sat\data\tiles\benv2_catalog` to your Google Drive folder
4. Let it sync automatically

---

## Step 2: Upload Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload Notebook**
3. Upload `colab_stage1_precompute.ipynb` from your project folder
4. ✓ Notebook is now in your Colab

---

## Step 3: Run on Colab

### 3.1 Connect to GPU
- Click **Runtime → Change runtime type**
- Select **T4 GPU** (or V100 if available)
- Click **Save**

### 3.2 Run All Cells
1. Click **Runtime → Run all**
2. When prompted, click **Connect to Google Drive** → Allow access
3. Update the `DATA_DIR` path in Cell 5 to match where you uploaded data
4. Let it run! (~15-18 hours)

### 3.3 Keep Colab Alive (Important!)
Colab disconnects after ~12 hours of inactivity. To prevent this:

**Option 1:** Use browser extension
- Install [Colab Auto Reconnect](https://chrome.google.com/webstore) extension

**Option 2:** Manual approach
- Check on it every few hours
- Click reconnect if needed
- Progress is saved to Google Drive, so you can resume

**Option 3:** Colab Pro ($10/month)
- Longer session times
- Faster GPUs (A100)
- Priority access

---

## Step 4: Download Results

Once complete:
1. Go to Google Drive → `Axion-Sat-Data/stage1_outputs`
2. Download all `.npz` files
3. Save to `D:\axion_stage1_precompute\` on your PC

---

## Alternative: Upload Your Project to GitHub First

If you want to skip manual file copying:

1. **Push your code to GitHub:**
   ```bash
   cd C:\Users\Dhenenjay\Axion-Sat
   git add .
   git commit -m "Ready for Colab"
   git push
   ```

2. **In Colab, clone directly:**
   ```python
   !git clone https://github.com/YOUR_USERNAME/Axion-Sat.git
   ```

This way, Colab always has your latest code!

---

## Troubleshooting

### "Data not found" error
- Check the path in Cell 5 matches where you uploaded in Google Drive
- Make sure folder structure is: `MyDrive/Axion-Sat-Data/tiles/benv2_catalog/`

### Colab disconnects
- Use Auto Reconnect extension
- Or upgrade to Colab Pro

### Out of memory
- Reduce `--batch-size` from 64 to 32
- T4 has 16GB VRAM, should handle batch size 64 easily

### Slow upload to Google Drive
- Use Google Drive Desktop app for initial upload
- Or split into smaller batches

---

## What Happens Next?

After precompute finishes:
1. Download outputs from Google Drive
2. Run Stage 2 training (can also do on Colab if you want!)
3. Your GAC pipeline will be fully trained

---

## Time Comparison

| Method | GPU | Time |
|--------|-----|------|
| Local (your PC) | RTX 4050 | ~51 hours |
| Colab Free | Tesla T4 | ~15-18 hours |
| Colab Pro | A100 | ~5-7 hours |

**Recommendation:** Start with Colab Free. If it's working well, consider upgrading to Pro for the final training runs.

---

## Questions?

Let me know if you hit any issues! The notebook is ready to run.
