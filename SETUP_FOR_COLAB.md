# Setup for Colab - Quick Commands

## What You Need to Do

Just **3 simple steps** before uploading to Colab:

---

## Step 1: Push Your Code to GitHub

Open your terminal here and run:

```bash
git add .
git commit -m "Ready for Colab pipeline"
git push
```

**That's it!** Your code is now on GitHub.

---

## Step 2: Upload Notebook to Colab

1. Go to: https://colab.research.google.com/
2. Click: **File → Upload Notebook**
3. Select: `colab_complete_pipeline.ipynb`
4. Done!

---

## Step 3: Edit GitHub URL in Notebook

In Colab, find Cell 3 (the one with `git clone`) and change:

```python
!git clone https://github.com/YOUR_USERNAME/Axion-Sat.git
```

To your actual GitHub username, for example:

```python
!git clone https://github.com/dhenenjay/Axion-Sat.git
```

---

## Then Just Click "Run All"!

Everything else happens automatically:

✅ **Downloads BigEarthNet** (~2 hours)  
✅ **Converts to tiles** (~3 hours)  
✅ **Runs precompute** (~15 hours)  
✅ **Saves to Google Drive**

---

## Total Time

**~20-25 hours** completely hands-off!

vs **51 hours** on your local PC

---

## Benefits

| What | Colab | Your PC |
|------|-------|---------|
| **Data upload** | None (downloads direct) | Hours/days |
| **Processing time** | ~20-25 hours | ~51 hours |
| **Your PC** | Free to use | Busy |
| **VRAM** | 16GB (T4) | 6GB |
| **Batch size** | 64 | 32 |

---

## After It Finishes

Download results from Google Drive:
- Location: `MyDrive/Axion-Sat-Outputs/stage1_precompute/`
- Save to your D: drive

Then run Stage 2 training (also can be done on Colab if you want!)

---

## Questions?

Let me know if you need help with any step!
