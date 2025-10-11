# 🐍 PythonAnywhere Deployment Guide (100% FREE)

## ✅ **Why PythonAnywhere is Perfect for You**

- **🆓 Truly FREE** - No credit card ever required
- **🐍 Built for Python** - Flask apps work perfectly  
- **🌐 Always-on hosting** - Your site stays live 24/7
- **👥 Beginner-friendly** - Web interface, no command line needed
- **💾 Persistent storage** - Your SQLite database persists
- **📊 500MB storage** - More than enough for your news aggregator

---

## 🚀 **Step-by-Step Deployment (15 minutes)**

### **Step 1: Create Free Account (3 minutes)**
1. Go to [**pythonanywhere.com**](https://www.pythonanywhere.com)
2. Click **"Pricing & signup"**
3. Click **"Create a Beginner account"** 
4. Fill in details (username will be your website URL)
5. **No credit card required!** ✅

---

### **Step 2: Upload Your Code (5 minutes)**

#### **Option A: Via Bash Console (Recommended)**
1. In your **PythonAnywhere Dashboard**, click **"Tasks" → "Bash"**
2. **Clone your repository**:
   ```bash
   git clone https://github.com/shadowscythe03/News_scrapper.git
   cd News_scrapper
   ls  # Verify files are there
   ```

#### **Option B: Upload Files**
1. Go to **"Files"** tab in dashboard
2. **Upload** your entire project folder
3. **Extract** if needed

---

### **Step 3: Install Dependencies (5 minutes)**
In the **Bash console**:
```bash
cd News_scrapper

# Install all required packages
pip3.10 install --user -r requirements.txt

# Create necessary directories
mkdir -p data/raw data/processed models evaluation logs

# Verify installation
python3.10 -c "import flask, pandas, sklearn; print('✅ Dependencies installed!')"
```

---

### **Step 4: Set Up Web Application (2 minutes)**
1. Go to **"Web"** tab in your dashboard
2. Click **"Add a new web app"**
3. **Choose settings**:
   - **Domain**: `yourusername.pythonanywhere.com` (automatic)
   - **Python framework**: **"Manual configuration"**
   - **Python version**: **"Python 3.10"**

---

### **Step 5: Configure WSGI File (3 minutes)**
1. **After creating the web app**, you'll see a **WSGI configuration file path**
2. **Click on the WSGI file link** to edit it
3. **Replace ALL content** with this:

```python
import sys
import os

# Add your project directory to Python path
project_home = '/home/YOURUSERNAME/News_scrapper'  # Replace YOURUSERNAME
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set environment variables
os.environ['FLASK_ENV'] = 'production'
os.environ['WEB_APP_HOST'] = '0.0.0.0'
os.environ['WEB_APP_PORT'] = '5000'
os.environ['DATABASE_URL'] = 'sqlite:///data/news_database.db'

# Import your Flask application
from src.web_app.app import create_production_app
application = create_production_app()
```

**⚠️ Important**: Replace `YOURUSERNAME` with your actual PythonAnywhere username!

---

### **Step 6: Initialize Data & Launch (2 minutes)**
1. **In Bash console**, run initial setup:
   ```bash
   cd News_scrapper
   python3.10 main.py pipeline  # This might take 5-10 minutes
   ```

2. **Go back to Web tab** and click **"Reload"** button

3. **🎉 Your website is now LIVE!** 
   - **URL**: `https://yourusername.pythonanywhere.com`
   - **Click the link** to visit your news aggregator!

---

## 🔧 **Troubleshooting**

### **Common Issues & Solutions:**

**❌ "Import Error" or "Module not found"**
```bash
# Re-install dependencies
cd News_scrapper
pip3.10 install --user -r requirements.txt
```

**❌ "No articles found" on website**
```bash
# Run the pipeline to get articles
python3.10 main.py scrape
python3.10 main.py classify
```

**❌ "Application error" on website**
- Check the **"Error log"** in Web tab
- Verify WSGI file path is correct
- Make sure you replaced `YOURUSERNAME`

**❌ "Database locked" error**
```bash
# Fix permissions
chmod 664 data/news_database.db
```

---

## 🎯 **After Deployment - What You'll Have**

Your live website at `https://yourusername.pythonanywhere.com` will feature:

- ✅ **📰 Latest News Articles** - Categorized and classified
- ✅ **🤖 AI Chatbot** - Ask questions about current news
- ✅ **📊 Category Filtering** - Politics, Tech, Sports, Business
- ✅ **📱 Mobile Responsive** - Works on all devices  
- ✅ **🔍 Search Functionality** - Find specific articles
- ✅ **📈 Real-time Updates** - Via GitHub Actions automation

---

## 🔄 **Automatic Updates**

Your **GitHub Actions** will continue running weekly and:
- ✅ **Scrape new articles** every Sunday
- ✅ **Retrain ML models** with fresh data
- ✅ **Update your live website** automatically

**No maintenance required!** Your news aggregator runs itself.

---

## 📊 **Usage Limits (Free Tier)**

- **💾 Storage**: 512MB (plenty for your app)
- **⏱️ CPU seconds**: 100/day (enough for news site)
- **🌐 Bandwidth**: No limits on free tier
- **⏰ Always-on**: Yes! Your site never goes down

---

## 🎉 **Success Checklist**

- [ ] ✅ PythonAnywhere account created (free)
- [ ] ✅ Code uploaded to `/home/username/News_scrapper`
- [ ] ✅ Dependencies installed with `pip3.10 install --user -r requirements.txt`
- [ ] ✅ Web app created and configured
- [ ] ✅ WSGI file updated with correct paths
- [ ] ✅ Initial data pipeline run completed
- [ ] ✅ Website reloaded and live
- [ ] ✅ Visited `https://yourusername.pythonanywhere.com` and confirmed it works

---

## 💡 **Pro Tips**

1. **Custom Domain**: You can add your own domain later for $5/month
2. **More Storage**: Upgrade to $5/month for 1GB if needed
3. **Scheduled Tasks**: Use PythonAnywhere's task scheduler for additional automation
4. **Monitoring**: Check error logs in Web tab if something breaks

---

## 🆘 **Need Help?**

- **PythonAnywhere Help**: [help.pythonanywhere.com](https://help.pythonanywhere.com)
- **Community Forum**: Very responsive community support
- **Documentation**: Excellent beginner guides available

**Your news aggregator will be live and professional-quality - completely free! 🚀**