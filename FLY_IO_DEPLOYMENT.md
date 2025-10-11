# 🚀 Fly.io Deployment Guide (FREE)

## ✅ Why Fly.io is Great for Your Project

- **FREE tier** - $0/month for small apps
- **Fast global deployment** - Edge locations worldwide  
- **Persistent storage** - Your SQLite database persists
- **Auto-scaling** - Handles traffic spikes
- **Easy deployment** - One command setup

## 🆓 Free Tier Limits (More Than Enough!)
- **3 shared-cpu VMs** (you need 1)
- **3GB persistent storage** (your app needs ~100MB)
- **160GB bandwidth/month** (plenty for news site)
- **No time limits** (unlike other platforms)

---

## 🛠 **Option 1: Deploy via Web Dashboard (No Installation)**

### Step 1: Setup Account
1. Go to [fly.io](https://fly.io)
2. **Sign up with GitHub** (easiest)
3. **No credit card required** for free tier

### Step 2: Deploy from Dashboard
1. **Connect GitHub**: Link your `News_scrapper` repository
2. **Configure app**:
   - **App name**: `news-aggregator-yourname`
   - **Region**: Choose closest to you
   - **Build**: Select "Python"
3. **Environment variables** (add these in dashboard):
   ```
   FLASK_ENV=production
   WEB_APP_HOST=0.0.0.0
   WEB_APP_PORT=8080
   DATABASE_URL=sqlite:///data/news_database.db
   ```
4. **Deploy**: Click "Deploy" button
5. **Your site**: `https://news-aggregator-yourname.fly.dev`

---

## 🛠 **Option 2: Deploy via CLI (More Control)**

### Step 1: Install Fly CLI
**Windows (PowerShell):**
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

**Or download installer**: [fly.io/docs/hands-on/install-flyctl/](https://fly.io/docs/hands-on/install-flyctl/)

### Step 2: Deploy Your App
```bash
# 1. Login to Fly.io
fly auth login

# 2. In your project directory
cd News_scrapper

# 3. Launch your app (uses fly.toml config we created)
fly launch

# 4. Deploy
fly deploy

# 5. Your app is live!
fly open
```

**That's it!** Your news aggregator will be live at `https://your-app-name.fly.dev`

---

## 📊 **Fly.io vs Other Options**

| Platform | Free Tier | Setup Time | Reliability | Best For |
|----------|-----------|------------|-------------|----------|
| **Fly.io** | Permanent | 5-10 min | ⭐⭐⭐⭐⭐ | Advanced users |
| **PythonAnywhere** | Permanent | 10 min | ⭐⭐⭐⭐⭐ | Beginners |
| **Replit** | Limited | 2 min | ⭐⭐⭐ | Quick testing |

---

## 🎯 **Recommendation**

**For your first deployment**: **Fly.io is excellent** if you:
- Want professional-grade hosting
- Don't mind installing one tool (flyctl)
- Want global edge deployment
- Like having more control

**Choose PythonAnywhere** if you:
- Want zero installation
- Prefer web-based setup only
- Are completely new to deployment

---

## 🔧 **After Deployment**

Your live website will have:
- ✅ **Global CDN** - Fast loading worldwide
- ✅ **Automatic HTTPS** - Secure connections
- ✅ **Custom domain support** - Add your own domain later
- ✅ **Auto-scaling** - Handles traffic increases
- ✅ **Persistent storage** - Your database survives restarts

---

## 🆘 **Troubleshooting**

### Common Issues:

**❌ "App won't start"**
- Check your `fly.toml` configuration
- Verify environment variables are set

**❌ "Build failed"**  
- Ensure `requirements.txt` is complete
- Check Python version compatibility

**❌ "Database not found"**
- Run `python main.py pipeline` to populate data
- Check persistent volume is mounted

### Getting Help:
- Fly.io has excellent documentation
- Community forum for support
- GitHub Issues for technical problems

---

## 💰 **Cost Breakdown**

- **Monthly cost**: **$0** (free tier)
- **Setup time**: 10 minutes
- **Maintenance**: Zero (auto-updates via GitHub Actions)
- **Performance**: Production-grade
- **Scalability**: Can upgrade if needed

**Perfect for your news aggregator!** 🎉