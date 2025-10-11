# 🌐 Deploy Your News Aggregator Website Online

## Quick Deployment Options (All Free!)

### Option 1: Render.com (Recommended - Easiest)

**Why Render?**
- ✅ **Free tier available**
- ✅ **Supports Flask apps**  
- ✅ **Automatic deployments from GitHub**
- ✅ **Custom domain support**
- ✅ **HTTPS included**

**Steps:**
1. **Sign up** at [render.com](https://render.com)
2. **Connect GitHub** - authorize Render to access your repository
3. **Create Web Service:**
   - Click "New +" → "Web Service"
   - Select your `news-aggregator` repository
   - Fill in settings:
     - **Name**: `news-aggregator` (or your choice)
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt && mkdir -p data logs`
     - **Start Command**: `python main.py web`
     - **Plan**: `Free` (0$/month)

4. **Environment Variables** (Add these in Render dashboard):
   ```
   FLASK_ENV=production
   FLASK_DEBUG=false
   WEB_APP_HOST=0.0.0.0
   WEB_APP_PORT=10000
   DATABASE_URL=sqlite:///data/news_database.db
   ```

5. **Deploy!** - Click "Create Web Service"
   - Takes 5-10 minutes for first deployment
   - Your site will be live at: `https://your-app-name.onrender.com`

### Option 2: Railway.app (Alternative)

1. **Sign up** at [railway.app](https://railway.app)
2. **Deploy from GitHub**:
   - Click "Deploy from GitHub"
   - Select your repository
   - Railway auto-detects Python and Flask
3. **Environment Variables**:
   ```
   FLASK_ENV=production
   WEB_APP_PORT=8080
   ```
4. **Your site**: `https://your-app-name.up.railway.app`

### Option 3: Heroku Alternative - Fly.io

1. **Install Fly CLI**: [fly.io/docs/hands-on/install-flyctl/](https://fly.io/docs/hands-on/install-flyctl/)
2. **Sign up**: `fly auth signup`
3. **In your project directory**:
   ```bash
   fly launch
   fly deploy
   ```
4. **Your site**: `https://your-app-name.fly.dev`

## 🔧 Configuration Files Included

Your repository now includes:
- **`render.yaml`** - Render.com deployment configuration
- **Docker setup** - For container deployment
- **Environment variables** - Production-ready settings

## 🚀 After Deployment

### Your Live Website Will Have:
- **📰 Latest News Articles** - Refreshed via GitHub Actions
- **🔍 Category Filtering** - Politics, Tech, Sports, etc.
- **🤖 AI Chatbot** - Ask questions about news
- **📊 Real-time Stats** - Article counts, sentiment analysis
- **📱 Mobile Responsive** - Works on all devices

### Automatic Updates:
- **GitHub Actions** runs weekly
- **New articles** scraped automatically  
- **Models retrained** with fresh data
- **Website updates** automatically (if deployed from main branch)

## 💰 Cost Breakdown

### Free Tier Limits:
- **Render.com**: 750 hours/month free (enough for always-on)
- **Railway.app**: $5/month after trial, but very generous free tier
- **Fly.io**: 2340 hours/month free

### Expected Usage:
- **Always-on website**: ~730 hours/month
- **GitHub Actions**: ~60 minutes/month  
- **Total cost**: **$0/month** on free tiers! 🎉

## 🔗 Custom Domain (Optional)

Once deployed, you can add your own domain:

1. **Buy domain** (e.g., from Namecheap, GoDaddy)
2. **In Render dashboard**: Settings → Custom Domains
3. **Add your domain**: `www.yournewssite.com`
4. **Update DNS**: Point CNAME to Render URL
5. **HTTPS**: Automatically enabled by Render

## 📊 Monitoring Your Live Site

### Check if it's working:
- Visit your deployed URL
- Should see news articles and chatbot
- Try the chatbot: "What's the latest in tech?"
- Check different categories

### Logs and debugging:
- **Render**: Dashboard → your service → Logs
- **Railway**: Dashboard → your project → Deployments
- **GitHub Actions**: Actions tab for pipeline logs

## 🛠 Troubleshooting Deployment

### Common Issues:

**❌ "Build failed"**
- Check your `requirements.txt` is complete
- Verify Python version compatibility

**❌ "App won't start"**  
- Ensure start command is: `python main.py web`
- Check environment variables are set

**❌ "Database errors"**
- SQLite works on most platforms
- Data persists via disk storage on Render

**❌ "No articles showing"**
- Run GitHub Actions to populate database
- Or manually trigger: `python main.py scrape`

## 🎯 Next Steps After Deployment

1. **Share your live site!** 🎉
2. **Monitor performance** in hosting dashboard
3. **Add custom domain** if desired
4. **Scale up** if you get traffic (upgrade hosting plan)
5. **Add more features** to your chatbot

---

## 🚀 Quick Start Summary

```bash
# 1. Push latest code to GitHub
git add .
git commit -m "Add deployment configuration"
git push

# 2. Sign up at render.com
# 3. Connect GitHub repository  
# 4. Create Web Service with provided settings
# 5. Your site goes live automatically! 🌐
```

**Estimated deployment time: 15 minutes**  
**Monthly cost: $0 (free tier)**  
**Result: Professional news website with AI chatbot! ✨**