# GitHub Actions Setup Guide

## 🚀 Setting up CI/CD for News Aggregator

This guide will help you set up automated ML pipeline execution using GitHub Actions.

### What the GitHub Actions Workflow Does

The workflow (`ml-pipeline.yml`) automatically:

- **Scrapes latest news** from RSS feeds
- **Trains ML models** for article classification
- **Evaluates model performance**
- **Runs tests** on the application
- **Uploads artifacts** (models, evaluation results)
- **Commits results** back to the repository

### Triggers

The pipeline runs on:

- **Push** to `main` or `develop` branches
- **Pull requests** to `main` branch
- **Weekly schedule** (Sundays at 2 AM UTC)
- **Manual trigger** (workflow_dispatch)

### Setup Steps

1. **Push your code to GitHub:**

   ```bash
   git init
   git add .
   git commit -m "Initial commit: News Aggregator ML Pipeline"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
2. **Enable GitHub Actions:**

   - Go to your repository on GitHub
   - Click on "Actions" tab
   - GitHub will automatically detect the workflow file
   - Click "Enable GitHub Actions" if prompted
3. **Manual Trigger (Optional):**

   - Go to Actions tab in your repository
   - Select "News Aggregator ML Pipeline"
   - Click "Run workflow" button
   - Choose branch and click "Run workflow"

### What Happens During Execution

#### Job 1: ML Pipeline

- Sets up Python 3.10 environment
- Installs dependencies from `requirements.txt`
- Creates necessary directories
- Runs DVC pipeline (`dvc repro --force`)
- Scrapes news articles
- Trains classification models
- Evaluates model performance
- Uploads results as artifacts

#### Job 2: Test Application

- Downloads ML artifacts from Job 1
- Tests application module imports
- Verifies database functionality
- Ensures all components work together

#### Job 3: Deployment Ready

- Runs only on `main` branch
- Confirms all tests passed
- Marks system as production-ready

### Monitoring Results

1. **Check workflow status:**

   - Go to Actions tab in your repository
   - Click on the latest workflow run
   - View logs for each job
2. **Download artifacts:**

   - In the workflow run page
   - Scroll to "Artifacts" section
   - Download `ml-pipeline-results.zip`
   - Contains: models, evaluation results, processed data
3. **View evaluation metrics:**

   - Check the "Show evaluation results" step
   - Look for accuracy and F1 scores in logs

### Cost Considerations

This setup is **completely free** because:

- Uses GitHub Actions free tier (2000 minutes/month)
- No external paid APIs required
- All processing runs on GitHub's servers
- Runs weekly, using ~10-15 minutes per run

### Customization

To modify the schedule:

```yaml
schedule:
  # Run daily at 6 AM UTC
  - cron: '0 6 * * *'
  
  # Run every Monday and Friday at 3 PM UTC  
  - cron: '0 15 * * 1,5'
```

To change Python version:

```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'  # Change to desired version
```

### Troubleshooting

**Pipeline fails with dependency errors:**

- Check `requirements.txt` has all needed packages
- Verify package versions are compatible

**No new articles scraped:**

- RSS feeds might be temporarily down
- Check scraping logs for specific errors
- This is normal and pipeline will continue

**Model training fails:**

- Usually due to insufficient data
- Pipeline uses synthetic data as fallback
- Check if database has enough articles

**Workflow doesn't trigger:**

- Ensure workflow file is in `.github/workflows/`
- Check YAML syntax is valid
- Verify branch names match triggers

### Next Steps

Once GitHub Actions is working:

1. **Monitor weekly reports** in Actions tab
2. **Set up notifications** for failures
3. **Consider deployment** to cloud platforms
4. **Add more sophisticated testing**

The automated pipeline ensures your news aggregator stays up-to-date with fresh data and retrained models every week! 🎉
