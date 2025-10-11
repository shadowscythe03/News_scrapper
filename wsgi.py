# PythonAnywhere WSGI Configuration
import sys
import os

# Add your project directory to Python path
project_home = '/home/yourusername/News_scrapper'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set environment variables
os.environ['FLASK_ENV'] = 'production'
os.environ['WEB_APP_HOST'] = '0.0.0.0'
os.environ['WEB_APP_PORT'] = '5000'

# Import your application
from src.web_app.app import create_production_app
application = create_production_app()

# Initialize components
if __name__ == "__main__":
    application.run()