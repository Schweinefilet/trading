import os
from flask import Flask
from flask_cors import CORS
from models import db
from routes.market import market_bp
from routes.portfolio import portfolio_bp
from routes.analytics import analytics_bp

app = Flask(__name__)
CORS(app)

# Database Configuration
# sqlite file lives at backend/data/stock_cache.db
basedir = os.path.abspath(os.path.dirname(__file__))
db_dir = os.path.join(basedir, 'data')
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

db_path = os.path.join(db_dir, 'stock_cache.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Register Blueprints
app.register_blueprint(market_bp, url_prefix='/api')
app.register_blueprint(portfolio_bp, url_prefix='/api')
app.register_blueprint(analytics_bp, url_prefix='/api')

@app.route('/api/health')
def health_check():
    return {'status': 'healthy'}

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    # Default Flask port is 5000, but macOS uses it for AirPlay. Using 5001.
    app.run(host='0.0.0.0', port=5001, debug=True)
