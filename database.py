"""
Database models and setup for user authentication
⚠️ DEMO: Stores plain text passwords for educational purposes only
"""
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    plain_password = db.Column(db.String(200), nullable=True)  # ⚠️ DEMO ONLY
    user_type = db.Column(db.String(20), default='normal')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
        self.plain_password = password  # ⚠️ DEMO ONLY
    
    def check_password(self, password):
        """Check if password is correct"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self, include_password=False):
        """Convert user to dictionary"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'user_type': self.user_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
        
        if include_password:
            data['password'] = self.plain_password
        
        return data
    
    def __repr__(self):
        return f'<User {self.username}>'


class ResearchTask(db.Model):
    """Model to store research task history"""
    __tablename__ = 'research_tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(50), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    task_description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')
    completed_steps = db.Column(db.Integer, default=0)
    files_created = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    user = db.relationship('User', backref=db.backref('tasks', lazy=True))
    
    def to_dict(self):
        """Convert task to dictionary"""
        return {
            'id': self.id,
            'task_id': self.task_id,
            'user_id': self.user_id,
            'user_email': self.user.email if self.user else None,
            'task_description': self.task_description,
            'status': self.status,
            'completed_steps': self.completed_steps,
            'files_created': self.files_created,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
    
    def __repr__(self):
        return f'<ResearchTask {self.task_id}>'


def init_db(app):
    """Initialize database"""
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        
        # Create default admin
        admin = User.query.filter_by(email='nivi303.jk@gmail.com').first()
        if not admin:
            admin = User(
                username='admin',
                email='nivi303.jk@gmail.com',
                user_type='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("✓ Default admin account created")
        else:
            print("✓ Admin account already exists")
        
        # Create demo users
        user_count = User.query.filter_by(user_type='normal').count()
        if user_count == 0:
            demo_users = [
                {'username': 'demo_user', 'email': 'demo@research.ai', 'password': 'demo123'},
                {'username': 'john_doe', 'email': 'john@research.ai', 'password': 'john2024'},
                {'username': 'researcher1', 'email': 'researcher@ai.com', 'password': 'research99'}
            ]
            
            for user_data in demo_users:
                user = User(
                    username=user_data['username'],
                    email=user_data['email'],
                    user_type='normal'
                )
                user.set_password(user_data['password'])
                db.session.add(user)
            
            db.session.commit()
            print("✓ Demo users created")