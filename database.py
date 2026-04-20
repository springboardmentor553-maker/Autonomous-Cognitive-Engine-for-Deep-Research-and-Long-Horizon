"""
Database models and setup for user authentication + conversation history
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
    plain_password = db.Column(db.String(200), nullable=True)
    user_type = db.Column(db.String(20), default='normal')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        self.plain_password = password
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self, include_password=False):
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'type': self.user_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
        if include_password:
            data['password'] = self.plain_password
        return data
    
    def __repr__(self):
        return f'<User {self.username}>'


class Conversation(db.Model):
    """Stores chat conversations per user"""
    __tablename__ = 'conversations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), default='New Chat')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('conversations', lazy=True, order_by='Conversation.updated_at.desc()'))
    messages = db.relationship('Message', backref='conversation', lazy=True, order_by='Message.created_at', cascade='all, delete-orphan')
    
    def to_dict(self, include_messages=False):
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_messages:
            data['messages'] = [m.to_dict() for m in self.messages]
        return data
    
    def __repr__(self):
        return f'<Conversation {self.id}: {self.title}>'


class Message(db.Model):
    """Stores individual messages within a conversation"""
    __tablename__ = 'messages'
    
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    search_sources = db.Column(db.Text, nullable=True)  # JSON string of source URLs
    confidence_score = db.Column(db.Float, nullable=True)  # Hallucination guard confidence
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        import json
        sources = None
        if self.search_sources:
            try:
                sources = json.loads(self.search_sources)
            except (json.JSONDecodeError, TypeError):
                sources = None
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'role': self.role,
            'content': self.content,
            'search_sources': sources,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
    
    def __repr__(self):
        return f'<Message {self.id} ({self.role})>'


class ResearchTask(db.Model):
    """Model to store research task history"""
    __tablename__ = 'research_tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(50), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=True)
    task_description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')
    completed_steps = db.Column(db.Integer, default=0)
    files_created = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    user = db.relationship('User', backref=db.backref('tasks', lazy=True))
    conversation = db.relationship('Conversation', backref=db.backref('tasks', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'task_id': self.task_id,
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
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


def _safe_alter_table(app):
    """Safely add new columns to existing tables (for DB migration)."""
    from sqlalchemy import text, inspect
    with app.app_context():
        inspector = inspect(db.engine)
        existing_columns = [col['name'] for col in inspector.get_columns('messages')]
        
        migrations = [
            ('search_sources', 'ALTER TABLE messages ADD COLUMN search_sources TEXT'),
            ('confidence_score', 'ALTER TABLE messages ADD COLUMN confidence_score FLOAT'),
        ]
        
        for col_name, sql in migrations:
            if col_name not in existing_columns:
                try:
                    db.session.execute(text(sql))
                    db.session.commit()
                    print(f"[OK] Added column '{col_name}' to messages table")
                except Exception as e:
                    db.session.rollback()
                    print(f"[WARN] Could not add column '{col_name}': {e}")


def init_db(app):
    """Initialize database"""
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        
        # Run safe migrations for existing DBs
        _safe_alter_table(app)
        
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
            print("[OK] Default admin account created")
        else:
            print("[OK] Admin account already exists")
        
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
            print("[OK] Demo users created")