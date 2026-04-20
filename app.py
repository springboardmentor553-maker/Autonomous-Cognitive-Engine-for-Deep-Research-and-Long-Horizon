import os
import uuid
import json
import threading
import time
import re
import traceback
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(override=True)

# LangChain/OpenAI
from langchain_openai import ChatOpenAI

# Database Imports
from database import db, ResearchTask, User, Conversation, Message, init_db
from brains.filetools import FILE_SYSTEM_DIR, clear_virtual_fs

# New modules
from web_search import search_web, format_sources_for_llm, format_sources_for_display, get_source_urls
from query_matcher import find_matching_conversation
from garbage_collector import GarbageCollector
import hallucination_guard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepresearch.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
init_db(app)

tasks = {}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "meta/llama-3.1-8b-instruct")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize Garbage Collector
gc = GarbageCollector(tasks_dict=tasks, fs_dir=FILE_SYSTEM_DIR)

class TaskStatus:
    def __init__(self, task_id, description):
        self.task_id = task_id
        self.description = description
        self.status = 'running'
        self.completed_steps = 0
        self.current_step = 1
        self.agent_status = {'supervisor': 'idle', 'researcher': 'idle', 'writer': 'idle', 'reviewer': 'idle'}
        self.logs = ["System initialized."]
        self.files = []
        self.final_result = None
        self.search_results = []
        self.confidence_score = None
        self.completed_at = None  # Timestamp for GC

def _get_relevant_slug(text: str) -> str:
    stopwords = {"the","a","an","of","in","on","at","to","for","and","or","is","are","was","impact","research","analyze"}
    words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
    candidates = [w for w in words if w not in stopwords and len(w) > 3]
    return max(candidates, key=len) if candidates else "research_task"

def internal_write(filename, content):
    """Helper to bypass the missing fs_write import"""
    path = Path(FILE_SYSTEM_DIR) / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

def _generate_title(text: str) -> str:
    """Generate a short conversation title from the first user message"""
    words = text.strip().split()
    if len(words) <= 6:
        return text.strip()
    return ' '.join(words[:6]) + '...'

def run_workflow(task_id, description, conversation_id=None):
    with app.app_context():
        t = tasks[task_id]
        db_task = ResearchTask.query.filter_by(task_id=task_id).first()
        try:
            clear_virtual_fs()
            slug = _get_relevant_slug(description)
            llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.7, base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
            
            fnames = {
                "bg": f"{slug}_background.txt", "find": f"{slug}_findings.txt",
                "outlook": f"{slug}_future_outlook.txt", "report": f"{slug}_final_report.txt",
                "audit": f"{slug}_quality_audit.txt"
            }

            # 1. SUPERVISOR
            t.agent_status['supervisor'] = 'working'
            t.logs.append("🛡️ Supervisor: Planning 5-step architecture.")
            time.sleep(1)

            # 2. RESEARCHER (with Web Search)
            t.agent_status['supervisor'] = 'idle'; t.agent_status['researcher'] = 'working'
            t.current_step = 2
            t.logs.append(f"🔍 Researcher: Searching the web for {slug}...")

            # --- Tavily Web Search ---
            web_results = search_web(description, max_results=5)
            t.search_results = web_results
            source_context = format_sources_for_llm(web_results)
            source_display = format_sources_for_display(web_results)

            if web_results:
                t.logs.append(f"🌐 Found {len(web_results)} web sources.")
                research_prompt = (
                    f"Based on these real web sources:\n{source_context}\n\n"
                    f"Write a comprehensive, well-structured research report about: {description}\n"
                    f"Ground your findings in the sources above. Cite specific data points."
                )
            else:
                t.logs.append("⚠️ No web sources found — using knowledge base only.")
                research_prompt = f"Provide background, findings, and trends for: {description}"

            res = llm.invoke(research_prompt)
            
            internal_write(fnames['bg'], f"Background Intel:\n{res.content[:300]}")
            internal_write(fnames['find'], f"Detailed Findings:\n{res.content}")
            internal_write(fnames['outlook'], "Future Projections: High growth expected.")
            t.completed_steps = 3

            # 3. WRITER
            t.agent_status['researcher'] = 'idle'; t.agent_status['writer'] = 'working'
            t.current_step = 4
            t.logs.append("✍️ Writer: Drafting synthesis report.")
            report = llm.invoke(f"Summarize this into a polished, well-organized report: {res.content}")
            internal_write(fnames['report'], report.content)
            t.completed_steps = 4

            # 4. HALLUCINATION GUARD
            t.agent_status['writer'] = 'idle'
            t.current_step = 5
            t.logs.append("🔬 Verifying claims against web sources...")

            guard_result = hallucination_guard.guard(report.content, web_results, llm)
            final_report = guard_result["report"]
            t.confidence_score = guard_result["confidence_score"]

            if t.confidence_score is not None:
                t.logs.append(f"🔬 Claim verification: {int(t.confidence_score * 100)}% confidence")
            t.completed_steps = 5

            # 5. REVIEWER
            t.agent_status['reviewer'] = 'working'
            t.current_step = 6
            t.logs.append("✅ Reviewer: Finalizing quality verification.")
            audit = llm.invoke(f"Audit this report: {final_report[:2000]}")
            internal_write(fnames['audit'], audit.content)
            t.completed_steps = 6

            # Append sources to the final report
            if source_display:
                final_report += source_display

            # Finalize
            t.status = 'complete'
            t.final_result = final_report
            t.completed_at = time.time()
            t.agent_status = {k:'idle' for k in t.agent_status}
            db_task.status = 'complete'
            
            # Save assistant response to conversation with sources
            if conversation_id:
                sources_json = json.dumps(get_source_urls(web_results)) if web_results else None
                msg = Message(
                    conversation_id=conversation_id,
                    role='assistant',
                    content=final_report,
                    search_sources=sources_json,
                    confidence_score=t.confidence_score
                )
                db.session.add(msg)
            
            db.session.commit()

        except Exception as e:
            print("\n" + "="*50 + "\nWORKFLOW FAILED\n" + "="*50)
            traceback.print_exc() 
            print("="*50)
            t.status = 'error'
            t.final_result = f"Error: {str(e)}"
            t.completed_at = time.time()
            db_task.status = 'error'
            
            # Save error as assistant message
            if conversation_id:
                msg = Message(
                    conversation_id=conversation_id,
                    role='assistant',
                    content=f"⚠️ Research failed: {str(e)}"
                )
                db.session.add(msg)
            
            db.session.commit()

# ==================== STATIC ROUTES ====================

@app.route('/')
def index():
    return send_file('ui/app.html')

@app.route('/dashboard')
def dashboard():
    return send_file('ui/dashboard.html')

@app.route('/admin')
def admin():
    return send_file('admin/admin_dashboard.html')

# ==================== AUTH ROUTES ====================

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.json
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'success': False, 'error': 'Email already exists'})
    
    user = User(username=data['username'], email=data['email'])
    user.set_password(data['password'])
    db.session.add(user)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    if user and user.check_password(data['password']):
        user.last_login = datetime.utcnow()
        db.session.commit()
        return jsonify({
            'success': True,
            'user': user.to_dict()
        })
    return jsonify({'success': False, 'error': 'Invalid credentials'})

# ==================== CONVERSATION ROUTES ====================

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """List all conversations for a user"""
    email = request.args.get('email')
    if not email:
        return jsonify({'error': 'Email required'}), 400
    
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    convos = Conversation.query.filter_by(user_id=user.id).order_by(Conversation.updated_at.desc()).all()
    return jsonify([c.to_dict() for c in convos])

@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation"""
    data = request.json
    email = data.get('email')
    
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    convo = Conversation(
        user_id=user.id,
        title=data.get('title', 'New Chat')
    )
    db.session.add(convo)
    db.session.commit()
    
    return jsonify(convo.to_dict())

@app.route('/api/conversations/<int:convo_id>', methods=['GET'])
def get_conversation(convo_id):
    """Get a conversation with all its messages"""
    convo = Conversation.query.get_or_404(convo_id)
    return jsonify(convo.to_dict(include_messages=True))

@app.route('/api/conversations/<int:convo_id>', methods=['DELETE'])
def delete_conversation(convo_id):
    """Delete a conversation and all its messages"""
    convo = Conversation.query.get_or_404(convo_id)
    db.session.delete(convo)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/conversations/<int:convo_id>/rename', methods=['PUT'])
def rename_conversation(convo_id):
    """Rename a conversation"""
    data = request.json
    convo = Conversation.query.get_or_404(convo_id)
    convo.title = data.get('title', convo.title)
    db.session.commit()
    return jsonify(convo.to_dict())

@app.route('/api/conversations/search', methods=['GET'])
def search_conversations():
    """Search conversations by title"""
    email = request.args.get('email')
    query = request.args.get('q', '')
    
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify([])
    
    convos = Conversation.query.filter(
        Conversation.user_id == user.id,
        Conversation.title.ilike(f'%{query}%')
    ).order_by(Conversation.updated_at.desc()).all()
    
    return jsonify([c.to_dict() for c in convos])

# ==================== RESEARCH ROUTES ====================

@app.route('/api/research/submit', methods=['POST'])
def submit():
    data = request.json
    user_email = data.get('userEmail')
    conversation_id = data.get('conversationId')
    force = data.get('force', False)
    user = User.query.filter_by(email=user_email).first()
    
    if not user:
        user = User.query.filter_by(email='nivi303.jk@gmail.com').first()
    
    # --- Ambiguity Detection (only for new conversations) ---
    if not conversation_id and not force:
        match = find_matching_conversation(user.id, data['task'])
        if match:
            return jsonify({
                'redirect': True,
                'conversation_id': match['conversation_id'],
                'message': f"You've already researched a similar topic: \"{match['title']}\". Opening that conversation...",
                'similarity': match['similarity']
            })
    
    # Create or use existing conversation
    if not conversation_id:
        convo = Conversation(
            user_id=user.id,
            title=_generate_title(data['task'])
        )
        db.session.add(convo)
        db.session.flush()
        conversation_id = convo.id
    else:
        convo = Conversation.query.get(conversation_id)
        if convo:
            convo.updated_at = datetime.utcnow()
    
    # Save user message
    user_msg = Message(
        conversation_id=conversation_id,
        role='user',
        content=data['task']
    )
    db.session.add(user_msg)
    
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = TaskStatus(task_id, data['task'])
    db_task = ResearchTask(
        task_id=task_id, 
        task_description=data['task'], 
        status='running',
        user_id=user.id,
        conversation_id=conversation_id
    )
    db.session.add(db_task)
    db.session.commit()
    
    threading.Thread(target=run_workflow, args=(task_id, data['task'], conversation_id)).start()
    return jsonify({'task_id': task_id, 'conversation_id': conversation_id})

@app.route('/api/research/status/<task_id>')
def status(task_id):
    t = tasks.get(task_id)
    if not t: return jsonify({'error': 'Not found'}), 404
    live_files = [{'name': f.name} for f in Path(FILE_SYSTEM_DIR).iterdir() if f.is_file()]
    return jsonify({
        'status': t.status, 'completed_steps': t.completed_steps, 'current_step': t.current_step,
        'agent_status': t.agent_status, 'logs': t.logs, 'files': live_files, 'final_result': t.final_result,
        'confidence_score': t.confidence_score,
        'web_sources_count': len(t.search_results) if t.search_results else 0
    })

# ==================== ADMIN ROUTES ====================

@app.route('/api/tasks')
def get_all_tasks():
    tasks_list = ResearchTask.query.order_by(ResearchTask.created_at.desc()).all()
    return jsonify([t.to_dict() for t in tasks_list])

@app.route('/api/files/<filename>')
def get_file(filename):
    path = Path(FILE_SYSTEM_DIR) / filename
    if not path.is_file():
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, as_attachment=False)

@app.route('/api/users')
def list_users():
    users = User.query.all()
    return jsonify([{
        'id': u.id,
        'username': u.username,
        'email': u.email,
        'user_type': u.user_type,
        'created_at': u.created_at.isoformat()
    } for u in users])

@app.route('/api/user/<int:user_id>/tasks')
def user_tasks(user_id):
    tasks = ResearchTask.query.filter_by(user_id=user_id).order_by(ResearchTask.created_at.desc()).all()
    return jsonify([t.to_dict() for t in tasks])

# ==================== GARBAGE COLLECTOR ADMIN ====================

@app.route('/api/admin/gc', methods=['GET'])
def trigger_gc():
    """Manually trigger garbage collection and return stats."""
    result = gc.run_collection()
    stats = gc.get_stats()
    return jsonify({
        'collection_result': result,
        'stats': stats
    })


if __name__ == '__main__':
    # Start the garbage collector daemon
    gc.start(interval=1800)  # Every 30 minutes
    logger.info("[STARTUP] Garbage Collector daemon started.")
    app.run(debug=True, port=5000, use_reloader=False)