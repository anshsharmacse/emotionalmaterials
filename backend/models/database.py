"""
PolyMind AI — SQLAlchemy Database Models
Author: Ansh Sharma | B230825MT
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./polymind.db"
# For PostgreSQL: DATABASE_URL = "postgresql://user:pass@localhost/polymind"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(100), unique=True, nullable=False)
    password   = Column(String(255), nullable=False)   # hash in production
    role       = Column(String(20), default="user")    # "user" | "admin"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class SimulationLog(Base):
    __tablename__ = "simulation_logs"
    id           = Column(Integer, primary_key=True, index=True)
    user         = Column(String(100), nullable=False)
    polymer      = Column(String(100), default="PEDOT:PSS")
    strain       = Column(Float, default=0.0)
    temperature  = Column(Float, default=300.0)
    conductivity = Column(Float, default=0.0)
    band_gap     = Column(Float, default=0.0)
    mental_state = Column(String(50), default="unknown")
    confidence   = Column(Float, default=0.0)
    lammps_log   = Column(Text, nullable=True)
    qe_log       = Column(Text, nullable=True)
    timestamp    = Column(DateTime, default=datetime.datetime.utcnow)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id         = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True)
    user       = Column(String(100), nullable=True)
    messages   = Column(Text, default="[]")   # JSON list
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# Seed admin on first run
def seed_admin():
    db = SessionLocal()
    if not db.query(User).filter(User.username == "admin").first():
        admin = User(username="admin", password="polymind2024",
                     role="admin", created_at=datetime.datetime.utcnow())
        db.add(admin); db.commit()
    db.close()

Base.metadata.create_all(bind=engine)
seed_admin()
