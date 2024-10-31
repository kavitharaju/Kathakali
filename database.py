'''Database connection module and table definitions'''
import urllib
import os
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import Column, Integer, String, JSON, Float, text
from sqlalchemy import Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship, Session
from pgvector.sqlalchemy import Vector


postgres_host = os.environ.get("KATHAKALI_APP_DB_HOST", "localhost")
postgres_user = os.environ.get("KATHAKALI_APP_DB_USER", "dbusername")
postgres_database = os.environ.get("KATHAKALI_APP_DB_NAME", "dbname")
postgres_password = os.environ.get("KATHAKALI_APP_DB_PASS", "dbpassword")
postgres_port = os.environ.get("KATHAKALI_APP_DB_PORT", "5432")
postgres_schema = os.environ.get("KATHAKALI_APP_DB_SCHEMA", "public")

SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg2://{postgres_user}:\
{urllib.parse.quote(postgres_password)}@"\
        f"{postgres_host}:{postgres_port}/{postgres_database}"


engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=10, max_overflow=20)
conn = engine.connect()
conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
conn.commit()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    '''To start a DB connection(session)'''
    db_ = SessionLocal()
    try:
        yield db_
    finally:
        # pass
        db_.close()

Base = declarative_base()

#pylint: disable=too-few-public-methods
class Media(Base):
    '''Media table with all images and videos uploaded to server'''
    __tablename__ = 'media'

    mediaId = Column('media_id', Integer, primary_key=True)
    mediaType = Column('media_type', String)
    filename = Column('filename', String)
    filepath = Column('filepath', String)
    tasks = Column('tasks', ARRAY(String))
    label = Column('label', String)
    createdTime = Column('created_at', DateTime,
                        default=datetime.now)
    updatedTime = Column('updated_at', DateTime,
                        default=datetime.now,
                        onupdate=datetime.now)

class KathakaliMudraVectors(Base):
    '''Kathakali Mudra vectors table's mapping'''
    __tablename__ = 'kathakali_mudra_vectors'

    vectorId = Column('vector_id', Integer, primary_key=True)
    mediaId = Column('media_id', Integer, ForeignKey('media.media_id'))
    media = relationship(Media)
    label = Column('label', String)
    embedding = Column('embedding', Vector(63))
    createdTime = Column('created_at', DateTime,
                        default=datetime.now)


class Trials(Base):
    '''Table trials with each task's run outcome'''
    __tablename__ = 'trials'

    trialId = Column('trial_id', Integer, primary_key=True)
    mediaId = Column('media_id', Integer, ForeignKey('media.media_id'))
    media = relationship(Media)
    label = Column('label', String)
    score = Column('score', Float, nullable=True)
    outputMediaId = Column('output_media_id', Integer, nullable=True)
    # outputMedia = relationship(Media)
    userFeedback = Column('user_feedback', Boolean, nullable=True)
    task = Column('task', String)
    createdTime = Column('created_at', DateTime,
                        default=datetime.now)

Base.metadata.create_all(bind=engine)
