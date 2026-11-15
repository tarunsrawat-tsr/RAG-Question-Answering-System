# schema.py
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates
from sqlalchemy import ForeignKey, JSON, String, Text
from pgvector.sqlalchemy import Vector
from typing import List, Dict, Any

class Base(DeclarativeBase):
    pass

class Users(Base):
    __tablename__ = "user_accounts"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30), nullable=False)
    email_address: Mapped[str] = mapped_column(String(255), nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)

    sessions: Mapped[List["Sessions"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    @validates("email_address")
    def validate_email(self, key, val):
        if not val or not val.lower().endswith("@gmail.com"):
            raise ValueError("Invalid email address found!")
        return val #[cite: 1]

class Sessions(Base):
    __tablename__ = "session"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(30), nullable=False)
    messages: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    user: Mapped["Users"] = relationship(back_populates="sessions") #[cite: 2]

class VectorDB(Base):
    __tablename__ = "vector_db"
    index: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False) 
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(3072), nullable=False)