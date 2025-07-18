from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, CheckConstraint, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
   __tablename__ = "users"

   id = Column(Integer, primary_key=True, index=True)
   username = Column(String(50), unique=True, index=True, nullable=False)
   name = Column(String(100), nullable=False)
   email = Column(String(255), unique=True, nullable=False)
   password_hash = Column(String(255), nullable=False)
   personal_color_name = Column(String(50))

class StylingSummary(Base):
   __tablename__ = "styling_summary"

   id = Column(Integer, primary_key=True, index=True)
   user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
   budget = Column(Integer, nullable=False)
   occasion = Column(String(100), nullable=False)
   height = Column(Integer, nullable=False)
   gender = Column(String(50), nullable=False)
   top_size = Column(String(10), nullable=False)  # 5 -> 10으로 증가
   bottom_size = Column(Integer, nullable=False)
   shoe_size = Column(Integer, nullable=False)
   body_feature = Column(JSON, nullable=False)
   preferred_styles = Column(JSON, nullable=False)

   __table_args__ = (
       CheckConstraint("top_size IN ('XS','S','M','L','XL','XXL','2XL','3XL')"),
   )

class Item(Base):
   __tablename__ = "items"

   product_id = Column(Integer, primary_key=True, index=True)
   product_name = Column(String(100), nullable=False)
   image_url = Column(String(255))
   price = Column(Integer)

class Favorite(Base):
   __tablename__ = "favorites"

   id = Column(Integer, primary_key=True, index=True)
   user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
   outfit_name = Column(String(200), nullable=False)  
   outfit_dev = Column(String(1000)) 
   product_id = Column(Integer, ForeignKey("items.product_id", ondelete="CASCADE"))

class FavoriteOutfitItem(Base):
   __tablename__ = "favorites_outfit_items"

   favorite_id = Column(Integer, ForeignKey("favorites.id", ondelete="CASCADE"), primary_key=True)
   product_id = Column(Integer, ForeignKey("items.product_id", ondelete="CASCADE"), primary_key=True)

class UserFavoriteItem(Base):
   __tablename__ = "user_favorite_item"

   user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
   product_id = Column(Integer, ForeignKey("items.product_id", ondelete="CASCADE"), primary_key=True)

