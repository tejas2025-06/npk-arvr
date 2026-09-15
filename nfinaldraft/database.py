import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NPKDatabase:
    def __init__(self):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(os.getenv('MONGODB_URI'))
            self.db = self.client.npk_sensor_db
            self.collection = self.db.sensor_readings
            
            # Test connection
            self.client.admin.command('ping')
            print("✅ Connected to MongoDB Atlas successfully")
            
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")
            self.client = None
            self.db = None
            self.collection = None
    
    def store_npk_reading(self, nitrogen, phosphorus, potassium, analysis=None):
        """Store NPK sensor reading with timestamp"""
        if self.collection is None:
            print("❌ Database not connected")
            return None
            
        try:
            document = {
                "timestamp": datetime.utcnow(),
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "analysis": analysis,
                "device_id": "arduino_npk_sensor_01"  # You can make this dynamic
            }
            
            result = self.collection.insert_one(document)
            print(f"✅ Stored reading with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Error storing data: {e}")
            return None
    
    def get_recent_readings(self, limit=10):
        """Get recent NPK readings"""
        if self.collection is None:
            return []
            
        try:
            readings = list(self.collection.find()
                          .sort("timestamp", -1)
                          .limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for reading in readings:
                reading['_id'] = str(reading['_id'])
                
            return readings
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return []
    
    def get_readings_by_date_range(self, start_date, end_date):
        """Get readings within a date range"""
        if self.collection is None:
            return []
            
        try:
            readings = list(self.collection.find({
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }).sort("timestamp", -1))
            
            # Convert ObjectId to string
            for reading in readings:
                reading['_id'] = str(reading['_id'])
                
            return readings
            
        except Exception as e:
            print(f"❌ Error fetching data by date: {e}")
            return []
    
    def get_average_values(self, hours=24):
        """Get average NPK values for the last N hours"""
        if self.collection is None:
            return None
            
        try:
            from datetime import timedelta
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            pipeline = [
                {"$match": {"timestamp": {"$gte": cutoff_time}}},
                {"$group": {
                    "_id": None,
                    "avg_nitrogen": {"$avg": "$nitrogen"},
                    "avg_phosphorus": {"$avg": "$phosphorus"},
                    "avg_potassium": {"$avg": "$potassium"},
                    "count": {"$sum": 1}
                }}
            ]
            
            result = list(self.collection.aggregate(pipeline))
            return result[0] if result else None
            
        except Exception as e:
            print(f"❌ Error calculating averages: {e}")
            return None
    
    def close_connection(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("🔌 Database connection closed")