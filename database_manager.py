import json
import os
from datetime import datetime
from typing import Dict, List, Any

class DatabaseManager:
    """Simple file-based database manager for storing analysis history"""
    
    def __init__(self, db_file: str = "analysis_history.json"):
        """Initialize the database manager
        
        Args:
            db_file: Path to the JSON file to store analysis history
        """
        self.db_file = db_file
        self.ensure_db_file_exists()
    
    def ensure_db_file_exists(self):
        """Ensure the database file exists"""
        if not os.path.exists(self.db_file):
            with open(self.db_file, 'w') as f:
                json.dump([], f)
    
    def save_analysis(self, analysis_record: Dict[str, Any]) -> bool:
        """Save an analysis record to the database
        
        Args:
            analysis_record: Dictionary containing analysis data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read existing data
            existing_data = self.load_all_data()
            
            # Add timestamp if not present
            if 'timestamp' not in analysis_record:
                analysis_record['timestamp'] = datetime.now().isoformat()
            
            # Append new record
            existing_data.append(analysis_record)
            
            # Write back to file
            with open(self.db_file, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return False
    
    def get_history(self, session_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get analysis history
        
        Args:
            session_id: Optional session ID to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of analysis records
        """
        try:
            data = self.load_all_data()
            
            # Filter by session_id if provided
            if session_id:
                data = [record for record in data if record.get('session_id') == session_id]
            
            # Sort by timestamp (newest first)
            data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Apply limit
            return data[:limit]
            
        except Exception as e:
            print(f"Error getting history: {e}")
            return []
    
    def clear_history(self, session_id: str = None) -> bool:
        """Clear analysis history
        
        Args:
            session_id: Optional session ID to clear specific session data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if session_id:
                # Clear only specific session data
                data = self.load_all_data()
                filtered_data = [record for record in data if record.get('session_id') != session_id]
                
                with open(self.db_file, 'w') as f:
                    json.dump(filtered_data, f, indent=2, default=str)
            else:
                # Clear all data
                with open(self.db_file, 'w') as f:
                    json.dump([], f)
            
            return True
            
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False
    
    def load_all_data(self) -> List[Dict[str, Any]]:
        """Load all data from the database file
        
        Returns:
            List of all records
        """
        try:
            with open(self.db_file, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def get_analysis_by_type(self, analysis_type: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Get analyses by type
        
        Args:
            analysis_type: Type of analysis (e.g., 'EDA', 'Single Query Analysis')
            session_id: Optional session ID to filter by
            
        Returns:
            List of matching analysis records
        """
        try:
            data = self.load_all_data()
            
            # Filter by type
            filtered_data = [record for record in data if record.get('type') == analysis_type]
            
            # Filter by session_id if provided
            if session_id:
                filtered_data = [record for record in filtered_data if record.get('session_id') == session_id]
            
            # Sort by timestamp (newest first)
            filtered_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return filtered_data
            
        except Exception as e:
            print(f"Error getting analysis by type: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        try:
            data = self.load_all_data()
            
            stats = {
                'total_records': len(data),
                'unique_sessions': len(set(record.get('session_id', '') for record in data)),
                'analysis_types': {},
                'oldest_record': None,
                'newest_record': None
            }
            
            # Count analysis types
            for record in data:
                analysis_type = record.get('type', 'Unknown')
                stats['analysis_types'][analysis_type] = stats['analysis_types'].get(analysis_type, 0) + 1
            
            # Find oldest and newest records
            if data:
                timestamps = [record.get('timestamp', '') for record in data if record.get('timestamp')]
                if timestamps:
                    timestamps.sort()
                    stats['oldest_record'] = timestamps[0]
                    stats['newest_record'] = timestamps[-1]
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                'total_records': 0,
                'unique_sessions': 0,
                'analysis_types': {},
                'oldest_record': None,
                'newest_record': None,
                'error': str(e)
            }
    
    def backup_database(self, backup_file: str = None) -> bool:
        """Create a backup of the database
        
        Args:
            backup_file: Path for backup file. If None, uses timestamp-based name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if backup_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"analysis_history_backup_{timestamp}.json"
            
            data = self.load_all_data()
            
            with open(backup_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    def restore_from_backup(self, backup_file: str) -> bool:
        """Restore database from backup
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(backup_file):
                print(f"Backup file not found: {backup_file}")
                return False
            
            with open(backup_file, 'r') as f:
                data = json.load(f)
            
            # Validate data format
            if not isinstance(data, list):
                print("Invalid backup file format")
                return False
            
            # Write to main database file
            with open(self.db_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error restoring from backup: {e}")
            return False
    
    def delete_old_records(self, days_old: int = 30) -> int:
        """Delete records older than specified days
        
        Args:
            days_old: Number of days to keep records
            
        Returns:
            int: Number of records deleted
        """
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat()
            
            data = self.load_all_data()
            original_count = len(data)
            
            # Filter out old records
            filtered_data = []
            for record in data:
                record_time = record.get('timestamp', '')
                if record_time >= cutoff_str:
                    filtered_data.append(record)
            
            # Write filtered data back
            with open(self.db_file, 'w') as f:
                json.dump(filtered_data, f, indent=2, default=str)
            
            deleted_count = original_count - len(filtered_data)
            return deleted_count
            
        except Exception as e:
            print(f"Error deleting old records: {e}")
            return 0