'''
================================================================================
ML FINANCIAL FORECASTING - MAIN APPLICATION
================================================================================

OVERVIEW:
This is the primary entry point for the ML Financial Forecasting system, a 
comprehensive machine learning platform designed for financial market prediction
and automated trading signal generation.

ROLE IN CODEBASE:
- Central application orchestrator and GUI launcher
- Initialises all system components and dependencies  
- Manages the main application lifecycle and error handling
- Serves as the bridge between user interface and core ML functionality

SYSTEM ARCHITECTURE:
The application integrates multiple subsystems:
1. Data processing pipeline (feature engineering from Data_Exporter2.0 format)
2. Multiple ML trainers (RandomForest, XGBoost, LightGBM, SVM, LSTM, Ensemble)
3. Model evaluation and threshold optimisation 
4. MQH header generation for MetaTrader 5 integration
5. GUI interface for user interaction and progress monitoring

DEPENDENCIES:
- tkinter for GUI interface
- ML trainer modules in src/trainers/
- Data processing modules in src/data_processing/
- Utility modules in src/utils/
- Configuration management system

USAGE:
Run this file directly to launch the GUI application:
python main.py

The system processes trading data, trains ML models, and generates MQL5 headers
for real-time trading in MetaTrader 5 platforms.
================================================================================
'''

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class MLFinancialForecastingApp:
    """Main application class"""
    
    def __init__(self):
        self.root = None
        self.gui = None
        
        # Core components
        self.data_manager = None
        self.config_manager = None
        self.current_model = None
        
        # Application state
        self.training_in_progress = False
        
        self.initialise_components()
    
    def initialise_components(self):
        """Initialise core application components"""
        try:
            # Initialise configuration manager
            from src.utils.config_manager import ConfigManager
            self.config_manager = ConfigManager()
            
            # Initialise data manager
            from src.utils.data_manager import DataManager
            data_dir = self.config_manager.get_config('data.default_data_dir', 'data')
            self.data_manager = DataManager(data_dir)
            
            # Setup logging
            from src.utils.logging_utils import setup_logging
            log_config = self.config_manager.get_logging_config()
            setup_logging(
                log_level=log_config.get('log_level', 'INFO'),
                log_file=log_config.get('log_file', 'logs/app.log'),
                max_size_mb=log_config.get('max_log_size_mb', 10),
                backup_count=log_config.get('backup_count', 5)
            )
            
            print("Core components initialised successfully")
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialise application components:\n{e}")
    
    def create_gui(self):
        """Create and setup the GUI"""
        try:
            # Create main window
            self.root = tk.Tk()
            self.root.title("ML Financial Forecasting - University of Essex")
            self.root.iconname("ML Forecast")
            
            # Set window size from config
            gui_config = self.config_manager.get_gui_config()
            window_size = gui_config.get('window_size', [1000, 700])
            self.root.geometry(f"{window_size[0]}x{window_size[1]}")
            
            # Center window on screen
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f"{width}x{height}+{x}+{y}")
            
            # Create GUI interface
            from src.gui.main_interface import MainInterface
            self.gui = MainInterface(self.root, self)
            
            # Setup window close handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            print("GUI created successfully")
            
        except Exception as e:
            print(f"Error creating GUI: {e}")
            messagebox.showerror("GUI Error", 
                               f"Failed to create GUI:\n{e}")
            return False
        
        return True
    
    def on_closing(self):
        """Handle application closing"""
        if self.training_in_progress:
            if messagebox.askokcancel("Quit", "Training is in progress. Do you want to quit?"):
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Run the application"""
        print("Starting ML Financial Forecasting Application...")
        
        # Validate configuration
        is_valid, errors = self.config_manager.validate_config()
        if not is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            print(error_msg)
            messagebox.showwarning("Configuration Warning", error_msg)
        
        # Create and run GUI
        if self.create_gui():
            print("Application ready!")
            self.root.mainloop()
        else:
            print("Failed to start application")

def main():
    '''Main entry point'''
    try:
        app = MLFinancialForecastingApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Critical error: {e}")
        messagebox.showerror("Critical Error", f"Application failed to start:\n{e}")

if __name__ == "__main__":
    main()
