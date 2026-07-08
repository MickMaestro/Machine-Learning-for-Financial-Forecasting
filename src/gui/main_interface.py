'''
================================================================================
MAIN GUI INTERFACE - USER INTERACTION AND WORKFLOW MANAGEMENT
================================================================================

OVERVIEW:
This module provides the primary graphical user interface for the ML Financial
Forecasting system. It presents an intuitive workflow for training ML models,
generating trading signals, and creating MetaTrader 5 integration files.

ROLE IN CODEBASE:
- Primary user interaction layer for the entire system
- Orchestrates the complete ML training and deployment workflow
- Provides progress monitoring and real-time feedback to users
- Manages file selection, algorithm choice, and parameter configuration
- Handles error display and user guidance throughout the process

GUI ARCHITECTURE:
- Tkinter-based interface with modern styling and layout
- Multi-step wizard approach for guided user experience
- Real-time progress bars and status updates during training
- Tabbed interface for different system functions
- Responsive design with proper error handling and validation

WORKFLOW MANAGEMENT:
1. Data Loading: CSV file selection and validation
2. Algorithm Selection: Choice of ML algorithm with dependency checking
3. Training Execution: Model training with progress monitoring
4. Model Evaluation: Performance metrics display and analysis
5. MQH Generation: Automatic creation of MetaTrader 5 header files
6. File Organisation: Automatic file management and directory creation

ALGORITHM SUPPORT:
- RandomForest (scikit-learn based)
- XGBoost (gradient boosting)
- LightGBM (Microsoft gradient boosting)
- SVM (Support Vector Machine)
- LSTM (TensorFlow deep learning)
- Ensemble (multiple model combination)

INTEGRATION POINTS:
- Calls all trainer modules in src/trainers/
- Uses MQH generator for MetaTrader 5 integration
- Manages configuration through config_manager
- Handles logging and error reporting
- Coordinates with data processing modules

USER EXPERIENCE FEATURES:
- Dependency checking and installation guidance
- Algorithm availability detection based on installed packages
- Progress tracking with detailed status messages
- Error handling with user-friendly explanations
- Automatic file organisation and cleanup
================================================================================
'''
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class MainInterface:
    """Main GUI interface with tabbed layout"""
    
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.setup_ui()
        self.setup_styles()
        
    def setup_styles(self):
        """Configure UI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
        
    def setup_ui(self):
        """Setup the main UI components"""
        # Main title
        
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = ttk.Label(title_frame, text="ML Financial Forecasting System", 
                               style='Title.TLabel')
        title_label.pack(side='left')
        
        # Status indicator
        self.status_label = ttk.Label(title_frame, text="Ready", 
                                     style='Success.TLabel')
        self.status_label.pack(side='right')
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create tabs
        self.create_data_tab()
        self.create_training_tab()
        self.create_results_tab()
        
        # Bottom status bar
        self.create_status_bar()
        
    def create_data_tab(self):
        """Create data management tab"""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Management")
        
        # Data loading section
        load_frame = ttk.LabelFrame(self.data_frame, text="Load Data", padding=10)
        load_frame.pack(fill='x', padx=10, pady=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(load_frame)
        buttons_frame.pack(fill='x')
        
        ttk.Button(buttons_frame, text="Load CSV File", 
                  command=self.load_csv_file).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Clear Data Preview", 
                  command=self.clear_data_preview).pack(side='left', padx=5)
        
        # Help button for data tab
        def show_data_help():
            help_text = """
Data Management Help
===================

Load CSV File:
* Click to select a CSV file containing financial signal data
* Supported formats: CSV files with Data_Exporter2.0 structure
* Required columns: Signal_Type, Entry_Price, Profit_Loss, 24 Data_Exporter2.0 features
* IMPORTANT: Use Data_Exporter2.0.mq5 script to generate compatible CSV files

Data Preview:
* Displays loaded data structure and sample values
* Shows column types and non-null counts
* Use 'Clear Data Preview' to reset the display

Data Requirements:
* CSV files with financial trading signal data
* Proper column structure for ML training
* Clean data without missing critical values
"""
            messagebox.showinfo("Data Management Help", help_text)
        
        ttk.Button(buttons_frame, text="Help", command=show_data_help).pack(side='right', padx=5)
        
        # Data preview section
        preview_frame = ttk.LabelFrame(self.data_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Treeview for data preview
        columns = ('Column', 'Type', 'Non-Null', 'Sample')
        self.data_tree = ttk.Treeview(preview_frame, columns=columns, show='headings')
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=150)
        
        # Scrollbars for treeview
        tree_frame = ttk.Frame(preview_frame)
        tree_frame.pack(fill='both', expand=True)
        
        v_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.data_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.data_tree.pack(side='left', fill='both', expand=True)
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        
        # Data info (read-only)
        self.data_info_text = tk.Text(preview_frame, height=5, width=50, state='disabled')
        self.data_info_text.pack(fill='x', pady=5)

    def create_training_tab(self):
        """Create model training tab with enhanced algorithm selection"""
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Model Training")
        
        # Training configuration
        config_frame = ttk.LabelFrame(self.training_frame, text="Training Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        # Algorithm selection with availability checking
        ttk.Label(config_frame, text="Algorithm:").grid(row=0, column=0, sticky='w', padx=5)
        self.algorithm_var = tk.StringVar(value="Random Forest")
        
        # Get available algorithms
        try:
            from src.models.ml_algorithms import AlgorithmFactory
            status = AlgorithmFactory.get_algorithm_status()
            available_algorithms = [algo for algo, available in status.items() if available]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_algorithms = []
            for algo in available_algorithms:
                if algo not in seen:
                    unique_algorithms.append(algo)
                    seen.add(algo)
            
        except ImportError:
            # Fallback list
            unique_algorithms = [
                "Random Forest", 
                "XGBoost", 
                "LightGBM", 
                "SVM",
                "Ensemble"
            ]
        
        algorithm_combo = ttk.Combobox(config_frame, textvariable=self.algorithm_var,
                                     values=unique_algorithms, state="readonly")
        algorithm_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Add algorithm info label
        algo_info_label = ttk.Label(config_frame, text="", font=('Arial', 8), foreground='gray')
        algo_info_label.grid(row=0, column=2, padx=10, pady=2, sticky='w')
        
        def update_algorithm_info(*args):
            """Update algorithm information when selection changes"""
            algo = self.algorithm_var.get()
            info_text = {
                "Random Forest": "Ensemble of decision trees - Fast & reliable",
                "XGBoost": "Gradient boosting - High performance",
                "LightGBM": "Fast gradient boosting - Memory efficient", 
                "SVM": "Support Vector Machine - Good for non-linear patterns",
                "LSTM": "Deep learning - Sequential pattern recognition",
                "Ensemble": "Combines multiple algorithms"
            }
            algo_info_label.config(text=info_text.get(algo, ""))
        
        self.algorithm_var.trace('w', update_algorithm_info)
        update_algorithm_info()  # Set initial info
        
        # Test size configuration
        ttk.Label(config_frame, text="Test Size (%):").grid(row=1, column=0, sticky='w', padx=5)
        self.test_size_var = tk.IntVar(value=20)
        test_size_spin = ttk.Spinbox(config_frame, from_=10, to=50, textvariable=self.test_size_var, width=10)
        test_size_spin.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(config_frame, text="Percentage of data for testing", 
                 font=('Arial', 8), foreground='gray').grid(row=1, column=2, padx=10, pady=2, sticky='w')
        
        # Confidence threshold
        # Training controls
        control_frame = ttk.Frame(config_frame)
        control_frame.grid(row=2, column=0, columnspan=3, pady=15)
        
        # Enhanced buttons
        train_btn = ttk.Button(control_frame, text="Start Training", 
                              command=self.start_training)
        train_btn.pack(side='left', padx=5)
        
        stop_btn = ttk.Button(control_frame, text="Stop Training", 
                             command=self.stop_training)
        stop_btn.pack(side='left', padx=5)
        
        load_btn = ttk.Button(control_frame, text="Load Model", 
                             command=self.load_model)
        load_btn.pack(side='left', padx=5)
        
        # Add help button
        def show_training_help():
            help_text = """
Training Help
=============

Algorithm Selection:
* Random Forest: Fast, reliable baseline algorithm
* XGBoost: High performance, may require more memory
* LightGBM: Fast and memory efficient, good for large datasets
* SVM: Good for complex patterns, slower training

Parameters:
* Test Size: 20% recommended for most datasets

Training Process:
1. Data validation and preprocessing
2. Feature engineering (24 financial indicators)
3. Time-based train/test split (chronological order)
4. Model training with progress monitoring
5. Performance evaluation with confidence scores
6. Model saving and prediction export
7. Threshold analysis to find optimal confidence level

Tips:
* Ensure at least 1000 samples for reliable results
* Use time-based split to avoid data leakage
* Monitor training progress in real-time
* After training, use 'Results & Analysis' -> 'Confidence Score Analysis' to find optimal confidence score
* Different thresholds optimize for different goals (precision vs recall)
"""
            messagebox.showinfo("Training Help", help_text)
        
        help_btn = ttk.Button(control_frame, text="Help", command=show_training_help)
        help_btn.pack(side='right', padx=5)
        
        # Training progress section
        progress_frame = ttk.LabelFrame(self.training_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)
        
        # Progress text (read-only) - increased height and made expandable
        self.progress_text = tk.Text(progress_frame, height=12, state='disabled')
        self.progress_text.pack(fill='both', expand=True, pady=5)
        
        # Training results section
        results_frame = ttk.LabelFrame(self.training_frame, text="Training Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Results text (read-only) - increased height
        self.results_text = tk.Text(results_frame, height=12, state='disabled')
        self.results_text.pack(fill='both', expand=True)

    def create_results_tab(self):
        """Create results visualization tab"""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results & Analysis")
        
        # Chart selection
        chart_frame = ttk.LabelFrame(self.results_frame, text="Visualization", padding=10)
        chart_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(chart_frame, text="Chart Type:").pack(side='left', padx=5)
        self.chart_type_var = tk.StringVar(value="Performance Metrics")
        chart_combo = ttk.Combobox(chart_frame, textvariable=self.chart_type_var,
                                 values=["Performance Metrics", "Feature Importance", 
                                        "Confidence Score Analysis"])
        chart_combo.pack(side='left', padx=5)
        
        ttk.Button(chart_frame, text="Generate Chart", 
                  command=self.generate_chart).pack(side='left', padx=10)
        
        ttk.Button(chart_frame, text="Save Chart", 
                  command=self.save_chart).pack(side='left', padx=10)
        
        ttk.Button(chart_frame, text="Save All Charts", 
                  command=self.save_all_charts).pack(side='left', padx=10)
        
        # MQH Model Export
        ttk.Button(chart_frame, text="Generate MQH", 
                  command=self.generate_mqh_model).pack(side='left', padx=10)
        
        # Help button for results tab
        def show_results_help():
            help_text = """
Results & Analysis Help
=======================

Chart Types:
* Performance Metrics: Shows accuracy, precision, recall, F1-score for trained models
* Feature Importance: Displays all features ranked by importance for predictions
* Confidence Score Analysis: Analyses how confidence thresholds affect performance

Chart Operations:
* Generate Chart: Creates visualization based on selected chart type
* Save Chart: Exports current chart as PNG image to plots/ folder
* Save All Charts: Automatically generates and saves all three chart types to respective folders

MQL5 Integration:
* Generate MQH: Creates .mqh header files for ML_FF_AI Expert Advisor
* Exports .mqh files to header_files/ folder with ML confidence logic
* Converts ML model confidence thresholds into MetaTrader 5 preset format

Understanding Performance Metrics:
* Accuracy: Overall correctness (correct predictions / total predictions)
* Precision: Of all positive predictions, how many were correct (reduces false alarms)
* Recall: Of all actual positives, how many were caught (reduces missed opportunities)
* F1-Score: Balanced measure combining precision and recall (harmonic mean)
* Higher values = better performance (range: 0.0 to 1.0)

Understanding Charts:
* Feature Importance: Higher bars = more influential features for predictions
* Confidence Analysis: Shows optimal confidence thresholds for trading decisions

Workflow:
1. Train a model in the 'Model Training' tab first
2. Select desired chart type from dropdown
3. Click 'Generate Chart' to visualize results
4. Use 'Save Chart' to export visualizations
5. Use 'Generate MQH' for trading integration

Note: Charts require a trained model to display meaningful data.
Load data and train a model first if you see "No model available" messages.
"""
            messagebox.showinfo("Results & Analysis Help", help_text)
        
        ttk.Button(chart_frame, text="Help", command=show_results_help).pack(side='right', padx=5)
        
        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.results_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_status_bar(self):
        """Create bottom status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill='x', side='bottom')
        
        self.status_message = ttk.Label(self.status_bar, text="Ready")
        self.status_message.pack(side='left', padx=10)
        
        self.progress_message = ttk.Label(self.status_bar, text="")
        self.progress_message.pack(side='right', padx=10)

    # Event handlers
    def load_csv_file(self):
        """Load CSV file with proper default directory"""
        # Get project root directory more reliably
        try:
            # Try to get from main.py location
            import main
            project_root = os.path.dirname(os.path.abspath(main.__file__))
        except (ImportError, AttributeError):
            # Fallback to current working directory
            project_root = os.getcwd()
        
        default_dir = os.path.join(project_root, 'data', 'signal')
        
        # Create directory if it doesn't exist
        os.makedirs(default_dir, exist_ok=True)
        
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            initialdir=default_dir,
            filetypes=[
                ("CSV files", "*.csv"), 
                ("Excel files", "*.xlsx;*.xls"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Show loading message
                self.update_status("Loading CSV file...", "info")
                self.root.update()
                
                # Load the file
                self.app.data_manager.load_csv(file_path)
                self.update_data_preview()
                self.update_status(f"CSV file loaded: {os.path.basename(file_path)}", "success")
                
            except Exception as e:
                error_msg = f"Failed to load CSV file: {str(e)}"
                messagebox.showerror("Data Loading Error", error_msg)
                self.update_status("Error loading CSV file", "error")
                self.add_progress_message(f"Error: {error_msg}")
    
    def clear_data_preview(self):
        """Clear data preview and reset data manager"""
        # Clear the treeview
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Clear the info text
        self.data_info_text.config(state='normal')
        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.config(state='disabled')
        
        # Reset data manager
        self.app.data_manager.data = None
        self.app.data_manager.data_info = {}
        
        self.update_status("Data preview cleared", "success")


    def start_training(self):
        """Start model training with comprehensive validation"""
        
        # Check if data is loaded
        if not self.app.data_manager.has_data():
            messagebox.showwarning(
                "No Data", 
                "Please load data first using the 'Data Management' tab.\n\n"
                "Steps:\n"
                "1. Go to 'Data Management' tab\n"
                "2. Click 'Load CSV File'\n"
                "3. Select your financial data file\n"
                "4. Return to this tab to start training"
            )
            self.notebook.select(0)  # Switch to data tab
            return
        
        # Check if training is already in progress
        if self.app.training_in_progress:
            result = messagebox.askyesnocancel(
                "Training in Progress", 
                "Training is already in progress.\n\n"
                "Choose an action:\n"
                "- Yes: Stop current training and start new\n"
                "- No: Continue current training\n"
                "- Cancel: Do nothing"
            )
            
            if result is True:  # Yes - stop and restart
                self.stop_training()
                # Wait a moment for cleanup
                self.root.after(1000, self.start_training)
                return
            else:  # No or Cancel
                return
        
        # Validate data quality
        try:
            is_valid, issues = self.app.data_manager.validate_data()
            if not is_valid:
                issue_text = "\n".join([f"- {issue}" for issue in issues])
                result = messagebox.askyesno(
                    "Data Quality Issues", 
                    f"Data validation found the following issues:\n\n{issue_text}\n\n"
                    "These issues may affect training performance.\n\n"
                    "Do you want to continue training anyway?\n\n"
                    "Recommendation: Fix data issues for better results."
                )
                if not result:
                    return
        except Exception as e:
            messagebox.showerror("Validation Error", f"Error validating data:\n{str(e)}")
            return
        
        # Get selected algorithm
        algorithm = self.algorithm_var.get()
        
        # Check algorithm-specific dependencies
        dependency_checks = {
            "XGBoost": ("xgboost", "pip install xgboost"),
            "LightGBM": ("lightgbm", "pip install lightgbm"), 
            "LSTM": ("tensorflow", "pip install tensorflow")
        }
        
        if algorithm in dependency_checks:
            module_name, install_cmd = dependency_checks[algorithm]
            try:
                __import__(module_name)
            except ImportError:
                result = messagebox.askyesnocancel(
                    "Missing Dependency", 
                    f"{algorithm} requires the '{module_name}' library.\n\n"
                    f"To install it, run:\n{install_cmd}\n\n"
                    "Options:\n"
                    "- Yes: Continue with a different algorithm\n"
                    "- No: Cancel training\n"
                    "- Cancel: Show installation guide"
                )
                
                if result is True:  # Show alternative algorithms
                    available_algos = ["Random Forest", "XGBoost", "SVM"]
                    algo_text = "\n".join([f"- {algo}" for algo in available_algos])
                    messagebox.showinfo(
                        "Available Algorithms",
                        f"You can use these algorithms without additional dependencies:\n\n{algo_text}\n\n"
                        "Please select one from the dropdown and try again."
                    )
                elif result is None:  # Show installation guide
                    messagebox.showinfo(
                        "Installation Guide",
                        f"To install {algorithm} support:\n\n"
                        f"1. Open command prompt/terminal\n"
                        f"2. Run: {install_cmd}\n"
                        f"3. Restart the application\n"
                        f"4. Try training again\n\n"
                        f"Note: Some systems may require additional setup.\n"
                        f"Check the documentation for platform-specific instructions."
                    )
                return
        
        # Validate training parameters
        test_size = self.test_size_var.get()
        if not (10 <= test_size <= 50):
            messagebox.showwarning(
                "Invalid Test Size",
                f"Test size must be between 10% and 50%.\n"
                f"Current value: {test_size}%\n\n"
                f"Recommended: 20% for most datasets"
            )
            return
        
        # Note: Confidence threshold will be determined through analysis after training
        # Default threshold of 0.5 will be used during training for basic evaluation
        
        # Check dataset size recommendations
        df = self.app.data_manager.get_data()
        dataset_size = len(df)
        
        if dataset_size < 100:
            messagebox.showerror(
                "Dataset Too Small",
                f"Dataset has only {dataset_size} rows.\n\n"
                f"Minimum required: 100 rows\n"
                f"Recommended: 1,000+ rows for reliable results\n\n"
                f"Please load a larger dataset."
            )
            return
        elif dataset_size < 500:
            result = messagebox.askyesno(
                "Small Dataset Warning",
                f"Dataset has only {dataset_size} rows.\n\n"
                f"This may result in:\n"
                f"- Poor model performance\n"
                f"- Unreliable results\n"
                f"- Overfitting\n\n"
                f"Recommended: 1,000+ rows\n\n"
                f"Continue anyway?"
            )
            if not result:
                return
        
        # Clear previous results and prepare UI
        self.progress_var.set(0)
        self.progress_text.config(state='normal')
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.config(state='disabled')
        
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')
        
        # Show training start message
        self.add_progress_message(f"Starting {algorithm} training...")
        self.add_progress_message(f"Dataset: {dataset_size:,} samples")
        self.add_progress_message(f"Test size: {test_size}%")
        self.add_progress_message("Confidence threshold will be optimized after training")
        
        # Start training in separate thread
        training_thread = threading.Thread(target=self._training_worker)
        training_thread.daemon = True
        training_thread.start()
        
        # Update UI state
        self.update_status(f"Training {algorithm}...", "warning")

    def stop_training(self):
        """Stop training process"""
        if self.app.training_in_progress:
            self.app.training_in_progress = False
            self.update_status("Training stopped by user", "warning")
            self.add_progress_message("Training stopped by user")
        else:
            self.update_status("No training in progress", "info")

    def _get_model_save_path(self, algorithm_name):
        """Generate organised model save path based on data source"""
        try:
            import os
            # Get data info from data manager
            data_info = self.app.data_manager.data_info
            source_path = data_info.get('source', '')
            
            # Extract symbol and timeframe from filename
            # Handle multiple formats: {SYMBOL}_{TIMEFRAME}_signal_data.csv OR {SYMBOL}_signal_data-{TIMEFRAME}.csv
            filename = os.path.basename(source_path)
            
            # Default values
            symbol = "UNKNOWN"
            timeframe = "UNKNOWN"
            
            if filename and filename.endswith('_signal_data.csv'):
                # Format: NAS100_TEST_H1_signal_data.csv
                parts = filename.replace('_signal_data.csv', '').split('_')
                if len(parts) >= 2:
                    # Last part is timeframe, everything else is symbol
                    timeframe = parts[-1]
                    symbol = '_'.join(parts[:-1])
            elif filename and '_signal_data-' in filename:
                # Format: NAS100_TEST_signal_data-H1.csv
                symbol = filename.split('_signal_data-')[0]
                timeframe_part = filename.split('_signal_data-')[1]
                timeframe = timeframe_part.replace('.csv', '')
            
            # Map symbol names to folder names
            symbol_mapping = {
                'GER40': 'DAX40',  # GER40 files go to DAX40 folder
                'JPN225': 'JPN225',
                'NAS100': 'NAS100',
                'SP500': 'SP500',
                'US30': 'US30'
            }
            
            folder_name = symbol_mapping.get(symbol, symbol)
            
            # Create model filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{algorithm_name}_{timeframe}_{timestamp}.pkl"
            
            # Create full path
            model_folder = os.path.join("models", folder_name)
            os.makedirs(model_folder, exist_ok=True)
            
            model_path = os.path.join(model_folder, model_filename)
            
            return model_path, symbol, timeframe
            
        except Exception as e:
            # Fallback to simple naming if extraction fails
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fallback_path = os.path.join("models", f"{algorithm_name}_{timestamp}.pkl")
            return fallback_path, "UNKNOWN", "UNKNOWN"

    def _training_worker(self):
        """Worker thread for training with comprehensive error handling"""
        try:
            self.app.training_in_progress = True
            self.update_status("Training in progress...", "warning")
            
            algorithm = self.algorithm_var.get()
            test_size = self.test_size_var.get() / 100
            
            if algorithm == "Random Forest":
                # Use the Random Forest trainer
                from src.trainers.random_forest_trainer import RandomForestTrainer
                
                trainer = RandomForestTrainer(csv_file=None)
                trainer.df = self.app.data_manager.get_data().copy()
                
                # Run training with progress updates
                success = trainer.run_complete_training_with_gui(
                    progress_callback=self.update_training_progress
                )
                
                if success:
                    # Save model with organised path
                    model_path, symbol, timeframe = self._get_model_save_path("RandomForest")
                    trainer.save_model(model_path)
                    
                    self.app.current_model = trainer
                    self.display_training_results(trainer)
                    self.update_status("Random Forest training completed", "success")
                    self.add_progress_message(f"Model saved: {symbol}_{timeframe} -> {model_path}")
                else:
                    self.update_status("Random Forest training failed", "error")
                        
            elif algorithm == "Random Forest":
                # Use the Random Forest trainer
                from src.trainers.random_forest_trainer import RandomForestTrainer
                
                # Get Random Forest parameters from config
                rf_params = self.app.config_manager.get_algorithm_config("Random Forest")
                
                trainer = RandomForestTrainer(csv_file=None, **rf_params)
                trainer.df = self.app.data_manager.get_data().copy()
                
                # Run training with progress updates
                success = trainer.run_complete_training_with_gui(
                    progress_callback=self.update_training_progress
                )
                
                if success:
                    # Save model with organised path
                    model_path, symbol, timeframe = self._get_model_save_path("RandomForest")
                    trainer.save_model(model_path)
                    
                    self.app.current_model = trainer
                    self.display_random_forest_results(trainer)
                    self.update_status("Random Forest training completed", "success")
                    self.add_progress_message(f"Model saved: {symbol}_{timeframe} -> {model_path}")
                else:
                    self.update_status("Random Forest training failed", "error")
                        
            elif algorithm == "XGBoost":
                # Use the XGBoost trainer
                from src.trainers.xgboost_trainer import XGBoostTrainer
                
                # Get XGBoost parameters from config
                xgb_params = self.app.config_manager.get_algorithm_config("XGBoost")
                
                trainer = XGBoostTrainer(csv_file=None, **xgb_params)
                trainer.df = self.app.data_manager.get_data().copy()
                
                # Run training with progress updates
                success = trainer.run_complete_training_with_gui(
                    progress_callback=self.update_training_progress
                )
                
                if success:
                    # Save model with organised path
                    model_path, symbol, timeframe = self._get_model_save_path("XGBoost")
                    trainer.save_model(model_path)
                    
                    self.app.current_model = trainer
                    self.display_xgboost_results(trainer)
                    self.update_status("XGBoost training completed", "success")
                    self.add_progress_message(f"Model saved: {symbol}_{timeframe} -> {model_path}")
                else:
                    self.update_status("XGBoost training failed", "error")
                        
            elif algorithm == "SVM":
                # Use the SVM trainer
                from src.trainers.svm_trainer import SVMTrainer
                
                # Get SVM parameters from config
                svm_params = self.app.config_manager.get_algorithm_config("SVM")
                
                trainer = SVMTrainer(csv_file=None, **svm_params)
                trainer.df = self.app.data_manager.get_data().copy()
                
                # Run training with progress updates
                success = trainer.run_complete_training_with_gui(
                    progress_callback=self.update_training_progress
                )
                
                if success:
                    # Save model with organised path
                    model_path, symbol, timeframe = self._get_model_save_path("SVM")
                    trainer.save_model(model_path)
                    
                    self.app.current_model = trainer
                    self.display_svm_results(trainer)
                    self.update_status("SVM training completed", "success")
                    self.add_progress_message(f"Model saved: {symbol}_{timeframe} -> {model_path}")
                else:
                    self.update_status("SVM training failed", "error")
                        
            elif algorithm == "LightGBM":
                # Use the LightGBM trainer
                from src.trainers.lightgbm_trainer import LightGBMTrainer
                
                # Get LightGBM parameters from config
                lgb_params = self.app.config_manager.get_algorithm_config("LightGBM")
                
                trainer = LightGBMTrainer(csv_file=None, **lgb_params)
                trainer.df = self.app.data_manager.get_data().copy()
                
                # Run training with progress updates
                success = trainer.run_complete_training_with_gui(
                    progress_callback=self.update_training_progress
                )
                
                if success:
                    # Save model with organised path
                    model_path, symbol, timeframe = self._get_model_save_path("LightGBM")
                    trainer.save_model(model_path)
                    
                    self.app.current_model = trainer
                    self.display_lightgbm_results(trainer)
                    self.update_status("LightGBM training completed", "success")
                    self.add_progress_message(f"Model saved: {symbol}_{timeframe} -> {model_path}")
                else:
                    self.update_status("LightGBM training failed", "error")
                        
            elif algorithm == "LSTM":
                try:
                    # Use the LSTM trainer
                    from src.trainers.lstm_trainer import LSTMTrainer
                    print("DEBUG: LSTM trainer imported successfully")
                except ImportError as ie:
                    self.add_progress_message(f"LSTM Import Error: {str(ie)}")
                    raise ie
                from datetime import datetime
                import os
                
                try:
                    # Get LSTM parameters from config
                    lstm_params = self.app.config_manager.get_algorithm_config("LSTM")
                    
                    trainer = LSTMTrainer(csv_file=None, **lstm_params)
                    data = self.app.data_manager.get_data().copy()
                    
                    self.add_progress_message("Preparing LSTM data...")
                    
                    # Use data directly with proper signal quality creation
                    if data is None or data.empty:
                        raise ValueError("No data available for training")
                    
                    # Create Signal_Quality column if it doesn't exist
                    if 'Signal_Quality' not in data.columns:
                        # Import and use the signal labeling function
                        from src.data_processing.feature_engineering import create_signal_labels
                        self.add_progress_message("Creating signal quality labels...")
                        data = create_signal_labels(data)
                    
                    # Data_Exporter2.0 compatible feature engineering
                    self.add_progress_message("Engineering features with Data_Exporter2.0...")
                    
                    from src.data_processing.feature_engineering import FeatureEngineer
                    
                    feature_engineer = FeatureEngineer()
                    
                    # Validate Data_Exporter2.0 format
                    is_valid, missing = feature_engineer.validate_data_exporter_2_features(data)
                    if not is_valid:
                        self.add_progress_message("WARNING: Not in Data_Exporter2.0 format. Training may fail.")
                        self.add_progress_message(f"Missing features: {missing[:3]}{'...' if len(missing) > 3 else ''}")
                        # For backward compatibility, try to continue with available data
                        raise ValueError(f"Data must be in Data_Exporter2.0 format.\nMissing features: {missing[:5]}{'...' if len(missing) > 5 else ''}\n\nPlease use Data_Exporter2.0.mq5 script to generate compatible CSV files.")
                    
                    # Apply comprehensive feature engineering
                    data = feature_engineer.engineer_all_features(data)
                    
                    self.add_progress_message(f"Created {len(feature_engineer.get_feature_columns())} features from Data_Exporter2.0 format")
                    
                    # Remove any rows with NaN in Signal_Quality
                    processed_df = data.dropna(subset=['Signal_Quality']).copy()
                    
                    if processed_df.empty:
                        raise ValueError("No valid data after removing NaN values")
                    
                    self.add_progress_message("Creating train/test splits...")
                    total_samples = len(processed_df)
                    train_size = int(total_samples * 0.8)  # 80% train, 20% test
                    
                    if train_size == 0:
                        raise ValueError("Not enough data for training")
                    
                    # Sort by index to maintain temporal order
                    processed_df = processed_df.sort_index()
                    train_df = processed_df.iloc[:train_size]
                    test_df = processed_df.iloc[train_size:]
                    
                    # Get feature columns from feature engineer
                    feature_cols = feature_engineer.get_feature_columns()
                    
                    # Ensure all features exist in the data
                    missing_features = [col for col in feature_cols if col not in processed_df.columns]
                    if missing_features:
                        self.add_progress_message(f"WARNING: Missing {len(missing_features)} expected features")
                    
                    # Use only available features
                    feature_cols = [col for col in feature_cols if col in processed_df.columns]
                    
                    if not feature_cols:
                        raise ValueError("No feature columns found for training")
                    
                    trainer.feature_columns = feature_cols
                    
                    X_train = train_df[feature_cols]
                    y_train = train_df['Signal_Quality']
                    X_test = test_df[feature_cols]
                    y_test = test_df['Signal_Quality']
                    
                    train_samples = len(X_train) if X_train is not None else 0
                    self.add_progress_message(f"Training LSTM with {train_samples} train samples...")
                    
                    # Train the LSTM model
                    result = trainer.train_model(X_train, y_train, X_test, y_test, 
                                                progress_callback=self.update_training_progress)
                    
                    # LSTM train_model returns a tuple (X_test_seq, y_test_seq, y_pred_proba) or raises exception
                    if result is not None and len(result) == 3:
                        # Save model with organised path
                        model_path, symbol, timeframe = self._get_model_save_path("LSTM")
                        trainer.save_model(model_path)
                        
                        self.app.current_model = trainer
                        self.display_training_results(trainer)
                        self.update_status("LSTM training completed", "success")
                        self.add_progress_message(f"Model saved: {symbol}_{timeframe} -> {model_path}")
                    else:
                        self.update_status("LSTM training failed", "error")
                        
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"LSTM training exception details:\n{error_details}")
                    self.update_status(f"LSTM training failed: {str(e)}", "error")
                    self.add_progress_message(f"LSTM Error: {str(e)} (Type: {type(e).__name__})")
                
            elif algorithm == "Ensemble":
                # Use the Ensemble trainer
                from src.trainers.ensemble_trainer import EnsembleTrainer
                
                # Get Ensemble parameters from config
                ensemble_params = self.app.config_manager.get_algorithm_config("Ensemble")
                
                trainer = EnsembleTrainer(csv_file=None, **ensemble_params)
                trainer.df = self.app.data_manager.get_data().copy()
                
                # Run training with progress updates
                success = trainer.run_complete_training_with_gui(
                    progress_callback=self.update_training_progress
                )
                
                if success:
                    # Save model with organised path
                    model_path, symbol, timeframe = self._get_model_save_path("Ensemble")
                    trainer.save_model(model_path)
                    
                    self.app.current_model = trainer
                    self.display_ensemble_results(trainer)
                    self.update_status("Ensemble training completed", "success")
                    self.add_progress_message(f"Model saved: {symbol}_{timeframe} -> {model_path}")
                else:
                    self.update_status("Ensemble training failed", "error")
                        
            elif algorithm == "Random Forest":
                # Use the Random Forest trainer  
                from src.trainers.random_forest_trainer import RandomForestTrainer
                
                # Get Random Forest parameters from config
                rf_params = self.app.config_manager.get_algorithm_config("Random Forest")
                
                trainer = RandomForestTrainer(csv_file=None, **rf_params)
                trainer.df = self.app.data_manager.get_data().copy()
                
                # Run training with progress updates
                success = trainer.run_complete_training_with_gui(
                    progress_callback=self.update_training_progress
                )
                
                if success:
                    # Save model with organised path
                    model_path, symbol, timeframe = self._get_model_save_path("Signal_Filter_Enhancement")
                    trainer.save_model(model_path)
                    
                    self.app.current_model = trainer
                    self.display_training_results(trainer)
                    self.update_status("Random Forest training completed", "success")
                    self.add_progress_message(f"Model saved: {symbol}_{timeframe} -> {model_path}")
                else:
                    self.update_status("Random Forest training failed", "error")
            
            else:
                # Unknown algorithm
                self.update_status(f"Unknown algorithm: {algorithm}", "error")
                self.add_progress_message(f"Error: {algorithm} is not implemented")
            
            self.app.training_in_progress = False
            
        except Exception as e:
            self.update_status(f"Training failed: {str(e)}", "error")
            self.app.training_in_progress = False
            self.add_progress_message(f"Critical Error: {str(e)}")
            
            # Show detailed error to user
            import traceback
            error_details = traceback.format_exc()
            self.add_progress_message(f"Detailed error information:")
            for line in error_details.split('\n'):
                if line.strip():
                    self.add_progress_message(f"  {line}")

    def display_training_results(self, trainer):
        """Display generic training results"""
        if not hasattr(trainer, 'model') or trainer.model is None:
            print("DEBUG: No model found in trainer")
            return
        
        # Check for performance metrics
        if not hasattr(trainer, 'performance_metrics') or not trainer.performance_metrics:
            print("DEBUG: No performance metrics found in trainer")
            # Show basic info even without metrics
            algorithm_name = getattr(trainer, 'algorithm', 'Unknown Algorithm')
            results = f"""
    {algorithm_name} Training Results
    {'=' * (len(algorithm_name) + 17)}

    Algorithm: {algorithm_name}
    Status: Training completed but no performance metrics available
    
    Note: This may indicate the training process didn't complete properly
    or performance metrics weren't calculated.
    """
            
            # Update results text
            self.results_text.config(state='normal')
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            self.results_text.config(state='disabled')
            return
        
        # Get algorithm name
        algorithm_name = getattr(trainer, 'algorithm', 'Unknown Algorithm')
        
        # Get performance metrics
        results = f"""
    {algorithm_name} Training Results
    {'=' * (len(algorithm_name) + 17)}

    Algorithm: {algorithm_name}
    Features: {len(trainer.feature_columns) if trainer.feature_columns else 0} financial features

    Performance Metrics:
    - Accuracy: {trainer.performance_metrics.get('accuracy', 0):.3f}
    - Precision: {trainer.performance_metrics.get('precision', 0):.3f}
    - Recall: {trainer.performance_metrics.get('recall', 0):.3f}
    - F1-Score: {trainer.performance_metrics.get('f1_score', 0):.3f}
    - ROC-AUC: {trainer.performance_metrics.get('roc_auc', 0):.3f}

    Note: Use 'Results & Analysis' tab -> 'Confidence Score Analysis' to find optimal confidence score

    Files Generated:
    - Model: signal_filter_model.pkl
    - Features: feature_columns.txt
    - Predictions: data/exports/signal_filter_enhancement_predictions_[timestamp].csv

    Predictions Export:
    - Test set predictions saved to data/exports/
    - Includes confidence scores and recommendations
    - Ready for MQL5 integration
    - Contains EXECUTE/FILTER recommendations

    Next steps:
    1. Use this model for real-time signal filtering
    2. Integrate with your MQL5 trading EA
    3. Monitor performance in live trading
    4. Compare with other algorithms
    """
        
        # Update results text
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.results_text.config(state='disabled')

    def display_random_forest_results(self, trainer):
        """Display Random Forest training results"""
        if not hasattr(trainer, 'model') or trainer.model is None:
            return
        
        # Get performance metrics
        results = f"""
Random Forest Training Results
=============================

Algorithm: Random Forest Classifier
Features: {len(trainer.feature_columns)} financial features

Performance Metrics:
- Accuracy: {trainer.performance_metrics.get('accuracy', 0):.3f}
- Precision: {trainer.performance_metrics.get('precision', 0):.3f}
- Recall: {trainer.performance_metrics.get('recall', 0):.3f}
- F1-Score: {trainer.performance_metrics.get('f1_score', 0):.3f}
- ROC-AUC: {trainer.performance_metrics.get('roc_auc', 0):.3f}

Model Parameters:
- n_estimators: {trainer.rf_params.get('n_estimators', 200)}
- max_depth: {trainer.rf_params.get('max_depth', 15)}
- min_samples_split: {trainer.rf_params.get('min_samples_split', 4)}
- min_samples_leaf: {trainer.rf_params.get('min_samples_leaf', 2)}

Files Generated:
- Model: random_forest_model.pkl
- Features: rf_feature_columns.txt
- Predictions: data/exports/random_forest_predictions_[timestamp].csv

Predictions Export:
- Test set predictions saved to data/exports/
- Includes confidence scores and recommendations
- Ready for MQL5 integration
- Contains EXECUTE/FILTER recommendations

Top 5 Important Features:
"""
        
        # Add top features if available
        if 'feature_importance' in trainer.performance_metrics:
            top_features = trainer.performance_metrics['feature_importance'].head(5)
            for idx, row in top_features.iterrows():
                results += f"- {row['feature']}: {row['importance']:.3f}\n"
        
        # Update results text
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.results_text.config(state='disabled')

    def display_xgboost_results(self, trainer):
        """Display XGBoost training results"""
        if not hasattr(trainer, 'model') or trainer.model is None:
            return
        
        # Get performance metrics
        results = f"""
XGBoost Training Results
=======================

Algorithm: XGBoost Classifier
Features: {len(trainer.feature_columns)} financial features

Performance Metrics:
- Accuracy: {trainer.performance_metrics.get('accuracy', 0):.3f}
- Precision: {trainer.performance_metrics.get('precision', 0):.3f}
- Recall: {trainer.performance_metrics.get('recall', 0):.3f}
- F1-Score: {trainer.performance_metrics.get('f1_score', 0):.3f}
- ROC-AUC: {trainer.performance_metrics.get('roc_auc', 0):.3f}

Model Parameters:
- n_estimators: {trainer.xgb_params.get('n_estimators', 200)}
- max_depth: {trainer.xgb_params.get('max_depth', 6)}
- learning_rate: {trainer.xgb_params.get('learning_rate', 0.1)}
- subsample: {trainer.xgb_params.get('subsample', 0.8)}
- colsample_bytree: {trainer.xgb_params.get('colsample_bytree', 0.8)}

Files Generated:
- Model: xgboost_model.pkl
- Features: xgb_feature_columns.txt
- Predictions: data/exports/xgboost_predictions_[timestamp].csv

Predictions Export:
- Test set predictions saved to data/exports/
- Includes confidence scores and recommendations
- Ready for MQL5 integration
- Contains EXECUTE/FILTER recommendations

Training Features:
- Early stopping enabled (50 rounds)
- Validation during training
- Time-based split used

Top 5 Important Features:
"""
        
        # Add top features if available
        if 'feature_importance' in trainer.performance_metrics:
            top_features = trainer.performance_metrics['feature_importance'].head(5)
            for idx, row in top_features.iterrows():
                results += f"- {row['feature']}: {row['importance']:.3f}\n"
        
        # Update results text
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.results_text.config(state='disabled')

    def display_svm_results(self, trainer):
        """Display SVM training results"""
        if not hasattr(trainer, 'model') or trainer.model is None:
            print("DEBUG: No model found in SVM trainer")
            return
            
        # Check for required attributes
        if not hasattr(trainer, 'svm_params'):
            print("DEBUG: SVM trainer missing svm_params attribute")
            # Use generic display method as fallback
            self.display_training_results(trainer)
            return
        
        # Get performance metrics
        results = f"""
SVM Training Results
===================

Algorithm: Support Vector Machine
Features: {len(trainer.feature_columns)} financial features

Performance Metrics:
- Accuracy: {trainer.performance_metrics.get('accuracy', 0):.3f}
- Precision: {trainer.performance_metrics.get('precision', 0):.3f}
- Recall: {trainer.performance_metrics.get('recall', 0):.3f}
- F1-Score: {trainer.performance_metrics.get('f1_score', 0):.3f}
- ROC-AUC: {trainer.performance_metrics.get('roc_auc', 0):.3f}

Model Parameters:
- C (regularization): {trainer.svm_params.get('C', 1.0)}
- Kernel: {trainer.svm_params.get('kernel', 'rbf')}
- Gamma: {trainer.svm_params.get('gamma', 'scale')}
- Class weight: {trainer.svm_params.get('class_weight', 'balanced')}

Files Generated:
- Model: svm_model.pkl
- Features: svm_feature_columns.txt
- Predictions: data/exports/svm_predictions_[timestamp].csv

Predictions Export:
- Test set predictions saved to data/exports/
- Includes confidence scores and recommendations
- Ready for MQL5 integration
- Contains EXECUTE/FILTER recommendations

Training Features:
- Features standardized (required for SVM)
- RBF kernel for non-linear classification
- Balanced class weights for imbalanced data
- Time-based split used

Training Notes:
- SVM requires feature scaling (automatically applied)
- RBF kernel handles non-linear relationships
- Training time increases with dataset size
- Good performance on complex decision boundaries
"""
        
        # Update results text
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.results_text.config(state='disabled')

    def display_ensemble_results(self, trainer):
        """Display Ensemble training results"""
        if not hasattr(trainer, 'model') or trainer.model is None:
            print("DEBUG: No model found in Ensemble trainer")
            return
            
        # Check for required attributes
        if not hasattr(trainer, 'ensemble_params'):
            print("DEBUG: Ensemble trainer missing ensemble_params attribute")
            # Use generic display method as fallback
            self.display_training_results(trainer)
            return
        
        # Get performance metrics
        base_estimators = trainer.performance_metrics.get('base_estimators', [])
        ensemble_method = trainer.performance_metrics.get('ensemble_method', 'voting')
        num_base_models = trainer.performance_metrics.get('num_base_models', 0)
        
        results = f"""
Ensemble Training Results
=========================

Algorithm: {ensemble_method.title()} Ensemble
Base Estimators: {', '.join(base_estimators)}
Number of Base Models: {num_base_models}
Features: {len(trainer.feature_columns)} financial features

Performance Metrics:
- Accuracy: {trainer.performance_metrics.get('accuracy', 0):.3f}
- Precision: {trainer.performance_metrics.get('precision', 0):.3f}
- Recall: {trainer.performance_metrics.get('recall', 0):.3f}
- F1-Score: {trainer.performance_metrics.get('f1_score', 0):.3f}
- ROC-AUC: {trainer.performance_metrics.get('roc_auc', 0):.3f}

Ensemble Configuration:
- Method: {ensemble_method}
- Voting: {trainer.ensemble_params.get('voting', 'soft')}
- CV Folds: {trainer.ensemble_params.get('cv_folds', 5)}
- Base Algorithms: {trainer.ensemble_params.get('base_algorithms', [])}

Model Combination:
- Leverages strengths of multiple algorithms
- Reduces overfitting through model diversity
- Improved generalization performance
- Robust predictions across market conditions

Files Generated:
- Model: ensemble_model_[timestamp].pkl
- Features: ensemble_feature_columns.txt
- Predictions: data/exports/ensemble_predictions_[timestamp].csv

Training Notes:
- Ensemble combines predictions from base models
- {'Soft voting uses predicted probabilities' if ensemble_method == 'voting' else 'Stacking uses meta-learner for final predictions'}
- Performance typically exceeds individual algorithms
- Requires more computational resources
"""
        
        # Update results text
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.results_text.config(state='disabled')

    def display_lightgbm_results(self, trainer):
        """Display LightGBM training results"""
        if not hasattr(trainer, 'model') or trainer.model is None:
            print("DEBUG: No model found in LightGBM trainer")
            return
            
        # Check for required attributes
        if not hasattr(trainer, 'lgb_params'):
            print("DEBUG: LightGBM trainer missing lgb_params attribute")
            # Use generic display method as fallback
            self.display_training_results(trainer)
            return
        
        # Get performance metrics
        results = f"""
LightGBM Training Results
========================

Algorithm: LightGBM Classifier
Features: {len(trainer.feature_columns)} financial features

Performance Metrics:
- Accuracy: {trainer.performance_metrics.get('accuracy', 0):.3f}
- Precision: {trainer.performance_metrics.get('precision', 0):.3f}
- Recall: {trainer.performance_metrics.get('recall', 0):.3f}
- F1-Score: {trainer.performance_metrics.get('f1_score', 0):.3f}
- ROC-AUC: {trainer.performance_metrics.get('roc_auc', 0):.3f}

Model Parameters:
- n_estimators: {trainer.lgb_params.get('n_estimators', 200)}
- max_depth: {trainer.lgb_params.get('max_depth', 6)}
- learning_rate: {trainer.lgb_params.get('learning_rate', 0.1)}
- subsample: {trainer.lgb_params.get('subsample', 0.8)}
- colsample_bytree: {trainer.lgb_params.get('colsample_bytree', 0.8)}
- num_leaves: {trainer.lgb_params.get('num_leaves', 31)}

Files Generated:
- Model: lightgbm_model.pkl
- Features: lgb_feature_columns.txt
- Predictions: data/exports/lightgbm_predictions_[timestamp].csv

Predictions Export:
- Test set predictions saved to data/exports/
- Includes confidence scores and recommendations
- Ready for MQL5 integration
- Contains EXECUTE/FILTER recommendations

Training Features:
- Early stopping enabled (50 rounds)
- Validation during training
- Time-based split used
- Feature scaling applied

Top 5 Important Features:
"""
        
        # Add top features if available
        if 'feature_importance' in trainer.performance_metrics:
            top_features = trainer.performance_metrics['feature_importance'].head(5)
            for idx, row in top_features.iterrows():
                results += f"- {row['feature']}: {row['importance']:.3f}\n"
        
        results += f"""

Training Notes:
- LightGBM is fast and memory efficient
- Handles categorical features natively  
- Less prone to overfitting than XGBoost
- Good performance on tabular data
- Uses gradient-based one-side sampling (GOSS)
- Exclusive feature bundling (EFB) for efficiency
"""
        
        # Update results text
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.results_text.config(state='disabled')

    def load_model(self):
        """Load saved model with better error handling and validation"""
        
        # Check if models directory exists
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            initialdir=models_dir,
            filetypes=[
                ("Pickle files", "*.pkl"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Show loading message
            self.update_status("Loading model...", "warning")
            self.root.update()
            
            # Try to load the model
            from src.models.model_predictor import ModelPredictor
            predictor = ModelPredictor.load_model_static(file_path)
            
            if predictor:
                self.app.current_model = predictor
                
                # Get model info
                model_info = predictor.get_model_info()
                
                # Show success message with model details
                messagebox.showinfo(
                    "Model Loaded Successfully",
                    f"Model loaded from: {os.path.basename(file_path)}\n\n{model_info}"
                )
                
                self.update_status("Model loaded successfully", "success")
                self.add_progress_message(f"Model loaded from {os.path.basename(file_path)}")
                
                # Update results display with model info
                self.results_text.config(state='normal')
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(1.0, f"""
Model Loading Results
====================

File: {os.path.basename(file_path)}
Path: {file_path}

{model_info}

Status: Ready for analysis
Use the 'Results & Analysis' tab to view model performance

Note: Loaded models retain their original training configuration
and performance characteristics.
""")
                self.results_text.config(state='disabled')
                
            else:
                raise ValueError("Failed to create model predictor")
                
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror(
                "Model Loading Error", 
                f"Failed to load model from:\n{os.path.basename(file_path)}\n\n"
                f"Error: {error_msg}\n\n"
                f"Possible causes:\n"
                f"- Model file is corrupted\n"
                f"- Model was created with incompatible version\n" 
                f"- Missing dependencies\n"
                f"- File is not a valid model file\n\n"
                f"Try loading a different model or retrain a new one."
            )
            
            self.update_status("Error loading model", "error")
            print(f"Model loading error: {e}")
            
            # Clear current model if loading failed
            self.app.current_model = None

    def _is_loaded_model(self):
        """Check if current model was loaded from file (not freshly trained)"""
        try:
            if not self.app.current_model:
                return False
            
            # ModelPredictor objects are created when loading from file
            # Trainer objects are created when training fresh models
            from src.models.model_predictor import ModelPredictor
            
            # Check if it's a ModelPredictor instance (loaded from file)
            is_predictor = isinstance(self.app.current_model, ModelPredictor)
            
            # Additional check: loaded models should have a model_info attribute
            has_model_info = hasattr(self.app.current_model, 'model_info')
            
            return is_predictor and has_model_info
            
        except Exception as e:
            print(f"Error checking model type: {e}")
            return False


    # Helper methods
    def update_data_preview(self):
        """Update data preview with enhanced information"""
        if self.app.data_manager.has_data():
            df = self.app.data_manager.get_data()
            if df is None:
                return
            
            # Clear existing items
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Add column information with better formatting
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].notna().sum()
                null_count = df[col].isnull().sum()
                
                # Format sample value
                if len(df) > 0 and df[col].notna().any():
                    sample_val = df[col].dropna().iloc[0]
                    if isinstance(sample_val, float):
                        sample = f"{sample_val:.6f}" if abs(sample_val) < 1 else f"{sample_val:.2f}"
                    else:
                        sample = str(sample_val)[:20]  # Truncate long strings
                else:
                    sample = "N/A"
                
                # Create display text
                non_null_text = f"{non_null}/{len(df)}"
                if null_count > 0:
                    non_null_text += f" ({null_count} null)"
                
                self.data_tree.insert("", "end", values=(col, dtype, non_null_text, sample))
            
            # Update info text with comprehensive statistics
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Check for required financial columns (Data_Exporter2.0 format)
            required_cols = ['Signal_Type', 'Entry_Price', 'Profit_Loss', 'Gann_Zone_Width_Pips', 'Trading_Session']
            missing_required = [col for col in required_cols if col not in df.columns]
            
            info_text = f"""Dataset Overview:
- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
- Memory Usage: {memory_mb:.2f} MB
- Total Missing Values: {df.isnull().sum().sum():,}
- Complete Rows: {df.dropna().shape[0]:,} ({df.dropna().shape[0]/len(df)*100:.1f}%)
- Duplicate Rows: {df.duplicated().sum():,}

Financial Columns Status:
- Required Present: {len(required_cols) - len(missing_required)}/{len(required_cols)}"""

            if missing_required:
                info_text += f"\n- Missing Required: {', '.join(missing_required)}"
            
            # Add date range if timestamp column exists
            timestamp_cols = ['Timestamp', 'Time', 'Date', 'DateTime']
            for col in timestamp_cols:
                if col in df.columns:
                    try:
                        dates = pd.to_datetime(df[col], errors='coerce')
                        if dates.notna().any():
                            date_range = dates.max() - dates.min()
                            info_text += f"\n\nTime Period:\n- From: {dates.min()}\n- To: {dates.max()}\n- Duration: {date_range}"
                            break
                    except (ValueError, TypeError, KeyError):
                        continue
            
            self.data_info_text.config(state='normal')
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, info_text)
            self.data_info_text.config(state='disabled')
            
        else:
            # Clear preview when no data
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            self.data_info_text.config(state='normal')
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, "No data loaded. Please load a CSV file.")
            self.data_info_text.config(state='disabled')
    
    def update_status(self, message, status_type="info"):
        """Update status message with improved visual feedback"""
        # Truncate very long messages
        if len(message) > 100:
            message = message[:97] + "..."
        
        self.status_message.config(text=message)
        
        # Set colors based on status type
        color_map = {
            "success": "#2E7D32",  # Dark green
            "error": "#C62828",    # Dark red  
            "warning": "#F57C00",  # Dark orange
            "info": "#1565C0",     # Dark blue
            "default": "#000000"   # Black
        }
        
        color = color_map.get(status_type, color_map["default"])
        self.status_message.config(foreground=color)
        
        # Auto-clear status after delay for non-error messages
        if status_type not in ["error", "warning"]:
            self.root.after(5000, lambda: self.status_message.config(text="Ready", foreground="#2E7D32"))

    def update_training_progress(self, progress, message):
        """Update training progress"""
        self.progress_var.set(progress)
        self.progress_message.config(text=f"Progress: {progress:.1f}%")
        
        # Add message to progress text
        self.add_progress_message(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def add_progress_message(self, message):
        """Add message to progress text with timestamp and auto-scroll"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        
        self.progress_text.config(state='normal')
        self.progress_text.insert(tk.END, f"{formatted_message}\n")
        
        # Auto-scroll to bottom
        self.progress_text.see(tk.END)
        
        # Limit text length to prevent memory issues
        line_count = int(self.progress_text.index('end-1c').split('.')[0])
        if line_count > 1000:  # Keep last 1000 lines
            self.progress_text.delete('1.0', '500.0')
        
        self.progress_text.config(state='disabled')
        
        # Force GUI update
        self.root.update_idletasks()

        # Process predictions in chunks
        def process_batch():
                try:
                    batch_progress['maximum'] = len(df)
                    results = []
                    
                    for i, row in df.iterrows():
                        # Create market data (Data_Exporter2.0 compatible)
                        market_data = {
                            'entry_price': row['Entry_Price'],
                            'signal_type': row['Signal_Type'],
                            'gann_zone_width_pips': row.get('Gann_Zone_Width_Pips', 50.0),
                            'trading_session': row.get('Trading_Session', 2),
                            'hour': row.get('Hour', 12),
                            'spread': row.get('Spread', 0.0001)
                        }
                        
                        # Make prediction
                        try:
                            prediction = self.app.current_model.predict_signal_quality(market_data)
                            
                            # Add to results
                            result_row = {
                                'Index': i,
                                'Signal_Type': 'BUY' if row['Signal_Type'] == 1 else 'SELL',
                                'Entry_Price': f"{row['Entry_Price']:.5f}",
                                'Confidence': f"{prediction['confidence']:.3f}",
                                'Recommendation': prediction['recommendation']
                            }
                            results.append(result_row)
                            
                            # Add to treeview
                            tree.insert('', 'end', values=list(result_row.values()))
                            
                        except Exception as e:
                            # Handle individual prediction errors
                            error_row = {
                                'Index': i,
                                'Signal_Type': 'BUY' if row['Signal_Type'] == 1 else 'SELL',
                                'Entry_Price': f"{row['Entry_Price']:.5f}",
                                'Confidence': 'ERROR',
                                'Recommendation': 'ERROR'
                            }
                            results.append(error_row)
                            tree.insert('', 'end', values=list(error_row.values()))
                        
                        # Update progress
                        batch_progress['value'] = i + 1
                        batch_window.update()
                    
                    # Create results dataframe for export
                    results_df = pd.DataFrame(results)
                    export_results.results_df = results_df
                    
                    # Update summary
                    total_predictions = len(results)
                    execute_count = sum(1 for r in results if r['Recommendation'] == 'EXECUTE')
                    filter_count = total_predictions - execute_count
                    
                    summary_text = f"Total: {total_predictions} | Execute: {execute_count} | Filter: {filter_count}"
                    summary_label.config(text=summary_text)
                    
                    messagebox.showinfo(
                        "Batch Prediction Complete",
                        f"Processed {total_predictions} predictions:\n"
                        f"- Execute: {execute_count} ({execute_count/total_predictions:.1%})\n"
                        f"- Filter: {filter_count} ({filter_count/total_predictions:.1%})\n\n"
                        f"Use 'Export Results' to save the predictions."
                    )
                    
                except Exception as e:
                    messagebox.showerror("Batch Prediction Error", f"Error processing batch predictions:\n{str(e)}")
            
                    # Start processing in a separate thread
                    import threading
                    process_thread = threading.Thread(target=process_batch)
                    process_thread.daemon = True
                    process_thread.start()
            
                except Exception as e:
                    messagebox.showerror("File Loading Error", f"Error loading file:\n{str(e)}")

    # Helper methods
    def update_data_preview(self):
        """Update data preview with enhanced information"""
        if self.app.data_manager.has_data():
            df = self.app.data_manager.get_data()
            if df is None:
                return
            
            # Clear existing items
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Add column information with better formatting
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].notna().sum()
                null_count = df[col].isnull().sum()
                
                # Format sample value
                if len(df) > 0 and df[col].notna().any():
                    sample_val = df[col].dropna().iloc[0]
                    if isinstance(sample_val, float):
                        sample = f"{sample_val:.6f}" if abs(sample_val) < 1 else f"{sample_val:.2f}"
                    else:
                        sample = str(sample_val)[:20]  # Truncate long strings
                else:
                    sample = "N/A"
                
                # Create display text
                non_null_text = f"{non_null}/{len(df)}"
                if null_count > 0:
                    non_null_text += f" ({null_count} null)"
                
                self.data_tree.insert("", "end", values=(col, dtype, non_null_text, sample))
            
            # Update info text with comprehensive statistics
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Check for required financial columns (Data_Exporter2.0 format)
            required_cols = ['Signal_Type', 'Entry_Price', 'Profit_Loss', 'Gann_Zone_Width_Pips', 'Trading_Session']
            missing_required = [col for col in required_cols if col not in df.columns]
            
            info_text = f"""Dataset Overview:
- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
- Memory Usage: {memory_mb:.2f} MB
- Total Missing Values: {df.isnull().sum().sum():,}
- Complete Rows: {df.dropna().shape[0]:,} ({df.dropna().shape[0]/len(df)*100:.1f}%)
- Duplicate Rows: {df.duplicated().sum():,}

Financial Columns Status:
- Required Present: {len(required_cols) - len(missing_required)}/{len(required_cols)}"""

            if missing_required:
                info_text += f"\n- Missing Required: {', '.join(missing_required)}"
            
            # Add date range if timestamp column exists
            timestamp_cols = ['Timestamp', 'Time', 'Date', 'DateTime']
            for col in timestamp_cols:
                if col in df.columns:
                    try:
                        dates = pd.to_datetime(df[col], errors='coerce')
                        if dates.notna().any():
                            date_range = dates.max() - dates.min()
                            info_text += f"\n\nTime Period:\n- From: {dates.min()}\n- To: {dates.max()}\n- Duration: {date_range}"
                            break
                    except (ValueError, TypeError, KeyError):
                        continue
            
            self.data_info_text.config(state='normal')
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, info_text)
            self.data_info_text.config(state='disabled')
            
        else:
            # Clear preview when no data
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            self.data_info_text.config(state='normal')
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, "No data loaded. Please load a CSV file.")
            self.data_info_text.config(state='disabled')
    
    def update_status(self, message, status_type="info"):
        """Update status message with improved visual feedback"""
        # Truncate very long messages
        if len(message) > 100:
            message = message[:97] + "..."
        
        self.status_message.config(text=message)
        
        # Set colors based on status type
        color_map = {
            "success": "#2E7D32",  # Dark green
            "error": "#C62828",    # Dark red  
            "warning": "#F57C00",  # Dark orange
            "info": "#1565C0",     # Dark blue
            "default": "#000000"   # Black
        }
        
        color = color_map.get(status_type, color_map["default"])
        self.status_message.config(foreground=color)
        
        # Auto-clear status after delay for non-error messages
        if status_type not in ["error", "warning"]:
            self.root.after(5000, lambda: self.status_message.config(text="Ready", foreground="#2E7D32"))

    def update_training_progress(self, progress, message):
        """Update training progress"""
        self.progress_var.set(progress)
        self.progress_message.config(text=f"Progress: {progress:.1f}%")
        
        # Add message to progress text
        self.add_progress_message(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def add_progress_message(self, message):
        """Add message to progress text with timestamp and auto-scroll"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        
        self.progress_text.config(state='normal')
        self.progress_text.insert(tk.END, f"{formatted_message}\n")
        
        # Auto-scroll to bottom
        self.progress_text.see(tk.END)
        
        # Limit text length to prevent memory issues
        line_count = int(self.progress_text.index('end-1c').split('.')[0])
        if line_count > 1000:  # Keep last 1000 lines
            self.progress_text.delete('1.0', '500.0')
        
        self.progress_text.config(state='disabled')
        
        # Force GUI update
        self.root.update_idletasks()

    def _get_test_data_for_analysis(self):
        """Get data for threshold analysis - uses newly trained model data"""
        try:
            print("DEBUG: Getting data for threshold analysis...")
            
            # Use newly trained model data
            return self._get_fresh_model_data()
            
        except Exception as e:
            print(f"Error getting test data for analysis: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _get_fresh_model_data(self):
        """Get data from newly trained model (fallback method)"""
        try:
            print("DEBUG: Trying to get data from freshly trained model...")
            
            if not hasattr(self.app, 'current_model') or self.app.current_model is None:
                print("DEBUG: No current model")
                return None
                
            model = self.app.current_model
            print(f"DEBUG: Model found: {type(model)}")
            
            # Check if model has stored test data from recent training
            if (hasattr(model, 'X_test') and hasattr(model, 'y_test') and 
                model.X_test is not None and model.y_test is not None):
                print("DEBUG: Using stored test data from recent training")
                
                # Use pre-computed probabilities if available (much faster and more reliable)
                if hasattr(model, 'y_pred_proba') and model.y_pred_proba is not None:
                    print("DEBUG: Using pre-computed prediction probabilities")
                    confidence_scores = model.y_pred_proba
                else:
                    print("DEBUG: Computing prediction probabilities on the fly")
                    X_test = model.X_test
                    
                    # Get prediction probabilities
                    if hasattr(model, 'model') and model.model is not None:
                        try:
                            if hasattr(model, 'scaler') and model.scaler is not None:
                                X_test_scaled = model.scaler.transform(X_test)
                            else:
                                X_test_scaled = X_test
                                
                            confidence_scores = model.model.predict_proba(X_test_scaled)[:, 1]
                        except Exception as e:
                            print(f"DEBUG: Error computing probabilities: {e}")
                            return None
                    else:
                        print("DEBUG: No model available for predictions")
                        return None
                
                return {
                    'type': 'fresh_model',
                    'confidence_scores': confidence_scores,
                    'actual_outcomes': model.y_test.values if hasattr(model.y_test, 'values') else model.y_test,
                    'source_file': 'newly_trained_model',
                    'n_samples': len(confidence_scores)
                }
            
            print("DEBUG: No usable data from fresh model")
            return None
            
        except Exception as e:
            print(f"DEBUG: Error getting fresh model data: {e}")
            return None


    def _generate_threshold_analysis(self):
        """Generate threshold analysis visualization - works with prediction CSV or fresh models"""
        try:
            # Clear figure first
            self.fig.clear()
            print("DEBUG: Starting threshold analysis...")
            
            # Get analysis data (either from prediction CSV or fresh model)
            analysis_data = self._get_test_data_for_analysis()
            if analysis_data is None:
                self._show_no_data_message("No data available for threshold analysis")
                return
            
            print(f"DEBUG: Using data source: {analysis_data['type']}")
            print(f"DEBUG: Source: {analysis_data['source_file']}")
            
            # Extract data
            confidence_scores = analysis_data['confidence_scores']
            actual_outcomes = analysis_data['actual_outcomes']
            
            print(f"DEBUG: Data shape - confidence: {len(confidence_scores)}, outcomes: {len(actual_outcomes)}")
            print(f"DEBUG: Confidence range: {confidence_scores.min():.3f} - {confidence_scores.max():.3f}")
            
            # Ensure we have enough data
            if len(confidence_scores) < 10:
                print(f"DEBUG: Insufficient data: {len(confidence_scores)} samples")
                self._show_no_data_message("Insufficient data (need at least 10 samples)")
                return
            
            # Calculate threshold analysis
            thresholds = np.arange(0.1, 1.0, 0.05)
            metrics_data = self._calculate_threshold_metrics_from_confidence(
                actual_outcomes, confidence_scores, thresholds)
            print(f"DEBUG: Calculated metrics for {len(thresholds)} thresholds")
            
            # Create the 4-panel visualization
            self._plot_threshold_metrics_new(thresholds, metrics_data, analysis_data)
            print("DEBUG: Threshold analysis completed successfully")
            
        except Exception as e:
            print(f"Error in threshold analysis: {e}")
            import traceback
            traceback.print_exc()
            self._show_error_message(f"Error generating threshold analysis: {str(e)}")

    def _calculate_threshold_metrics_from_confidence(self, y_true, confidence_scores, thresholds):
        """Calculate threshold metrics directly from confidence scores and outcomes"""
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': [],
            'signal_retention': []
        }
        
        for threshold in thresholds:
            # Apply threshold to confidence scores
            y_pred = (confidence_scores >= threshold).astype(int)
            
            # Calculate metrics with zero_division handling
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            retention = np.mean(y_pred)  # Percentage of signals that would be executed
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1)
            metrics['accuracy'].append(accuracy)
            metrics['signal_retention'].append(retention)
        
        return metrics

    def _plot_threshold_metrics_new(self, thresholds, metrics_data, analysis_data):
        """Create the 4-panel threshold analysis plot with new data structure"""
        # Set figure size
        self.fig.set_size_inches(12, 8)
        
        # Create 2x2 subplot grid
        axes = self.fig.subplots(2, 2)
        axes = axes.flatten()
        
        # Define the 4 metrics to plot
        plot_configs = [
            ('precision', 'Precision', '#2E8B57', 'o-'),
            ('recall', 'Recall', '#4682B4', 's-'),
            ('f1_score', 'F1-Score', '#FF6347', '^-'),
            ('signal_retention', 'Signal Retention Rate', '#9932CC', 'd-')
        ]
        
        # Get current threshold (default 0.6 if not available)
        current_threshold = 0.6
        if analysis_data.get('type') == 'prediction_csv' and 'data' in analysis_data:
            df = analysis_data['data']
            if 'Confidence_Threshold' in df.columns:
                current_threshold = df['Confidence_Threshold'].iloc[0]
        
        for i, (metric_key, metric_name, color, marker) in enumerate(plot_configs):
            ax = axes[i]
            values = metrics_data[metric_key]
            
            # Plot the metric curve
            ax.plot(thresholds, values, marker, color=color, linewidth=2, markersize=4, alpha=0.8)
            
            # Find and highlight optimal threshold for this metric
            optimal_idx = np.argmax(values)
            optimal_threshold = thresholds[optimal_idx]
            optimal_value = values[optimal_idx]
            
            # Optimal threshold line
            ax.axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.7, 
                      label=f'Optimal: {optimal_threshold:.2f}')
            
            # Current threshold line
            ax.axvline(x=current_threshold, color='red', linestyle='--', alpha=0.7,
                      label=f'Current: {current_threshold:.2f}')
            
            # Highlight optimal point
            ax.plot(optimal_threshold, optimal_value, 'go', markersize=8, alpha=0.8)
            ax.annotate(f'{optimal_value:.3f}', 
                       xy=(optimal_threshold, optimal_value),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.8))
            
            # Show current threshold performance
            current_idx = np.abs(thresholds - current_threshold).argmin()
            current_value = values[current_idx]
            ax.plot(current_threshold, current_value, 'ro', markersize=8, alpha=0.8)
            ax.annotate(f'{current_value:.3f}', 
                       xy=(current_threshold, current_value),
                       xytext=(-15, -15), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.8))
            
            # Formatting
            ax.set_xlabel('Confidence Score', fontsize=10)
            ax.set_ylabel(f'{metric_name}', fontsize=10)
            ax.set_title(f'{metric_name} vs Confidence Score', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.05, 0.95)
            
            # Set y-limits based on metric type
            if metric_key == 'signal_retention':
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, 1.05)
            
            ax.legend(loc='best', fontsize=8)
            
            # Add some statistics text
            max_val = np.max(values)
            min_val = np.min(values)
            ax.text(0.02, 0.98, f'Max: {max_val:.3f}\nMin: {min_val:.3f}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Overall title with data source info
        import os
        data_source = analysis_data.get('source_file', 'unknown')
        if data_source != 'newly_trained_model':
            data_source = os.path.basename(data_source)
        
        self.fig.suptitle(f'Confidence Score Analysis: Model Performance vs Confidence Score\nData: {data_source}', 
                         fontsize=12, fontweight='bold', y=0.95)
        
        # Adjust layout
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.90])


    def _calculate_threshold_metrics(self, y_true, y_pred_proba, thresholds):
        """Calculate precision, recall, F1, and accuracy for each threshold"""
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': [],
            'signal_retention': []
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics with zero_division handling
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            retention = np.mean(y_pred)  # Percentage of signals retained
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1)
            metrics['accuracy'].append(accuracy)
            metrics['signal_retention'].append(retention)
        
        return metrics

    def _plot_threshold_metrics(self, thresholds, metrics_data, model):
        """Create the 4-panel threshold analysis plot"""
        # Create 2x2 subplot grid
        axes = self.fig.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # Define the 4 metrics to plot
        plot_configs = [
            ('precision', 'Precision', '#2E8B57', 'o-'),
            ('recall', 'Recall', '#4682B4', 's-'),
            ('f1_score', 'F1-Score', '#FF6347', '^-'),
            ('signal_retention', 'Signal Retention Rate', '#9932CC', 'd-')
        ]
        
        current_threshold = getattr(model, 'confidence_threshold', 0.6)
        
        for i, (metric_key, metric_name, color, marker) in enumerate(plot_configs):
            ax = axes[i]
            values = metrics_data[metric_key]
            
            # Plot the metric curve
            ax.plot(thresholds, values, marker, color=color, linewidth=2, markersize=4, alpha=0.8)
            
            # Find and highlight optimal threshold for this metric
            optimal_idx = np.argmax(values)
            optimal_threshold = thresholds[optimal_idx]
            optimal_value = values[optimal_idx]
            
            # Optimal threshold line
            ax.axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.7, 
                      label=f'Optimal: {optimal_threshold:.2f}')
            
            # Current threshold line
            ax.axvline(x=current_threshold, color='red', linestyle='--', alpha=0.7,
                      label=f'Current: {current_threshold:.2f}')
            
            # Highlight optimal point
            ax.plot(optimal_threshold, optimal_value, 'go', markersize=8, alpha=0.8)
            ax.annotate(f'{optimal_value:.3f}', 
                       xy=(optimal_threshold, optimal_value),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.8))
            
            # Formatting
            ax.set_xlabel('Confidence Score', fontsize=10)
            ax.set_ylabel(f'{metric_name}', fontsize=10)
            ax.set_title(f'{metric_name} vs Confidence Score', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.05, 0.95)
            
            # Set y-limits based on metric type
            if metric_key == 'signal_retention':
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, 1.05)
            
            ax.legend(loc='best', fontsize=8)
            
            # Add some statistics text
            max_val = np.max(values)
            min_val = np.min(values)
            ax.text(0.02, 0.98, f'Max: {max_val:.3f}\nMin: {min_val:.3f}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Overall title
        self.fig.suptitle('Confidence Score Analysis: Model Performance Across Confidence Scores', 
                         fontsize=14, fontweight='bold', y=0.95)
        
        # Adjust layout
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _show_no_model_message(self, message):
        """Show no model available message"""
        ax = self.fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'{message}\n\nLoad a model first', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Threshold Analysis - No Model')
        self._clean_axes(ax)

    def _show_no_data_message(self, message):
        """Show no data available message"""
        ax = self.fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'{message}\n\nTo perform threshold analysis:\n1. Go to Data Management tab\n2. Load a CSV file with trading data\n3. Come back and try again\n\n(Threshold analysis needs actual trading data\nto show how confidence affects performance)', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax.set_title('Threshold Analysis - Load Trading Data First')
        self._clean_axes(ax)

    def _show_error_message(self, message):
        """Show error message"""
        ax = self.fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'{message}\n\nCheck console for details', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.set_title('Threshold Analysis - Error')
        self._clean_axes(ax)

    def _clean_axes(self, ax):
        """Remove ticks and spines from axes"""
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


    def save_chart(self):
        """Save the current chart to organised folder structure"""
        try:
            # Check if there's a chart to save
            if not hasattr(self, 'ax') or self.ax is None:
                messagebox.showwarning("No Chart", "No chart available to save. Generate a chart first.")
                return
            
            # Get chart type for filename
            chart_type = self.chart_type_var.get()
            
            # Get symbol and timeframe from current data
            symbol = "UNKNOWN"
            timeframe = "UNKNOWN" 
            algorithm = "UNKNOWN"
            
            # Try to extract from data manager
            if hasattr(self.app, 'data_manager') and self.app.data_manager and hasattr(self.app.data_manager, 'data_info'):
                data_info = self.app.data_manager.data_info
                source_path = data_info.get('source', '')
                
                if source_path:
                    filename = os.path.basename(source_path)
                    
                    # Parse filename: NAS100_TEST_H1_signal_data.csv
                    if filename.endswith('_signal_data.csv'):
                        parts = filename.replace('_signal_data.csv', '').split('_')
                        if len(parts) >= 2:
                            timeframe = parts[-1]  # Last part is timeframe
                            symbol = '_'.join(parts[:-1])  # Everything else is symbol
            
            # Get algorithm from current model
            if hasattr(self.app, 'current_model') and self.app.current_model:
                if hasattr(self.app.current_model, 'algorithm'):
                    algorithm = self.app.current_model.algorithm
                    # Clean algorithm name for folder (remove spaces, special chars)
                    algorithm = algorithm.replace(' ', '_').replace('-', '_')
            
            # Create organised folder structure: plots/symbol/algorithm/timeframe/
            base_dir = "plots"
            full_dir = os.path.join(base_dir, symbol, algorithm, timeframe)
            
            # Create directories if they don't exist
            os.makedirs(full_dir, exist_ok=True)
            
            # Generate filename: ChartType_Algorithm_Symbol_Timeframe.png
            chart_filename = f"{chart_type.replace(' ', '_')}_{algorithm}_{symbol}_{timeframe}.png"
            full_path = os.path.join(full_dir, chart_filename)
            
            # Save the chart
            self.fig.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            # Show success message
            messagebox.showinfo(
                "Chart Saved", 
                f"Chart saved successfully!\n\nLocation: {full_path}\n\n"
                f"Symbol: {symbol}\n"
                f"Algorithm: {algorithm}\n" 
                f"Timeframe: {timeframe}\n"
                f"Chart Type: {chart_type}"
            )
            
            print(f"Chart saved: {full_path}")
            
        except Exception as e:
            error_msg = f"Error saving chart: {str(e)}"
            messagebox.showerror("Save Error", error_msg)
            print(error_msg)
    
    def save_all_charts(self):
        """Generate and save all three chart types automatically"""
        try:
            # Check if we have a trained model
            if not hasattr(self.app, 'current_model') or self.app.current_model is None:
                messagebox.showwarning(
                    "No Model", 
                    "No trained model available.\n\n"
                    "Please train a model first in the 'Model Training' tab."
                )
                return
            
            # Chart types to generate and save
            chart_types = ["Performance Metrics", "Feature Importance", "Confidence Score Analysis"]
            
            # Keep track of original chart type
            original_chart_type = self.chart_type_var.get()
            
            saved_charts = []
            failed_charts = []
            
            # Generate and save each chart type
            for chart_type in chart_types:
                try:
                    # Set the chart type
                    self.chart_type_var.set(chart_type)
                    
                    # Generate the chart (this updates self.fig and self.ax)
                    self.generate_chart()
                    
                    # Get metadata for folder structure
                    symbol = "UNKNOWN"
                    timeframe = "UNKNOWN" 
                    algorithm = "UNKNOWN"
                    
                    # Try to extract from data manager
                    if hasattr(self.app, 'data_manager') and self.app.data_manager and hasattr(self.app.data_manager, 'data_info'):
                        data_info = self.app.data_manager.data_info
                        source_path = data_info.get('source', '')
                        
                        if source_path:
                            filename = os.path.basename(source_path)
                            
                            # Parse filename: NAS100_TEST_H1_signal_data.csv
                            if filename.endswith('_signal_data.csv'):
                                parts = filename.replace('_signal_data.csv', '').split('_')
                                if len(parts) >= 2:
                                    timeframe = parts[-1]  # Last part is timeframe
                                    symbol = '_'.join(parts[:-1])  # Everything else is symbol
                    
                    # Get algorithm from current model
                    if hasattr(self.app, 'current_model') and self.app.current_model:
                        if hasattr(self.app.current_model, 'algorithm'):
                            algorithm = self.app.current_model.algorithm
                            # Clean algorithm name for folder (remove spaces, special chars)
                            algorithm = algorithm.replace(' ', '_').replace('-', '_')
                    
                    # Create organised folder structure: plots/symbol/algorithm/timeframe/
                    base_dir = "plots"
                    full_dir = os.path.join(base_dir, symbol, algorithm, timeframe)
                    
                    # Create directories if they don't exist
                    os.makedirs(full_dir, exist_ok=True)
                    
                    # Generate filename: ChartType_Algorithm_Symbol_Timeframe.png
                    chart_filename = f"{chart_type.replace(' ', '_')}_{algorithm}_{symbol}_{timeframe}.png"
                    full_path = os.path.join(full_dir, chart_filename)
                    
                    # Save the chart
                    self.fig.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
                    
                    saved_charts.append({
                        'type': chart_type,
                        'path': full_path
                    })
                    
                    print(f"Saved {chart_type}: {full_path}")
                    
                except Exception as e:
                    failed_charts.append({
                        'type': chart_type,
                        'error': str(e)
                    })
                    print(f"Failed to save {chart_type}: {str(e)}")
            
            # Restore original chart type and regenerate that chart for display
            self.chart_type_var.set(original_chart_type)
            self.generate_chart()
            
            # Show results message
            if saved_charts and not failed_charts:
                # All charts saved successfully
                chart_list = "\n".join([f"✓ {chart['type']}" for chart in saved_charts])
                messagebox.showinfo(
                    "All Charts Saved", 
                    f"Successfully saved all {len(saved_charts)} charts!\n\n"
                    f"Charts saved:\n{chart_list}\n\n"
                    f"Location: {os.path.join(base_dir, symbol, algorithm, timeframe)}\n\n"
                    f"Symbol: {symbol}\n"
                    f"Algorithm: {algorithm}\n" 
                    f"Timeframe: {timeframe}"
                )
            elif saved_charts and failed_charts:
                # Some charts saved, some failed
                success_list = "\n".join([f"✓ {chart['type']}" for chart in saved_charts])
                fail_list = "\n".join([f"✗ {chart['type']}: {chart['error']}" for chart in failed_charts])
                messagebox.showwarning(
                    "Partial Success", 
                    f"Saved {len(saved_charts)} of {len(chart_types)} charts.\n\n"
                    f"Successful:\n{success_list}\n\n"
                    f"Failed:\n{fail_list}"
                )
            else:
                # All charts failed
                fail_list = "\n".join([f"✗ {chart['type']}: {chart['error']}" for chart in failed_charts])
                messagebox.showerror(
                    "Save Failed", 
                    f"Failed to save any charts.\n\n"
                    f"Errors:\n{fail_list}"
                )
            
        except Exception as e:
            error_msg = f"Error in save_all_charts: {str(e)}"
            messagebox.showerror("Save All Charts Error", error_msg)
            print(error_msg)
    
    def generate_chart(self):
        """Generate visualization chart with better error handling"""
        chart_type = self.chart_type_var.get()
        
        # Clear previous chart completely (including figure-level settings)
        self.fig.clear()
        # Reset figure size to default before creating new chart
        self.fig.set_size_inches(10, 6)  # Reset to default size
        # Create subplot with proper spacing
        self.ax = self.fig.add_subplot(111)
        # Ensure the subplot fills the figure properly
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9)
        
        try:
            if chart_type == "Performance Metrics":
                if hasattr(self.app.current_model, 'performance_metrics') and self.app.current_model.performance_metrics:
                    metrics = self.app.current_model.performance_metrics
                    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                    values = [metrics.get(m, 0) for m in metric_names]
                    
                    bars = self.ax.bar(metric_names, values, color=['#2E8B57', '#4682B4', '#FF6347', '#32CD32', '#9370DB'])
                    self.ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
                    self.ax.set_ylabel('Score', fontsize=12)
                    self.ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                    
                    # Rotate x-axis labels for better readability
                    self.ax.tick_params(axis='x', rotation=45)
                    
                else:
                    self.ax.text(0.5, 0.5, 'No performance metrics available\nTrain a model first', 
                               ha='center', va='center', transform=self.ax.transAxes,
                               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    self.ax.set_title('Performance Metrics - No Data')
                    
            elif chart_type == "Feature Importance":
                # Debug information
                print(f"DEBUG: Current model type: {type(self.app.current_model)}")
                if hasattr(self.app.current_model, 'performance_metrics'):
                    print(f"DEBUG: Performance metrics keys: {self.app.current_model.performance_metrics.keys()}")
                    if 'feature_importance' in self.app.current_model.performance_metrics:
                        fi = self.app.current_model.performance_metrics['feature_importance']
                        print(f"DEBUG: Feature importance type: {type(fi)}")
                        print(f"DEBUG: Feature importance shape: {fi.shape if hasattr(fi, 'shape') else 'No shape'}")
                        print(f"DEBUG: Feature importance columns: {fi.columns.tolist() if hasattr(fi, 'columns') else 'No columns'}")
                else:
                    print("DEBUG: No performance_metrics attribute")
                
                if (hasattr(self.app.current_model, 'performance_metrics') and 
                    'feature_importance' in self.app.current_model.performance_metrics):
                    
                    feature_imp = self.app.current_model.performance_metrics['feature_importance']
                    
                    # Adjust figure size based on number of features
                    num_features = len(feature_imp)
                    fig_height = max(6, min(20, num_features * 0.4))  # Minimum 6, maximum 20
                    self.fig.set_size_inches(10, fig_height)
                    
                    bars = self.ax.barh(feature_imp['feature'], feature_imp['importance'], 
                                      color='#4682B4')
                    self.ax.set_title('Feature Importance (All Features)', fontsize=14, fontweight='bold')
                    self.ax.set_xlabel('Importance Score', fontsize=12)
                    
                    # Add value labels (only if not too many features to avoid clutter)
                    if num_features <= 30:
                        for i, (bar, value) in enumerate(zip(bars, feature_imp['importance'])):
                            width = bar.get_width()
                            self.ax.text(width + max(feature_imp['importance']) * 0.01, bar.get_y() + bar.get_height()/2,
                                       f'{value:.3f}', ha='left', va='center', fontweight='bold', fontsize=8)
                    
                    # Adjust layout based on number of features
                    if num_features > 15:
                        self.ax.tick_params(axis='y', labelsize=7)  # Smaller font for many features
                    else:
                        self.ax.tick_params(axis='y', labelsize=9)
                    
                else:
                    # Try to generate feature importance if model exists but doesn't have it
                    if (hasattr(self.app.current_model, 'feature_columns') and 
                        hasattr(self.app.current_model, 'model') and
                        self.app.current_model.feature_columns and 
                        self.app.current_model.model is not None):
                        
                        try:
                            # Generate placeholder feature importance
                            if hasattr(self.app.current_model.model, 'feature_importances_'):
                                # Use actual feature importances if available
                                feature_importance = pd.DataFrame({
                                    'feature': self.app.current_model.feature_columns,
                                    'importance': self.app.current_model.model.feature_importances_
                                }).sort_values('importance', ascending=False)
                            else:
                                # Create uniform importance for visualization
                                n_features = len(self.app.current_model.feature_columns)
                                uniform_importance = np.ones(n_features) / n_features
                                feature_importance = pd.DataFrame({
                                    'feature': self.app.current_model.feature_columns,
                                    'importance': uniform_importance
                                }).sort_values('importance', ascending=False)
                            
                            # Store it for future use
                            if not hasattr(self.app.current_model, 'performance_metrics'):
                                self.app.current_model.performance_metrics = {}
                            self.app.current_model.performance_metrics['feature_importance'] = feature_importance
                            
                            # Display the chart
                            feature_imp = feature_importance
                            
                            # Adjust figure size based on number of features
                            num_features = len(feature_imp)
                            fig_height = max(6, min(20, num_features * 0.4))  # Minimum 6, maximum 20
                            self.fig.set_size_inches(10, fig_height)
                            
                            bars = self.ax.barh(feature_imp['feature'], feature_imp['importance'], 
                                              color='#4682B4')
                            self.ax.set_title('Feature Importance (All Features)', fontsize=14, fontweight='bold')
                            self.ax.set_xlabel('Importance Score', fontsize=12)
                            
                            # Add value labels (only if not too many features to avoid clutter)
                            if num_features <= 30:
                                for bar, value in zip(bars, feature_imp['importance']):
                                    width = bar.get_width()
                                    self.ax.text(width + max(feature_imp['importance']) * 0.01, 
                                               bar.get_y() + bar.get_height()/2,
                                               f'{value:.3f}', ha='left', va='center', fontweight='bold', fontsize=8)
                            
                            # Adjust layout based on number of features
                            if num_features > 15:
                                self.ax.tick_params(axis='y', labelsize=7)  # Smaller font for many features
                            else:
                                self.ax.tick_params(axis='y', labelsize=9)
                            
                        except Exception as e:
                            print(f"Error generating feature importance: {e}")
                            self.ax.text(0.5, 0.5, 'No feature importance available\nTrain a model first', 
                                       ha='center', va='center', transform=self.ax.transAxes,
                                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                            self.ax.set_title('Feature Importance - No Data')
                    else:
                        self.ax.text(0.5, 0.5, 'No feature importance available\nTrain a model first', 
                                   ha='center', va='center', transform=self.ax.transAxes,
                                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                        self.ax.set_title('Feature Importance - No Data')
            
            elif chart_type == "Confidence Score Analysis":
                self._generate_threshold_analysis()
            
            
            else:
                self.ax.text(0.5, 0.5, f'{chart_type}\nFeature not yet implemented', 
                           ha='center', va='center', transform=self.ax.transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                self.ax.set_title(f'{chart_type} - Coming Soon')
            
            # Adjust layout and refresh - use padding to fill space properly
            self.fig.tight_layout(pad=2.0)
            
            # Ensure canvas updates its size
            self.canvas.get_tk_widget().update()
            self.canvas.draw()
            self.canvas.flush_events()  # Ensure complete refresh
            
        except Exception as e:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Error generating chart:\n{str(e)}\n\nPlease try again or select a different chart type', 
                       ha='center', va='center', transform=self.ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            self.ax.set_title('Chart Generation Error')
            self.canvas.draw()
            
            # Log the error
            print(f"Chart generation error: {e}")
    
    def generate_mqh_model(self):
        """Generate MQH header file for ML model"""
        try:
            # Check if models directory exists
            if not os.path.exists('models'):
                messagebox.showwarning("No Models", 
                    "No models directory found.\n\n"
                    "Please train some models first.")
                return
            
            # Find the latest model file (search recursively in subdirectories)
            model_files = []
            for root, dirs, files in os.walk('models'):
                for file in files:
                    if file.endswith('.pkl'):
                        model_files.append(os.path.join(root, file))
            
            if not model_files:
                messagebox.showwarning("No Models", "No trained models found. Please train a model first.")
                return
            
            # Sort by modification time to get the latest
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_model = model_files[0]
            
            # Auto-detect symbol and timeframe from loaded data
            symbol, timeframe = self._extract_symbol_timeframe()
            
            if symbol == "UNKNOWN" or timeframe == "UNKNOWN":
                messagebox.showwarning("Data Required", 
                    "Cannot detect symbol and timeframe.\n\n"
                    "Please load training data first in the 'Data Management' tab.\n"
                    "File format should be: {SYMBOL}_{TIMEFRAME}_signal_data.csv")
                return
            
            # Generate MQH file directly
            self._generate_mqh_file(latest_model, symbol, timeframe)
            
        except Exception as e:
            error_msg = f"Error generating MQH model: {str(e)}"
            print(error_msg)
            messagebox.showerror("Generation Error", error_msg)
            self.update_status("MQH generation failed", "error")
    
    def _extract_symbol_timeframe(self):
        """Extract symbol and timeframe from loaded data file"""
        # Default values
        symbol = "UNKNOWN"
        timeframe = "UNKNOWN"
        
        try:
            # Check if data is loaded
            if not hasattr(self.app, 'data_manager') or not self.app.data_manager or not hasattr(self.app.data_manager, 'data_info'):
                return symbol, timeframe
                
            # Get data info from data manager
            data_info = self.app.data_manager.data_info
            source_path = data_info.get('source', '')
            
            if not source_path:
                return symbol, timeframe
            
            # Extract symbol and timeframe from filename
            filename = os.path.basename(source_path)
            
            if filename and filename.endswith('_signal_data.csv'):
                # Format: NAS100_TEST_H1_signal_data.csv or GER40test_M5_signal_data.csv
                parts = filename.replace('_signal_data.csv', '').split('_')
                if len(parts) >= 2:
                    # Last part is timeframe, everything else is symbol
                    timeframe = parts[-1]
                    symbol = '_'.join(parts[:-1])
                    
                    # Clean up symbol (remove common suffixes like 'test', 'TEST')
                    if symbol.lower().endswith('_test'):
                        symbol = symbol[:-5]  # Remove '_test'
                    elif symbol.lower().endswith('test'):
                        symbol = symbol[:-4]  # Remove 'test'
                        
                    # Ensure symbol is uppercase
                    symbol = symbol.upper()
            
            print(f"Auto-detected symbol: {symbol}, timeframe: {timeframe}")
            
        except Exception as e:
            print(f"Error extracting symbol/timeframe: {e}")
            
        return symbol, timeframe
    
    def _show_mqh_settings_dialog(self, model_path):
        """Show dialog for MQH generation settings"""
        # Extract default symbol and timeframe
        symbol = "EURUSD"
        timeframe = "H1"
        
        # Try to get from current data file name
        if hasattr(self, 'current_file') and self.current_file:
            import re
            match = re.match(r'([A-Z]+)_([HMD]\d+)_', os.path.basename(self.current_file))
            if match:
                symbol = match.group(1)
                timeframe = match.group(2)
        
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("MQH Model Generation Settings")
        dialog.geometry("350x200")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Symbol input
        tk.Label(dialog, text="Symbol:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        symbol_var = tk.StringVar(value=symbol)
        tk.Entry(dialog, textvariable=symbol_var, width=15).grid(row=0, column=1, padx=10, pady=10)
        
        # Timeframe input
        tk.Label(dialog, text="Timeframe:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        timeframe_var = tk.StringVar(value=timeframe)
        timeframe_combo = ttk.Combobox(dialog, textvariable=timeframe_var,
                                      values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                                      state="readonly", width=12)
        timeframe_combo.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        
        # Model info
        model_name = os.path.basename(model_path)
        tk.Label(dialog, text="Model:", font=('Arial', 8)).grid(row=2, column=0, padx=10, pady=5, sticky='e')
        tk.Label(dialog, text=model_name[:30], font=('Arial', 8), foreground='gray').grid(row=2, column=1, padx=10, pady=5, sticky='w')
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        def on_generate():
            sym = symbol_var.get().upper()
            tf = timeframe_var.get()
            
            if not sym or not tf:
                messagebox.showwarning("Input Required", "Please enter both symbol and timeframe")
                return
                
            dialog.destroy()
            self._generate_mqh_file(model_path, sym, tf)
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(button_frame, text="Generate", command=on_generate).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side='left', padx=5)
        
        # Focus on dialog
        dialog.focus_set()
    
    def _generate_mqh_file(self, model_path, symbol, timeframe):
        """Generate the MQH file in background thread"""
        try:
            self.update_status("Generating MQH model...", "info")
            self.add_progress_message(f"Generating MQH for {symbol} {timeframe}...")
            self.root.update()
            
            # Run in background thread
            import threading
            thread = threading.Thread(target=self._mqh_generation_worker, 
                                    args=(model_path, symbol, timeframe), 
                                    daemon=True)
            thread.start()
            
        except Exception as e:
            self._show_mqh_error(str(e))
    
    def _mqh_generation_worker(self, model_path, symbol, timeframe):
        """Background worker for MQH generation"""
        try:
            from src.utils.mqh_generator import MQHGenerator
            
            # Create generator
            generator = MQHGenerator(model_path, symbol, timeframe)
            
            # Load model
            if not generator.load_model():
                error_msg = "Failed to load model data"
                self.root.after_idle(self._show_mqh_error, error_msg)
                return
            
            # Generate and save MQH file
            output_path = generator.save_mqh_file("header_files")
            
            # Process results on main thread
            self.root.after_idle(self._process_mqh_results, output_path, generator.algorithm, symbol, timeframe)
            
        except Exception as e:
            error_msg = f"Error generating MQH: {str(e)}"
            self.root.after_idle(self._show_mqh_error, error_msg)
    
    def _process_mqh_results(self, output_path, algorithm, symbol, timeframe):
        """Process MQH generation results on main GUI thread"""
        try:
            self.add_progress_message(f"MQH file generated successfully!")
            
            filename = os.path.basename(output_path)
            
            success_msg = f"MQH model header generated successfully!\n\n"
            success_msg += f"File: {filename}\n"
            success_msg += f"Location: header_files/\n\n"
            success_msg += f"Algorithm: {algorithm}\n"
            success_msg += f"Symbol: {symbol} (auto-detected)\n"
            success_msg += f"Timeframe: {timeframe} (auto-detected)\n\n"
            success_msg += "To use in MetaTrader 5:\n"
            success_msg += "1. Copy the .mqh file to MQL5/Include/\n"
            success_msg += "2. In ML_FF_AI EA settings, set:\n"
            success_msg += f'   MLModelHeader = "{filename}"'
            
            messagebox.showinfo("Generation Complete", success_msg)
            self.update_status("MQH model generated successfully", "success")
            
        except Exception as e:
            self._show_mqh_error(str(e))
    
    def _show_mql5_error(self, error_msg):
        """Show MQL5 generation error on main GUI thread"""
        self.add_progress_message(f"MQL5 Error: {error_msg}")
        messagebox.showerror("Generation Error", f"Error generating MQL5 rules: {error_msg}")
        self.update_status("MQL5 rule generation failed", "error")
    
    def _show_mqh_error(self, error_msg):
        """Show MQH generation error on main GUI thread"""
        self.add_progress_message(f"MQH Error: {error_msg}")
        messagebox.showerror("Generation Error", f"Error generating MQH header: {error_msg}")
        self.update_status("MQH generation failed", "error")
    
