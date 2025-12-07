"""
AI Answer Aggregator - GUI Version
A graphical user interface for the AI Answer Aggregator program.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
from datetime import datetime
import os

# Import the core functionality from the main module
try:
    from ai_aggregator import (
        query_all_models, 
        aggregate_answers, 
        AI_MODELS,
        save_results_to_file
    )
except ImportError:
    messagebox.showerror("Error", "Could not import ai_aggregator.py. Make sure it's in the same directory.")
    exit(1)


class AIAggregatorGUI:
    def __init__(self, root):
        try:
            self.root = root
            self.root.title("AI Answer Aggregator")
            self.root.geometry("1000x700")
            self.root.configure(bg="#f0f0f0")
            
            # Prevent window from closing unexpectedly
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Variables
            self.is_processing = False
            self.responses = []
            self.final_answer = ""
            self.final_answer_dict = {}
            
            self.setup_ui()
        except Exception as e:
            import traceback
            error_msg = f"Failed to initialize GUI:\n{str(e)}\n\n{traceback.format_exc()}"
            try:
                messagebox.showerror("Initialization Error", error_msg)
            except:
                print(error_msg)
            raise
    
    def on_closing(self):
        """Handle window closing gracefully."""
        if self.is_processing:
            if not messagebox.askokcancel("Quit", "A query is in progress. Do you want to quit anyway?"):
                return
        self.root.destroy()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=10)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame,
            text="🤖 AI Answer Aggregator",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Query multiple AI models and get a synthesized answer",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        subtitle_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Question input section
        question_frame = tk.LabelFrame(
            main_frame,
            text="Enter Your Question",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            padx=10,
            pady=10
        )
        question_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.question_text = scrolledtext.ScrolledText(
            question_frame,
            height=4,
            font=("Arial", 11),
            wrap=tk.WORD
        )
        self.question_text.pack(fill=tk.X)
        
        # Model selection section
        model_frame = tk.LabelFrame(
            main_frame,
            text="Select AI Models (Default: All)",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            padx=10,
            pady=10
        )
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_vars = {}
        models_container = tk.Frame(model_frame, bg="#f0f0f0")
        models_container.pack(fill=tk.X)
        
        # Create checkboxes for each model
        cols = 3
        row, col = 0, 0
        for model_name in AI_MODELS.keys():
            var = tk.BooleanVar(value=True)  # All selected by default
            self.model_vars[model_name] = var
            
            checkbox = tk.Checkbutton(
                models_container,
                text=model_name.replace("_", " ").title(),
                variable=var,
                font=("Arial", 10),
                bg="#f0f0f0"
            )
            checkbox.grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
            
            col += 1
            if col >= cols:
                col = 0
                row += 1
        
        # Options frame
        options_frame = tk.Frame(main_frame, bg="#f0f0f0")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.use_cache_var = tk.BooleanVar(value=True)
        cache_check = tk.Checkbutton(
            options_frame,
            text="Use Response Cache",
            variable=self.use_cache_var,
            font=("Arial", 10),
            bg="#f0f0f0"
        )
        cache_check.pack(side=tk.LEFT, padx=10)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.query_button = tk.Button(
            button_frame,
            text="🚀 Query AI Models",
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            command=self.start_query,
            cursor="hand2"
        )
        self.query_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(
            button_frame,
            text="💾 Save Results",
            font=("Arial", 12),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            command=self.save_results,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(
            button_frame,
            text="🗑️ Clear",
            font=("Arial", 12),
            bg="#e74c3c",
            fg="white",
            padx=20,
            pady=10,
            command=self.clear_all,
            cursor="hand2"
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(
            main_frame,
            text="Ready to query",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#7f8c8d"
        )
        self.status_label.pack()
        
        # Results section
        results_frame = tk.LabelFrame(
            main_frame,
            text="Results",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Final Answer tab
        final_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(final_tab, text="Final Answer")
        
        self.final_answer_text = scrolledtext.ScrolledText(
            final_tab,
            font=("Arial", 11),
            wrap=tk.WORD,
            bg="white",
            fg="#2c3e50"
        )
        self.final_answer_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Individual Responses tab
        individual_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(individual_tab, text="Individual Responses")
        
        self.individual_responses_text = scrolledtext.ScrolledText(
            individual_tab,
            font=("Arial", 10),
            wrap=tk.WORD,
            bg="white",
            fg="#2c3e50"
        )
        self.individual_responses_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def get_selected_models(self):
        """Get list of selected model names."""
        return [name for name, var in self.model_vars.items() if var.get()]
    
    def start_query(self):
        """Start the query process in a separate thread."""
        if self.is_processing:
            messagebox.showwarning("Warning", "A query is already in progress.")
            return
        
        question = self.question_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("Warning", "Please enter a question.")
            return
        
        selected_models = self.get_selected_models()
        if not selected_models:
            messagebox.showwarning("Warning", "Please select at least one AI model.")
            return
        
        # Disable button and start progress
        self.query_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.is_processing = True
        self.progress.start()
        self.status_label.config(text="Querying AI models...", fg="#3498db")
        
        # Clear previous results
        self.final_answer_text.delete("1.0", tk.END)
        self.individual_responses_text.delete("1.0", tk.END)
        
        # Start query in separate thread
        thread = threading.Thread(
            target=self.query_models,
            args=(question, selected_models),
            daemon=True
        )
        thread.start()
    
    def query_models(self, question, selected_models):
        """Query the AI models (runs in separate thread)."""
        try:
            import traceback
            from ai_aggregator import AI_MODELS as ORIGINAL_MODELS
            
            # Filter to only selected models
            models_to_query = {k: v for k, v in ORIGINAL_MODELS.items() if k in selected_models}
            
            if not models_to_query:
                self.root.after(0, lambda: self.show_error("No valid models selected"))
                return
            
            # Temporarily modify AI_MODELS in the module
            import ai_aggregator
            original_ai_models = ai_aggregator.AI_MODELS
            ai_aggregator.AI_MODELS = models_to_query
            
            try:
                # Query models
                self.responses = query_all_models(question, use_cache=self.use_cache_var.get())
                
                # Filter responses to only selected models (safety check)
                self.responses = [r for r in self.responses if r.get("model_name") in selected_models]
                
                # Aggregate answers
                result = aggregate_answers(self.responses, question)
                # Handle both dict and string returns
                if isinstance(result, dict):
                    self.final_answer = result.get("final_answer", str(result))
                    self.final_answer_dict = result
                else:
                    self.final_answer = str(result)
                    self.final_answer_dict = {"final_answer": self.final_answer}
                
                # Update UI in main thread
                self.root.after(0, self.display_results)
            finally:
                # Restore original AI_MODELS
                ai_aggregator.AI_MODELS = original_ai_models
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self.show_error(error_msg))
    
    def display_results(self):
        """Display the results in the UI."""
        # Stop progress
        self.progress.stop()
        self.is_processing = False
        self.query_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.status_label.config(text="Query completed successfully!", fg="#27ae60")
        
        # Display final answer
        self.final_answer_text.delete("1.0", tk.END)
        self.final_answer_text.insert("1.0", self.final_answer)
        
        # Display individual responses
        self.individual_responses_text.delete("1.0", tk.END)
        for resp in self.responses:
            model_name = resp.get("model_name", "Unknown")
            response = resp.get("response", "No response")
            error = resp.get("error")
            
            self.individual_responses_text.insert(tk.END, f"\n{'='*60}\n")
            self.individual_responses_text.insert(tk.END, f"Model: {model_name}\n", "bold")
            self.individual_responses_text.insert(tk.END, f"{'='*60}\n\n")
            
            if error:
                self.individual_responses_text.insert(tk.END, f"❌ Error: {error}\n\n", "error")
            else:
                self.individual_responses_text.insert(tk.END, f"{response}\n\n")
        
        # Configure text tags for styling
        self.individual_responses_text.tag_config("bold", font=("Arial", 10, "bold"))
        self.individual_responses_text.tag_config("error", foreground="red")
        
        # Switch to final answer tab
        self.notebook.select(0)
        
        messagebox.showinfo("Success", "Query completed! Check the Results section.")
    
    def show_error(self, error_msg):
        """Show error message."""
        try:
            self.progress.stop()
        except:
            pass
        self.is_processing = False
        self.query_button.config(state=tk.NORMAL)
        # Truncate long error messages for status label
        short_msg = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
        self.status_label.config(text=f"Error: {short_msg}", fg="#e74c3c")
        # Show full error in message box
        messagebox.showerror("Error", f"An error occurred:\n\n{error_msg}")
    
    def save_results(self):
        """Save results to a file."""
        if not self.final_answer:
            messagebox.showwarning("Warning", "No results to save. Please run a query first.")
            return
        
        question = self.question_text.get("1.0", tk.END).strip()
        try:
            # Use the dict format if available, otherwise create one
            final_answer_data = getattr(self, 'final_answer_dict', {"final_answer": self.final_answer})
            filename = save_results_to_file(question, self.responses, final_answer_data)
            messagebox.showinfo("Success", f"Results saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
    
    def clear_all(self):
        """Clear all inputs and results."""
        if self.is_processing:
            messagebox.showwarning("Warning", "Cannot clear while query is in progress.")
            return
        
        self.question_text.delete("1.0", tk.END)
        self.final_answer_text.delete("1.0", tk.END)
        self.individual_responses_text.delete("1.0", tk.END)
        self.responses = []
        self.final_answer = ""
        self.final_answer_dict = {}
        self.save_button.config(state=tk.DISABLED)
        self.status_label.config(text="Ready to query", fg="#7f8c8d")
    
    def on_closing(self):
        """Handle window closing gracefully."""
        if self.is_processing:
            if not messagebox.askokcancel("Quit", "A query is in progress. Do you want to quit anyway?"):
                return
        self.root.destroy()


def main():
    """Launch the GUI application."""
    try:
        root = tk.Tk()
        app = AIAggregatorGUI(root)
        root.mainloop()
    except Exception as e:
        import traceback
        error_msg = f"Failed to start GUI:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        try:
            messagebox.showerror("Fatal Error", error_msg)
        except:
            pass


if __name__ == "__main__":
    main()

