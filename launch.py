"""
HYBRIDVFL Dashboard & Training Launcher
"""

import subprocess
import time
import os
import sys
import signal
import argparse
from threading import Thread

class HybridVFLLauncher:
    def __init__(self):
        self.dashboard_process = None
        self.training_process = None
        self.running = True

    def launch_dashboard(self):
        """Launch the dashboard server"""
        try:
            print("Starting dashboard server...")
            self.dashboard_process = subprocess.Popen(
                [sys.executable, 'dashboard.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            return True
        except Exception as e:
            print(f"Failed to start dashboard: {e}")
            return False

    def launch_training(self, delay=5):
        """Launch the training process after delay"""
        if delay > 0:
            print(f"Waiting {delay} seconds for dashboard to initialize...")
            time.sleep(delay)
        
        if not os.path.exists('main.py'):
            print("Error: main.py not found in current directory")
            return False
        
        try:
            print("Starting training process...")
            self.training_process = subprocess.Popen(
                [sys.executable, 'main.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream training output
            while self.running and self.training_process.poll() is None:
                line = self.training_process.stdout.readline()
                if line:
                    print(f"[TRAINING] {line.rstrip()}")
            
            if self.training_process.returncode == 0:
                print("Training completed successfully!")
            elif self.training_process.returncode is not None:
                print(f"Training ended with code: {self.training_process.returncode}")
                
            return True
            
        except Exception as e:
            print(f"Failed to start training: {e}")
            return False

    def cleanup(self):
        """Clean up all processes"""
        self.running = False
        print("\nShutting down processes...")
        
        if self.training_process and self.training_process.poll() is None:
            print("  Stopping training...")
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.training_process.kill()
        
        if self.dashboard_process and self.dashboard_process.poll() is None:
            print("  Stopping dashboard...")
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
        
        print("Cleanup complete")

    def run(self, auto_train=True, train_delay=5):
        """Main launcher logic"""
        try:
            # Start dashboard
            if not self.launch_dashboard():
                return False
            
            # Start training automatically or wait for manual start
            if auto_train:
                training_thread = Thread(
                    target=self.launch_training, 
                    args=(train_delay,), 
                    daemon=True
                )
                training_thread.start()
            else:
                print("Dashboard ready. Run 'python main.py' manually when ready.")
            
            # Keep dashboard running
            try:
                self.dashboard_process.wait()
            except KeyboardInterrupt:
                pass
                
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(
        description="Launch HYBRIDVFL Dashboard and Training"
    )
    parser.add_argument(
        '--no-auto-train', 
        action='store_true',
        help="Start dashboard only, don't auto-launch training"
    )
    parser.add_argument(
        '--delay', 
        type=int, 
        default=5,
        help="Seconds to wait before starting training (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("HYBRIDVFL Launcher")
    print("=" * 40)
    print("Dashboard: http://localhost:5050")
    if not args.no_auto_train:
        print(f"Training: Auto-start in {args.delay}s")
    else:
        print("Training: Manual start")
    print("=" * 40)
    
    # Ctrl+C gracefully
    launcher = HybridVFLLauncher()
    
    def signal_handler(signum, frame):
        launcher.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run launcher
    launcher.run(
        auto_train=not args.no_auto_train,
        train_delay=args.delay
    )

if __name__ == '__main__':
    main() 
