#!/usr/bin/env python3
"""
Comprehensive Test Suite for All PyTorch GPU Models
Tests all 6 models to ensure they work properly
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class ModelTester:
    def __init__(self):
        self.base_dir = Path("/home/ganesh/pytorch_GPU")
        self.results = {}
        self.passed = 0
        self.failed = 0
        
        print("🧪 PyTorch GPU Models Test Suite")
        print("=" * 60)
        print(f"Base directory: {self.base_dir}")
        print(f"Testing all models for basic functionality...")
        print("=" * 60)
    
    def run_test(self, model_name, script_path, timeout=120):
        """Run a single model test"""
        print(f"\n🔄 Testing {model_name}...")
        print(f"   Script: {script_path}")
        
        start_time = time.time()
        
        try:
            # Change to the model directory
            model_dir = script_path.parent
            original_dir = os.getcwd()
            os.chdir(model_dir)
            
            # Run the script with timeout

            # Copy current environment and add LD_PRELOAD
            env = os.environ.copy()
            env["LD_PRELOAD"] = "/home/ganesh/nixnan.so"  

            # Run the subprocess with the modified environment
            subprocess.run([sys.executable, str(script_path.resolve())], env=env)

            
            print("---> Running :", script_path.name, " with LD_PRELOAD set.")
            result = subprocess.run(
                [sys.executable, script_path.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            print("Run of :", script_path.name, " finished <---")
            
            # Change back to original directory
            os.chdir(original_dir)
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"   ✅ PASSED ({elapsed_time:.1f}s)")
                self.results[model_name] = {
                    'status': 'PASSED',
                    'time': elapsed_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                self.passed += 1
                return True
            else:
                print(f"   ❌ FAILED ({elapsed_time:.1f}s)")
                print(f"   Error: {result.stderr}")
                self.results[model_name] = {
                    'status': 'FAILED',
                    'time': elapsed_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
                self.failed += 1
                return False
                
        except subprocess.TimeoutExpired:
            os.chdir(original_dir)
            print(f"   ⏰ TIMEOUT ({timeout}s)")
            self.results[model_name] = {
                'status': 'TIMEOUT',
                'time': timeout,
                'error': 'Test exceeded timeout limit'
            }
            self.failed += 1
            return False
            
        except FileNotFoundError:
            print(f"   📁 FILE NOT FOUND")
            self.results[model_name] = {
                'status': 'FILE_NOT_FOUND',
                'time': 0,
                'error': f'Script not found: {script_path}'
            }
            self.failed += 1
            return False
            
        except Exception as e:
            os.chdir(original_dir)
            print(f"   ❌ ERROR: {e}")
            self.results[model_name] = {
                'status': 'ERROR',
                'time': time.time() - start_time,
                'error': str(e)
            }
            self.failed += 1
            return False
    
    def test_all_models(self):
        """Test all models"""
        
        # Define all models and their test scripts
        models = [
            {
                'name': 'Simple ResNet',
                'script': self.base_dir / 'simple_resnet' / 'inference_fixed.py',
                'timeout': 180
            },
            {
                'name': 'Simple GAN', 
                'script': self.base_dir / 'simple_gan' / 'simple_gan.py',
                'timeout': 300
            },
            {
                'name': 'Autoencoder',
                'script': self.base_dir / 'autoencoder' / 'autoencoder.py', 
                'timeout': 180
            },
            {
                'name': 'CNN Autoencoder',
                'script': self.base_dir / 'autoencoder' / 'cnn_autoencoder.py',
                'timeout': 300
            },
            {
                'name': 'Vision Transformer',
                'script': self.base_dir / 'vision_transformer' / 'vision_transformer.py',
                'timeout': 600
            },
            {
                'name': 'Stock LSTM',
                'script': self.base_dir / 'stock_lstm' / 'stock_lstm.py',
                'timeout': 300
            }
        ]
        
        total_start_time = time.time()
        
        # Test each model
        for model in models:
            if model['script'].exists():
                self.run_test(
                    model['name'],
                    model['script'], 
                    model['timeout']
                )
            else:
                print(f"\n⚠️  {model['name']} script not found: {model['script']}")
                self.results[model['name']] = {
                    'status': 'NOT_FOUND',
                    'time': 0,
                    'error': f"Script not found: {model['script']}"
                }
                self.failed += 1
        
        total_time = time.time() - total_start_time
        
        # Print summary
        self.print_summary(total_time)
    
    def print_summary(self, total_time):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = self.passed + self.failed
        pass_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print(f"Total Time: {total_time:.1f}s")
        
        print("\n📊 DETAILED RESULTS:")
        print("-" * 60)
        
        for model_name, result in self.results.items():
            status = result['status']
            time_taken = result.get('time', 0)
            
            if status == 'PASSED':
                icon = "✅"
            elif status == 'FAILED':
                icon = "❌"
            elif status == 'TIMEOUT':
                icon = "⏰"
            elif status == 'NOT_FOUND':
                icon = "📁"
            else:
                icon = "❓"
            
            print(f"{icon} {model_name:<20} {status:<12} ({time_taken:.1f}s)")
            
            # Show error details for failed tests
            if status in ['FAILED', 'ERROR'] and 'stderr' in result:
                stderr_lines = result['stderr'].strip().split('\n')
                if stderr_lines and stderr_lines[0]:
                    error_preview = stderr_lines[-1][:80] + "..." if len(stderr_lines[-1]) > 80 else stderr_lines[-1]
                    print(f"   └─ Error: {error_preview}")
        
        print("\n" + "=" * 60)
        
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED! All models are working correctly.")
        else:
            print(f"⚠️  {self.failed} test(s) failed. Check the details above.")
            print("\nTroubleshooting tips:")
            print("- Make sure all required packages are installed")
            print("- Check if models have been trained (some may need training first)")
            print("- Verify GPU is available and has sufficient memory")
            print("- Check file paths and permissions")
        
        return self.failed == 0
    
    def create_detailed_report(self):
        """Create a detailed test report file"""
        report_path = self.base_dir / "test_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("PyTorch GPU Models Test Report\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Time: {result.get('time', 0):.1f}s\n")
                
                if 'stdout' in result:
                    f.write(f"Output:\n{result['stdout']}\n")
                
                if 'stderr' in result:
                    f.write(f"Errors:\n{result['stderr']}\n")
                
                f.write("-" * 30 + "\n\n")
        
        print(f"📄 Detailed report saved to: {report_path}")

def main():
    """Main test function"""
    try:
        tester = ModelTester()
        tester.test_all_models()
        tester.create_detailed_report()
        
        # Exit with appropriate code
        sys.exit(0 if tester.failed == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
