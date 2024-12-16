import subprocess
from concurrent.futures import ThreadPoolExecutor

# Define the test commands
test_commands = [
    ["python", "-m", "unittest", "-v", "./Tests/hypergraph_constructors.py"],
    ["python", "-m", "unittest", "-v", "./Tests/control_can.py"]
]

# Function to run a test command
def run_test(command):
    subprocess.run(command)

# Run the tests simultaneously using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    executor.map(run_test, test_commands)
